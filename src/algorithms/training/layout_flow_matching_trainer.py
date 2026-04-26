"""Trainer for bbox-conditioned pixel-space flow matching."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.algorithms.inference.layout_flow_matching_sampler import LayoutFlowMatchingSampler
from src.algorithms.training.flow_matching_trainer import (
    FlowMatchingTrainer,
    _resolve_unet_sample_size,
)
from src.core.ot import build_layout_descriptor, match_target_batch, pairwise_mean_squared_cost
from src.core.training_utils import (
    EMAState,
    autocast_context,
    build_grad_scaler,
    build_scheduler,
    grad_norm,
    move_optimizer_state_to_device,
    resolve_precision_settings,
)
from src.core.visualization.layout_debug import (
    draw_bbox_overlays,
    draw_mask_overlays,
    ensure_rgb,
    make_side_by_side_panel,
    normalize_feature_map,
    render_mask_composite,
    render_class_layout,
    save_image_batch,
)
from src.models.fm_unet import load_unet_config, save_unet_config
from src.models.layout_conditioned_unet import (
    build_layout_conditioned_pixel_unet,
    save_layout_conditioning_metadata,
)
from src.models.stay_layout_conditioned_unet import build_stay_layout_conditioned_pixel_unet


class LayoutFMTrainer(FlowMatchingTrainer):
    """Pixel-space FM trainer for bbox/layout conditioned generation."""

    def __init__(
        self,
        unet,
        *,
        layout_config,
        device: Optional[str] = None,
        t_scale: float = 1.0,
        train_target: str = "v",
        model_dir: str = "./artifacts/checkpoints/flow_matching/layout_fm/",
        from_norm_to_display=None,
        unet_config: Optional[Dict[str, Any]] = None,
        vae=None,
        vae_config: Optional[Dict[str, Any]] = None,
        path_mode: str = "independent",
        path_solver: str = "hungarian",
        layout_cost_resolution: int = 16,
        condition_weight: float = 1.0,
    ) -> None:
        super().__init__(
            unet,
            device=device,
            t_scale=t_scale,
            train_target=train_target,
            model_dir=model_dir,
            from_norm_to_display=from_norm_to_display,
            unet_config=unet_config,
            vae=vae,
            vae_config=vae_config,
            conditioner=None,
            path_mode=path_mode,
            path_solver=path_solver,
            layout_cost_resolution=layout_cost_resolution,
            condition_weight=condition_weight,
        )
        self.layout_config = layout_config
        self._last_loss_components: Dict[str, float] = {}

    @classmethod
    def from_config(
        cls,
        config,
        *,
        from_norm_to_display=None,
    ) -> "LayoutFMTrainer":
        from src.models.vae import build_vae_from_config, resolve_vae_config_from_model_config

        device = config.resolved_device() if hasattr(config, "resolved_device") else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        num_classes = int(config.layout_conditioning.num_classes or 0)
        if num_classes <= 0:
            raise ValueError(
                "layout_conditioning.num_classes must be set before building LayoutFMTrainer"
            )

        base_unet_cfg = load_unet_config(config.model.unet_config)
        effective_unet_cfg = dict(base_unet_cfg)
        image_in_channels = int(base_unet_cfg.get("in_channels", 1))
        variant = str(getattr(config.layout_conditioning, "variant", "raster_v1"))
        vae_cfg = None
        vae = None
        if (
            getattr(config.model, "vae_config", None) is not None
            or getattr(config.model, "vae_pretrained_model_name_or_path", None) is not None
        ):
            vae_cfg = resolve_vae_config_from_model_config(config.model)
            effective_unet_cfg["sample_size"] = _resolve_unet_sample_size(config, vae_cfg)
            effective_unet_cfg["in_channels"] = int(vae_cfg.get("latent_channels", base_unet_cfg.get("in_channels", 4)))
            effective_unet_cfg["out_channels"] = int(vae_cfg.get("latent_channels", base_unet_cfg.get("out_channels", 4)))
            vae = build_vae_from_config(vae_cfg, device=device)
        else:
            effective_unet_cfg["sample_size"] = int(config.data.image_size)
            effective_unet_cfg["out_channels"] = image_in_channels

        if variant == "stay_v2":
            wrapped_unet = build_stay_layout_conditioned_pixel_unet(
                effective_unet_cfg,
                image_in_channels=int(effective_unet_cfg["in_channels"]),
                num_classes=num_classes,
                class_embed_dim=int(config.layout_conditioning.class_embed_dim),
                bbox_embed_dim=int(config.layout_conditioning.bbox_embed_dim),
                object_embed_dim=int(config.layout_conditioning.object_embed_dim),
                use_style_latent=bool(config.layout_conditioning.use_style_latent),
                style_latent_dim=int(config.layout_conditioning.style_latent_dim),
                style_seed=int(config.layout_conditioning.style_seed),
                mask_resolution=int(config.layout_conditioning.mask_resolution),
                mask_hidden_channels=int(config.layout_conditioning.mask_hidden_channels),
                mask_threshold=float(config.layout_conditioning.mask_threshold),
                edge_dilation=int(config.layout_conditioning.edge_dilation),
                injection_mode=str(config.layout_conditioning.injection_mode),
                use_masked_context=bool(config.layout_conditioning.use_masked_context),
                mask_overlap_loss_weight=float(config.layout_conditioning.mask_overlap_loss_weight),
                mask_sharpness_loss_weight=float(config.layout_conditioning.mask_sharpness_loss_weight),
                mask_activation_loss_weight=float(config.layout_conditioning.mask_activation_loss_weight),
                category_id_to_name=getattr(config.layout_conditioning, "category_id_to_name", {}),
                device=device,
            )
        else:
            effective_unet_cfg["in_channels"] = int(effective_unet_cfg["in_channels"]) + int(config.layout_conditioning.spatial_channels)
            wrapped_unet = build_layout_conditioned_pixel_unet(
                effective_unet_cfg,
                image_in_channels=int(base_unet_cfg.get("in_channels", 1) if vae is None else effective_unet_cfg["out_channels"]),
                num_classes=num_classes,
                class_embed_dim=int(config.layout_conditioning.class_embed_dim),
                bbox_embed_dim=int(config.layout_conditioning.bbox_embed_dim),
                spatial_channels=int(config.layout_conditioning.spatial_channels),
                raster_mode=str(config.layout_conditioning.raster_mode),
                category_id_to_name=getattr(config.layout_conditioning, "category_id_to_name", {}),
                device=device,
            )

        return cls(
            wrapped_unet,
            layout_config=config.layout_conditioning,
            device=device,
            t_scale=config.training.t_scale,
            train_target=config.training.train_target,
            model_dir=config.output.model_dir,
            from_norm_to_display=from_norm_to_display,
            unet_config=effective_unet_cfg,
            vae=vae,
            vae_config=vae_cfg,
            path_mode=getattr(config.path, "mode", "independent"),
            path_solver=getattr(config.path, "solver", "hungarian"),
            layout_cost_resolution=getattr(config.path, "layout_cost_resolution", 16),
            condition_weight=getattr(config.path, "condition_weight", 1.0),
        )

    def train_from_config(
        self,
        config,
        dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> None:
        self.train(
            dataloader=dataloader,
            epochs=config.training.epochs,
            eval_dataloader=eval_dataloader,
            pretrained_vae_path=config.model.vae_weights,
            pretrained_unet_path=config.model.pretrained_unet_path,
            strict_load=config.training.strict_load,
            log_dir=config.output.resolved_log_dir(),
            debug_dir=config.output.resolved_debug_dir(),
            lr=config.resolved_lr() if hasattr(config, "resolved_lr") else config.training.lr,
            sample_every=config.sampling.sample_every,
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            sample_steps=config.sampling.sample_steps,
            sample_shape=config.sampling.sample_shape,
            save_every_n_epochs=config.training.save_every_n_epochs,
            eval_every=getattr(config.training, "eval_every", 1),
            resume_from_checkpoint=config.output.resume,
            scalar_every_steps=config.logging.scalar_every_steps,
            image_every_steps=config.logging.image_every_steps,
            max_logged_images=config.logging.max_logged_images,
            fixed_validation_examples=config.sampling.fixed_validation_examples,
            sample_every_steps=config.sampling.sample_every_steps,
            save_debug_images=config.sampling.save_debug_images,
            log_internal_maps=config.layout_conditioning.log_internal_maps,
            optimizer_name=getattr(config.optimizer, "name", "adamw"),
            weight_decay=getattr(config.optimizer, "weight_decay", 0.01),
            beta1=getattr(config.optimizer, "beta1", 0.9),
            beta2=getattr(config.optimizer, "beta2", 0.999),
            scheduler_name=getattr(config.scheduler, "name", "warmup_cosine"),
            warmup_ratio=getattr(config.scheduler, "warmup_ratio", 0.05),
            min_lr_ratio=getattr(config.scheduler, "min_lr_ratio", 0.1),
            ema_enabled=getattr(config.ema, "enabled", True),
            ema_decay=getattr(config.ema, "decay", 0.999),
            ema_start_step=getattr(config.ema, "start_step", 100),
            mixed_precision=getattr(config.precision, "mixed_precision", "auto"),
            max_grad_norm=getattr(config.training, "max_grad_norm", 1.0),
        )

    def _save_configs(self) -> None:
        if self.unet_config is not None:
            save_unet_config(self.unet_config, os.path.join(self._unet_dir(), "config.json"))
        if self.vae_config is not None:
            os.makedirs(self._vae_dir(), exist_ok=True)
            import json
            with open(os.path.join(self._vae_dir(), "config.json"), "w", encoding="utf-8") as handle:
                json.dump(self.vae_config, handle, indent=2, sort_keys=True)

        layout_metadata = {
            "variant": str(getattr(self.layout_config, "variant", "raster_v1")),
            "num_classes": int(self.unet.num_classes),
            "class_embed_dim": int(self.unet.class_embed_dim),
            "bbox_embed_dim": int(self.unet.bbox_embed_dim),
            "image_in_channels": int(self.unet.image_in_channels),
            "category_id_to_name": {
                str(key): value for key, value in self.unet.category_id_to_name.items()
            },
        }
        if hasattr(self.unet, "spatial_channels"):
            layout_metadata["spatial_channels"] = int(self.unet.spatial_channels)
            layout_metadata["raster_mode"] = str(self.unet.raster_mode)
        if hasattr(self.unet, "object_embed_dim"):
            layout_metadata.update(
                {
                    "object_embed_dim": int(self.unet.object_embed_dim),
                    "use_style_latent": bool(self.unet.use_style_latent),
                    "style_latent_dim": int(self.unet.style_latent_dim),
                    "style_seed": int(self.unet.style_seed),
                    "mask_resolution": int(self.unet.mask_resolution),
                    "mask_hidden_channels": int(self.unet.mask_hidden_channels),
                    "mask_threshold": float(self.unet.mask_threshold),
                    "edge_dilation": int(self.unet.edge_dilation),
                    "injection_mode": str(self.unet.injection_mode),
                    "use_masked_context": bool(self.unet.use_masked_context),
                    "mask_overlap_loss_weight": float(self.unet.mask_overlap_loss_weight),
                    "mask_sharpness_loss_weight": float(self.unet.mask_sharpness_loss_weight),
                    "mask_activation_loss_weight": float(self.unet.mask_activation_loss_weight),
                }
            )
        save_layout_conditioning_metadata(
            layout_metadata,
            os.path.join(self.model_dir, "layout_conditioning.json"),
        )

    def _make_sampler(self) -> LayoutFlowMatchingSampler:
        if self.vae is not None:
            return LayoutFlowMatchingSampler.from_stable(
                self.unet,
                self.vae,
                device=self.device,
                t_scale=self.t_scale,
                train_target=self.train_target,
                from_norm_to_display=self.from_norm_to_display,
            )
        return LayoutFlowMatchingSampler(
            self.unet,
            device=self.device,
            t_scale=self.t_scale,
            train_target=self.train_target,
            from_norm_to_display=self.from_norm_to_display,
        )

    def prepare_conditioning_kwargs(
        self,
        batch: Dict[str, Any],
        device: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        target_device = device or self.device
        cond_kwargs = {
            "boxes_xyxy_norm": batch["boxes_xyxy_norm"].to(target_device),
            "labels": batch["labels"].to(target_device),
            "object_mask": batch["object_mask"].to(target_device),
        }
        variant = str(getattr(self.layout_config, "variant", "raster_v1"))
        if variant == "stay_v2" and bool(getattr(self.layout_config, "use_style_latent", False)):
            if "style_noise" in batch:
                cond_kwargs["style_noise"] = batch["style_noise"].to(target_device)
            else:
                batch_size, max_objects = int(batch["labels"].shape[0]), int(batch["labels"].shape[1])
                cond_kwargs["style_noise"] = self.unet.sample_style_noise(
                    batch_size=batch_size,
                    max_objects=max_objects,
                    device=target_device,
                    dtype=batch["boxes_xyxy_norm"].dtype,
                )
        return cond_kwargs

    def _attach_fixed_style_noise(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        variant = str(getattr(self.layout_config, "variant", "raster_v1"))
        if variant != "stay_v2" or not bool(getattr(self.layout_config, "use_style_latent", False)):
            return batch
        if "style_noise" in batch:
            return batch

        seeded = dict(batch)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(getattr(self.layout_config, "style_seed", 1234)))
        batch_size, max_objects = int(batch["labels"].shape[0]), int(batch["labels"].shape[1])
        seeded["style_noise"] = self.unet.sample_style_noise(
            batch_size=batch_size,
            max_objects=max_objects,
            device="cpu",
            dtype=batch["boxes_xyxy_norm"].dtype,
            generator=generator,
        ).cpu()
        return seeded

    def _match_flow_targets(
        self,
        z0: torch.Tensor,
        x_fm: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if self.path_mode != "conditional_ot":
            return super()._match_flow_targets(z0, x_fm, cond_kwargs)
        if cond_kwargs is None:
            raise ValueError("conditional_ot requires layout conditioning kwargs.")

        descriptor = build_layout_descriptor(
            boxes_xyxy_norm=cond_kwargs["boxes_xyxy_norm"],
            labels=cond_kwargs["labels"],
            object_mask=cond_kwargs["object_mask"],
            num_classes=int(getattr(self.layout_config, "num_classes", self.unet.num_classes) or self.unet.num_classes),
            resolution=int(self.layout_cost_resolution),
        )
        layout_cost = pairwise_mean_squared_cost(descriptor, descriptor)
        matched, _, _ = match_target_batch(
            z0,
            x_fm,
            solver=self.path_solver,
            extra_cost=float(self.condition_weight) * layout_cost,
        )
        return matched

    def flow_matching_step(self, x_fm: torch.Tensor, cond_kwargs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        if cond_kwargs is None:
            cond_kwargs = {}
        batch_size = x_fm.shape[0]
        z0 = torch.randn_like(x_fm)
        t = torch.rand(batch_size, device=x_fm.device)
        t_expanded = t[:, None, None, None]
        x_target = self._match_flow_targets(z0, x_fm, cond_kwargs)

        zt = (1.0 - t_expanded) * z0 + t_expanded * x_target
        v_target = x_target - z0

        unet_output = self.unet(zt, t * self.t_scale, **cond_kwargs)
        unet_sample = unet_output.sample
        if self.train_target == "x0":
            fm_prediction = (unet_sample - zt) / (1.0 - t_expanded).clamp(min=1e-5)
        else:
            fm_prediction = unet_sample

        fm_loss = F.mse_loss(fm_prediction, v_target)
        aux_loss = getattr(unet_output, "aux_loss", None)
        if aux_loss is None:
            aux_loss = torch.zeros((), device=x_fm.device, dtype=fm_loss.dtype)
        total_loss = fm_loss + aux_loss

        self._last_loss_components = {
            "total_loss": float(total_loss.detach().item()),
            "fm_loss": float(fm_loss.detach().item()),
            "aux_loss": float(aux_loss.detach().item()),
            "mask_overlap_loss": float(getattr(unet_output, "mask_overlap_loss", aux_loss * 0.0).detach().item()),
            "mask_sharpness_loss": float(getattr(unet_output, "mask_sharpness_loss", aux_loss * 0.0).detach().item()),
            "mask_activation_loss": float(getattr(unet_output, "mask_activation_loss", aux_loss * 0.0).detach().item()),
        }
        return total_loss

    @staticmethod
    def _slice_batch(batch: Dict[str, Any], max_items: int) -> Dict[str, Any]:
        sliced: Dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                sliced[key] = value[:max_items]
            elif isinstance(value, list):
                sliced[key] = value[:max_items]
            else:
                sliced[key] = value
        return sliced

    @staticmethod
    def _build_fixed_validation_batch(
        dataloader: Optional[DataLoader],
        num_examples: int,
    ) -> Optional[Dict[str, Any]]:
        if dataloader is None or num_examples <= 0:
            return None
        dataset = dataloader.dataset
        num_items = min(num_examples, len(dataset))
        if num_items == 0:
            return None
        samples = [dataset[idx] for idx in range(num_items)]
        collate_fn = getattr(dataloader, "collate_fn", None)
        if collate_fn is None:
            raise ValueError("layout validation batching requires an explicit collate_fn")
        return collate_fn(samples)

    @staticmethod
    def _grad_norm(parameters) -> float:
        return grad_norm(parameters)

    @staticmethod
    def _should_log(step: int, every: int) -> bool:
        return every > 0 and (step == 1 or step % every == 0)

    @staticmethod
    def _should_log_epoch(epoch_idx: int, every: int) -> bool:
        return every > 0 and (epoch_idx + 1) % every == 0

    def _log_training_visuals(
        self,
        writer: SummaryWriter,
        *,
        batch: Dict[str, Any],
        cond_kw: Optional[Dict[str, torch.Tensor]],
        global_step: int,
        max_logged_images: int,
        log_internal_maps: bool,
    ) -> None:
        vis_batch = self._slice_batch(batch, max_logged_images)
        pixel_values = vis_batch["pixel_values"]
        boxes_xyxy = vis_batch["boxes_xyxy"]
        labels = vis_batch["labels"]
        object_mask = vis_batch["object_mask"]
        image_size = int(pixel_values.shape[-1])

        display_images = self.from_norm_to_display(pixel_values).clamp(0.0, 1.0)
        overlay_images = draw_bbox_overlays(
            display_images,
            boxes_xyxy=boxes_xyxy,
            labels=labels,
            object_mask=object_mask,
        )
        class_layout = render_class_layout(
            boxes_xyxy=boxes_xyxy,
            labels=labels,
            object_mask=object_mask,
            image_size=image_size,
        )

        writer.add_images("layout_fm/train/input", ensure_rgb(display_images), global_step)
        writer.add_images("layout_fm/train/input_boxes", overlay_images, global_step)
        writer.add_images("layout_fm/train/class_layout", class_layout, global_step)

        if not log_internal_maps:
            return

        vis_cond_kw = None
        if cond_kw is not None:
            vis_cond_kw = {
                key: value[:max_logged_images] if torch.is_tensor(value) else value
                for key, value in cond_kw.items()
            }

        with torch.no_grad():
            debug = self.unet.build_conditioning(
                boxes_xyxy_norm=vis_batch["boxes_xyxy_norm"].to(self.device),
                labels=labels.to(self.device),
                object_mask=object_mask.to(self.device),
                spatial_size=(image_size, image_size),
                style_noise=None if vis_cond_kw is None else vis_cond_kw.get("style_noise"),
            )

        writer.add_images(
            "layout_fm/train/objectness",
            ensure_rgb(debug["objectness_map"]),
            global_step,
        )
        writer.add_images(
            "layout_fm/train/condition_energy",
            normalize_feature_map(debug["feature_energy_map"]),
            global_step,
        )
        writer.add_images(
            "layout_fm/train/conditioning_maps",
            normalize_feature_map(debug["conditioning_maps"]),
            global_step,
        )
        variant = str(getattr(self.layout_config, "variant", "raster_v1"))
        if variant == "stay_v2":
            soft_mask_composite = render_mask_composite(
                debug["soft_masks_full"],
                object_mask=object_mask,
                labels=labels,
            )
            hard_mask_composite = render_mask_composite(
                debug["hard_masks_full"],
                object_mask=object_mask,
                labels=labels,
            )
            owner_mask_composite = render_mask_composite(
                debug["non_overlap_masks_full"],
                object_mask=object_mask,
                labels=labels,
            )
            writer.add_images("layout_fm/train/soft_masks", soft_mask_composite, global_step)
            writer.add_images("layout_fm/train/hard_masks", hard_mask_composite, global_step)
            writer.add_images("layout_fm/train/non_overlap_masks", owner_mask_composite, global_step)
            writer.add_images(
                "layout_fm/train/semantic_map",
                normalize_feature_map(debug["semantic_map"]),
                global_step,
            )
            writer.add_images(
                "layout_fm/train/edge_map",
                normalize_feature_map(debug["edge_map"]),
                global_step,
            )
            writer.add_images(
                "layout_fm/train/overlap_map",
                ensure_rgb(debug["overlap_map"]),
                global_step,
            )
            writer.add_images(
                "layout_fm/train/masked_context",
                normalize_feature_map(debug["masked_context_map"]),
                global_step,
            )
            writer.add_images(
                "layout_fm/train/input_mask_overlay",
                draw_mask_overlays(
                    display_images,
                    masks=debug["hard_masks_full"],
                    object_mask=object_mask,
                    labels=labels,
                ),
                global_step,
            )

    def _log_fixed_validation_samples(
        self,
        writer: SummaryWriter,
        *,
        sampler: LayoutFlowMatchingSampler,
        fixed_batch: Dict[str, Any],
        global_step: int,
        steps: int,
        sample_shape: Optional[Tuple[int, int, int]],
        max_logged_images: int,
        save_debug_images: bool,
        debug_dir: str,
        log_internal_maps: bool,
    ) -> None:
        vis_batch = self._slice_batch(fixed_batch, max_logged_images)
        variant = str(getattr(self.layout_config, "variant", "raster_v1"))
        generated_mask_overlay = None
        generated_latents = sampler.sample_euler_layout(
            vis_batch,
            steps=steps,
            sample_shape=sample_shape,
        )
        generated_images = sampler.decode(generated_latents)
        generated_display = self.from_norm_to_display(generated_images).clamp(0.0, 1.0)
        ground_truth_display = self.from_norm_to_display(vis_batch["pixel_values"]).clamp(0.0, 1.0)

        generated_overlay = draw_bbox_overlays(
            generated_display,
            boxes_xyxy=vis_batch["boxes_xyxy"],
            labels=vis_batch["labels"],
            object_mask=vis_batch["object_mask"],
        )
        ground_truth_overlay = draw_bbox_overlays(
            ground_truth_display,
            boxes_xyxy=vis_batch["boxes_xyxy"],
            labels=vis_batch["labels"],
            object_mask=vis_batch["object_mask"],
        )
        class_layout = render_class_layout(
            boxes_xyxy=vis_batch["boxes_xyxy"],
            labels=vis_batch["labels"],
            object_mask=vis_batch["object_mask"],
            image_size=int(vis_batch["pixel_values"].shape[-1]),
        )
        writer.add_images("layout_fm/val/input", ensure_rgb(ground_truth_display), global_step)
        writer.add_images("layout_fm/val/input_boxes", ground_truth_overlay, global_step)
        writer.add_images("layout_fm/val/generated", ensure_rgb(generated_display), global_step)
        writer.add_images("layout_fm/val/generated_boxes", generated_overlay, global_step)
        writer.add_images("layout_fm/val/ground_truth_boxes", ground_truth_overlay, global_step)

        if log_internal_maps:
            with torch.no_grad():
                debug = self.unet.build_conditioning(
                    boxes_xyxy_norm=vis_batch["boxes_xyxy_norm"].to(self.device),
                    labels=vis_batch["labels"].to(self.device),
                    object_mask=vis_batch["object_mask"].to(self.device),
                    spatial_size=(int(vis_batch["pixel_values"].shape[-2]), int(vis_batch["pixel_values"].shape[-1])),
                    style_noise=vis_batch.get("style_noise", None).to(self.device)
                    if isinstance(vis_batch.get("style_noise", None), torch.Tensor)
                    else None,
                )
            writer.add_images(
                "layout_fm/val/objectness",
                ensure_rgb(debug["objectness_map"]),
                global_step,
            )
            writer.add_images(
                "layout_fm/val/condition_energy",
                normalize_feature_map(debug["feature_energy_map"]),
                global_step,
            )
            writer.add_images(
                "layout_fm/val/conditioning_maps",
                normalize_feature_map(debug["conditioning_maps"]),
                global_step,
            )
            if variant == "stay_v2":
                soft_mask_composite = render_mask_composite(
                    debug["soft_masks_full"],
                    object_mask=vis_batch["object_mask"],
                    labels=vis_batch["labels"],
                )
                hard_mask_composite = render_mask_composite(
                    debug["hard_masks_full"],
                    object_mask=vis_batch["object_mask"],
                    labels=vis_batch["labels"],
                )
                owner_mask_composite = render_mask_composite(
                    debug["non_overlap_masks_full"],
                    object_mask=vis_batch["object_mask"],
                    labels=vis_batch["labels"],
                )
                generated_mask_overlay = draw_mask_overlays(
                    generated_display,
                    masks=debug["hard_masks_full"],
                    object_mask=vis_batch["object_mask"],
                    labels=vis_batch["labels"],
                )
                writer.add_images("layout_fm/val/soft_masks", soft_mask_composite, global_step)
                writer.add_images("layout_fm/val/hard_masks", hard_mask_composite, global_step)
                writer.add_images("layout_fm/val/non_overlap_masks", owner_mask_composite, global_step)
                writer.add_images(
                    "layout_fm/val/semantic_map",
                    normalize_feature_map(debug["semantic_map"]),
                    global_step,
                )
                writer.add_images(
                    "layout_fm/val/edge_map",
                    normalize_feature_map(debug["edge_map"]),
                    global_step,
                )
                writer.add_images(
                    "layout_fm/val/overlap_map",
                    ensure_rgb(debug["overlap_map"]),
                    global_step,
                )
                writer.add_images(
                    "layout_fm/val/masked_context",
                    normalize_feature_map(debug["masked_context_map"]),
                    global_step,
                )
                writer.add_images(
                    "layout_fm/val/generated_mask_overlay",
                    generated_mask_overlay,
                    global_step,
                )
                panel = make_side_by_side_panel(
                    [class_layout, soft_mask_composite, generated_overlay, generated_mask_overlay, ground_truth_overlay]
                )
            else:
                panel = make_side_by_side_panel(
                    [class_layout, generated_display, generated_overlay, ground_truth_overlay]
                )
        else:
            panel = make_side_by_side_panel(
                [class_layout, generated_display, generated_overlay, ground_truth_overlay]
            )

        writer.add_images("layout_fm/val/panel", panel, global_step)

        if save_debug_images:
            step_dir = os.path.join(debug_dir, f"step_{global_step:06d}")
            save_image_batch(panel, output_dir=step_dir, prefix="panel")
            save_image_batch(generated_overlay, output_dir=step_dir, prefix="generated_boxes")
            if generated_mask_overlay is not None:
                save_image_batch(generated_mask_overlay, output_dir=step_dir, prefix="generated_masks")

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        *,
        pretrained_vae_path: Optional[str] = None,
        pretrained_unet_path: Optional[str] = None,
        strict_load: bool = True,
        log_dir: str = "./artifacts/runs/main/layout_fm",
        debug_dir: str = "./artifacts/debug/layout_fm",
        lr: float = 1e-4,
        sample_every: int = 0,
        sample_steps: int = 50,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        save_every_n_epochs: int = 1,
        eval_every: int = 1,
        resume_from_checkpoint: Optional[str] = None,
        scalar_every_steps: int = 10,
        image_every_steps: int = 200,
        max_logged_images: int = 4,
        fixed_validation_examples: int = 4,
        sample_every_steps: int = 0,
        save_debug_images: bool = False,
        log_internal_maps: bool = True,
        optimizer_name: str = "adamw",
        weight_decay: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
        scheduler_name: str = "warmup_cosine",
        warmup_ratio: float = 0.05,
        min_lr_ratio: float = 0.1,
        ema_enabled: bool = True,
        ema_decay: float = 0.999,
        ema_start_step: int = 100,
        mixed_precision: str = "auto",
        max_grad_norm: float = 1.0,
    ) -> None:
        self._ensure_dirs()
        self._save_configs()
        os.makedirs(debug_dir, exist_ok=True)
        eval_every = int(eval_every)

        if str(optimizer_name).lower() != "adamw":
            raise ValueError(f"Unsupported optimizer_name={optimizer_name!r}. Only 'adamw' is implemented.")
        if patience is not None and eval_dataloader is None:
            raise ValueError("eval_dataloader must be provided when using patience early stopping.")
        if patience is not None and eval_every <= 0:
            raise ValueError("eval_every must be > 0 when using patience early stopping.")

        if pretrained_vae_path is not None and self.vae is not None:
            self.load_vae_weights(pretrained_vae_path, strict=strict_load)
            self.vae.eval()
            for param in self.vae.parameters():
                param.requires_grad = False

        if pretrained_unet_path is not None:
            self.load_unet_weights(pretrained_unet_path, strict=strict_load)

        total_steps = max(1, epochs * len(dataloader))
        precision = resolve_precision_settings(self.device, mixed_precision)
        scaler = build_grad_scaler(precision)
        optimizer = torch.optim.AdamW(
            self.unet.parameters(),
            lr=lr,
            betas=(float(beta1), float(beta2)),
            weight_decay=float(weight_decay),
        )
        scheduler = build_scheduler(
            optimizer,
            scheduler_name=scheduler_name,
            total_steps=total_steps,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
        )
        ema = EMAState(self.unet, decay=ema_decay) if ema_enabled and ema_decay > 0.0 else None

        global_step = 0
        best_eval = float("inf")
        best_epoch = -1
        bad_epochs = 0
        start_epoch = 0

        if resume_from_checkpoint is not None:
            print(f"[Resume] Loading checkpoint from {resume_from_checkpoint}")
            ckpt = torch.load(resume_from_checkpoint, map_location=self.device)
            self.unet.load_state_dict(ckpt["unet_state"])
            optimizer.load_state_dict(ckpt["optimizer_state"])
            move_optimizer_state_to_device(optimizer, self.device)
            if scheduler is not None and ckpt.get("scheduler_state") is not None:
                scheduler.load_state_dict(ckpt["scheduler_state"])
            if scaler is not None and ckpt.get("scaler_state") is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            if ema is not None:
                if ckpt.get("ema_state") is not None:
                    ema.load_state_dict(ckpt["ema_state"], device=self.device)
                else:
                    ema = EMAState(self.unet, decay=ema_decay)
            start_epoch = ckpt["epoch"] + 1
            global_step = ckpt["global_step"]
            best_eval = ckpt.get("best_eval", float("inf"))
            best_epoch = ckpt.get("best_epoch", -1)
            bad_epochs = ckpt.get("bad_epochs", 0)
            print(
                f"[Resume] epoch={start_epoch}, step={global_step}, best_eval={best_eval:.6f}"
            )

        writer = SummaryWriter(log_dir)
        sampler = self._make_sampler()
        fixed_batch = self._build_fixed_validation_batch(
            eval_dataloader or dataloader,
            fixed_validation_examples,
        )
        if fixed_batch is not None:
            fixed_batch = self._attach_fixed_style_noise(fixed_batch)

        epoch_image_logging = sample_every > 0
        if epoch_image_logging and (image_every_steps > 0 or sample_every_steps > 0):
            print(
                "[LayoutFM] Epoch-based image logging enabled via "
                f"sampling.sample_every={sample_every}; ignoring legacy "
                f"logging.image_every_steps={image_every_steps} and "
                f"sampling.sample_every_steps={sample_every_steps}."
            )
        if precision.requested != precision.mode:
            print(
                f"[LayoutFM Precision] requested={precision.requested!r} -> using {precision.mode!r} "
                f"on device={precision.device_type}"
            )

        def _save_checkpoint(path: str, epoch_idx: int) -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            ckpt = {
                "epoch": epoch_idx,
                "global_step": global_step,
                "unet_state": self.unet.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                "scaler_state": None if scaler is None else scaler.state_dict(),
                "ema_state": None if ema is None else ema.state_dict(),
                "best_eval": best_eval,
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
                "t_scale": self.t_scale,
                "train_target": self.train_target,
            }
            torch.save(ckpt, path)

        for epoch in range(start_epoch, epochs):
            self.unet.train()
            total_loss = 0.0
            last_batch: Optional[Dict[str, Any]] = None
            last_cond_kw: Optional[Dict[str, torch.Tensor]] = None

            for batch in tqdm(dataloader, desc=f"LayoutFM Epoch {epoch + 1}/{epochs}"):
                pixel_values = batch["pixel_values"].to(self.device)
                cond_kw = self.prepare_conditioning_kwargs(batch)
                last_batch = batch
                last_cond_kw = cond_kw
                with torch.no_grad():
                    x_fm = self.encode_fm_input(pixel_values)

                optimizer.zero_grad(set_to_none=True)
                with autocast_context(precision):
                    loss = self.flow_matching_step(x_fm, cond_kw)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                if max_grad_norm and max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_grad_norm)
                step_grad_norm = self._grad_norm(self.unet.parameters())

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                if ema is not None and global_step >= int(ema_start_step):
                    ema.update(self.unet)

                global_step += 1
                total_loss += float(loss.item())

                if self._should_log(global_step, scalar_every_steps):
                    with torch.no_grad():
                        layout_debug = self.unet.build_conditioning(
                            boxes_xyxy_norm=cond_kw["boxes_xyxy_norm"],
                            labels=cond_kw["labels"],
                            object_mask=cond_kw["object_mask"],
                            spatial_size=(int(x_fm.shape[-2]), int(x_fm.shape[-1])),
                            style_noise=cond_kw.get("style_noise"),
                        )
                    n_objects = batch["n_objects"].to(torch.float32)
                    writer.add_scalar("layout_fm/loss_step", float(loss.item()), global_step)
                    writer.add_scalar(
                        "layout_fm/fm_loss_step",
                        float(self._last_loss_components.get("fm_loss", float(loss.item()))),
                        global_step,
                    )
                    writer.add_scalar(
                        "layout_fm/aux_loss_step",
                        float(self._last_loss_components.get("aux_loss", 0.0)),
                        global_step,
                    )
                    writer.add_scalar("layout_fm/lr", float(optimizer.param_groups[0]["lr"]), global_step)
                    writer.add_scalar("layout_fm/grad_norm", float(step_grad_norm), global_step)
                    writer.add_scalar("layout_fm/mean_objects", float(n_objects.mean().item()), global_step)
                    writer.add_scalar("layout_fm/max_objects", float(n_objects.max().item()), global_step)
                    writer.add_scalar(
                        "layout_fm/empty_layout_fraction",
                        float((n_objects == 0).to(torch.float32).mean().item()),
                        global_step,
                    )
                    writer.add_scalar(
                        "layout_fm/layout_coverage",
                        float(layout_debug["objectness_map"].mean().item()),
                        global_step,
                    )
                    writer.add_scalar(
                        "layout_fm/mask_overlap_loss",
                        float(self._last_loss_components.get("mask_overlap_loss", 0.0)),
                        global_step,
                    )
                    writer.add_scalar(
                        "layout_fm/mask_sharpness_loss",
                        float(self._last_loss_components.get("mask_sharpness_loss", 0.0)),
                        global_step,
                    )
                    writer.add_scalar(
                        "layout_fm/mask_activation_loss",
                        float(self._last_loss_components.get("mask_activation_loss", 0.0)),
                        global_step,
                    )
                    soft_masks = layout_debug.get("soft_masks_full")
                    if soft_masks is not None:
                        if soft_masks.numel() > 0:
                            writer.add_scalar(
                                "layout_fm/mask_mean",
                                float(soft_masks.mean().item()),
                                global_step,
                            )
                            writer.add_scalar(
                                "layout_fm/overlap_ratio",
                                float(layout_debug["overlap_map"].mean().item()),
                                global_step,
                            )
                        else:
                            writer.add_scalar("layout_fm/mask_mean", 0.0, global_step)
                            writer.add_scalar("layout_fm/overlap_ratio", 0.0, global_step)
                    if "edge_map" in layout_debug:
                        writer.add_scalar(
                            "layout_fm/edge_map_energy",
                            float(layout_debug["edge_map"].abs().mean().item()),
                            global_step,
                        )
                    del layout_debug, soft_masks

                if (not epoch_image_logging) and self._should_log(global_step, image_every_steps):
                    ema_context = ema.average_parameters(self.unet) if ema is not None and global_step >= int(ema_start_step) else torch.no_grad()
                    with ema_context:
                        self._log_training_visuals(
                            writer,
                            batch=batch,
                            cond_kw=cond_kw,
                            global_step=global_step,
                            max_logged_images=max_logged_images,
                            log_internal_maps=log_internal_maps,
                        )

                if (
                    (not epoch_image_logging)
                    and fixed_batch is not None
                    and self._should_log(global_step, sample_every_steps)
                ):
                    ema_context = ema.average_parameters(self.unet) if ema is not None and global_step >= int(ema_start_step) else torch.no_grad()
                    with ema_context:
                        self._log_fixed_validation_samples(
                            writer,
                            sampler=sampler,
                            fixed_batch=fixed_batch,
                            global_step=global_step,
                            steps=sample_steps,
                            sample_shape=sample_shape,
                            max_logged_images=max_logged_images,
                            save_debug_images=save_debug_images,
                            debug_dir=debug_dir,
                            log_internal_maps=log_internal_maps,
                        )

            avg_loss = total_loss / max(1, len(dataloader))
            print(f"[LayoutFM Epoch {epoch + 1}] loss={avg_loss:.6f}")
            writer.add_scalar("layout_fm/loss_epoch", avg_loss, epoch)

            if save_every_n_epochs and (epoch + 1) % save_every_n_epochs == 0:
                self.save_unet_weights(os.path.join(self._unet_dir(), f"unet_fm_epoch_{epoch + 1}.pt"))
                _save_checkpoint(
                    os.path.join(self._unet_dir(), f"unet_fm_epoch_{epoch + 1}_ckpt.pt"),
                    epoch_idx=epoch,
                )

            should_run_eval = (
                eval_dataloader is not None
                and eval_every > 0
                and (epoch + 1) % eval_every == 0
            )
            if should_run_eval:
                self.unet.eval()
                eval_loss = 0.0
                n_eval = 0

                ema_context = ema.average_parameters(self.unet) if ema is not None and global_step >= int(ema_start_step) else torch.no_grad()
                with ema_context:
                    with torch.no_grad():
                        for batch in tqdm(eval_dataloader, desc=f"LayoutFM Eval {epoch + 1}/{epochs}"):
                            pixel_values = batch["pixel_values"].to(self.device)
                            cond_kw = self.prepare_conditioning_kwargs(batch)
                            x_fm = self.encode_fm_input(pixel_values)
                            with autocast_context(precision):
                                loss = self.flow_matching_step(x_fm, cond_kw)
                            batch_size = int(pixel_values.shape[0])
                            eval_loss += float(loss.item()) * batch_size
                            n_eval += batch_size

                avg_eval = eval_loss / max(1, n_eval)
                writer.add_scalar("layout_fm/eval_loss_epoch", avg_eval, epoch)
                print(f"  [Eval loss: {avg_eval:.6f}]")

                improved = (best_eval - avg_eval) > min_delta
                if improved:
                    best_eval = avg_eval
                    best_epoch = epoch
                    bad_epochs = 0
                    if ema is not None and global_step >= int(ema_start_step):
                        with ema.average_parameters(self.unet):
                            self.save_unet_weights(os.path.join(self._unet_dir(), "unet_fm_best.pt"))
                    else:
                        self.save_unet_weights(os.path.join(self._unet_dir(), "unet_fm_best.pt"))
                    print(f"  New best eval={best_eval:.6f} at epoch {epoch + 1}")
                elif patience is not None:
                    bad_epochs += 1
                    print(f"  No improvement (best={best_eval:.6f}), bad_epochs={bad_epochs}/{patience}")
                    if bad_epochs >= patience:
                        _save_checkpoint(os.path.join(self._unet_dir(), "unet_last_ckpt.pt"), epoch_idx=epoch)
                        print(f"Early stopping. Best epoch: {best_epoch + 1}")
                        break

            if epoch_image_logging and self._should_log_epoch(epoch, sample_every):
                self.unet.eval()
                ema_context = ema.average_parameters(self.unet) if ema is not None and global_step >= int(ema_start_step) else torch.no_grad()
                with ema_context:
                    if last_batch is not None:
                        self._log_training_visuals(
                            writer,
                            batch=last_batch,
                            cond_kw=last_cond_kw,
                            global_step=global_step,
                            max_logged_images=max_logged_images,
                            log_internal_maps=log_internal_maps,
                        )
                    if fixed_batch is not None:
                        self._log_fixed_validation_samples(
                            writer,
                            sampler=sampler,
                            fixed_batch=fixed_batch,
                            global_step=global_step,
                            steps=sample_steps,
                            sample_shape=sample_shape,
                            max_logged_images=max_logged_images,
                            save_debug_images=save_debug_images,
                            debug_dir=debug_dir,
                            log_internal_maps=log_internal_maps,
                        )

            _save_checkpoint(os.path.join(self._unet_dir(), "unet_last_ckpt.pt"), epoch_idx=epoch)

        writer.close()


from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.trainer.register("layout_fm")(LayoutFMTrainer)

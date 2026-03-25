"""Trainer for bbox-conditioned pixel-space flow matching."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.algorithms.inference.layout_flow_matching_sampler import LayoutFlowMatchingSampler
from src.algorithms.training.flow_matching_trainer import FlowMatchingTrainer
from src.core.visualization.layout_debug import (
    draw_bbox_overlays,
    ensure_rgb,
    make_side_by_side_panel,
    normalize_feature_map,
    render_class_layout,
    save_image_batch,
)
from src.models.fm_unet import load_unet_config, save_unet_config
from src.models.layout_conditioned_unet import (
    build_layout_conditioned_pixel_unet,
    save_layout_conditioning_metadata,
)


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
    ) -> None:
        super().__init__(
            unet,
            device=device,
            t_scale=t_scale,
            train_target=train_target,
            model_dir=model_dir,
            from_norm_to_display=from_norm_to_display,
            unet_config=unet_config,
            vae=None,
            vae_config=None,
            conditioner=None,
        )
        self.layout_config = layout_config

    @classmethod
    def from_config(
        cls,
        config,
        *,
        from_norm_to_display=None,
    ) -> "LayoutFMTrainer":
        device = config.resolved_device() if hasattr(config, "resolved_device") else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        num_classes = int(config.layout_conditioning.num_classes or 0)
        if num_classes <= 0:
            raise ValueError(
                "layout_conditioning.num_classes must be set before building LayoutFMTrainer"
            )

        base_unet_cfg = load_unet_config(config.model.unet_config)
        image_in_channels = int(base_unet_cfg.get("in_channels", 1))
        effective_unet_cfg = dict(base_unet_cfg)
        effective_unet_cfg["sample_size"] = int(config.data.image_size)
        effective_unet_cfg["in_channels"] = image_in_channels + int(config.layout_conditioning.spatial_channels)
        effective_unet_cfg["out_channels"] = image_in_channels

        wrapped_unet = build_layout_conditioned_pixel_unet(
            effective_unet_cfg,
            image_in_channels=image_in_channels,
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
            pretrained_unet_path=config.model.pretrained_unet_path,
            strict_load=config.training.strict_load,
            log_dir=config.output.resolved_log_dir(),
            debug_dir=config.output.resolved_debug_dir(),
            lr=config.training.lr,
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            sample_steps=config.sampling.sample_steps,
            sample_shape=config.sampling.sample_shape,
            save_every_n_epochs=config.training.save_every_n_epochs,
            resume_from_checkpoint=config.output.resume,
            scalar_every_steps=config.logging.scalar_every_steps,
            image_every_steps=config.logging.image_every_steps,
            max_logged_images=config.logging.max_logged_images,
            fixed_validation_examples=config.sampling.fixed_validation_examples,
            sample_every_steps=config.sampling.sample_every_steps,
            save_debug_images=config.sampling.save_debug_images,
            log_internal_maps=config.layout_conditioning.log_internal_maps,
        )

    def _save_configs(self) -> None:
        if self.unet_config is not None:
            save_unet_config(self.unet_config, os.path.join(self._unet_dir(), "config.json"))

        layout_metadata = {
            "num_classes": int(self.unet.num_classes),
            "class_embed_dim": int(self.unet.class_embed_dim),
            "bbox_embed_dim": int(self.unet.bbox_embed_dim),
            "spatial_channels": int(self.unet.spatial_channels),
            "raster_mode": str(self.unet.raster_mode),
            "image_in_channels": int(self.unet.image_in_channels),
            "category_id_to_name": {
                str(key): value for key, value in self.unet.category_id_to_name.items()
            },
        }
        save_layout_conditioning_metadata(
            layout_metadata,
            os.path.join(self.model_dir, "layout_conditioning.json"),
        )

    def _make_sampler(self) -> LayoutFlowMatchingSampler:
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
        return {
            "boxes_xyxy_norm": batch["boxes_xyxy_norm"].to(target_device),
            "labels": batch["labels"].to(target_device),
            "object_mask": batch["object_mask"].to(target_device),
        }

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
        total = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            grad_norm = param.grad.detach().pow(2).sum().item()
            total += grad_norm
        return float(total ** 0.5)

    @staticmethod
    def _should_log(step: int, every: int) -> bool:
        return every > 0 and (step == 1 or step % every == 0)

    def _log_training_visuals(
        self,
        writer: SummaryWriter,
        *,
        batch: Dict[str, Any],
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

        with torch.no_grad():
            debug = self.unet.build_conditioning(
                boxes_xyxy_norm=vis_batch["boxes_xyxy_norm"].to(self.device),
                labels=labels.to(self.device),
                object_mask=object_mask.to(self.device),
                spatial_size=(image_size, image_size),
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
        panel = make_side_by_side_panel(
            [class_layout, generated_display, generated_overlay, ground_truth_overlay]
        )

        writer.add_images("layout_fm/val/generated", ensure_rgb(generated_display), global_step)
        writer.add_images("layout_fm/val/generated_boxes", generated_overlay, global_step)
        writer.add_images("layout_fm/val/ground_truth_boxes", ground_truth_overlay, global_step)
        writer.add_images("layout_fm/val/panel", panel, global_step)

        if log_internal_maps:
            with torch.no_grad():
                debug = self.unet.build_conditioning(
                    boxes_xyxy_norm=vis_batch["boxes_xyxy_norm"].to(self.device),
                    labels=vis_batch["labels"].to(self.device),
                    object_mask=vis_batch["object_mask"].to(self.device),
                    spatial_size=(int(vis_batch["pixel_values"].shape[-2]), int(vis_batch["pixel_values"].shape[-1])),
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

        if save_debug_images:
            step_dir = os.path.join(debug_dir, f"step_{global_step:06d}")
            save_image_batch(panel, output_dir=step_dir, prefix="panel")
            save_image_batch(generated_overlay, output_dir=step_dir, prefix="generated_boxes")

    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        *,
        pretrained_unet_path: Optional[str] = None,
        strict_load: bool = True,
        log_dir: str = "./artifacts/runs/main/layout_fm",
        debug_dir: str = "./artifacts/debug/layout_fm",
        lr: float = 1e-4,
        sample_steps: int = 50,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        save_every_n_epochs: int = 1,
        resume_from_checkpoint: Optional[str] = None,
        scalar_every_steps: int = 10,
        image_every_steps: int = 200,
        max_logged_images: int = 4,
        fixed_validation_examples: int = 4,
        sample_every_steps: int = 0,
        save_debug_images: bool = False,
        log_internal_maps: bool = True,
    ) -> None:
        self._ensure_dirs()
        self._save_configs()
        os.makedirs(debug_dir, exist_ok=True)

        if pretrained_unet_path is not None:
            self.load_unet_weights(pretrained_unet_path, strict=strict_load)

        optimizer = Adam(self.unet.parameters(), lr=lr)

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
            for state in optimizer.state.values():
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = value.to(self.device)
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

        def _save_checkpoint(path: str, epoch_idx: int) -> None:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            ckpt = {
                "epoch": epoch_idx,
                "global_step": global_step,
                "unet_state": self.unet.state_dict(),
                "optimizer_state": optimizer.state_dict(),
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

            for batch in tqdm(dataloader, desc=f"LayoutFM Epoch {epoch + 1}/{epochs}"):
                pixel_values = batch["pixel_values"].to(self.device)
                cond_kw = self.prepare_conditioning_kwargs(batch)
                x_fm = self.encode_fm_input(pixel_values)
                loss = self.flow_matching_step(x_fm, cond_kw)

                optimizer.zero_grad()
                loss.backward()
                grad_norm = self._grad_norm(self.unet.parameters())
                optimizer.step()

                global_step += 1
                total_loss += float(loss.item())

                if self._should_log(global_step, scalar_every_steps):
                    with torch.no_grad():
                        layout_debug = self.unet.build_conditioning(
                            boxes_xyxy_norm=cond_kw["boxes_xyxy_norm"],
                            labels=cond_kw["labels"],
                            object_mask=cond_kw["object_mask"],
                            spatial_size=(int(pixel_values.shape[-2]), int(pixel_values.shape[-1])),
                        )
                    n_objects = batch["n_objects"].to(torch.float32)
                    writer.add_scalar("layout_fm/loss_step", float(loss.item()), global_step)
                    writer.add_scalar("layout_fm/lr", float(optimizer.param_groups[0]["lr"]), global_step)
                    writer.add_scalar("layout_fm/grad_norm", float(grad_norm), global_step)
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

                if self._should_log(global_step, image_every_steps):
                    self._log_training_visuals(
                        writer,
                        batch=batch,
                        global_step=global_step,
                        max_logged_images=max_logged_images,
                        log_internal_maps=log_internal_maps,
                    )

                if fixed_batch is not None and self._should_log(global_step, sample_every_steps):
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

            if eval_dataloader is not None:
                self.unet.eval()
                eval_loss = 0.0
                n_eval = 0

                with torch.no_grad():
                    for batch in tqdm(eval_dataloader, desc=f"LayoutFM Eval {epoch + 1}/{epochs}"):
                        pixel_values = batch["pixel_values"].to(self.device)
                        cond_kw = self.prepare_conditioning_kwargs(batch)
                        x_fm = self.encode_fm_input(pixel_values)
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
                    self.save_unet_weights(os.path.join(self._unet_dir(), "unet_fm_best.pt"))
                    print(f"  New best eval={best_eval:.6f} at epoch {epoch + 1}")
                elif patience is not None:
                    bad_epochs += 1
                    print(f"  No improvement (best={best_eval:.6f}), bad_epochs={bad_epochs}/{patience}")
                    if bad_epochs >= patience:
                        print(f"Early stopping. Best epoch: {best_epoch + 1}")
                        break

        writer.close()


from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.trainer.register("layout_fm")(LayoutFMTrainer)

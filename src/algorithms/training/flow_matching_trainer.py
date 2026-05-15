"""Training logic for flow-matching models.

This module is the sole source of truth for flow-matching training.
It replaces the training methods formerly embedded in
``fm_src.pipelines.flow_matching_pipeline``.

Two modes are supported:
* **pixel-space** — UNet operates directly on normalised images.
* **latent-space** (stable) — images are first encoded by a VAE;
  the UNet operates in the VAE latent space.
"""

from __future__ import annotations

import os
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler, validate_t_scale
from src.algorithms.tasks.flow_matching import FlowMatchingTask
from src.algorithms.training.regiondiff_attention_distillation import (
    RegionDiffAttentionRecorder,
    compute_region_attention_distillation_loss,
    load_regiondiff_attention_teacher,
)
from src.core.artifacts import ArtifactManifest, write_artifact_manifest
from src.core.training_utils import (
    TrainingProgressState,
    autocast_context,
    build_training_checkpoint,
    build_summary_writer,
    module_state_dict_cpu,
    release_cuda_cache,
    restore_training_checkpoint,
    save_training_checkpoint,
)
from src.core.training_runtime import (
    build_ema,
    build_lr_scheduler,
    build_optimizer,
    set_epoch_for_dataloader,
    setup_precision,
)
from src.core.visualization.layout_debug import (
    draw_bbox_overlays,
    render_class_layout,
    save_image_batch,
)
from src.models.fm_unet import save_unet_config
from src.models.regiondiffusion import build_area_weight_map
from src.models.regiondiffusion_factory import (
    build_regiondiff_wrapper,
    configure_regiondiff_trainability,
    regiondiff_optimizer_param_groups,
    save_regiondiff_metadata,
)

if TYPE_CHECKING:
    from diffusers import UNet2DModel
    from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Default display helper (same as sampler)
# ---------------------------------------------------------------------------
def _default_from_norm_to_display(x: torch.Tensor) -> torch.Tensor:
    return (x + 1) / 2


def _infer_vae_downsample_factor(vae_config: Dict[str, Any]) -> int:
    """Infer the VAE spatial downsample factor from its channel schedule."""
    num_channels = vae_config.get("num_channels")
    if isinstance(num_channels, (list, tuple)) and num_channels:
        return 2 ** max(0, len(num_channels) - 1)

    block_out_channels = vae_config.get("block_out_channels")
    if isinstance(block_out_channels, (list, tuple)) and block_out_channels:
        return 2 ** max(0, len(block_out_channels) - 1)

    down_block_types = vae_config.get("down_block_types")
    if isinstance(down_block_types, (list, tuple)) and down_block_types:
        return 2 ** max(0, len(down_block_types) - 1)

    raise ValueError(
        "VAE config must define a non-empty num_channels, block_out_channels, "
        "or down_block_types sequence to infer sample_size."
    )


def _resolve_unet_sample_size(config, vae_config: Dict[str, Any]) -> int | None:
    """Map the configured training image size to the latent UNet sample size."""
    data_cfg = getattr(config, "data", None)
    image_size = getattr(data_cfg, "image_size", None)
    if image_size is None:
        return None

    downsample_factor = _infer_vae_downsample_factor(vae_config)
    image_size = int(image_size)
    if image_size % downsample_factor != 0:
        raise ValueError(
            f"image_size={image_size} is not divisible by VAE downsample factor "
            f"{downsample_factor}"
        )
    return image_size // downsample_factor


# ═══════════════════════════════════════════════════════════════════════════
# FlowMatchingTrainer
# ═══════════════════════════════════════════════════════════════════════════

class FlowMatchingTrainer:
    """Self-contained trainer for flow-matching UNets.

    Parameters
    ----------
    unet : UNet2DModel
        The UNet to train.
    device : str or torch.device
    t_scale : float
        Time-scaling factor for UNet timestep input.
    train_target : ``"v"`` | ``"x0"``
    model_dir : str
        Root output directory. Weights are saved under ``model_dir/UNET/``.
    from_norm_to_display : callable, optional
        [-1,1] → [0,1] for TensorBoard visualisation.
    unet_config : dict, optional
        If provided, saved as ``model_dir/UNET/config.json``.
    vae : AutoencoderKL, optional
        If provided, training happens in latent space.
        The VAE is frozen automatically.
    vae_config : dict, optional
        Saved as ``model_dir/VAE/config.json`` when given.
    conditioner : BaseConditioner, optional
        Conditioning module.  When ``None``, no extra kwargs are fed
        to the UNet (unconditional training).
    """

    def __init__(
        self,
        unet: UNet2DModel,
        *,
        device: Optional[Union[str, torch.device]] = None,
        t_scale: float = 1.0,
        train_target: str = "v",
        model_dir: str = "./artifacts/checkpoints/legacy/pipeline_model",
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        unet_config: Optional[Dict[str, Any]] = None,
        vae=None,
        vae_config: Optional[Dict[str, Any]] = None,
        conditioner=None,
        path_mode: str = "independent",
        path_solver: str = "hungarian",
        layout_cost_resolution: int = 16,
        condition_weight: float = 1.0,
        layout_config=None,
        regiondiff_trainability_info: Optional[Dict[str, Any]] = None,
        distillation_config=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.unet = unet
        self.t_scale = validate_t_scale(t_scale)
        assert train_target in ("v", "x0"), f"train_target must be 'v' or 'x0', got '{train_target}'"
        self.train_target = train_target
        self.model_dir = model_dir
        self.from_norm_to_display = from_norm_to_display or _default_from_norm_to_display
        self.unet_config = unet_config
        self.vae = vae
        self.vae_config = vae_config
        self.conditioner = conditioner
        self.path_mode = str(path_mode)
        self.path_solver = str(path_solver)
        self.layout_cost_resolution = int(layout_cost_resolution)
        self.condition_weight = float(condition_weight)
        self.layout_config = layout_config
        self.regiondiff_trainability_info = dict(regiondiff_trainability_info or {})
        self.distillation_config = distillation_config
        self._attention_teacher = None
        self._current_epoch = 0
        self._current_global_step = 0
        self._kd_training_enabled = False
        self._last_loss_components: Dict[str, Any] = {}
        self.flow_task = FlowMatchingTask(
            train_target=self.train_target,
            path_mode=self.path_mode,
            path_solver=self.path_solver,
            area_loss_fn=self._apply_regiondiff_area_loss_weights,
        )

        # Freeze VAE if present
        if self.vae is not None:
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

    def _metric_prefix(self) -> str:
        return "fm"

    def _checkpoint_stem(self) -> str:
        return "unet_fm"

    def _progress_label(self) -> str:
        return "FM"

    def _sample_tensorboard_tag(self) -> str:
        return f"{self._metric_prefix()}/generated"

    def _save_additional_configs(self) -> None:
        """Persist algorithm-specific config artifacts."""
        if self._uses_regiondiff_layout():
            save_regiondiff_metadata(
                self.unet,
                self.model_dir,
                extra={"trainability": self.regiondiff_trainability_info},
            )

    def _checkpoint_metadata(self) -> Dict[str, Any]:
        metadata = {
            "t_scale": self.t_scale,
            "train_target": self.train_target,
        }
        if self._uses_regiondiff_layout():
            metadata["layout_conditioning"] = {
                "variant": "regiondiff_v1",
                "trainability": self.regiondiff_trainability_info,
            }
        return metadata

    def _uses_regiondiff_layout(self) -> bool:
        return (
            self.layout_config is not None
            and bool(getattr(self.layout_config, "enabled", False))
            and str(getattr(self.layout_config, "variant", "")) == "regiondiff_v1"
        )

    def _uses_regiondiff_area_loss(self) -> bool:
        return self._uses_regiondiff_layout() and bool(
            getattr(self.layout_config, "area_loss_enabled", False)
        )

    def _uses_attention_distillation(self) -> bool:
        return bool(
            self.distillation_config is not None
            and getattr(self.distillation_config, "enabled", False)
        )

    def _ensure_attention_teacher(self, *, torch_dtype: Optional[torch.dtype] = None):
        if not self._uses_attention_distillation():
            return None
        if not self._uses_regiondiff_layout():
            raise ValueError(
                "RegionDiff attention distillation requires "
                "layout_conditioning.variant='regiondiff_v1'."
            )
        checkpoint = getattr(self.distillation_config, "teacher_checkpoint", None)
        if checkpoint is None or not str(checkpoint).strip():
            raise ValueError(
                "distillation.teacher_checkpoint must be set when "
                "distillation.enabled=true."
            )
        if self._attention_teacher is None:
            self._attention_teacher = load_regiondiff_attention_teacher(
                str(checkpoint),
                device=self.device,
                torch_dtype=torch_dtype,
            )
        return self._attention_teacher

    def _apply_regiondiff_area_loss_weights(
        self,
        loss: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]],
    ) -> torch.Tensor:
        """Apply RegionDiff object-focused spatial weights to unreduced loss."""
        if not self._uses_regiondiff_area_loss():
            return loss
        if cond_kwargs is None:
            cond_kwargs = {}
        required = ("boxes_xyxy_norm", "object_mask")
        missing = [key for key in required if key not in cond_kwargs]
        if missing:
            raise KeyError(f"RegionDiff area loss requires conditioning keys: {missing}")

        weights = build_area_weight_map(
            boxes_xyxy_norm=cond_kwargs["boxes_xyxy_norm"],
            object_mask=cond_kwargs["object_mask"],
            latent_height=int(loss.shape[-2]),
            latent_width=int(loss.shape[-1]),
            alpha=float(getattr(self.layout_config, "area_loss_alpha", 1.0)),
            background_weight=float(getattr(self.layout_config, "area_loss_background_weight", 0.5)),
            min_weight=float(getattr(self.layout_config, "area_loss_min_weight", 0.5)),
            max_weight=float(getattr(self.layout_config, "area_loss_max_weight", 4.0)),
        )
        return loss * weights.to(device=loss.device, dtype=loss.dtype)

    def _validate_resume_checkpoint(self, checkpoint: Dict[str, Any], path: str) -> None:
        """Reject full-checkpoint resumes from another training family."""
        if not isinstance(checkpoint, dict) or "unet_state" not in checkpoint:
            raise ValueError(
                f"{self._progress_label()} resume expects a full checkpoint with "
                f"'unet_state', got {path!r}. Use model.pretrained_unet_path for "
                "weights-only initialization."
            )

        basename = os.path.basename(path)
        expected_stem = self._checkpoint_stem()
        if not basename.startswith(f"{expected_stem}_"):
            raise ValueError(
                f"{self._progress_label()} resume checkpoint family mismatch: "
                f"expected filename starting with {expected_stem!r}, got {basename!r}."
            )

        if expected_stem == "unet_fm":
            required = ("t_scale", "train_target")
            foreign = ("num_train_timesteps", "beta_schedule", "prediction_type")
        elif expected_stem == "unet_sd_uncond":
            required = ("num_train_timesteps", "beta_schedule", "prediction_type")
            foreign = ("t_scale", "train_target")
        else:
            return

        missing = [key for key in required if key not in checkpoint]
        present_foreign = [key for key in foreign if key in checkpoint]
        if missing or present_foreign:
            details = []
            if missing:
                details.append(f"missing expected metadata {missing}")
            if present_foreign:
                details.append(f"found foreign metadata {present_foreign}")
            raise ValueError(
                f"{self._progress_label()} resume checkpoint family mismatch for {path!r}: "
                + "; ".join(details)
            )

    def _best_weights_path(self) -> str:
        return os.path.join(self._unet_dir(), f"{self._checkpoint_stem()}_best.pt")

    def _epoch_weights_path(self, epoch_num: int) -> str:
        return os.path.join(self._unet_dir(), f"{self._checkpoint_stem()}_epoch_{epoch_num}.pt")

    def _epoch_checkpoint_path(self, epoch_num: int) -> str:
        return os.path.join(self._unet_dir(), f"{self._checkpoint_stem()}_epoch_{epoch_num}_ckpt.pt")

    def _resolve_resume_path(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]],
    ) -> Optional[str]:
        if resume_from_checkpoint in (None, False, ""):
            return None

        if resume_from_checkpoint is True or resume_from_checkpoint == "latest":
            latest_epoch = -1
            latest_path = None
            pattern = re.compile(
                rf"^{re.escape(self._checkpoint_stem())}_epoch_(\d+)_ckpt\.pt$"
            )
            unet_dir = self._unet_dir()
            if os.path.isdir(unet_dir):
                for filename in os.listdir(unet_dir):
                    match = pattern.match(filename)
                    if match is None:
                        continue
                    epoch = int(match.group(1))
                    if epoch > latest_epoch:
                        latest_epoch = epoch
                        latest_path = os.path.join(unet_dir, filename)
            if latest_path is None:
                raise FileNotFoundError(
                    f"No {self._progress_label()} checkpoint found in {unet_dir} to resume from."
                )
            return latest_path

        path = str(resume_from_checkpoint)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    # ------------------------------------------------------------------
    # Config-driven constructor
    # ------------------------------------------------------------------
    @classmethod
    def from_config(
        cls,
        config,
        *,
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> "FlowMatchingTrainer":
        """Build a trainer from an :class:`FMTrainConfig`.

        Parameters
        ----------
        config : FMTrainConfig
            Structured configuration object.
        from_norm_to_display : callable, optional
            Override display normalisation if needed.
        """
        from src.models.adapters.fm import FMModelAdapter

        device = config.resolved_device() if hasattr(config, "resolved_device") else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model_bundle = FMModelAdapter().build_from_train_config(config, device=device)
        fm_adapter = model_bundle.components["fm_adapter"]
        unet = fm_adapter.unet
        vae = fm_adapter.vae
        unet_cfg = fm_adapter.unet_config
        vae_cfg = fm_adapter.vae_config

        layout_config = getattr(config, "layout_conditioning", None)
        regiondiff_trainability_info = None
        if (
            layout_config is not None
            and bool(getattr(layout_config, "enabled", False))
            and str(getattr(layout_config, "variant", "")) == "regiondiff_v1"
        ):
            unet = build_regiondiff_wrapper(
                base_model=unet,
                region_config=layout_config,
                category_id_to_name=getattr(layout_config, "category_id_to_name", {}),
                num_classes=getattr(layout_config, "num_classes", None),
                backbone_kind="fm_unet2d",
                attachment_kind="attention",
            ).to(device)
            regiondiff_trainability_info = configure_regiondiff_trainability(
                wrapper=unet,
                train_mode=str(getattr(layout_config, "train_mode", "adapters_only")),
                partial_backbone_modules=getattr(layout_config, "partial_backbone_modules", []),
                mixed_precision=getattr(getattr(config, "precision", None), "mixed_precision", None),
            )

        return cls(
            unet,
            device=device,
            t_scale=config.training.t_scale,
            train_target=config.training.train_target,
            model_dir=config.output.model_dir,
            from_norm_to_display=from_norm_to_display,
            unet_config=unet_cfg,
            vae=vae,
            vae_config=vae_cfg,
            path_mode=getattr(config.path, "mode", "independent"),
            path_solver=getattr(config.path, "solver", "hungarian"),
            layout_cost_resolution=getattr(config.path, "layout_cost_resolution", 16),
            condition_weight=getattr(config.path, "condition_weight", 1.0),
            layout_config=layout_config,
            regiondiff_trainability_info=regiondiff_trainability_info,
            distillation_config=getattr(config, "distillation", None),
        )

    def train_from_config(
        self,
        config,
        dataloader: DataLoader,
        eval_dataloader: Optional[DataLoader] = None,
    ) -> None:
        """Launch training driven by an :class:`FMTrainConfig`.

        Extracts all keyword arguments from the config sub-objects and
        delegates to :meth:`train`.
        """
        self.train(
            dataloader=dataloader,
            epochs=config.training.epochs,
            eval_dataloader=eval_dataloader,
            pretrained_vae_path=config.model.vae_weights,
            pretrained_unet_path=config.model.pretrained_unet_path,
            strict_load=config.training.strict_load,
            log_dir=config.output.resolved_log_dir(),
            sample_every=config.sampling.sample_every,
            sample_steps=config.sampling.sample_steps,
            sample_batch_size=config.sampling.sample_batch_size,
            patience=config.training.patience,
            min_delta=config.training.min_delta,
            sample_shape=config.sampling.sample_shape,
            save_every_n_epochs=config.training.save_every_n_epochs,
            eval_every=getattr(config.training, "eval_every", 1),
            resume_from_checkpoint=config.output.resume,
            lr=config.resolved_lr() if hasattr(config, "resolved_lr") else getattr(config.training, "lr", 1e-4),
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
            fixed_validation_examples=getattr(config.sampling, "fixed_validation_examples", 0),
            early_sanity_sample_epoch=getattr(config.sampling, "early_sanity_sample_epoch", 0),
            save_debug_images=getattr(config.sampling, "save_debug_images", False),
            debug_dir=config.output.resolved_debug_dir(),
        )

    # ------------------------------------------------------------------
    # Directory helpers
    # ------------------------------------------------------------------
    def _unet_dir(self) -> str:
        return os.path.join(self.model_dir, "UNET")

    def _vae_dir(self) -> str:
        return os.path.join(self.model_dir, "VAE")

    def _ensure_dirs(self) -> None:
        os.makedirs(self._unet_dir(), exist_ok=True)
        if self.vae is not None:
            os.makedirs(self._vae_dir(), exist_ok=True)

    def _save_configs(self) -> None:
        if self.unet_config is not None:
            save_unet_config(self.unet_config, os.path.join(self._unet_dir(), "config.json"))
        if self.vae_config is not None:
            os.makedirs(self._vae_dir(), exist_ok=True)
            import json
            path = os.path.join(self._vae_dir(), "config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.vae_config, f, indent=2, sort_keys=True)
        self._save_additional_configs()
        write_artifact_manifest(self.model_dir, self._build_artifact_manifest())

    def _build_artifact_manifest(self) -> ArtifactManifest:
        layout_variant = None
        if self.layout_config is not None:
            layout_variant = getattr(self.layout_config, "variant", None)

        components: Dict[str, Any] = {}
        if self.unet_config is not None:
            components["unet"] = {
                "kind": "unet",
                "config": "UNET/config.json",
            }
        if self.vae_config is not None:
            components["vae"] = {
                "kind": "vae",
                "config": "VAE/config.json",
            }

        adapters = []
        if layout_variant:
            adapters.append(
                {
                    "kind": "layout_conditioning",
                    "variant": str(layout_variant),
                    "trainability": self.regiondiff_trainability_info,
                }
            )

        base_model = None
        if isinstance(self.vae_config, dict):
            base_model = (
                self.vae_config.get("pretrained_model_name_or_path")
                or self.vae_config.get("_pretrained_model_name_or_path")
            )

        return ArtifactManifest(
            schema_version=1,
            model_kind="native_fm_unet",
            model_family="flow_matching",
            base_model=base_model,
            components=components,
            adapters=adapters,
            task={
                "kind": "flow_matching",
                "train_target": self.train_target,
                "t_scale": self.t_scale,
                "path_mode": self.path_mode,
                "path_solver": self.path_solver,
                "layout_conditioning": str(layout_variant) if layout_variant else None,
            },
            dataset={},
            normalization={},
            checkpoints={
                "best": os.path.join("UNET", f"{self._checkpoint_stem()}_best.pt"),
                "epoch_pattern": os.path.join("UNET", f"{self._checkpoint_stem()}_epoch_{{epoch}}.pt"),
                "resume_pattern": os.path.join("UNET", f"{self._checkpoint_stem()}_epoch_{{epoch}}_ckpt.pt"),
            },
            metadata={
                "checkpoint_metadata": self._checkpoint_metadata(),
                "legacy_config_files": {
                    "unet": "UNET/config.json" if self.unet_config is not None else None,
                    "vae": "VAE/config.json" if self.vae_config is not None else None,
                },
            },
        )

    # ------------------------------------------------------------------
    # Weight I/O
    # ------------------------------------------------------------------
    def load_unet_weights(self, path: str, *, strict: bool = True) -> None:
        state = torch.load(path, map_location=self.device)
        if isinstance(state, dict) and "unet_state" in state:
            state = state["unet_state"]
        load_target = self.unet
        load_label = "unet"
        allow_missing_regiondiff_adapters = False
        base_attr = None
        if hasattr(self.unet, "base_model"):
            base_attr = "base_model"
        elif hasattr(self.unet, "base_unet"):
            base_attr = "base_unet"
        if isinstance(state, dict) and base_attr is not None:
            state_keys = [str(key) for key in state.keys()]
            is_wrapped_state = any(
                key.startswith(("base_model.", "base_unet.", "layout_tokenizer."))
                or ".region_adapter." in key
                for key in state_keys
            )
            if not is_wrapped_state:
                load_target = getattr(self.unet, base_attr)
                load_label = f"unet.{base_attr}"
                allow_missing_regiondiff_adapters = True
                target_keys = set(load_target.state_dict().keys())
                remapped_state = {}
                for key, value in state.items():
                    key_str = str(key)
                    mapped_key = key_str
                    if key_str not in target_keys and ".attentions." in key_str:
                        prefix, suffix = key_str.split(".attentions.", 1)
                        parts = suffix.split(".", 1)
                        if len(parts) == 2:
                            mapped_key = f"{prefix}.attentions.{parts[0]}.base_attention.{parts[1]}"
                    remapped_state[mapped_key if mapped_key in target_keys else key_str] = value
                state = remapped_state

        effective_strict = strict and not allow_missing_regiondiff_adapters
        missing, unexpected = load_target.load_state_dict(state, strict=effective_strict)
        if strict and allow_missing_regiondiff_adapters:
            disallowed_missing = [
                key for key in missing if ".region_adapter." not in str(key)
            ]
            if disallowed_missing or unexpected:
                details = []
                if disallowed_missing:
                    details.append(f"Missing non-RegionDiff keys: {disallowed_missing}")
                if unexpected:
                    details.append(f"Unexpected keys: {unexpected}")
                raise RuntimeError(
                    "Error(s) in loading base UNet checkpoint into RegionDiff model: "
                    + "; ".join(details)
                )
        if (not strict) or missing or unexpected:
            print(f"[load_unet_weights] target={load_label} strict={strict}")
            if strict and allow_missing_regiondiff_adapters and missing and not unexpected:
                print(
                    "  Missing RegionDiff adapter keys were initialized from scratch "
                    f"({len(missing)} keys)."
                )
            elif missing:
                print("  Missing keys:", missing)
            if unexpected:
                print("  Unexpected keys:", unexpected)

    def save_unet_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(module_state_dict_cpu(self.unet), path)
        release_cuda_cache()

    def load_vae_weights(self, path: str, *, strict: bool = True) -> None:
        assert self.vae is not None, "VAE not set."
        from src.models.vae import load_vae_weights as _load_vae_weights

        _load_vae_weights(
            self.vae,
            path,
            strict=strict,
            map_location=self.device,
        )

    # ------------------------------------------------------------------
    # Encode helper (pixel-space passthrough or VAE)
    # ------------------------------------------------------------------
    def encode_fm_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.vae is None:
            return x
        with torch.no_grad():
            z_mu, z_sigma = self.vae.encode(x)
            return self.vae.sampling(z_mu, z_sigma)

    def _prepare_batch(self, batch) -> tuple[torch.Tensor, Dict[str, Any]]:
        if torch.is_tensor(batch):
            return batch.to(self.device), {}
        if isinstance(batch, dict) and "pixel_values" in batch:
            cond_kwargs: Dict[str, Any] = {}
            if self._uses_regiondiff_layout():
                required = ("boxes_xyxy_norm", "labels", "object_mask")
                missing = [key for key in required if key not in batch]
                if missing:
                    raise KeyError(f"RegionDiff layout batch is missing keys: {missing}")
                cond_kwargs = {
                    "boxes_xyxy_norm": batch["boxes_xyxy_norm"].to(self.device),
                    "labels": batch["labels"].to(self.device),
                    "object_mask": batch["object_mask"].to(self.device),
                }
            return batch["pixel_values"].to(self.device), cond_kwargs
        raise TypeError(
            "Expected a tensor batch or a dict containing 'pixel_values' for "
            f"{self.__class__.__name__}, got {type(batch)!r}."
        )

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
            raise ValueError("RegionDiff validation batching requires an explicit collate_fn")
        return collate_fn(samples)

    @torch.no_grad()
    def _log_regiondiff_validation_samples(
        self,
        writer: SummaryWriter,
        *,
        sampler,
        fixed_batch: Dict[str, Any],
        epoch: int,
        steps: int,
        sample_shape: Optional[Tuple[int, int, int]],
        max_logged_images: int,
        save_debug_images: bool,
        debug_dir: str,
        tag: Optional[str] = None,
    ) -> None:
        vis_batch = self._slice_batch(fixed_batch, max_logged_images)
        if hasattr(sampler, "sample_euler_layout"):
            generated_latents = sampler.sample_euler_layout(
                vis_batch,
                steps=steps,
                sample_shape=sample_shape,
            )
        elif hasattr(sampler, "sample_layout"):
            generated_latents = sampler.sample_layout(
                vis_batch,
                steps=steps,
                sample_shape=sample_shape,
            )
        else:
            raise TypeError(
                f"{sampler.__class__.__name__} does not support RegionDiff layout sampling."
            )

        generated_images = sampler.decode(generated_latents)
        generated_display = self.from_norm_to_display(generated_images).clamp(0.0, 1.0)
        image_h, image_w = generated_display.shape[-2:]
        overlay_boxes = vis_batch["boxes_xyxy_norm"].detach().cpu().to(torch.float32).clone()
        overlay_boxes[..., 0::2] *= float(image_w)
        overlay_boxes[..., 1::2] *= float(image_h)
        overlay_line_width = max(2, int(round(min(image_h, image_w) / 128.0)))
        generated_overlay = draw_bbox_overlays(
            generated_display,
            boxes_xyxy=overlay_boxes,
            labels=vis_batch["labels"],
            object_mask=vis_batch["object_mask"],
            line_width=overlay_line_width,
        )
        layout_canvas = render_class_layout(
            boxes_xyxy=overlay_boxes,
            labels=vis_batch["labels"],
            object_mask=vis_batch["object_mask"],
            image_size=int(image_h),
        )

        prefix = tag or self._sample_tensorboard_tag()
        writer.add_images(prefix, generated_overlay.detach().cpu(), epoch)
        writer.add_images(f"{prefix}_clean", generated_display.detach().cpu(), epoch)
        writer.add_images(f"{prefix}_boxes", generated_overlay.detach().cpu(), epoch)
        writer.add_images(f"{prefix}_layout", layout_canvas.detach().cpu(), epoch)

        if save_debug_images:
            step_dir = os.path.join(debug_dir, f"epoch_{epoch + 1:04d}")
            save_image_batch(generated_display.detach().cpu(), output_dir=step_dir, prefix="generated_clean")
            save_image_batch(generated_overlay.detach().cpu(), output_dir=step_dir, prefix="generated")
            save_image_batch(generated_overlay.detach().cpu(), output_dir=step_dir, prefix="generated_boxes")
            save_image_batch(layout_canvas.detach().cpu(), output_dir=step_dir, prefix="layout")
        release_cuda_cache()

    def _compute_batch_loss(
        self,
        x_fm: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        if not self._uses_attention_distillation():
            return self.flow_matching_step(x_fm, cond_kwargs)

        state = self._sample_flow_matching_state(x_fm, cond_kwargs)
        kd_enabled = self._should_apply_attention_kd(state["t"])
        student_recorder = (
            RegionDiffAttentionRecorder(
                self.unet,
                selected_layers=getattr(self.distillation_config, "selected_region_layers", []),
                detach=False,
            )
            if kd_enabled
            else None
        )

        if student_recorder is None:
            unet_out = self._forward_flow_matching_unet(state)
        else:
            with student_recorder:
                unet_out = self._forward_flow_matching_unet(state)
        base_loss = self._compute_flow_matching_loss_from_prediction(unet_out, state)

        kd_loss = base_loss.new_zeros(())
        kd_diagnostics: Dict[str, Any] = {
            "matched_layers": 0,
            "selected_instances": 0,
            "skipped_layers_shape": 0,
            "skipped_layers_missing": 0,
            "loss_by_layer": {},
        }
        if kd_enabled and student_recorder is not None:
            teacher = self._ensure_attention_teacher(
                torch_dtype=getattr(self, "_distillation_teacher_torch_dtype", None)
            )
            teacher_recorder = RegionDiffAttentionRecorder(
                teacher.unet,
                selected_layers=getattr(self.distillation_config, "selected_region_layers", []),
                detach=bool(getattr(self.distillation_config, "detach_teacher", True)),
            )
            with teacher_recorder:
                teacher.forward_attention(
                    noisy_latents=state["zt"],
                    fm_t=state["t"],
                    cond_kwargs=state["cond_kwargs"],
                    detach_teacher=bool(getattr(self.distillation_config, "detach_teacher", True)),
                )
            kd_loss, kd_diagnostics = compute_region_attention_distillation_loss(
                teacher_attention_maps=teacher_recorder.records,
                student_attention_maps=student_recorder.records,
                boxes_xyxy_norm=state["cond_kwargs"]["boxes_xyxy_norm"],
                labels=state["cond_kwargs"]["labels"],
                object_mask=state["cond_kwargs"]["object_mask"],
                timesteps=state["t"],
                distillation_config=self.distillation_config,
                category_id_to_name=getattr(self.layout_config, "category_id_to_name", {}),
            )

        kd_weight = float(getattr(self.distillation_config, "lambda_attn", 0.0))
        weighted_kd = kd_loss * kd_weight
        total_loss = base_loss + weighted_kd
        self._last_loss_components = {
            "base_loss": base_loss.detach(),
            "attention_kd_loss": kd_loss.detach(),
            "attention_kd_weighted": weighted_kd.detach(),
            "total_loss": total_loss.detach(),
            "attention_kd_diagnostics": kd_diagnostics,
        }
        return total_loss

    # ------------------------------------------------------------------
    # Flow-matching loss
    # ------------------------------------------------------------------
    @staticmethod
    def _permute_conditioning_kwargs(
        cond_kwargs: Dict[str, Any],
        permutation: Optional[torch.Tensor],
        batch_size: int,
    ) -> Dict[str, Any]:
        """Keep batch-aligned conditioning tensors paired with matched targets."""
        return FlowMatchingTask.permute_conditioning_kwargs(
            cond_kwargs,
            permutation,
            batch_size,
        )

    def _match_flow_targets_with_permutation(
        self,
        z0: torch.Tensor,
        x_fm: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Apply the configured path coupling and return targets plus permutation."""
        return self.flow_task.match_targets_with_permutation(z0, x_fm, cond_kwargs)

    def _match_flow_targets(
        self,
        z0: torch.Tensor,
        x_fm: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Apply the configured path coupling and return the matched targets."""
        return self.flow_task.match_targets(z0, x_fm, cond_kwargs)

    def _sample_flow_matching_state(
        self,
        x_fm: torch.Tensor,
        cond_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Sample FM path variables and keep layout kwargs aligned to targets."""
        if cond_kwargs is None:
            cond_kwargs = {}
        batch_size = x_fm.shape[0]
        z0 = torch.randn_like(x_fm)
        t = torch.rand(batch_size, device=x_fm.device)
        t_expanded = t[:, None, None, None]
        x_target, target_permutation = self._match_flow_targets_with_permutation(z0, x_fm, cond_kwargs)
        cond_kwargs = self._permute_conditioning_kwargs(cond_kwargs, target_permutation, batch_size)

        zt = (1 - t_expanded) * z0 + t_expanded * x_target
        v_target = x_target - z0
        return {
            "z0": z0,
            "t": t,
            "t_expanded": t_expanded,
            "x_target": x_target,
            "target_permutation": target_permutation,
            "cond_kwargs": cond_kwargs,
            "zt": zt,
            "v_target": v_target,
        }

    def _forward_flow_matching_unet(self, state: Dict[str, Any]) -> torch.Tensor:
        return self.unet(
            state["zt"],
            state["t"] * self.t_scale,
            **state["cond_kwargs"],
        ).sample

    def _compute_flow_matching_loss_from_prediction(
        self,
        unet_out: torch.Tensor,
        state: Dict[str, Any],
    ) -> torch.Tensor:
        self.flow_task.train_target = self.train_target
        self.flow_task.area_loss_fn = self._apply_regiondiff_area_loss_weights
        return self.flow_task.loss_from_prediction(unet_out, state)

    def _should_apply_attention_kd(self, t: torch.Tensor) -> bool:
        if not self._uses_attention_distillation() or not self._kd_training_enabled:
            return False
        if int(self._current_epoch) < int(getattr(self.distillation_config, "warmup_epochs", 0)):
            return False
        start, end = getattr(self.distillation_config, "timestep_range", (0.0, 1.0))
        in_range = (t.detach() >= float(start)) & (t.detach() <= float(end))
        return bool(in_range.any().item())

    def flow_matching_step(self, x_fm: torch.Tensor, cond_kwargs: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """Compute a single flow-matching loss on encoded input *x_fm*."""
        state = self._sample_flow_matching_state(x_fm, cond_kwargs)
        unet_out = self._forward_flow_matching_unet(state)
        loss = self._compute_flow_matching_loss_from_prediction(unet_out, state)
        self._last_loss_components = {
            "base_loss": loss.detach(),
            "attention_kd_loss": loss.detach() * 0.0,
            "attention_kd_weighted": loss.detach() * 0.0,
            "total_loss": loss.detach(),
            "attention_kd_diagnostics": {
                "matched_layers": 0,
                "selected_instances": 0,
                "skipped_layers_shape": 0,
                "skipped_layers_missing": 0,
            },
        }
        return loss

    # ------------------------------------------------------------------
    # Build a sampler for sample-at-epoch
    # ------------------------------------------------------------------
    def _make_sampler(self) -> FlowMatchingSampler:
        if self.vae is not None:
            return FlowMatchingSampler.from_stable(
                self.unet,
                self.vae,
                device=self.device,
                t_scale=self.t_scale,
                train_target=self.train_target,
                from_norm_to_display=self.from_norm_to_display,
            )
        return FlowMatchingSampler(
            self.unet,
            device=self.device,
            t_scale=self.t_scale,
            train_target=self.train_target,
            from_norm_to_display=self.from_norm_to_display,
        )

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------
    def train(
        self,
        dataloader: DataLoader,
        epochs: int,
        eval_dataloader: Optional[DataLoader] = None,
        *,
        pretrained_vae_path: Optional[str] = None,
        pretrained_unet_path: Optional[str] = None,
        strict_load: bool = True,
        log_dir: str = "./artifacts/runs/main/flow_matching",
        sample_every: int = 1,
        sample_steps: int = 50,
        sample_batch_size: int = 4,
        patience: Optional[int] = None,
        min_delta: float = 0.0,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        save_every_n_epochs: int = 1,
        eval_every: int = 1,
        resume_from_checkpoint: Optional[str] = None,
        lr: float = 1e-4,
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
        fixed_validation_examples: int = 0,
        early_sanity_sample_epoch: int = 0,
        save_debug_images: bool = False,
        debug_dir: str = "./artifacts/debug/flow_matching",
    ) -> None:
        eval_every = int(eval_every)
        if patience is not None and eval_dataloader is None:
            raise ValueError("eval_dataloader must be provided when using patience early stopping.")
        if patience is not None and eval_every <= 0:
            raise ValueError("eval_every must be > 0 when using patience early stopping.")
        self._ensure_dirs()
        self._save_configs()
        resume_path = self._resolve_resume_path(resume_from_checkpoint)

        # Pre-load weights
        if pretrained_vae_path is not None and self.vae is not None:
            self.load_vae_weights(pretrained_vae_path, strict=strict_load)
            # Re-freeze after loading
            self.vae.eval()
            for p in self.vae.parameters():
                p.requires_grad = False

        if resume_path is not None and pretrained_unet_path is not None:
            print(
                f"[{self._progress_label()} Resume] Ignoring pretrained_unet_path because "
                "a full resume checkpoint was provided."
            )
        elif pretrained_unet_path is not None:
            self.load_unet_weights(pretrained_unet_path, strict=strict_load)

        total_steps = max(1, epochs * len(dataloader))
        precision, scaler = setup_precision(self.device, mixed_precision)
        self._distillation_teacher_torch_dtype = precision.dtype if precision.enabled else None
        optimizer_params = self.unet.parameters()
        if self._uses_regiondiff_layout():
            optimizer_params = regiondiff_optimizer_param_groups(
                wrapper=self.unet,
                adapter_learning_rate=getattr(self.layout_config, "adapter_learning_rate", lr),
                backbone_learning_rate=getattr(self.layout_config, "backbone_learning_rate", lr),
            )
        optimizer = build_optimizer(
            optimizer_params,
            optimizer_name=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            beta1=beta1,
            beta2=beta2,
        )
        scheduler = build_lr_scheduler(
            optimizer,
            scheduler_name=scheduler_name,
            total_steps=total_steps,
            warmup_ratio=warmup_ratio,
            min_lr_ratio=min_lr_ratio,
        )
        ema = build_ema(self.unet, enabled=ema_enabled, decay=ema_decay)
        if self._uses_attention_distillation():
            self._ensure_attention_teacher(torch_dtype=self._distillation_teacher_torch_dtype)

        # Resume state
        global_step = 0
        best_eval = float("inf")
        best_epoch = -1
        bad_epochs = 0
        start_epoch = 0

        if resume_path is not None:
            print(f"[{self._progress_label()} Resume] Loading checkpoint from {resume_path}")
            _ckpt, start_epoch, progress = restore_training_checkpoint(
                resume_path,
                device=self.device,
                model_states={"unet_state": self.unet},
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                ema_model=self.unet,
                restore_rng=True,
                validate_checkpoint=self._validate_resume_checkpoint,
            )
            global_step = progress.global_step
            best_eval = progress.best_eval
            best_epoch = progress.best_epoch
            bad_epochs = progress.bad_epochs
            print(
                f"[{self._progress_label()} Resume] Resuming from epoch {start_epoch}, "
                f"global_step={global_step}, best_eval={best_eval:.6f}"
            )

        writer = build_summary_writer(log_dir)

        def _save_checkpoint(path: str, epoch_idx: int) -> None:
            progress = TrainingProgressState(
                epoch=epoch_idx,
                global_step=global_step,
                best_eval=best_eval,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
            )
            ckpt = build_training_checkpoint(
                model_states={"unet_state": self.unet},
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                ema=ema,
                progress=progress,
                extra_metadata=self._checkpoint_metadata(),
                include_rng=True,
            )
            save_training_checkpoint(path, ckpt)

        early_sanity_sample_epoch = int(early_sanity_sample_epoch)
        sampler_obj = self._make_sampler() if sample_every > 0 or early_sanity_sample_epoch > 0 else None
        fixed_batch = None
        if sampler_obj is not None and self._uses_regiondiff_layout():
            fixed_batch = self._build_fixed_validation_batch(
                eval_dataloader or dataloader,
                min(int(fixed_validation_examples), int(sample_batch_size)),
            )
            if save_debug_images:
                os.makedirs(debug_dir, exist_ok=True)

        if precision.requested != precision.mode:
            print(
                f"[FM Precision] requested={precision.requested!r} -> using {precision.mode!r} "
                f"on device={precision.device_type}"
            )

        for epoch in range(start_epoch, epochs):
            set_epoch_for_dataloader(dataloader, epoch)
            set_epoch_for_dataloader(eval_dataloader, epoch)
            self.unet.train()
            total_loss = 0.0

            for batch in tqdm(dataloader, desc=f"{self._progress_label()} Epoch {epoch+1}/{epochs}"):
                x, cond_kw = self._prepare_batch(batch)
                with torch.no_grad():
                    x_fm = self.encode_fm_input(x)
                if self.conditioner is not None:
                    cond_kw.update(self.conditioner.prepare_for_training(x, self.device))
                optimizer.zero_grad(set_to_none=True)
                self._current_epoch = epoch
                self._current_global_step = global_step
                self._kd_training_enabled = True
                with autocast_context(precision):
                    loss = self._compute_batch_loss(x_fm, cond_kw)

                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()

                if max_grad_norm and max_grad_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(self.unet.parameters(), max_grad_norm)

                if scaler is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                if ema is not None and global_step >= int(ema_start_step):
                    ema.update(self.unet)

                total_loss += loss.item()
                components = getattr(self, "_last_loss_components", {})
                kd_diag = components.get("attention_kd_diagnostics", {}) if isinstance(components, dict) else {}

                def _scalar_component(name: str, default: float) -> float:
                    value = components.get(name, default) if isinstance(components, dict) else default
                    if torch.is_tensor(value):
                        return float(value.detach().float().cpu().item())
                    return float(value)

                writer.add_scalar(f"{self._metric_prefix()}/loss_step", loss.item(), global_step)
                writer.add_scalar(
                    f"{self._metric_prefix()}/base_loss_step",
                    _scalar_component("base_loss", loss.item()),
                    global_step,
                )
                writer.add_scalar(
                    f"{self._metric_prefix()}/attention_kd_loss_step",
                    _scalar_component("attention_kd_loss", 0.0),
                    global_step,
                )
                writer.add_scalar(
                    f"{self._metric_prefix()}/attention_kd_weighted_step",
                    _scalar_component("attention_kd_weighted", 0.0),
                    global_step,
                )
                writer.add_scalar(
                    f"{self._metric_prefix()}/total_loss_step",
                    _scalar_component("total_loss", loss.item()),
                    global_step,
                )
                writer.add_scalar(
                    f"{self._metric_prefix()}/attention_kd_matched_layers",
                    float(kd_diag.get("matched_layers", 0)),
                    global_step,
                )
                writer.add_scalar(
                    f"{self._metric_prefix()}/attention_kd_selected_instances",
                    float(kd_diag.get("selected_instances", 0)),
                    global_step,
                )
                writer.add_scalar(
                    f"{self._metric_prefix()}/attention_kd_skipped_shape",
                    float(kd_diag.get("skipped_layers_shape", 0)),
                    global_step,
                )
                writer.add_scalar(
                    f"{self._metric_prefix()}/attention_kd_skipped_missing",
                    float(kd_diag.get("skipped_layers_missing", 0)),
                    global_step,
                )
                writer.add_scalar(
                    f"{self._metric_prefix()}/lr",
                    float(optimizer.param_groups[0]["lr"]),
                    global_step,
                )
                global_step += 1

            avg_loss = total_loss / max(1, len(dataloader))
            print(f"[{self._progress_label()} Epoch {epoch+1}] loss: {avg_loss:.6f}")
            writer.add_scalar(f"{self._metric_prefix()}/loss_epoch", avg_loss, epoch)

            if (save_every_n_epochs is not None) and ((epoch + 1) % save_every_n_epochs == 0):
                self.save_unet_weights(self._epoch_weights_path(epoch + 1))
                _save_checkpoint(self._epoch_checkpoint_path(epoch + 1), epoch_idx=epoch)

            # Eval + early stopping + best save
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
                        for batch in tqdm(eval_dataloader, desc=f"{self._progress_label()} Eval  {epoch+1}/{epochs}"):
                            x, cond_kw = self._prepare_batch(batch)
                            x_fm = self.encode_fm_input(x)
                            if self.conditioner is not None:
                                cond_kw.update(self.conditioner.prepare_for_training(x, self.device))
                            self._current_epoch = epoch
                            self._current_global_step = global_step
                            self._kd_training_enabled = False
                            with autocast_context(precision):
                                loss = self._compute_batch_loss(x_fm, cond_kw)

                            bs = x.size(0)
                            eval_loss += loss.item() * bs
                            n_eval += bs

                avg_eval_loss = eval_loss / max(1, n_eval)
                print(f"  [Eval loss: {avg_eval_loss:.6f}]")
                writer.add_scalar(f"{self._metric_prefix()}/eval_loss_epoch", avg_eval_loss, epoch)
                release_cuda_cache()

                improved = (best_eval - avg_eval_loss) > min_delta
                if improved:
                    best_eval = avg_eval_loss
                    best_epoch = epoch
                    bad_epochs = 0
                    if ema is not None and global_step >= int(ema_start_step):
                        with ema.average_parameters(self.unet):
                            self.save_unet_weights(self._best_weights_path())
                    else:
                        self.save_unet_weights(self._best_weights_path())
                    print(
                        f"  ✅ New best eval_loss={best_eval:.6f} at epoch {epoch+1} "
                        f"-> saved UNET/{os.path.basename(self._best_weights_path())}"
                    )
                elif patience is not None:
                    bad_epochs += 1
                    print(f"  ⏳ No improvement (best={best_eval:.6f}), bad_epochs={bad_epochs}/{patience}")
                    if bad_epochs >= patience:
                        print(f"🛑 Early stopping triggered. Best epoch: {best_epoch+1} (eval_loss={best_eval:.6f})")
                        break

            should_run_early_sanity_sample = (
                sampler_obj is not None
                and early_sanity_sample_epoch > 0
                and (epoch + 1) % early_sanity_sample_epoch == 0
            )
            if should_run_early_sanity_sample:
                ema_context = ema.average_parameters(self.unet) if ema is not None and global_step >= int(ema_start_step) else torch.no_grad()
                with ema_context:
                    if self._uses_regiondiff_layout():
                        if fixed_batch is None:
                            print(
                                "[RegionDiff Sampling] Skipping early sanity layout sample logging because "
                                "sampling.fixed_validation_examples <= 0 or no validation samples are available."
                            )
                        else:
                            self._log_regiondiff_validation_samples(
                                writer,
                                sampler=sampler_obj,
                                fixed_batch=fixed_batch,
                                epoch=epoch,
                                steps=sample_steps,
                                sample_shape=sample_shape,
                                max_logged_images=sample_batch_size,
                                save_debug_images=save_debug_images,
                                debug_dir=debug_dir,
                                tag=f"{self._sample_tensorboard_tag()}_early_sanity",
                            )
                    else:
                        sampler_obj.log_samples_to_tensorboard(
                            writer=writer,
                            epoch=epoch,
                            steps=sample_steps,
                            batch_size=sample_batch_size,
                            tag=f"{self._sample_tensorboard_tag()}_early_sanity",
                            sample_shape=sample_shape,
                        )
                release_cuda_cache()

            # Sampling
            if sampler_obj is not None and sample_every > 0 and (epoch + 1) % sample_every == 0:
                ema_context = ema.average_parameters(self.unet) if ema is not None and global_step >= int(ema_start_step) else torch.no_grad()
                with ema_context:
                    if self._uses_regiondiff_layout():
                        if fixed_batch is None:
                            print(
                                "[RegionDiff Sampling] Skipping layout sample logging because "
                                "sampling.fixed_validation_examples <= 0 or no validation samples are available."
                            )
                        else:
                            self._log_regiondiff_validation_samples(
                                writer,
                                sampler=sampler_obj,
                                fixed_batch=fixed_batch,
                                epoch=epoch,
                                steps=sample_steps,
                                sample_shape=sample_shape,
                                max_logged_images=sample_batch_size,
                                save_debug_images=save_debug_images,
                                debug_dir=debug_dir,
                            )
                    else:
                        sampler_obj.log_samples_to_tensorboard(
                            writer=writer,
                            epoch=epoch,
                            steps=sample_steps,
                            batch_size=sample_batch_size,
                            tag=self._sample_tensorboard_tag(),
                            sample_shape=sample_shape,
                        )
                release_cuda_cache()

        writer.close()


# ── registry ──────────────────────────────────────────────────────────────────
from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.trainer.register("default_fm", default=True)(FlowMatchingTrainer)

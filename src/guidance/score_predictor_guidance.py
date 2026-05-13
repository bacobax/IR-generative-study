"""Energy-based score-predictor guidance for Flow Matching sampling."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F

from src.guidance.base_guidance import BaseGuidance
from train_surprise_predictor import SurprisePredictor


@dataclass
class ScoreGuidanceConfig:
    """Hyperparameters that control score-predictor-guided sampling."""

    ckpt_path: str = ""
    vae_config_path: Optional[str] = None
    vae_weights_path: Optional[str] = None
    dino_name: Optional[str] = None
    hidden_dim: Optional[int] = None

    energy_mode: str = "surprise"
    sign: str = "minimize"
    w_surprise: float = 1.0
    w_gmm: float = 1.0

    lambda_start: float = 1.0
    lambda_end: float = 1.0
    lambda_schedule: str = "constant"

    grad_clip_norm: Optional[float] = None
    normalize_grad: bool = False

    guidance_on: str = "latent"
    use_ddim_hat: bool = False

    use_amp: bool = False
    detach_base_velocity: bool = False

    num_refine_steps: int = 10
    refine_step_size: float = 0.01


def _compute_lambda(t: float, cfg: ScoreGuidanceConfig) -> float:
    """Return lambda(t) in [lambda_start, lambda_end] for normalized flow time."""
    start, end = cfg.lambda_start, cfg.lambda_end
    schedule = cfg.lambda_schedule
    if schedule == "constant":
        return start
    if schedule == "linear":
        return start + (end - start) * t
    if schedule == "cosine":
        return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * t))
    if schedule == "step":
        return start if t < 0.5 else end
    return start


class ScorePredictorGuidance(BaseGuidance):
    """Wrap a trained ``SurprisePredictor`` for gradient-based guidance."""

    def __init__(
        self,
        predictor: SurprisePredictor,
        config: ScoreGuidanceConfig,
        device: str = "cpu",
    ) -> None:
        self.predictor = predictor
        self.config = config
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        config: ScoreGuidanceConfig,
        device: str = "cpu",
    ) -> "ScorePredictorGuidance":
        """Build guidance from a ``best_model.pt`` checkpoint."""
        ckpt_path = config.ckpt_path
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(
                f"[ScorePredictorGuidance] Checkpoint not found: {ckpt_path}\n"
                "  If DINOv2 has never been downloaded, ensure internet access once;\n"
                "  it will be cached at ~/.cache/torch/hub/ for subsequent offline runs."
            )

        print(f"[ScorePredictorGuidance] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_args: dict = ckpt.get("args", {})

        vae_config_path = config.vae_config_path or ckpt_args.get("vae_config")
        vae_weights_path = config.vae_weights_path or ckpt_args.get("vae_weights")
        dino_name = config.dino_name or ckpt_args.get("dino_name", "dinov2_vits14")
        hidden_dim = config.hidden_dim or int(ckpt_args.get("hidden_dim", 256))

        if not vae_config_path or not vae_weights_path:
            raise ValueError(
                "[ScorePredictorGuidance] vae_config_path and vae_weights_path must be "
                "provided either via ScoreGuidanceConfig or embedded in ckpt['args']."
            )

        try:
            predictor = SurprisePredictor(
                vae_config_path=vae_config_path,
                vae_weights_path=vae_weights_path,
                dino_name=dino_name,
                hidden_dim=hidden_dim,
                device="cpu",
            )
        except Exception as exc:
            raise RuntimeError(
                "[ScorePredictorGuidance] Failed to build SurprisePredictor.\n"
                "  If DINOv2 weights are not cached yet, run once with internet access.\n"
                f"  Original error: {exc}"
            ) from exc

        missing, unexpected = predictor.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            print(f"  [WARNING] Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"  [WARNING] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

        predictor = predictor.to(device)
        predictor.eval()
        for param in predictor.parameters():
            param.requires_grad = False
        predictor.dino.eval()

        print(
            f"[ScorePredictorGuidance] Ready.  "
            f"energy_mode={config.energy_mode}  sign={config.sign}  "
            f"guidance_on={config.guidance_on}  schedule={config.lambda_schedule}"
        )
        return cls(predictor=predictor, config=config, device=device)

    def compute_lambda(self, t: float) -> float:
        """Return lambda(t) using the configured schedule."""
        return _compute_lambda(t, self.config)

    def predict(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the predictor on latents and return raw score tensors."""
        pred_surprise, pred_gmm = self.predictor(z)
        return {"surprise": pred_surprise, "gmm": pred_gmm}

    def energy(
        self,
        z: torch.Tensor,
        t: Optional[float] = None,
        pipeline: Any = None,
    ) -> torch.Tensor:
        """Compute per-sample energy with shape ``(B,)``."""
        cfg = self.config

        if cfg.guidance_on == "decoded" and pipeline is not None:
            x = pipeline.decode_fm_output(z)
            if x.shape[1] != 1:
                x = x[:, :1]
            x_256 = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x_norm = self.predictor._per_image_minmax(x_256)
            dino_in = self.predictor._to_dino_input(x_norm)
            tokens = self.predictor._extract_patch_tokens(dino_in)
            pooled = tokens.mean(dim=1)
            pooled = self.predictor.pool_norm(pooled)
            hidden = self.predictor.trunk(pooled)
            pred_surprise = self.predictor.head_surprise(hidden).squeeze(-1)
            pred_gmm = self.predictor.head_gmm(hidden).squeeze(-1)
        else:
            pred_surprise, pred_gmm = self.predictor(z)

        if cfg.energy_mode == "surprise":
            return pred_surprise
        if cfg.energy_mode == "gmm":
            return 1.0 - pred_gmm
        if cfg.energy_mode == "combo":
            return cfg.w_surprise * pred_surprise + cfg.w_gmm * (1.0 - pred_gmm)
        raise ValueError(
            f"[ScorePredictorGuidance] Unknown energy_mode={cfg.energy_mode!r}. "
            "Choose from 'surprise', 'gmm', 'combo'."
        )

    def guidance_grad(
        self,
        z: torch.Tensor,
        t: Optional[float] = None,
        pipeline: Any = None,
        velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the guidance gradient to add to the velocity field."""
        cfg = self.config
        z_leaf = z.detach().requires_grad_(True)

        z_energy = z_leaf
        if cfg.use_ddim_hat and velocity is not None:
            t_val = float(t if t is not None else 0.0)
            z_energy = z_leaf + (1.0 - t_val) * velocity.detach()

        with torch.amp.autocast(
            "cuda",
            enabled=cfg.use_amp and z.device.type == "cuda",
        ):
            scalar = self.energy(z_energy, t=t, pipeline=pipeline).mean()

        (raw_grad,) = torch.autograd.grad(scalar, z_leaf, create_graph=False)
        grad = -raw_grad if cfg.sign == "minimize" else raw_grad

        if cfg.normalize_grad:
            batch_size = grad.shape[0]
            norms = grad.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            grad = grad / norms.clamp(min=1e-8)

        if cfg.grad_clip_norm is not None:
            batch_size = grad.shape[0]
            norms = grad.view(batch_size, -1).norm(dim=1).view(batch_size, 1, 1, 1)
            scale = torch.where(
                norms > cfg.grad_clip_norm,
                torch.full_like(norms, cfg.grad_clip_norm) / norms.clamp(min=1e-8),
                torch.ones_like(norms),
            )
            grad = grad * scale

        lambda_t = _compute_lambda(t if t is not None else 0.0, cfg)
        return (grad * lambda_t).detach()

    @torch.no_grad()
    def log_scores(self, z: torch.Tensor) -> Dict[str, float]:
        """Return mean predicted surprise and GMM for ``z``."""
        preds = self.predict(z.detach())
        return {
            "mean_surprise": preds["surprise"].mean().item(),
            "mean_gmm": preds["gmm"].mean().item(),
        }

    @torch.no_grad()
    def log_scores_DIMM(
        self,
        z_t: torch.Tensor,
        velocity: torch.Tensor,
        t: float,
    ) -> Dict[str, float]:
        """Return mean predicted scores on the DDIM-approximated clean latent."""
        z_hat = z_t.detach() + (1.0 - t) * velocity.detach()
        preds = self.predict(z_hat)
        return {
            "mean_surprise": preds["surprise"].mean().item(),
            "mean_gmm": preds["gmm"].mean().item(),
        }


def run_sanity_check(
    guidance: ScorePredictorGuidance,
    latent_shape: tuple = (4, 32, 32),
    batch_size: int = 2,
    steps: int = 5,
) -> None:
    """Minimal unit-like verification for a loaded guidance object."""
    print("[sanity_check] Starting ...")
    device = guidance.device
    channels, height, width = latent_shape
    z = torch.randn(batch_size, channels, height, width, device=device)

    grad = guidance.guidance_grad(z, t=0.0)
    assert grad.shape == z.shape, f"Grad shape mismatch: {grad.shape} vs {z.shape}"
    assert torch.isfinite(grad).all(), "Guidance gradient contains non-finite values!"
    print(
        f"  guidance_grad: shape={list(grad.shape)}  "
        f"|g|_max={grad.abs().max().item():.4e}  OK"
    )

    dt = 1.0 / steps
    velocity = torch.zeros_like(z)
    z_new = z + (velocity + grad) * dt
    assert not torch.allclose(z_new, z), "z did not change after guided step!"
    print("  Guided step changes z  OK")

    scores = guidance.log_scores(z)
    print(
        f"  Scores before guidance:  "
        f"mean_surprise={scores['mean_surprise']:.4f}  "
        f"mean_gmm={scores['mean_gmm']:.4f}"
    )
    scores_new = guidance.log_scores(z_new)
    print(
        f"  Scores after one step:   "
        f"mean_surprise={scores_new['mean_surprise']:.4f}  "
        f"mean_gmm={scores_new['mean_gmm']:.4f}"
    )

    print("[sanity_check] All checks passed")

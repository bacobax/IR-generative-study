"""
fm_src/guidance/score_predictor_guidance.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Energy-based guidance for Stable Flow Matching using a trained
SurprisePredictor.

Usage::

    from fm_src.guidance import ScoreGuidanceConfig, ScorePredictorGuidance

    cfg = ScoreGuidanceConfig(
        ckpt_path="runs/surprise_predictor/best_model.pt",
        energy_mode="surprise",
        sign="minimize",
        lambda_start=2.0,
        lambda_schedule="cosine",
        grad_clip_norm=1.0,
    )
    guide = ScorePredictorGuidance.from_checkpoint(cfg, device="cuda")

    # inside a sampling loop:
    g = guide.guidance_grad(z, t=step_t)
    z_next = z + (v + guidance_scale * g) * dt

Architecture reminder
---------------------
  SurprisePredictor: z → (post_quant_conv + decoder) → per-image minmax →
    DINO (frozen) → mean-pool → LayerNorm → trunk MLP →
      head_surprise (B,) + head_gmm (B,)

Params are frozen (requires_grad=False), but ops are differentiable so
autograd flows back to the input latent z.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Ensure repo root is importable (handles both direct run and pkg import).
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# The SurprisePredictor lives in the repo-root training script.
from train_surprise_predictor import SurprisePredictor  # noqa: E402


# =============================================================================
# Configuration dataclass
# =============================================================================


@dataclass
class ScoreGuidanceConfig:
    """All hyper-parameters that control guided sampling.

    Fields
    ------
    ckpt_path : str
        Path to ``best_model.pt`` from ``train_surprise_predictor.py``.
    vae_config_path : str, optional
        Path to VAE JSON config.  If *None*, read from ``ckpt["args"]["vae_config"]``.
    vae_weights_path : str, optional
        Path to VAE weights ``.pt``.  If *None*, read from ``ckpt["args"]["vae_weights"]``.
    dino_name : str, optional
        DINOv2 variant, e.g. ``"dinov2_vits14"``.  Falls back to ckpt args.
    hidden_dim : int, optional
        MLP hidden dim.  Falls back to ckpt args.

    energy_mode : {"surprise", "gmm", "combo"}
        Which predictor output to use as energy.
        * ``"surprise"``  →  E = pred_surprise
        * ``"gmm"``       →  E = 1 – pred_gmm
        * ``"combo"``     →  E = w_surprise·pred_surprise + w_gmm·(1–pred_gmm)
    sign : {"minimize", "maximize"}
        * ``"minimize"``  →  guide toward *lower* energy (less surprising / better GMM fit)
        * ``"maximize"``  →  guide toward *higher* energy (more novel / worse GMM fit)
    w_surprise, w_gmm : float
        Combo-mode weights.

    lambda_start, lambda_end : float
        λ end-points.  The schedule interpolates between them over flow time t∈[0,1).
    lambda_schedule : {"constant", "linear", "cosine", "step"}
        Schedule type.

    grad_clip_norm : float or None
        Per-sample guidance gradient norm cap.  *None* = no clipping.
    normalize_grad : bool
        Divide guidance gradient by its per-sample norm before scaling.

    guidance_on : {"latent", "decoded"}
        * ``"latent"``   – pass z directly to predictor (uses predictor's decoder).
        * ``"decoded"``  – decode z via the pipeline's VAE first, then run the
          predictor DINO+trunk+heads, bypassing the predictor's own decoder.
          Requires ``pipeline`` to be passed to ``guidance_grad``.

    use_ddim_hat : bool
        If ``True``, evaluate energy on a DDIM-style clean latent estimate
        ``z_hat(z_t, t, v)`` while still differentiating w.r.t. the original
        current latent ``z_t``:

        ``g_t ∝ ∇_{z_t} E(z_hat(z_t, t, v_detached))``.

        Velocity is always detached before building ``z_hat`` so guidance does
        not backpropagate into the UNet.

    use_amp : bool
        Wrap predictor forward in ``torch.amp.autocast("cuda", ...)``.
    detach_base_velocity : bool
        Informational flag; used by pipeline sampling methods to decide whether
        to detach the UNet velocity before adding guidance.

    num_refine_steps : int
        Number of post-sampling gradient refinement steps.
    refine_step_size : float
        Step size for each refinement step.
    """

    # ---- checkpoint + model ------------------------------------------------
    ckpt_path: str = ""
    vae_config_path: Optional[str] = None
    vae_weights_path: Optional[str] = None
    dino_name: Optional[str] = None
    hidden_dim: Optional[int] = None

    # ---- energy configuration ----------------------------------------------
    energy_mode: str = "surprise"       # "surprise" | "gmm" | "combo"
    sign: str = "minimize"              # "minimize" | "maximize"
    w_surprise: float = 1.0
    w_gmm: float = 1.0

    # ---- lambda schedule ---------------------------------------------------
    lambda_start: float = 1.0
    lambda_end: float = 1.0
    lambda_schedule: str = "constant"   # "constant" | "linear" | "cosine" | "step"

    # ---- gradient control --------------------------------------------------
    grad_clip_norm: Optional[float] = None
    normalize_grad: bool = False

    # ---- guidance target ---------------------------------------------------
    guidance_on: str = "latent"         # "latent" | "decoded"
    use_ddim_hat: bool = False

    # ---- misc --------------------------------------------------------------
    use_amp: bool = False
    detach_base_velocity: bool = False

    # ---- post-sampling refinement ------------------------------------------
    num_refine_steps: int = 10
    refine_step_size: float = 0.01


# =============================================================================
# Lambda schedule helper
# =============================================================================


def _compute_lambda(t: float, cfg: ScoreGuidanceConfig) -> float:
    """Return λ(t) ∈ [λ_start, λ_end] at normalised flow time *t* ∈ [0, 1)."""
    s, e = cfg.lambda_start, cfg.lambda_end
    sched = cfg.lambda_schedule
    if sched == "constant":
        return s
    if sched == "linear":
        return s + (e - s) * t
    if sched == "cosine":
        # Cosine anneal from s (t=0) to e (t=1)
        return e + (s - e) * 0.5 * (1.0 + math.cos(math.pi * t))
    if sched == "step":
        return s if t < 0.5 else e
    return s  # fallback


# =============================================================================
# Main guidance wrapper
# =============================================================================


class ScorePredictorGuidance:
    """Wraps a trained :class:`SurprisePredictor` for gradient-based guidance.

    All predictor parameters are *frozen* (``requires_grad=False``) but the
    forward graph is still differentiable, allowing ``torch.autograd.grad``
    to propagate back into the input latent ``z``.

    The class is intentionally thin: no ``nn.Module`` overhead, no trainer
    state.  Instantiate with :meth:`from_checkpoint`.
    """

    def __init__(
        self,
        predictor: SurprisePredictor,
        config: ScoreGuidanceConfig,
        device: str = "cpu",
    ) -> None:
        self.predictor = predictor
        self.config = config
        self.device = device

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        config: ScoreGuidanceConfig,
        device: str = "cpu",
    ) -> "ScorePredictorGuidance":
        """Build a :class:`ScorePredictorGuidance` from a *best_model.pt* checkpoint.

        Model construction params (VAE config/weights, DINO name, hidden_dim)
        are resolved from ``ckpt["args"]`` when not explicitly provided in *config*.

        Parameters
        ----------
        config:
            :class:`ScoreGuidanceConfig` with at least ``ckpt_path`` set.
        device:
            Target device, e.g. ``"cuda"`` or ``"cpu"``.

        Returns
        -------
        ScorePredictorGuidance
        """
        ckpt_path = config.ckpt_path
        if not Path(ckpt_path).is_file():
            raise FileNotFoundError(
                f"[ScorePredictorGuidance] Checkpoint not found: {ckpt_path}\n"
                "  → If DINOv2 has never been downloaded, ensure internet access once;\n"
                "    it will be cached at ~/.cache/torch/hub/ for subsequent offline runs."
            )

        print(f"[ScorePredictorGuidance] Loading checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ckpt_args: dict = ckpt.get("args", {})

        # Resolve model construction params (explicit config overrides ckpt args).
        vae_config_path  = config.vae_config_path  or ckpt_args.get("vae_config")
        vae_weights_path = config.vae_weights_path or ckpt_args.get("vae_weights")
        dino_name        = config.dino_name        or ckpt_args.get("dino_name", "dinov2_vits14")
        hidden_dim       = config.hidden_dim       or int(ckpt_args.get("hidden_dim", 256))

        if not vae_config_path or not vae_weights_path:
            raise ValueError(
                "[ScorePredictorGuidance] vae_config_path and vae_weights_path must be "
                "provided either via ScoreGuidanceConfig or embedded in ckpt['args']."
            )

        # Build predictor (torch.hub loads DINOv2 from cache or internet).
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
                "  → If DINOv2 weights are not cached yet, run once with internet access.\n"
                f"  Original error: {exc}"
            ) from exc

        # Load trained weights (non-strict to tolerate minor architecture diffs).
        missing, unexpected = predictor.load_state_dict(ckpt["model_state"], strict=False)
        if missing:
            print(f"  [WARNING] Missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"  [WARNING] Unexpected keys ({len(unexpected)}): {unexpected[:5]}")

        # Move to target device, freeze all params, set eval mode.
        predictor = predictor.to(device)
        predictor.eval()
        for p in predictor.parameters():
            p.requires_grad = False

        # DINO must stay in eval (BatchNorm / dropout).
        predictor.dino.eval()

        print(
            f"[ScorePredictorGuidance] Ready.  "
            f"energy_mode={config.energy_mode}  sign={config.sign}  "
            f"guidance_on={config.guidance_on}  schedule={config.lambda_schedule}"
        )
        return cls(predictor=predictor, config=config, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_lambda(self, t: float) -> float:
        """Return λ(t) using the configured schedule."""
        return _compute_lambda(t, self.config)

    # -----------------------
    # predict
    # -----------------------

    def predict(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Run the predictor on latents and return raw score tensors.

        Parameters
        ----------
        z : (B, C, H, W)
            Latent tensor.  A detached tensor is recommended unless you need
            gradients for a custom use case.

        Returns
        -------
        dict with keys ``"surprise"`` and ``"gmm"``, each shape (B,).
        """
        pred_surprise, pred_gmm = self.predictor(z)
        return {"surprise": pred_surprise, "gmm": pred_gmm}

    # -----------------------
    # energy
    # -----------------------

    def energy(
        self,
        z: torch.Tensor,
        t: Optional[float] = None,
        pipeline=None,
    ) -> torch.Tensor:
        """Compute per-sample energy E(z) — shape (B,).

        The tensor is produced by a fully differentiable graph rooted at *z*
        (assuming *z* is a leaf with ``requires_grad=True``).

        Energy definitions
        ------------------
        * ``"surprise"`` mode  →  E = pred_surprise
        * ``"gmm"``      mode  →  E = 1 – pred_gmm   (high = low GMM probability)
        * ``"combo"``    mode  →  E = w_s · pred_surprise + w_g · (1 – pred_gmm)

        Parameters
        ----------
        z : (B, C, H, W) with ``requires_grad=True`` when used for gradient.
        t : optional flow time (unused by energy itself; included for API parity).
        pipeline : optional ``StableFlowMatchingPipeline``; required when
            ``guidance_on="decoded"`` to decode *z* via the pipeline's VAE.
        """
        cfg = self.config

        if cfg.guidance_on == "decoded" and pipeline is not None:
            # Decode via the pipeline's VAE (ops are differentiable).
            # pipeline.decode_fm_output must NOT be wrapped in torch.no_grad().
            # Since pipeline.vae has requires_grad=False on its params, the grad
            # flows back to z through purely differentiable ops.
            x = pipeline.decode_fm_output(z)           # (B, C_out, H, W)
            if x.shape[1] != 1:
                x = x[:, :1]                           # keep first channel
            # Reuse predictor normalisation + DINO path (skip predictor's decoder).
            x_256 = F.interpolate(x, size=(256, 256), mode="bilinear", align_corners=False)
            x_norm = self.predictor._per_image_minmax(x_256)
            dino_in = self.predictor._to_dino_input(x_norm)
            tokens = self.predictor._extract_patch_tokens(dino_in)  # (B, N, D)
            pooled = tokens.mean(dim=1)                 # (B, D)
            pooled = self.predictor.pool_norm(pooled)
            h = self.predictor.trunk(pooled)
            pred_surprise = self.predictor.head_surprise(h).squeeze(-1)
            pred_gmm      = self.predictor.head_gmm(h).squeeze(-1)
        else:
            # Default: latent mode – predictor handles its own decoding.
            pred_surprise, pred_gmm = self.predictor(z)

        # Compose energy.
        if cfg.energy_mode == "surprise":
            E = pred_surprise
        elif cfg.energy_mode == "gmm":
            E = 1.0 - pred_gmm
        elif cfg.energy_mode == "combo":
            E = cfg.w_surprise * pred_surprise + cfg.w_gmm * (1.0 - pred_gmm)
        else:
            raise ValueError(
                f"[ScorePredictorGuidance] Unknown energy_mode={cfg.energy_mode!r}. "
                "Choose from 'surprise', 'gmm', 'combo'."
            )

        return E   # (B,)

    # -----------------------
    # guidance_grad
    # -----------------------

    def guidance_grad(
        self,
        z: torch.Tensor,
        t: Optional[float] = None,
        pipeline=None,
        velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute guidance gradient g to add to the velocity field.

        Returns a tensor of same shape as *z* that has *already* been scaled
        by λ(t), so the caller simply does::

            z_next = z + (v + guidance_scale * g) * dt

        Sign convention
        ---------------
        * ``sign="minimize"`` →  g = –∇_z E  (move toward lower energy)
        * ``sign="maximize"`` →  g = +∇_z E  (move toward higher energy)

        Gradient post-processing (in order)
        ------------------------------------
        1. Normalize per-sample to unit norm (if ``normalize_grad=True``).
        2. Clip per-sample norm to ``grad_clip_norm`` (if set).
        3. Multiply by λ(t).

        Parameters
        ----------
        z : (B, C, H, W)
            Current latent.  Will be detached internally; the returned gradient
            is always detached.
        t : float, optional
            Current flow time t ∈ [0, 1).  Used for λ(t) schedule.
        pipeline : optional ``StableFlowMatchingPipeline``
            Required when ``guidance_on="decoded"``.
        velocity : (B, C, H, W), optional
            Current FM velocity v(z_t, t). Used only when
            ``config.use_ddim_hat=True`` to form a differentiable clean estimate
            ``z_hat = z_t + (1 - t) * v_detached``. Velocity is explicitly
            detached so guidance cannot backpropagate into UNet.
        """
        cfg = self.config

        # Create fresh leaf tensor so we don't pollute any existing graph.
        z_leaf = z.detach().requires_grad_(True)

        # Optional DDIM-style clean estimate from current latent and velocity.
        z_energy = z_leaf
        if cfg.use_ddim_hat and (velocity is not None):
            t_val = float(t if t is not None else 0.0)
            v_detached = velocity.detach()
            # Flow-matching interpolation:
            #   z_t = (1 - t) * z0 + t * x,  v = x - z0
            # Rearranged clean estimate:
            #   x_hat = z_t + (1 - t) * v
            # This remains differentiable wrt z_t but not wrt v.
            z_energy = z_leaf + (1.0 - t_val) * v_detached

        with torch.amp.autocast(
            "cuda",
            enabled=cfg.use_amp and (z.device.type == "cuda"),
        ):
            E = self.energy(z_energy, t=t, pipeline=pipeline)   # (B,)
            scalar = E.mean()

        (raw_grad,) = torch.autograd.grad(scalar, z_leaf, create_graph=False)

        # Sign: move in direction that decreases (minimize) or increases (maximize) E.
        g: torch.Tensor = -raw_grad if cfg.sign == "minimize" else raw_grad

        # 1. Normalize
        if cfg.normalize_grad:
            B = g.shape[0]
            norms = g.view(B, -1).norm(dim=1).view(B, 1, 1, 1).clamp(min=1e-8)
            g = g / norms

        # 2. Clip norm
        if cfg.grad_clip_norm is not None:
            B = g.shape[0]
            norms = g.view(B, -1).norm(dim=1).view(B, 1, 1, 1)
            scale = torch.where(
                norms > cfg.grad_clip_norm,
                torch.full_like(norms, cfg.grad_clip_norm) / norms.clamp(min=1e-8),
                torch.ones_like(norms),
            )
            g = g * scale

        # 3. Scale by λ(t)
        lambda_t = _compute_lambda(t if t is not None else 0.0, cfg)
        g = g * lambda_t

        return g.detach()

    # -----------------------
    # score logging
    # -----------------------

    @torch.no_grad()
    def log_scores(self, z: torch.Tensor) -> Dict[str, float]:
        """Return mean predicted surprise and gmm for *z* (no grad).

        Handy for comparing unguided vs guided samples::

            scores_unguided = guide.log_scores(z_plain)
            scores_guided   = guide.log_scores(z_guided)
            print(scores_unguided, scores_guided)
        """
        preds = self.predict(z.detach())
        return {
            "mean_surprise": preds["surprise"].mean().item(),
            "mean_gmm":      preds["gmm"].mean().item(),
        }

    @torch.no_grad()
    def log_scores_DIMM(
        self,
        z_t: torch.Tensor,
        velocity: torch.Tensor,
        t: float,
    ) -> Dict[str, float]:
        """Return mean predicted scores evaluated on the DDIM-approximated z_1.

        Uses the same flow-matching clean estimate as ``guidance_grad`` with
        ``use_ddim_hat=True``::

            z_hat = z_t + (1 - t) * v_detached

        This approximates the final sample ``z_1`` from the current noisy
        latent ``z_t`` and the predicted velocity ``v`` at time ``t``.

        Parameters
        ----------
        z_t : (B, C, H, W)
            Current latent at flow time *t*.
        velocity : (B, C, H, W)
            Velocity predicted by the FM model at (z_t, t).  Detached
            internally before forming the estimate.
        t : float
            Current flow time t ∈ [0, 1).

        Returns
        -------
        dict with keys ``"mean_surprise"`` and ``"mean_gmm"``.
        """
        z_hat = z_t.detach() + (1.0 - t) * velocity.detach()
        preds = self.predict(z_hat)
        return {
            "mean_surprise": preds["surprise"].mean().item(),
            "mean_gmm":      preds["gmm"].mean().item(),
        }


# =============================================================================
# Sanity check
# =============================================================================


def run_sanity_check(
    guidance: ScorePredictorGuidance,
    latent_shape: tuple = (4, 32, 32),
    batch_size: int = 2,
    steps: int = 5,
) -> None:
    """Minimal unit-like verification.

    1. Sample a tiny batch and compute a guidance gradient.
    2. Assert the gradient is finite and has the correct shape.
    3. Run one guided Euler step and confirm the latent changes.
    4. Log mean predicted scores for the batch.

    Parameters
    ----------
    guidance : ScorePredictorGuidance
    latent_shape : (C, H, W)
        Shape of each latent tensor (default matches a common 4-ch 32×32 latent).
    batch_size : int
    steps : int
    """
    print("[sanity_check] Starting …")
    device = guidance.device
    C, H, W = latent_shape
    z = torch.randn(batch_size, C, H, W, device=device)

    # --- gradient check ---
    g = guidance.guidance_grad(z, t=0.0)
    assert g.shape == z.shape, f"Grad shape mismatch: {g.shape} vs {z.shape}"
    assert torch.isfinite(g).all(), "Guidance gradient contains non-finite values!"
    print(
        f"  guidance_grad: shape={list(g.shape)}  "
        f"|g|_max={g.abs().max().item():.4e}  ✓"
    )

    # --- guided step check ---
    dt = 1.0 / steps
    v = torch.zeros_like(z)
    z_new = z + (v + g) * dt
    assert not torch.allclose(z_new, z), "z did not change after guided step!"
    print("  Guided step changes z  ✓")

    # --- score logging ---
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

    print("[sanity_check] All checks passed  ✓")

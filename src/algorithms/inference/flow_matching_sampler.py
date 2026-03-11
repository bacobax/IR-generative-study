"""Inference / sampling logic for flow-matching models.

This module is the sole source of truth for flow-matching sampling.
It replaces the inference methods formerly embedded in
``fm_src.pipelines.flow_matching_pipeline``.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from torch.utils.tensorboard import SummaryWriter


# ---------------------------------------------------------------------------
# Default display helper
# ---------------------------------------------------------------------------
def _default_from_norm_to_display(x: torch.Tensor) -> torch.Tensor:
    """[-1, 1] -> [0, 1] for TensorBoard visualisation."""
    return (x + 1) / 2


# ---------------------------------------------------------------------------
# Pick latest numbered checkpoint in a directory
# ---------------------------------------------------------------------------
def _pick_latest(folder: str, prefix: str, suffix: str = ".pt"):
    import os as _os
    if not _os.path.isdir(folder):
        return None
    best_i, best_path = None, None
    for fn in _os.listdir(folder):
        if not (fn.startswith(prefix) and fn.endswith(suffix)):
            continue
        mid = fn[len(prefix):-len(suffix)]
        try:
            i = int(mid)
        except ValueError:
            continue
        if best_i is None or i > best_i:
            best_i = i
            best_path = _os.path.join(folder, fn)
    return best_path


# ---------------------------------------------------------------------------
# Utility: derive sample shape from a UNet2DModel
# ---------------------------------------------------------------------------
def get_unet_sample_shape(
    unet: UNet2DModel,
    override: Optional[Tuple[int, int, int]] = None,
) -> Tuple[int, int, int]:
    """Return (C, H, W) from the UNet config or an explicit override."""
    if override is not None:
        return override
    in_channels = unet.config.in_channels
    sample_size = unet.config.sample_size
    if isinstance(sample_size, Sequence):
        h, w = sample_size
    else:
        h = w = sample_size
    return (in_channels, h, w)


# ═══════════════════════════════════════════════════════════════════════════
# FlowMatchingSampler — inference only
# ═══════════════════════════════════════════════════════════════════════════

class FlowMatchingSampler:
    """Standalone sampler for flow-matching models.

    Operates in *pixel space* by default.  For latent-space (VAE) workflows,
    pass ``encoder``/``decoder`` callables or use the convenience
    class method :meth:`from_stable` which wires a VAE automatically.

    Parameters
    ----------
    unet : UNet2DModel
        Trained UNet.
    device : str or torch.device
    t_scale : float
        Time-scaling factor fed to the UNet.
    train_target : ``"v"`` | ``"x0"``
        Whether the UNet predicts velocity or clean sample.
    from_norm_to_display : callable, optional
        Maps [-1,1] tensors to [0,1] for TensorBoard.
    sample_shape : (C, H, W), optional
        Override spatial size; derived from UNet config if omitted.
    encoder : callable, optional
        ``x -> z`` (e.g. VAE encode). Identity if ``None``.
    decoder : callable, optional
        ``z -> x`` (e.g. VAE decode). Identity if ``None``.
    guidance : BaseGuidance, optional
        Guidance module injected at construction time.  When ``None``,
        the sampler runs unguided (no import-time dependency on the
        guidance package).
    conditioner : BaseConditioner, optional
        Conditioning module.  When ``None``, no extra kwargs are
        passed to the UNet (unconditional sampling).
    """

    def __init__(
        self,
        unet: UNet2DModel,
        *,
        device: Optional[Union[str, torch.device]] = None,
        t_scale: float = 1.0,
        train_target: str = "v",
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        encoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        decoder: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        guidance=None,
        conditioner=None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.unet = unet
        self.t_scale = float(t_scale)
        assert train_target in ("v", "x0")
        self.train_target = train_target
        self.from_norm_to_display = from_norm_to_display or _default_from_norm_to_display
        self._sample_shape = sample_shape
        self._encoder = encoder or (lambda x: x)
        self._decoder = decoder or (lambda z: z)
        self.guidance = guidance
        self.conditioner = conditioner

    # ------------------------------------------------------------------
    # Convenience constructor for latent-space (VAE) setups
    # ------------------------------------------------------------------
    @classmethod
    def from_stable(
        cls,
        unet: UNet2DModel,
        vae,
        **kwargs,
    ) -> "FlowMatchingSampler":
        """Build a sampler wired to a frozen VAE for latent-space sampling."""

        @torch.no_grad()
        def _encode(x: torch.Tensor) -> torch.Tensor:
            z_mu, z_sigma = vae.encode(x)
            return vae.sampling(z_mu, z_sigma)

        @torch.no_grad()
        def _decode(z: torch.Tensor) -> torch.Tensor:
            return vae.decode(z)

        return cls(unet, encoder=_encode, decoder=_decode, **kwargs)

    # ------------------------------------------------------------------
    # Guidance wiring
    # ------------------------------------------------------------------
    def set_guidance(self, guidance) -> None:
        """Inject or replace the guidance module at runtime."""
        self.guidance = guidance

    def set_conditioner(self, conditioner) -> None:
        """Inject or replace the conditioner at runtime."""
        self.conditioner = conditioner

    def _cond_kwargs(self, batch_size: int) -> dict:
        """Get conditioning kwargs from the conditioner (empty if None)."""
        if self.conditioner is None:
            return {}
        return self.conditioner.prepare_for_sampling(batch_size, self.device)

    @classmethod
    def from_config(
        cls,
        config,
        *,
        from_norm_to_display: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> "FlowMatchingSampler":
        """Build a sampler from an :class:`FMSampleConfig`.

        Loads the UNet and VAE from the pipeline directory specified in
        *config*, then wires them into a latent-space sampler.
        """
        from src.models.fm_unet import load_unet_config as _load_unet_config, build_fm_unet_from_config as _build_unet
        from src.models.vae import load_vae_config as _load_vae_config, build_vae_from_config as _build_vae, load_vae_weights as _load_vae_w, freeze_vae as _freeze_vae
        import os as _os

        device = config.resolved_device()
        pipeline_dir = config.pipeline_dir
        unet_dir = _os.path.join(pipeline_dir, "UNET")
        vae_dir = _os.path.join(pipeline_dir, "VAE")

        # UNet
        unet_cfg = _load_unet_config(_os.path.join(unet_dir, "config.json"))
        unet = _build_unet(unet_cfg, device=device)
        unet_w = _os.path.join(unet_dir, "unet_fm_best.pt")
        if not _os.path.isfile(unet_w):
            unet_w = _pick_latest(unet_dir, "unet_fm_epoch_")
        if unet_w is None or not _os.path.isfile(unet_w):
            raise FileNotFoundError(f"No UNET weights in {unet_dir}")
        unet.load_state_dict(torch.load(unet_w, map_location=device))
        unet.eval()

        # VAE
        vae_cfg = _load_vae_config(_os.path.join(vae_dir, "config.json"))
        vae = _build_vae(vae_cfg, device=device)
        vae_w = _os.path.join(vae_dir, "vae_best.pt")
        if not _os.path.isfile(vae_w):
            vae_w = _pick_latest(vae_dir, "vae_epoch_")
        if vae_w is None or not _os.path.isfile(vae_w):
            raise FileNotFoundError(f"No VAE weights in {vae_dir}")
        _load_vae_w(vae, vae_w, map_location=device)

        if config.vae_weights is not None:
            _load_vae_w(vae, config.vae_weights, map_location=device)
        _freeze_vae(vae)

        return cls.from_stable(
            unet, vae,
            device=device,
            t_scale=config.t_scale,
            train_target=config.train_target,
            from_norm_to_display=from_norm_to_display,
            sample_shape=config.sample_shape,
        )

    # ------------------------------------------------------------------
    # Shape helpers
    # ------------------------------------------------------------------
    def _shape(self, override: Optional[Tuple[int, int, int]] = None) -> Tuple[int, int, int]:
        return get_unet_sample_shape(self.unet, override=override or self._sample_shape)

    # ------------------------------------------------------------------
    # Encode / decode wrappers
    # ------------------------------------------------------------------
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self._encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self._decoder(z)

    # ------------------------------------------------------------------
    # 1. Plain Euler ODE sampler
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_euler(
        self,
        steps: int = 50,
        batch_size: int = 4,
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        self.unet.eval()
        shape = self._shape(sample_shape)
        z = torch.randn(batch_size, *shape, device=self.device)
        dt = 1.0 / steps
        cond_kw = self._cond_kwargs(batch_size)

        for i in range(steps):
            t_val = i / steps
            t = torch.full((batch_size,), t_val, device=self.device)
            unet_out = self.unet(z, t * self.t_scale, **cond_kw).sample

            if self.train_target == "x0":
                t_exp = t[:, None, None, None]
                v = (unet_out - z) / (1 - t_exp).clamp(min=1e-5)
            else:
                v = unet_out

            z = z + v * dt
        return z

    # ------------------------------------------------------------------
    # 2. Euler + predictor guidance
    # ------------------------------------------------------------------
    def sample_euler_guided(
        self,
        steps: int = 50,
        batch_size: int = 4,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        guidance=None,
        guidance_scale: float = 1.0,
        return_logs: bool = False,
    ):
        self.unet.eval()
        shape = self._shape(sample_shape)
        z = torch.randn(batch_size, *shape, device=self.device)
        dt = 1.0 / steps
        logs: List[Dict[str, Any]] = []

        # Fall back to instance-level guidance if none passed explicitly
        active_guidance = guidance if guidance is not None else self.guidance
        cond_kw = self._cond_kwargs(batch_size)

        for i in range(steps):
            t_val = i / steps
            t = torch.full((batch_size,), t_val, device=self.device)

            with torch.no_grad():
                unet_out = self.unet(z, t * self.t_scale, **cond_kw).sample
                if self.train_target == "x0":
                    t_exp = t[:, None, None, None]
                    v = (unet_out - z) / (1.0 - t_exp).clamp(min=1e-5)
                else:
                    v = unet_out

            if active_guidance is not None and guidance_scale > 0.0:
                g = active_guidance.guidance_grad(z, t=t_val, pipeline=self, velocity=v)
                guided_v = v + guidance_scale * g

                if return_logs:
                    grad_norm = g.view(batch_size, -1).norm(dim=1).mean().item()
                    step_log: Dict[str, Any] = {"step": i, "t": t_val, "grad_norm": grad_norm}
                    with torch.no_grad():
                        scores = active_guidance.log_scores(z)
                    step_log.update(scores)
                    logs.append(step_log)
            else:
                guided_v = v

            z = (z + guided_v * dt).detach()

        if return_logs:
            return z, logs
        return z

    # ------------------------------------------------------------------
    # 3. Rejection / reranking over N candidates
    # ------------------------------------------------------------------
    def sample_euler_with_candidates(
        self,
        steps: int = 50,
        n_candidates: int = 8,
        keep_top_k: int = 1,
        batch_size: int = 4,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        guidance=None,
        guidance_scale: float = 0.0,
        return_all_scores: bool = False,
    ):
        if guidance_scale > 0.0 and guidance is None:
            raise ValueError("guidance must be provided when guidance_scale > 0.")

        shape = self._shape(sample_shape)
        all_z = []
        for _ in range(n_candidates):
            z_c = self.sample_euler_guided(
                steps=steps,
                batch_size=batch_size,
                sample_shape=shape,
                guidance=guidance,
                guidance_scale=guidance_scale,
            )
            all_z.append(z_c)

        all_z_cat = torch.cat(all_z, dim=0)

        if guidance is not None:
            with torch.no_grad():
                E = guidance.energy(all_z_cat)
        else:
            if return_all_scores:
                return all_z_cat[: keep_top_k * batch_size], None
            return all_z_cat[: keep_top_k * batch_size]

        E_mat = E.view(n_candidates, batch_size)
        z_mat = all_z_cat.view(n_candidates, batch_size, *shape)
        order = E_mat.argsort(dim=0)
        kept_indices = order[:keep_top_k]

        z_best_list = []
        for k in range(keep_top_k):
            idx = kept_indices[k]
            z_best_list.append(z_mat[idx, torch.arange(batch_size)])
        z_best = torch.cat(z_best_list, dim=0)

        if return_all_scores:
            return z_best, E
        return z_best

    # ------------------------------------------------------------------
    # 4. Beam / branching sampling
    # ------------------------------------------------------------------
    def sample_euler_beam(
        self,
        steps: int = 50,
        batch_size: int = 4,
        beam_size: int = 4,
        branch_factor: int = 2,
        sigma_perturb: float = 0.05,
        sample_shape: Optional[Tuple[int, int, int]] = None,
        guidance=None,
        guidance_scale: float = 0.0,
        return_all_beams: bool = False,
    ):
        self.unet.eval()
        shape = self._shape(sample_shape)
        C, H, W = shape
        dt = 1.0 / steps

        z_beams = torch.randn(beam_size, batch_size, C, H, W, device=self.device)
        cond_kw = self._cond_kwargs(batch_size)

        for i in range(steps):
            t_val = i / steps
            t_scalar = torch.full((batch_size,), t_val, device=self.device)
            expanded_beams = []

            for b in range(beam_size):
                z_b = z_beams[b]
                for _ in range(branch_factor):
                    if sigma_perturb > 0.0:
                        z_branch = z_b + sigma_perturb * torch.randn_like(z_b)
                    else:
                        z_branch = z_b.clone()

                    with torch.no_grad():
                        unet_out = self.unet(z_branch, t_scalar * self.t_scale, **cond_kw).sample
                        if self.train_target == "x0":
                            t_exp = t_scalar[:, None, None, None]
                            v = (unet_out - z_branch) / (1.0 - t_exp).clamp(min=1e-5)
                        else:
                            v = unet_out

                    if guidance is not None and guidance_scale > 0.0:
                        g = guidance.guidance_grad(z_branch, t=t_val, pipeline=self, velocity=v)
                        v = v + guidance_scale * g

                    z_next = (z_branch + v * dt).detach()
                    expanded_beams.append(z_next)

            z_expanded = torch.stack(expanded_beams, dim=0)
            K = z_expanded.shape[0]

            if guidance is not None:
                z_flat = z_expanded.view(K * batch_size, C, H, W)
                with torch.no_grad():
                    E_flat = guidance.energy(z_flat)
                E_mat = E_flat.view(K, batch_size)
                order = E_mat.argsort(dim=0)

                new_beams = []
                for b_new in range(min(beam_size, K)):
                    idx = order[b_new]
                    new_beams.append(z_expanded[idx, torch.arange(batch_size)])
                z_beams = torch.stack(new_beams, dim=0)
            else:
                z_beams = z_expanded[:beam_size]

        if return_all_beams:
            return z_beams.view(beam_size * batch_size, C, H, W)

        if guidance is not None:
            z_flat = z_beams.view(beam_size * batch_size, C, H, W)
            with torch.no_grad():
                E_final = guidance.energy(z_flat).view(beam_size, batch_size)
            best_idx = E_final.argmin(dim=0)
            z_best = z_beams[best_idx, torch.arange(batch_size)]
        else:
            z_best = z_beams[0]

        return z_best

    # ------------------------------------------------------------------
    # 5. Post-sampling latent refinement
    # ------------------------------------------------------------------
    def refine_latents_energy(
        self,
        z: torch.Tensor,
        guidance,
        num_steps: Optional[int] = None,
        step_size: Optional[float] = None,
        clamp_latent_norm: Optional[float] = None,
    ) -> torch.Tensor:
        cfg = guidance.config
        n_steps = num_steps if num_steps is not None else cfg.num_refine_steps
        step_sz = step_size if step_size is not None else cfg.refine_step_size
        B = z.shape[0]

        z = z.detach()
        for _ in range(n_steps):
            g = guidance.guidance_grad(z, t=1.0, pipeline=self)
            z = (z + step_sz * g).detach()

            if clamp_latent_norm is not None:
                norms = z.view(B, -1).norm(dim=1).view(B, 1, 1, 1).clamp(min=1e-8)
                scale = torch.where(
                    norms > clamp_latent_norm,
                    torch.full_like(norms, clamp_latent_norm) / norms,
                    torch.ones_like(norms),
                )
                z = (z * scale).detach()

        return z

    # ------------------------------------------------------------------
    # TensorBoard helpers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def log_samples_to_tensorboard(
        self,
        writer: SummaryWriter,
        epoch: int,
        steps: int = 50,
        batch_size: int = 4,
        tag: str = "fm_samples",
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ) -> None:
        z = self.sample_euler(steps=steps, batch_size=batch_size, sample_shape=sample_shape)
        x_gen = self.decode(z)
        x_vis = self.from_norm_to_display(x_gen).clamp(0, 1)
        writer.add_images(tag, x_vis, epoch)

    def log_samples_to_tensorboard_guided(
        self,
        writer: SummaryWriter,
        epoch: int,
        guidance=None,
        guidance_scale: float = 1.0,
        steps: int = 50,
        batch_size: int = 4,
        tag_unguided: str = "fm_samples/unguided",
        tag_guided: str = "fm_samples/guided",
        tag_suffix: str = "",
        sample_shape: Optional[Tuple[int, int, int]] = None,
        log_score_scalars: bool = True,
    ) -> None:
        self.unet.eval()

        t_ug = (tag_unguided + tag_suffix) if tag_suffix else tag_unguided
        t_g = (tag_guided + tag_suffix) if tag_suffix else tag_guided

        z_plain = self.sample_euler_guided(
            steps=steps, batch_size=batch_size, sample_shape=sample_shape,
            guidance=None, guidance_scale=0.0,
        )
        with torch.no_grad():
            x_plain = self.decode(z_plain)
        x_plain_vis = self.from_norm_to_display(x_plain).clamp(0, 1)
        writer.add_images(t_ug, x_plain_vis, epoch)

        if guidance is not None:
            z_guided = self.sample_euler_guided(
                steps=steps, batch_size=batch_size, sample_shape=sample_shape,
                guidance=guidance, guidance_scale=guidance_scale,
            )
            with torch.no_grad():
                x_guided = self.decode(z_guided)
            x_guided_vis = self.from_norm_to_display(x_guided).clamp(0, 1)
            writer.add_images(t_g, x_guided_vis, epoch)

            if log_score_scalars:
                scores_plain = guidance.log_scores(z_plain)
                scores_guided = guidance.log_scores(z_guided)
                writer.add_scalars(
                    f"guided_scores{tag_suffix}/mean_surprise",
                    {"unguided": scores_plain["mean_surprise"],
                     "guided": scores_guided["mean_surprise"]},
                    epoch,
                )
                writer.add_scalars(
                    f"guided_scores{tag_suffix}/mean_gmm",
                    {"unguided": scores_plain["mean_gmm"],
                     "guided": scores_guided["mean_gmm"]},
                    epoch,
                )
                print(
                    f"[log_guided] epoch={epoch}  "
                    f"unguided: surprise={scores_plain['mean_surprise']:.4f}  "
                    f"gmm={scores_plain['mean_gmm']:.4f}  |  "
                    f"guided:   surprise={scores_guided['mean_surprise']:.4f}  "
                    f"gmm={scores_guided['mean_gmm']:.4f}"
                )


# ── registry ──────────────────────────────────────────────────────────────────
from src.core.registry import REGISTRIES  # noqa: E402

REGISTRIES.sampler.register("default_fm", default=True)(FlowMatchingSampler)

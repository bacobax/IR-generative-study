"""DomainStudio auxiliary losses for Stage-1 Stable Diffusion adaptation."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def _prediction_type(noise_scheduler) -> str:
    return getattr(noise_scheduler.config, "prediction_type", "epsilon")


def predict_original_latents_from_epsilon(
    noisy_latents: torch.Tensor,
    timesteps: torch.Tensor,
    epsilon_pred: torch.Tensor,
    noise_scheduler,
) -> torch.Tensor:
    """Reconstruct predicted clean latents from epsilon/noise prediction."""
    if _prediction_type(noise_scheduler) != "epsilon":
        raise NotImplementedError(
            "DomainStudio clean-latent reconstruction currently supports only "
            "epsilon prediction."
        )

    alphas_cumprod = noise_scheduler.alphas_cumprod.to(
        device=noisy_latents.device,
        dtype=noisy_latents.dtype,
    )
    alpha_prod_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1)
    beta_prod_t = 1.0 - alpha_prod_t
    return (
        noisy_latents - beta_prod_t.sqrt() * epsilon_pred
    ) / alpha_prod_t.sqrt().clamp_min(torch.finfo(noisy_latents.dtype).eps)


def decode_latents_to_images(vae, z0_hat: torch.Tensor) -> torch.Tensor:
    """Decode predicted clean latents with gradients flowing to ``z0_hat``."""
    vae_dtype = next(vae.parameters()).dtype
    latents = (z0_hat / vae.config.scaling_factor).to(dtype=vae_dtype)
    decoded = vae.decode(latents)
    if hasattr(decoded, "sample"):
        images = decoded.sample
    elif isinstance(decoded, (tuple, list)):
        images = decoded[0]
    else:
        images = decoded
    return images.clamp(-1.0, 1.0)


def haar_high_frequency(x: torch.Tensor) -> torch.Tensor:
    """Return differentiable Haar LH+HL+HH high-frequency bands."""
    if x.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] input, got shape {tuple(x.shape)}")

    height = x.shape[-2] - (x.shape[-2] % 2)
    width = x.shape[-1] - (x.shape[-1] % 2)
    x = x[..., :height, :width]
    if height < 2 or width < 2:
        return x.new_zeros(x.shape[0], x.shape[1], 0, 0)

    inv_sqrt_2 = 1.0 / math.sqrt(2.0)
    low = torch.tensor([inv_sqrt_2, inv_sqrt_2], device=x.device, dtype=x.dtype)
    high = torch.tensor([-inv_sqrt_2, inv_sqrt_2], device=x.device, dtype=x.dtype)
    kernels = torch.stack(
        [
            torch.outer(low, high),
            torch.outer(high, low),
            torch.outer(high, high),
        ],
        dim=0,
    )
    channels = x.shape[1]
    weight = kernels[:, None, :, :].repeat(channels, 1, 1, 1)
    bands = F.conv2d(x, weight, stride=2, groups=channels)
    bands = bands.view(x.shape[0], channels, 3, height // 2, width // 2)
    return bands.sum(dim=2)


def pairwise_kl_loss(
    target_features: torch.Tensor,
    source_features: torch.Tensor,
    temperature: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """KL between off-diagonal pairwise relative-similarity distributions."""
    batch_size = min(target_features.shape[0], source_features.shape[0])
    if batch_size < 2:
        return target_features.sum() * 0.0 + source_features.sum() * 0.0
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    target = target_features[:batch_size].flatten(start_dim=1)
    source = source_features[:batch_size].flatten(start_dim=1)
    target = F.normalize(target, p=2, dim=1, eps=eps)
    source = F.normalize(source, p=2, dim=1, eps=eps)

    target_sim = target @ target.transpose(0, 1)
    source_sim = source @ source.transpose(0, 1)
    keep = ~torch.eye(batch_size, dtype=torch.bool, device=target_sim.device)
    target_sim = target_sim[keep].view(batch_size, batch_size - 1)
    source_sim = source_sim[keep].view(batch_size, batch_size - 1)

    log_target = F.log_softmax(target_sim / temperature, dim=1)
    log_source = F.log_softmax(source_sim / temperature, dim=1)
    target_prob = log_target.exp()
    return (target_prob * (log_target - log_source)).sum(dim=1).mean()


def _zero_like_loss(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def compute_domainstudio_losses(
    *,
    student_pred_prior: torch.Tensor,
    teacher_pred_prior: torch.Tensor,
    img_target_hat: torch.Tensor,
    img_prior_hat: torch.Tensor,
    pixel_values: torch.Tensor,
    temperature: float = 1.0,
    min_pairwise_batch: int = 2,
) -> Dict[str, torch.Tensor]:
    """Compute DomainStudio prior, image-pairwise, HF-pairwise, and HF-MSE losses."""
    prior_loss = F.mse_loss(
        student_pred_prior.float(),
        teacher_pred_prior.float(),
        reduction="mean",
    )

    pairwise_batch = min(img_target_hat.shape[0], img_prior_hat.shape[0])
    if pairwise_batch < min_pairwise_batch:
        img_pairwise = _zero_like_loss(img_target_hat) + _zero_like_loss(img_prior_hat)
        hf_pairwise = img_pairwise
    else:
        img_pairwise = pairwise_kl_loss(
            img_target_hat[:pairwise_batch],
            img_prior_hat[:pairwise_batch],
            temperature=temperature,
        )
        hf_target = haar_high_frequency(img_target_hat[:pairwise_batch])
        hf_prior = haar_high_frequency(img_prior_hat[:pairwise_batch])
        hf_pairwise = pairwise_kl_loss(
            hf_target,
            hf_prior,
            temperature=temperature,
        )

    if pixel_values.shape[-2:] != img_target_hat.shape[-2:]:
        pixel_values = F.interpolate(
            pixel_values,
            size=img_target_hat.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    hf_pred = haar_high_frequency(img_target_hat)
    hf_real = haar_high_frequency(pixel_values.to(device=img_target_hat.device, dtype=img_target_hat.dtype))
    hf_mse = F.mse_loss(hf_pred.float(), hf_real.float(), reduction="mean")

    return {
        "prior": prior_loss,
        "img_pairwise": img_pairwise,
        "hf_pairwise": hf_pairwise,
        "hf_mse": hf_mse,
    }

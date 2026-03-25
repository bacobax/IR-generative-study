"""Sampler for layout-conditioned pixel-space flow matching."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

from src.algorithms.inference.flow_matching_sampler import (
    FlowMatchingSampler,
    get_unet_sample_shape,
)


class LayoutFlowMatchingSampler(FlowMatchingSampler):
    """Flow-matching sampler that conditions on padded bbox/layout batches."""

    @torch.no_grad()
    def sample_euler_layout(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        steps: int = 50,
        sample_shape: Optional[Tuple[int, int, int]] = None,
    ) -> torch.Tensor:
        self.unet.eval()
        shape = get_unet_sample_shape(self.unet, override=sample_shape or self._sample_shape)
        batch_size = int(batch["pixel_values"].shape[0])
        z = torch.randn(batch_size, *shape, device=self.device)
        dt = 1.0 / steps

        cond_kw = {
            "boxes_xyxy_norm": batch["boxes_xyxy_norm"].to(self.device),
            "labels": batch["labels"].to(self.device),
            "object_mask": batch["object_mask"].to(self.device),
        }

        for step_idx in range(steps):
            t_val = step_idx / steps
            t = torch.full((batch_size,), t_val, device=self.device)
            unet_out = self.unet(z, t * self.t_scale, **cond_kw).sample

            if self.train_target == "x0":
                t_exp = t[:, None, None, None]
                velocity = (unet_out - z) / (1.0 - t_exp).clamp(min=1e-5)
            else:
                velocity = unet_out

            z = z + velocity * dt

        return z

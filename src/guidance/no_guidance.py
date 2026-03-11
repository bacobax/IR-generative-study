"""No-op guidance: reproduces the current unguided sampling behavior.

``NoGuidance`` is the default guidance object.  It returns zero gradients,
zero energy, and empty scores — meaning it has **no effect** on the sampling
trajectory.  This lets the sampler always go through the guidance code-path
without special-casing ``if guidance is not None``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch

from src.guidance.base_guidance import BaseGuidance
from src.core.registry import REGISTRIES


class NoGuidance(BaseGuidance):
    """Identity / no-op guidance module.

    When injected into the sampler, behavior is identical to unguided
    sampling because:

    * ``guidance_grad`` returns a zero tensor
    * ``energy`` returns zeros
    * ``log_scores`` returns an empty dict
    """

    def energy(
        self,
        z: torch.Tensor,
        t: Optional[float] = None,
        pipeline: Any = None,
    ) -> torch.Tensor:
        return torch.zeros(z.shape[0], device=z.device)

    def guidance_grad(
        self,
        z: torch.Tensor,
        t: Optional[float] = None,
        pipeline: Any = None,
        velocity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return torch.zeros_like(z)

    def log_scores(self, z: torch.Tensor) -> Dict[str, float]:
        return {}


# ── registry ──────────────────────────────────────────────────────────────
REGISTRIES.guidance.register("no_guidance", default=True)(NoGuidance)

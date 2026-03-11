"""No-op conditioner: reproduces the current unconditional behavior.

``NoConditioner`` always returns an empty dict, meaning no extra kwargs
are passed to the UNet.  This is the default, ensuring zero behavior
change when no conditioning is configured.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from src.conditioning.base_conditioner import BaseConditioner
from src.core.registry import REGISTRIES


class NoConditioner(BaseConditioner):
    """Pass-through conditioner for unconditional models.

    Both ``prepare_for_training`` and ``prepare_for_sampling`` return
    ``{}``, so the UNet receives only ``(sample, timestep)`` — exactly
    as it does today.
    """

    def prepare_for_training(
        self,
        batch: Any,
        device: torch.device | str = "cpu",
    ) -> Dict[str, Any]:
        return {}

    def prepare_for_sampling(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> Dict[str, Any]:
        return {}


# ── registry ──────────────────────────────────────────────────────────────
REGISTRIES.conditioning.register("no_conditioning", default=True)(NoConditioner)

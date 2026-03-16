"""Small gated correction module for MoE adapter outputs."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class GatedCorrection(nn.Module):
    """Predict a near-zero channel-wise correction from hidden and text context."""

    def __init__(
        self,
        channels: int,
        cond_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        gate_bias: float = -4.0,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = channels

        input_dim = channels * 2 + cond_dim
        self.channels = channels
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.gate_bias = gate_bias

        self.delta_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, channels),
        )

        self._init_heads()

    def _init_heads(self) -> None:
        delta_out = self.delta_mlp[-1]
        gate_out = self.gate_mlp[-1]
        nn.init.zeros_(delta_out.weight)
        nn.init.zeros_(delta_out.bias)
        nn.init.zeros_(gate_out.weight)
        nn.init.constant_(gate_out.bias, self.gate_bias)

    def forward(
        self,
        hidden_state: torch.Tensor,
        mixture_residual: torch.Tensor,
        pooled_text_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Return a broadcastable correction tensor of shape ``(B, C, 1, 1)``."""
        h_pool = hidden_state.mean(dim=(2, 3))
        x_pool = mixture_residual.mean(dim=(2, 3))
        features = torch.cat([h_pool, x_pool, pooled_text_embeds], dim=-1)

        delta = self.delta_mlp(features).to(hidden_state.dtype)
        gate = torch.sigmoid(self.gate_mlp(features)).to(hidden_state.dtype)
        return (gate * delta).unsqueeze(-1).unsqueeze(-1)

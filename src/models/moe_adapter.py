"""Mixture-of-experts adapter modules for UNet feature routing.

Defines a small residual ExpertAdapter and an AdapterBank that combines
multiple experts with fixed uniform weights.
"""

from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn


class ExpertAdapter(nn.Module):
    """Small residual adapter operating on intermediate feature maps.

    Parameters
    ----------
    channels : int
        Number of feature channels.
    hidden_dim : int, optional
        Hidden dimension for the bottleneck MLP/conv.
        Defaults to ``max(4, channels // 4)``.
    dropout : float
        Dropout probability applied after the activation.
    """

    def __init__(
        self,
        channels: int,
        *,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = max(4, channels // 4)

        self.channels = channels
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.proj_in = nn.Conv2d(channels, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj_out = nn.Conv2d(hidden_dim, channels, kernel_size=1)

        # Initialize close to identity but not exactly zero, so routing matters.
        nn.init.normal_(self.proj_out.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply a residual adapter to the feature map.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Output features of shape ``(B, C, H, W)``.
        """
        residual = self.proj_out(self.drop(self.act(self.proj_in(x))))
        return x + residual


class AdapterBank(nn.Module):
    """Collection of experts with optional routing weights.

    Parameters
    ----------
    channels : int
        Number of feature channels.
    num_experts : int
        Number of expert adapters.
    hidden_dim : int, optional
        Hidden dimension for each ExpertAdapter.
    dropout : float
        Dropout probability for each ExpertAdapter.
    """

    def __init__(
        self,
        channels: int,
        num_experts: int,
        *,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if num_experts <= 0:
            raise ValueError("num_experts must be >= 1")

        self.channels = channels
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.dropout = dropout

        self.experts = nn.ModuleList(
            [
                ExpertAdapter(
                    channels,
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply experts and return a weighted combination.

        Parameters
        ----------
        x : torch.Tensor
            Input features of shape ``(B, C, H, W)``.

        Returns
        -------
        torch.Tensor
            Output features of shape ``(B, C, H, W)``.
        """
        outputs: List[torch.Tensor] = [expert(x) for expert in self.experts]
        stacked = torch.stack(outputs, dim=1)  # (B, K, C, H, W)

        if weights is None:
            return stacked.mean(dim=1)

        if weights.ndim != 2 or weights.shape[1] != self.num_experts:
            raise ValueError(
                f"weights must have shape (B, {self.num_experts}), got {tuple(weights.shape)}"
            )

        w = weights.to(stacked.dtype)
        w = w[:, :, None, None, None]
        return (stacked * w).sum(dim=1)

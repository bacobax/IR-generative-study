"""Compact binary classifier for FLIR foreground/background crops."""

from __future__ import annotations

import torch
import torch.nn as nn


class ForegroundBackgroundClassifier(nn.Module):
    """Small 1-channel CNN for object-vs-background classification."""

    def __init__(
        self,
        *,
        in_channels: int = 1,
        channels: tuple[int, int, int, int] = (32, 64, 128, 256),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        blocks = []
        current_channels = int(in_channels)
        for next_channels in channels:
            blocks.extend(
                [
                    nn.Conv2d(current_channels, next_channels, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(next_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            current_channels = next_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(float(dropout))
        self.head = nn.Linear(current_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        pooled = self.pool(features).flatten(1)
        return self.head(self.dropout(pooled)).squeeze(-1)

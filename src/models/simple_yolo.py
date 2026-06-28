"""Configurable native PyTorch YOLO-style detector."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class SimpleYOLOConfig:
    """Architecture settings for :class:`SimpleYOLODetector`."""

    nc: int
    input_channels: int = 3
    base_channels: int = 16
    width_multiplier: float = 1.0
    channel_multipliers: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    blocks_per_stage: list[int] = field(default_factory=lambda: [1, 1, 1, 1])
    output_stride: int = 16
    boxes_per_cell: int = 2
    activation: str = "silu"
    dropout: float = 0.0

    @classmethod
    def from_mapping(cls, payload: Any, *, nc: int) -> "SimpleYOLOConfig":
        data = dict(payload)
        data.pop("nc", None)
        return cls(nc=int(nc), **data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nc": self.nc,
            "input_channels": self.input_channels,
            "base_channels": self.base_channels,
            "width_multiplier": self.width_multiplier,
            "channel_multipliers": list(self.channel_multipliers),
            "blocks_per_stage": list(self.blocks_per_stage),
            "output_stride": self.output_stride,
            "boxes_per_cell": self.boxes_per_cell,
            "activation": self.activation,
            "dropout": self.dropout,
        }


def _activation(name: str) -> nn.Module:
    lowered = str(name).strip().lower()
    if lowered == "silu":
        return nn.SiLU(inplace=True)
    if lowered == "relu":
        return nn.ReLU(inplace=True)
    if lowered == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError("simple.activation must be one of: silu, relu, leaky_relu.")


def _make_channels(base_channels: int, width_multiplier: float, multiplier: int) -> int:
    raw = int(round(float(base_channels) * float(width_multiplier) * int(multiplier)))
    return max(1, int(math.ceil(raw / 8.0) * 8))


class ConvBNAct(nn.Module):
    """Convolution block used by the tiny detector."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int, activation: str) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            _activation(activation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """Small residual block for depth scaling."""

    def __init__(self, channels: int, *, activation: str, dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            ConvBNAct(channels, channels, stride=1, activation=activation),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        ]
        if float(dropout) > 0.0:
            layers.append(nn.Dropout2d(float(dropout)))
        self.net = nn.Sequential(*layers)
        self.act = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class SimpleYOLODetector(nn.Module):
    """Single-scale anchor-free YOLO-style detector.

    The output shape is ``[B, boxes_per_cell, 5 + nc, grid_h, grid_w]`` where
    the five box channels are raw ``tx, ty, tw, th, objectness`` logits.
    """

    def __init__(self, config: SimpleYOLOConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config(config)

        in_channels = int(config.input_channels)
        layers: list[nn.Module] = []
        current_stride = 1
        for stage_idx, (multiplier, n_blocks) in enumerate(
            zip(config.channel_multipliers, config.blocks_per_stage)
        ):
            out_channels = _make_channels(config.base_channels, config.width_multiplier, multiplier)
            stride = 2 if current_stride < int(config.output_stride) else 1
            layers.append(ConvBNAct(in_channels, out_channels, stride=stride, activation=config.activation))
            if stride == 2:
                current_stride *= 2
            for _ in range(int(n_blocks)):
                layers.append(ResidualBlock(out_channels, activation=config.activation, dropout=config.dropout))
            in_channels = out_channels

        if current_stride != int(config.output_stride):
            raise ValueError(
                "simple.channel_multipliers/blocks_per_stage must provide enough stages "
                f"to reach output_stride={config.output_stride}; reached stride={current_stride}."
            )

        head_channels = _make_channels(config.base_channels, config.width_multiplier, config.channel_multipliers[-1])
        self.backbone = nn.Sequential(*layers)
        self.neck = ConvBNAct(in_channels, head_channels, stride=1, activation=config.activation)
        self.head = nn.Conv2d(head_channels, int(config.boxes_per_cell) * (5 + int(config.nc)), kernel_size=1)

    @staticmethod
    def _validate_config(config: SimpleYOLOConfig) -> None:
        if int(config.nc) <= 0:
            raise ValueError("simple YOLO requires nc > 0.")
        if int(config.input_channels) <= 0:
            raise ValueError("simple.input_channels must be positive.")
        if int(config.base_channels) <= 0:
            raise ValueError("simple.base_channels must be positive.")
        if float(config.width_multiplier) <= 0:
            raise ValueError("simple.width_multiplier must be > 0.")
        if int(config.boxes_per_cell) <= 0:
            raise ValueError("simple.boxes_per_cell must be positive.")
        if int(config.output_stride) <= 0 or int(config.output_stride) & (int(config.output_stride) - 1):
            raise ValueError("simple.output_stride must be a positive power of two.")
        if len(config.channel_multipliers) != len(config.blocks_per_stage):
            raise ValueError("simple.channel_multipliers and simple.blocks_per_stage must have equal length.")
        if not config.channel_multipliers:
            raise ValueError("simple.channel_multipliers must not be empty.")
        if any(int(multiplier) <= 0 for multiplier in config.channel_multipliers):
            raise ValueError("simple.channel_multipliers must contain positive integers.")
        if any(int(n_blocks) < 0 for n_blocks in config.blocks_per_stage):
            raise ValueError("simple.blocks_per_stage must contain non-negative integers.")
        if not 0.0 <= float(config.dropout) < 1.0:
            raise ValueError("simple.dropout must be in [0, 1).")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.neck(self.backbone(x))
        raw = self.head(features)
        b, _, h, w = raw.shape
        return raw.view(b, self.config.boxes_per_cell, 5 + self.config.nc, h, w)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters."""

    return sum(param.numel() for param in model.parameters() if param.requires_grad)


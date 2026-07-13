"""SSDLite detector with MobileNetV3-Small backbone and depthwise-separable heads."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


_HEAD_CHANNELS = 256  # projection width shared by all prediction heads
_VARIANCES = (0.1, 0.1, 0.2, 0.2)  # SSD box-encoding scale factors


@dataclass(frozen=True)
class SSDLiteConfig:
    """Architecture settings for :class:`SSDLiteDetector`."""

    nc: int
    input_channels: int = 3
    n_feature_maps: int = 3
    anchor_min_sizes: tuple[float, ...] = (0.07, 0.15, 0.33)
    anchor_max_sizes: tuple[float, ...] = (0.15, 0.33, 0.60)
    anchor_aspect_ratios: tuple[float, ...] = (2.0,)
    conf_threshold: float = 0.25
    nms_iou_threshold: float = 0.45
    iou_pos_threshold: float = 0.50
    iou_neg_threshold: float = 0.40

    @classmethod
    def from_mapping(cls, payload: Any, *, nc: int) -> "SSDLiteConfig":
        data = dict(payload)
        data.pop("nc", None)
        # Convert lists to tuples for frozen dataclass compatibility
        for key in ("anchor_min_sizes", "anchor_max_sizes", "anchor_aspect_ratios"):
            if key in data and isinstance(data[key], list):
                data[key] = tuple(data[key])
        return cls(nc=int(nc), **data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "nc": self.nc,
            "input_channels": self.input_channels,
            "n_feature_maps": self.n_feature_maps,
            "anchor_min_sizes": list(self.anchor_min_sizes),
            "anchor_max_sizes": list(self.anchor_max_sizes),
            "anchor_aspect_ratios": list(self.anchor_aspect_ratios),
            "conf_threshold": self.conf_threshold,
            "nms_iou_threshold": self.nms_iou_threshold,
            "iou_pos_threshold": self.iou_pos_threshold,
            "iou_neg_threshold": self.iou_neg_threshold,
        }

    @property
    def anchors_per_cell(self) -> int:
        """2 square anchors (min_size + sqrt(min*max)) + 2 per aspect ratio (r and 1/r)."""
        return 2 + 2 * len(self.anchor_aspect_ratios)


class _DepthwiseSeparableConv(nn.Module):
    """Depthwise + pointwise conv with BN and ReLU6 (SSDLite style)."""

    def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1,
                      groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _SSDLiteHead(nn.Module):
    """Prediction head for one feature map scale."""

    def __init__(self, in_channels: int, anchors_per_cell: int, nc: int) -> None:
        super().__init__()
        self.anchors_per_cell = anchors_per_cell
        self.nc = nc
        # Shared DW-sep feature refinement
        self.shared = _DepthwiseSeparableConv(in_channels, in_channels)
        # Separate output branches (1×1 conv — no activation)
        self.cls_head = nn.Conv2d(in_channels, anchors_per_cell * nc, kernel_size=1)
        self.box_head = nn.Conv2d(in_channels, anchors_per_cell * 4, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.shared(x)
        B, _, H, W = feat.shape
        # [B, A*nc, H, W] → [B, H*W*A, nc]
        cls = self.cls_head(feat).permute(0, 2, 3, 1).reshape(B, -1, self.nc)
        # [B, A*4, H, W] → [B, H*W*A, 4]
        box = self.box_head(feat).permute(0, 2, 3, 1).reshape(B, -1, 4)
        return cls, box


class _MBV3SmallExtractor(nn.Module):
    """Splits MobileNetV3-Small features into 3 stages by spatial resolution."""

    def __init__(self, features: nn.Sequential) -> None:
        super().__init__()
        # Discover split indices by running a dummy forward pass
        dummy = torch.zeros(1, 3, 256, 256)
        sizes: list[int] = []
        with torch.no_grad():
            for layer in features:
                dummy = layer(dummy)
                sizes.append(int(dummy.shape[2]))

        # Last layer index (inclusive) that outputs each target size
        def _last_idx_at_size(target: int) -> int:
            return max(i for i, s in enumerate(sizes) if s == target)

        end_32 = _last_idx_at_size(32) + 1   # exclusive
        end_16 = _last_idx_at_size(16) + 1
        # The rest of the layers bring us to 8×8

        self.stage1 = nn.Sequential(*features[:end_32])   # → 32×32
        self.stage2 = nn.Sequential(*features[end_32:end_16])  # → 16×16
        self.stage3 = nn.Sequential(*features[end_16:])   # → 8×8

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        f1 = self.stage1(x)
        f2 = self.stage2(f1)
        f3 = self.stage3(f2)
        return [f1, f2, f3]


class SSDLiteDetector(nn.Module):
    """Lightweight SSD detector with MobileNetV3-Small backbone.

    forward() returns (cls_logits, bbox_pred, anchors):
      cls_logits: [B, N_anchors, nc]  — raw class logits (no sigmoid)
      bbox_pred:  [B, N_anchors, 4]   — raw box deltas (SSD encoding)
      anchors:    [N_anchors, 4]      — precomputed cx/cy/w/h default boxes
    """

    def __init__(self, config: SSDLiteConfig) -> None:
        super().__init__()
        self.config = config
        self._validate_config(config)

        try:
            from torchvision.models import mobilenet_v3_small
        except ImportError as exc:
            raise ImportError(
                "SSDLite requires torchvision. Install it via: pip install torchvision"
            ) from exc

        mbv3 = mobilenet_v3_small(weights=None)
        # Adapt first conv if input_channels != 3
        if int(config.input_channels) != 3:
            original = mbv3.features[0][0]
            mbv3.features[0][0] = nn.Conv2d(
                int(config.input_channels),
                original.out_channels,
                kernel_size=original.kernel_size,
                stride=original.stride,
                padding=original.padding,
                bias=False,
            )

        self.backbone = _MBV3SmallExtractor(mbv3.features)

        # Discover backbone output channel counts
        dummy = torch.zeros(1, int(config.input_channels), 256, 256)
        with torch.no_grad():
            backbone_fms = self.backbone(dummy)
        backbone_channels = [int(fm.shape[1]) for fm in backbone_fms]

        # 1×1 projection convs: backbone_ch → _HEAD_CHANNELS
        self.projection = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, _HEAD_CHANNELS, kernel_size=1, bias=False),
                nn.BatchNorm2d(_HEAD_CHANNELS),
                nn.ReLU6(inplace=True),
            )
            for ch in backbone_channels
        ])

        # Extra stride-2 DW-sep layers for n_feature_maps > 3
        n_extra = max(0, int(config.n_feature_maps) - 3)
        self.extra_layers = nn.ModuleList([
            _DepthwiseSeparableConv(_HEAD_CHANNELS, _HEAD_CHANNELS, stride=2)
            for _ in range(n_extra)
        ])

        n_fms = int(config.n_feature_maps)
        anchors_per_cell = config.anchors_per_cell

        # One prediction head per feature map
        self.heads = nn.ModuleList([
            _SSDLiteHead(_HEAD_CHANNELS, anchors_per_cell, int(config.nc))
            for _ in range(n_fms)
        ])

        # Precompute and register anchor buffer
        dummy_fms = [proj(fm) for proj, fm in zip(self.projection, backbone_fms)]
        extra_input = dummy_fms[-1]
        for layer in self.extra_layers:
            extra_input = layer(extra_input)
            dummy_fms.append(extra_input)
        fm_sizes = [(int(fm.shape[2]), int(fm.shape[3])) for fm in dummy_fms[:n_fms]]
        anchors = generate_ssdlite_anchors(config, fm_sizes)
        self.register_buffer("anchors", anchors)

    @staticmethod
    def _validate_config(config: SSDLiteConfig) -> None:
        if int(config.nc) <= 0:
            raise ValueError("SSDLite requires nc > 0.")
        if int(config.n_feature_maps) < 1:
            raise ValueError("ssdlite.n_feature_maps must be >= 1.")
        if int(config.n_feature_maps) > 5:
            raise ValueError("ssdlite.n_feature_maps must be <= 5 (max extra layers: 2).")
        if not config.anchor_min_sizes:
            raise ValueError("ssdlite.anchor_min_sizes must not be empty.")
        if len(config.anchor_min_sizes) < int(config.n_feature_maps):
            raise ValueError(
                "ssdlite.anchor_min_sizes must have at least n_feature_maps entries."
            )
        if len(config.anchor_max_sizes) < int(config.n_feature_maps):
            raise ValueError(
                "ssdlite.anchor_max_sizes must have at least n_feature_maps entries."
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fms = self.backbone(x)
        projected = [proj(fm) for proj, fm in zip(self.projection, fms)]

        all_fms = list(projected)
        extra_in = projected[-1]
        for layer in self.extra_layers:
            extra_in = layer(extra_in)
            all_fms.append(extra_in)

        n_fms = int(self.config.n_feature_maps)
        cls_list: list[torch.Tensor] = []
        box_list: list[torch.Tensor] = []
        for fm, head in zip(all_fms[:n_fms], self.heads):
            cls, box = head(fm)
            cls_list.append(cls)
            box_list.append(box)

        cls_logits = torch.cat(cls_list, dim=1)   # [B, N, nc]
        bbox_pred = torch.cat(box_list, dim=1)    # [B, N, 4]
        return cls_logits, bbox_pred, self.anchors  # type: ignore[return-value]


def generate_ssdlite_anchors(
    config: SSDLiteConfig,
    feature_map_sizes: list[tuple[int, int]],
) -> torch.Tensor:
    """Generate SSD default boxes for all feature maps.

    Returns Tensor[N_total, 4] in cx/cy/w/h normalized format.
    Ordering: for each feature map, for each cell, all anchor shapes.
    """
    min_sizes = list(config.anchor_min_sizes)
    max_sizes = list(config.anchor_max_sizes)
    aspect_ratios = list(config.anchor_aspect_ratios)

    all_anchors: list[torch.Tensor] = []
    for fm_idx, (fh, fw) in enumerate(feature_map_sizes):
        s_min = float(min_sizes[fm_idx])
        s_max = float(max_sizes[fm_idx])

        # Box shapes for this scale: [(w, h), ...]
        box_wh: list[tuple[float, float]] = [
            (s_min, s_min),                                                  # square at min_size
            (math.sqrt(s_min * s_max), math.sqrt(s_min * s_max)),           # square at sqrt(min*max)
        ]
        for r in aspect_ratios:
            r = float(r)
            box_wh.append((s_min * math.sqrt(r), s_min / math.sqrt(r)))     # wide  (r:1)
            box_wh.append((s_min / math.sqrt(r), s_min * math.sqrt(r)))     # tall  (1:r)

        # Cell centers
        cy = (torch.arange(fh, dtype=torch.float32) + 0.5) / fh
        cx = (torch.arange(fw, dtype=torch.float32) + 0.5) / fw
        grid_cy, grid_cx = torch.meshgrid(cy, cx, indexing="ij")  # [fh, fw]
        cx_flat = grid_cx.reshape(-1)   # [n_cells]
        cy_flat = grid_cy.reshape(-1)

        n_cells = fh * fw
        n_boxes = len(box_wh)
        # [n_cells, n_boxes, 4]
        anchors_fm = torch.zeros(n_cells, n_boxes, 4)
        anchors_fm[:, :, 0] = cx_flat.unsqueeze(1)
        anchors_fm[:, :, 1] = cy_flat.unsqueeze(1)
        for i, (bw, bh) in enumerate(box_wh):
            anchors_fm[:, i, 2] = float(bw)
            anchors_fm[:, i, 3] = float(bh)

        all_anchors.append(anchors_fm.reshape(-1, 4))

    return torch.cat(all_anchors, dim=0).clamp(0.0, 1.0)


def build_ssdlite_model(cfg: Any, *, nc: int) -> SSDLiteDetector:
    """Instantiate :class:`SSDLiteDetector` from a training config."""
    payload = (
        vars(cfg.model.ssdlite)
        if hasattr(cfg.model.ssdlite, "__dataclass_fields__")
        else dict(cfg.model.ssdlite)
    )
    config = SSDLiteConfig.from_mapping(payload, nc=nc)
    return SSDLiteDetector(config)

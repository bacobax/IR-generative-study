"""Reusable dataset classes for loading .npy thermal images."""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class NPYImageDataset(Dataset):
    """Load single-channel ``.npy`` images and return tensors.

    Parameters
    ----------
    root_dir : str
        Directory containing ``.npy`` files.
    transform : callable, optional
        Applied to the raw float tensor after loading.
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.files = sorted(
            f for f in os.listdir(root_dir) if f.endswith(".npy")
        )
        if len(self.files) == 0:
            raise RuntimeError("No .npy files found")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> torch.Tensor:
        x = np.load(os.path.join(self.root_dir, self.files[idx]))
        if x.ndim == 2:
            x = x[None, ...]  # (1,H,W)
        x = torch.from_numpy(x).float()
        if self.transform:
            x = self.transform(x)
        return x


class NPYStemDataset(Dataset):
    """Load ``.npy`` images and return ``(tensor, stem)`` pairs.

    Parameters
    ----------
    root_dir : str
        Directory containing ``.npy`` files.
    transform : callable, optional
        Applied to the raw float tensor after loading.
    stem_list : list of str, optional
        If provided, only load these stems (in order).
    max_items : int
        If > 0, cap the number of items loaded.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        stem_list: Optional[List[str]] = None,
        max_items: int = 0,
    ):
        self.root_dir = root_dir
        self.transform = transform

        if stem_list is not None:
            files = [f"{s}.npy" for s in stem_list]
        else:
            files = sorted(f for f in os.listdir(root_dir) if f.endswith(".npy"))

        if max_items > 0:
            files = files[:max_items]

        self.files = files
        if not self.files:
            raise RuntimeError(f"No .npy files found in {root_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        x = np.load(os.path.join(self.root_dir, fname))
        if x.ndim == 2:
            x = x[None, ...]
        x = torch.from_numpy(x).float()
        if self.transform is not None:
            x = self.transform(x)
        stem = Path(fname).stem
        return x, stem


class BBoxConditioningDataset(Dataset):
    """Load ``.npy`` images with COCO-format bounding-box annotations.

    Returns a dict with keys ``pixel_values`` (normalised image) and
    ``conditioning_pixel_values`` (binary bbox mask), both resized to 256x256.

    Parameters
    ----------
    root_dir : str
        Directory containing ``.npy`` image files.
    annotations_path : str
        Path to a COCO-format JSON with ``images`` and ``annotations``.
    conditioning_dropout : float
        Probability of zeroing out the conditioning mask (for training).
    """

    def __init__(
        self,
        root_dir: str,
        annotations_path: str,
        conditioning_dropout: float = 0.0,
    ):
        self.root_dir = root_dir
        self.conditioning_dropout = conditioning_dropout

        self.files = sorted(
            f for f in os.listdir(root_dir) if f.endswith(".npy")
        )
        if not self.files:
            raise RuntimeError(f"No .npy files found in {root_dir}")

        with open(annotations_path, "r") as fh:
            data = json.load(fh)

        id_to_fname: Dict[str, str] = {}
        self.img_info: Dict[str, dict] = {}
        for img in data["images"]:
            id_to_fname[img["id"]] = img["file_name"]
            self.img_info[img["file_name"]] = {
                "width": img["width"],
                "height": img["height"],
                "boxes": [],
            }

        for annot in data["annotations"]:
            fname = id_to_fname.get(annot["image_id"])
            if fname is not None and fname in self.img_info:
                self.img_info[fname]["boxes"].append(annot["bbox"])

        print(
            f"[BBoxDataset] {len(self.files)} files, "
            f"{len(self.img_info)} annotated images"
        )

    def __len__(self) -> int:
        return len(self.files)

    @staticmethod
    def _make_bbox_mask(
        boxes: list, width: int, height: int,
    ) -> torch.Tensor:
        """Rasterise ``[x, y, w, h]`` boxes into a binary (1, H, W) mask."""
        mask = torch.zeros(1, height, width)
        for bx, by, bw, bh in boxes:
            x0 = max(0, int(bx))
            y0 = max(0, int(by))
            x1 = min(width, int(bx + bw + 0.5))
            y1 = min(height, int(by + bh + 0.5))
            mask[0, y0:y1, x0:x1] = 1.0
        return mask

    def __getitem__(self, idx: int) -> dict:
        fname = self.files[idx]

        x = np.load(os.path.join(self.root_dir, fname))
        if x.ndim == 2:
            x = x[None, ...]
        x = torch.from_numpy(x).float()

        info = self.img_info.get(fname)
        if info is not None:
            w, h = info["width"], info["height"]
            boxes = info["boxes"]
        else:
            _, h, w = x.shape
            boxes = []

        mask = self._make_bbox_mask(boxes, w, h)

        # Resize to 256x256
        x = F.interpolate(
            x.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=False,
        ).squeeze(0)
        x = x - x.min()
        x = x / (x.max() + 1e-8)
        x = 2 * x - 1

        mask = F.interpolate(
            mask.unsqueeze(0), size=(256, 256), mode="nearest",
        ).squeeze(0)

        if (
            self.conditioning_dropout > 0
            and torch.rand(()).item() < self.conditioning_dropout
        ):
            mask = torch.zeros_like(mask)

        return {
            "pixel_values": x,
            "conditioning_pixel_values": mask,
        }

"""Annotation-aware FM dataset with optional curriculum learning.

Provides a single dataset class that supports:
- Unconditional FM training (returns image tensors)
- Text-conditioned FM training (returns image + caption dicts)
- Optional curriculum learning with person-centered crops

When curriculum is disabled, returns full images (standard behaviour).
When curriculum is enabled, may return person-centered crops according
to a configurable epoch-driven schedule.

Captions (text-conditioned mode only) are derived dynamically from
``annotations.json`` — **never** from ``captions.json``.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.core.data.annotations import (
    caption_from_count,
    count_fully_contained,
    count_people_for_image,
    get_bboxes_xyxy_for_image,
    index_annotations,
    load_coco_annotations,
    sample_person_crop,
)
from src.core.normalization import resize_and_normalize_256


def _resolve_allowed_counts(
    count_filter,
    all_counts: set,
) -> Optional[set]:
    """Resolve a CountFilterConfig to an explicit set of allowed counts.

    Returns None when no filtering is requested.
    """
    if count_filter is None:
        return None
    seen = getattr(count_filter, "seen_counts", None)
    unseen = getattr(count_filter, "unseen_counts", None)
    if seen is None and unseen is None:
        return None
    if seen is not None and unseen is not None:
        raise ValueError(
            "CountFilterConfig: set either seen_counts or unseen_counts, not both."
        )
    if seen is not None:
        return set(seen)
    # unseen is not None
    return all_counts - set(unseen)


class AnnotationFMDataset(Dataset):
    """FM dataset with annotation-driven captions and curriculum crops.

    Parameters
    ----------
    root_dir : str
        Directory containing ``.npy`` image files.
    annotations_path : str
        Path to COCO-format ``annotations.json``.
    text_mode : bool
        If True, return ``{"pixel_values": ..., "text": ...}`` dicts.
        If False, return plain image tensors (unconditional mode).
    curriculum : CurriculumConfig or None
        Curriculum learning settings. If None or ``enabled=False``,
        standard full-image behaviour is used.
    transform : callable, optional
        Standard augmentation transform (e.g. ``ScheduledAugment256``).
        Applied to the image tensor **after** any curriculum crop.
        When curriculum crops are active, the transform's own crop is
        bypassed — the curriculum crop replaces it.
    resize_target : int
        Spatial size to resize crops to before normalisation.
    """

    def __init__(
        self,
        root_dir: str,
        annotations_path: str,
        *,
        text_mode: bool = False,
        curriculum: Optional[Any] = None,
        count_filter: Optional[Any] = None,
        transform: Optional[Callable] = None,
        resize_target: int = 256,
    ):
        self.root_dir = root_dir
        self.text_mode = text_mode
        self.transform = transform
        self.resize_target = resize_target
        self._epoch = 0

        # Load and index annotations
        coco = load_coco_annotations(annotations_path)
        self.images_by_id, self.anns_by_image_id, self.fname_to_imgid = (
            index_annotations(coco)
        )

        # Collect .npy files
        all_files = sorted(f for f in os.listdir(root_dir) if f.endswith(".npy"))
        if not all_files:
            raise RuntimeError(f"No .npy files found in {root_dir}")

        # ── Count filtering ───────────────────────────────────────────
        # Compute full-image person count for every file
        self._file_full_counts: Dict[str, int] = {}
        all_observed_counts: set = set()
        for fname in all_files:
            image_id = self.fname_to_imgid.get(fname)
            n = len(get_bboxes_xyxy_for_image(self.anns_by_image_id, image_id)) if image_id is not None else 0
            self._file_full_counts[fname] = n
            all_observed_counts.add(n)

        self._allowed_counts = _resolve_allowed_counts(count_filter, all_observed_counts)
        self._max_crop_retries = getattr(count_filter, "max_crop_retries", 5) if count_filter is not None else 5

        if self._allowed_counts is not None:
            self.files = [f for f in all_files if self._file_full_counts[f] in self._allowed_counts]
            n_excluded = len(all_files) - len(self.files)
            _seen = sorted(self._allowed_counts)
            print(
                f"[CountFilter] kept {len(self.files)}/{len(all_files)} files, "
                f"excluded {n_excluded}, allowed_counts={_seen}"
            )
            if not self.files:
                raise RuntimeError(
                    f"Count filter excluded ALL files. "
                    f"allowed_counts={_seen}, observed_counts={sorted(all_observed_counts)}"
                )
        else:
            self.files = all_files

        # Curriculum config
        self.curriculum_enabled = False
        self._crop_prob_start = 0.0
        self._crop_prob_end = 0.5
        self._crop_schedule = "fixed"
        self._crop_margin_range = (1.2, 2.0)
        self._crop_jitter = 0.15
        self._crop_force_square = False
        self._total_epochs = 1

        if curriculum is not None and getattr(curriculum, "enabled", False):
            self.curriculum_enabled = True
            self._crop_prob_start = getattr(curriculum, "crop_prob_start", 0.0)
            self._crop_prob_end = getattr(curriculum, "crop_prob_end", 0.5)
            self._crop_schedule = getattr(curriculum, "schedule", "linear")
            self._crop_margin_range = (
                getattr(curriculum, "margin_min", 1.2),
                getattr(curriculum, "margin_max", 2.0),
            )
            self._crop_jitter = getattr(curriculum, "center_jitter", 0.15)
            self._crop_force_square = getattr(curriculum, "force_square", False)
            self._total_epochs = max(1, getattr(curriculum, "total_epochs", 1))

        n_annotated = sum(
            1 for f in self.files if f in self.fname_to_imgid
        )
        print(
            f"[AnnotationFMDataset] {len(self.files)} files, "
            f"{n_annotated} annotated, text_mode={text_mode}, "
            f"curriculum={self.curriculum_enabled}"
        )

    # ── Epoch management ──────────────────────────────────────────────

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch for curriculum scheduling."""
        self._epoch = epoch

    def _current_crop_prob(self) -> float:
        """Compute current crop probability based on schedule."""
        if not self.curriculum_enabled:
            return 0.0
        if self._crop_schedule == "fixed":
            return self._crop_prob_end
        # Linear schedule: start -> end over total_epochs
        alpha = min(1.0, self._epoch / max(1, self._total_epochs - 1))
        return self._crop_prob_start + alpha * (self._crop_prob_end - self._crop_prob_start)

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.files)

    def _count_allowed(self, n: int) -> bool:
        """Return True if person count *n* passes the count filter."""
        if self._allowed_counts is None:
            return True
        return n in self._allowed_counts

    def __getitem__(self, idx: int):
        fname = self.files[idx]
        stem = Path(fname).stem

        # Load raw image
        arr = np.load(os.path.join(self.root_dir, fname))
        if arr.ndim == 2:
            arr = arr[None, ...]  # (1, H, W)
        img_h, img_w = arr.shape[-2], arr.shape[-1]

        image_id = self.fname_to_imgid.get(fname)
        bboxes_xyxy = (
            get_bboxes_xyxy_for_image(self.anns_by_image_id, image_id)
            if image_id is not None
            else []
        )

        # Decide: crop or full image
        use_crop = False
        crop_region = None
        if self.curriculum_enabled and bboxes_xyxy:
            p_crop = self._current_crop_prob()
            if random.random() < p_crop:
                use_crop = True

        if use_crop:
            # Try up to max_crop_retries times to get a crop with allowed count
            for _attempt in range(self._max_crop_retries):
                candidate = sample_person_crop(
                    (img_h, img_w),
                    bboxes_xyxy,
                    margin_scale_range=self._crop_margin_range,
                    center_jitter_frac=self._crop_jitter,
                    force_square=self._crop_force_square,
                )
                crop_count = count_fully_contained(bboxes_xyxy, candidate)
                if self._count_allowed(crop_count):
                    crop_region = candidate
                    break
            else:
                # All retries produced excluded counts — fall back to full image
                use_crop = False
                crop_region = None

        cropped_arr = arr
        if use_crop and crop_region is not None:
            x1, y1, x2, y2 = crop_region
            cropped_arr = arr[..., y1:y2, x1:x2]

        # Convert to tensor
        x = torch.from_numpy(cropped_arr.copy()).float()

        # Apply standard transform (augmentation + resize + normalise)
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = resize_and_normalize_256(x)

        # Produce output
        if not self.text_mode:
            return x

        # Text-conditioned: derive caption from annotations
        if use_crop and crop_region is not None:
            n_people = count_fully_contained(bboxes_xyxy, crop_region)
        else:
            n_people = len(bboxes_xyxy)

        caption = caption_from_count(n_people)
        return {"pixel_values": x, "text": caption}

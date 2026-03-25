"""Batch collation helpers for bbox-conditioned layout training.

The layout-conditioned FM path consumes batches with a variable number of
objects per image. This module pads objects within a batch while preserving
alignment between ``boxes_xyxy`` and ``labels``.

Padding behavior
----------------
* Boxes are padded with zeros to ``(B, max_objects_in_batch, 4)``.
* Labels are padded with zeros. Category id 0 may be a valid class, so padded
  entries must always be interpreted together with ``object_mask``.
* ``object_mask`` is a boolean tensor marking which rows correspond to real
  objects.

Box format
----------
The input dataset already returns resized boxes in ``xyxy`` pixel coordinates
aligned with ``pixel_values``. This collate helper also emits normalized boxes
in ``[0, 1]`` using the resized image size.
"""

from __future__ import annotations

from typing import Any, Dict, List

import torch


def collate_layout_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length bbox annotations into a single mini-batch."""
    if not batch:
        raise ValueError("layout batch must contain at least one sample")

    pixel_values = torch.stack([sample["pixel_values"] for sample in batch], dim=0)
    batch_size, _, image_h, image_w = pixel_values.shape
    max_objects = max(int(sample["n_objects"]) for sample in batch)

    boxes_xyxy = torch.zeros(batch_size, max_objects, 4, dtype=torch.float32)
    boxes_xyxy_norm = torch.zeros_like(boxes_xyxy)
    labels = torch.zeros(batch_size, max_objects, dtype=torch.long)
    object_mask = torch.zeros(batch_size, max_objects, dtype=torch.bool)
    n_objects = torch.zeros(batch_size, dtype=torch.long)

    file_names: List[str] = []
    image_ids: List[Any] = []
    label_names: List[List[str]] = []

    width_scale = max(float(image_w), 1.0)
    height_scale = max(float(image_h), 1.0)

    for batch_idx, sample in enumerate(batch):
        num_objects = int(sample["n_objects"])
        n_objects[batch_idx] = num_objects
        file_names.append(sample["file_name"])
        image_ids.append(sample["image_id"])
        label_names.append(list(sample.get("label_names", [])))

        if num_objects == 0:
            continue

        sample_boxes = sample["boxes_xyxy"].to(dtype=torch.float32)
        sample_labels = sample["labels"].to(dtype=torch.long)

        boxes_xyxy[batch_idx, :num_objects] = sample_boxes
        labels[batch_idx, :num_objects] = sample_labels
        object_mask[batch_idx, :num_objects] = True

        normalized = sample_boxes.clone()
        normalized[:, [0, 2]] /= width_scale
        normalized[:, [1, 3]] /= height_scale
        boxes_xyxy_norm[batch_idx, :num_objects] = normalized.clamp(0.0, 1.0)

    return {
        "pixel_values": pixel_values,
        "boxes_xyxy": boxes_xyxy,
        "boxes_xyxy_norm": boxes_xyxy_norm,
        "labels": labels,
        "object_mask": object_mask,
        "n_objects": n_objects,
        "file_name": file_names,
        "image_id": image_ids,
        "label_names": label_names,
    }

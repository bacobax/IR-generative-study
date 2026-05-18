"""FLIR layout-aware Stage-2 Stable Diffusion data loading."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.algorithms.stable_diffusion.data import get_transforms, ir_npy_to_normalized_rgb
from src.core.configs.sd_layout_config import SDLayoutTrainConfig
from src.core.data.annotations import (
    build_category_id_to_name,
    get_boxes_and_labels_for_image,
    index_annotations,
    load_coco_annotations,
)
from src.core.data.dataset_targets import resolve_dataset_target
from src.core.data.layout_batching import collate_layout_batch
from src.core.data.subset_manifest import filter_files_by_subset_manifest
from src.core.normalization import RAW_UINT16_PERCENTILE


def _normalize_label_name(label_name: str) -> str:
    return str(label_name).replace("_", " ").strip()


def build_layout_prompt(
    *,
    label_names: Sequence[str],
    prompt_mode: str,
    constant_prompt: str,
    thermal_scene_suffix: str,
    caption: Optional[str] = None,
    use_captions_if_available: bool = False,
) -> str:
    """Construct the prompt used for one layout-conditioned training sample."""
    if use_captions_if_available and caption:
        return str(caption).strip()

    if prompt_mode == "constant":
        return constant_prompt

    unique_names: List[str] = []
    seen = set()
    for label_name in label_names:
        normalized = _normalize_label_name(label_name)
        if not normalized or normalized in seen:
            continue
        unique_names.append(normalized)
        seen.add(normalized)

    if not unique_names:
        return constant_prompt

    suffix = str(thermal_scene_suffix).strip()
    if suffix and not suffix.endswith("."):
        suffix = f"{suffix}."
    objects = " and ".join(unique_names)
    prompt = f"An image of {objects}"
    if suffix:
        prompt = f"{prompt} {suffix}"
    return prompt.strip()


def transform_boxes_xyxy_square_pad_resize(
    boxes_xyxy: torch.Tensor,
    *,
    src_width: int,
    src_height: int,
    dst_size: int,
) -> torch.Tensor:
    """Apply the same square-pad-plus-resize geometry as the SD image path."""
    if boxes_xyxy.numel() == 0:
        return boxes_xyxy.reshape(0, 4)

    padded = boxes_xyxy.clone().to(torch.float32)
    if src_width < src_height:
        diff = int(src_height) - int(src_width)
        pad_left = diff // 2
        padded[:, [0, 2]] += float(pad_left)
        side = float(src_height)
    elif src_height < src_width:
        diff = int(src_width) - int(src_height)
        pad_top = diff // 2
        padded[:, [1, 3]] += float(pad_top)
        side = float(src_width)
    else:
        side = float(src_width)

    scale = float(dst_size) / max(side, 1.0)
    return padded * scale


def _load_optional_caption_map(split_dir: str) -> Dict[str, str]:
    captions_path = os.path.join(split_dir, "captions.json")
    if not os.path.isfile(captions_path):
        return {}
    with open(captions_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {str(key): str(value) for key, value in data.items()}


@dataclass(frozen=True)
class ResolvedLayoutSplit:
    """Resolved dataset split for SD layout training."""

    root_dir: str
    annotations_path: str
    normalization_mode: str


def resolve_layout_split(
    *,
    dataset_id: str,
    split: str,
    root_dir: Optional[str] = None,
    annotations_path: Optional[str] = None,
) -> ResolvedLayoutSplit:
    """Resolve one repo-native or explicit layout dataset split."""
    if root_dir is None or annotations_path is None:
        target = resolve_dataset_target(dataset_id)
        split_root = root_dir or str(target.split_dir(split))
        split_annotations = annotations_path or str(target.annotations_path(split))
        return ResolvedLayoutSplit(
            root_dir=split_root,
            annotations_path=split_annotations,
            normalization_mode=target.normalization_mode,
        )
    return ResolvedLayoutSplit(
        root_dir=root_dir,
        annotations_path=annotations_path,
        normalization_mode=RAW_UINT16_PERCENTILE,
    )


class StableDiffusionLayoutDataset(Dataset):
    """Layout-aware SD dataset that keeps bbox geometry aligned with square padding."""

    def __init__(
        self,
        *,
        root_dir: str,
        annotations_path: str,
        tokenizer,
        resolution: int,
        normalization_mode: str,
        prompt_mode: str,
        constant_prompt: str,
        thermal_scene_suffix: str,
        use_captions_if_available: bool,
        max_samples: Optional[int] = None,
        subset_manifest: Optional[str] = None,
    ) -> None:
        self.root_dir = root_dir
        self.annotations_path = annotations_path
        self.tokenizer = tokenizer
        self.resolution = int(resolution)
        self.normalization_mode = normalization_mode
        self.prompt_mode = prompt_mode
        self.constant_prompt = constant_prompt
        self.thermal_scene_suffix = thermal_scene_suffix
        self.use_captions_if_available = bool(use_captions_if_available)
        self.image_transforms = get_transforms(
            resolution=self.resolution,
            center_crop=False,
            random_flip=False,
            interpolation_mode="lanczos",
        )

        images_dir = os.path.join(root_dir, "images")
        self.images_dir = images_dir if os.path.isdir(images_dir) else root_dir
        files = sorted(
            file_name for file_name in os.listdir(self.images_dir) if file_name.endswith(".npy")
        )
        self.files = filter_files_by_subset_manifest(
            files,
            subset_manifest,
            split_dir=root_dir,
            context="StableDiffusionLayoutDataset",
        )
        if max_samples is not None:
            self.files = self.files[: int(max_samples)]
        if not self.files:
            raise RuntimeError(f"No .npy files found in {self.images_dir}")

        coco = load_coco_annotations(annotations_path)
        self.category_id_to_name = build_category_id_to_name(coco)
        self.num_categories = max(self.category_id_to_name.keys(), default=-1) + 1
        self.images_by_id, self.anns_by_image_id, self.fname_to_imgid = index_annotations(coco)
        self.caption_map = _load_optional_caption_map(root_dir)

    def __len__(self) -> int:
        return len(self.files)

    def _encode_prompt(self, prompt_text: str) -> torch.Tensor:
        return self.tokenizer(
            prompt_text,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        file_name = self.files[idx]
        image_path = os.path.join(self.images_dir, file_name)
        stem = Path(file_name).stem

        arr = np.load(image_path)
        if arr.ndim == 3:
            if arr.shape[0] == 1:
                arr = arr[0]
            elif arr.shape[-1] == 1:
                arr = arr[..., 0]
        if arr.ndim != 2:
            raise ValueError(f"Expected single-channel array for {image_path}, got shape {arr.shape}")

        image = ir_npy_to_normalized_rgb(arr, normalization_mode=self.normalization_mode)
        pixel_values = self.image_transforms(image)

        image_id = self.fname_to_imgid.get(file_name)
        image_info = self.images_by_id.get(image_id, {})
        src_width = int(image_info.get("width", arr.shape[1]))
        src_height = int(image_info.get("height", arr.shape[0]))

        boxes_xyxy_raw, labels, label_names = get_boxes_and_labels_for_image(
            self.anns_by_image_id,
            image_id,
            category_id_to_name=self.category_id_to_name,
            include_label_names=True,
        )
        boxes_xyxy = torch.tensor(boxes_xyxy_raw, dtype=torch.float32)
        if boxes_xyxy.numel() == 0:
            boxes_xyxy = boxes_xyxy.reshape(0, 4)
        boxes_xyxy = transform_boxes_xyxy_square_pad_resize(
            boxes_xyxy,
            src_width=src_width,
            src_height=src_height,
            dst_size=self.resolution,
        )
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        if labels_tensor.numel() == 0:
            labels_tensor = labels_tensor.reshape(0)

        caption = self.caption_map.get(stem)
        prompt_text = build_layout_prompt(
            label_names=label_names or [],
            prompt_mode=self.prompt_mode,
            constant_prompt=self.constant_prompt,
            thermal_scene_suffix=self.thermal_scene_suffix,
            caption=caption,
            use_captions_if_available=self.use_captions_if_available,
        )

        return {
            "pixel_values": pixel_values,
            "boxes_xyxy": boxes_xyxy,
            "labels": labels_tensor,
            "image_id": image_id,
            "file_name": file_name,
            "n_objects": int(labels_tensor.numel()),
            "label_names": list(label_names or []),
            "prompt_text": prompt_text,
            "caption_text": caption or "",
            "input_ids": self._encode_prompt(prompt_text),
        }


def collate_sd_layout_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Extend the shared layout collate with prompt-token payloads."""
    collated = collate_layout_batch(batch)
    collated["input_ids"] = torch.stack([sample["input_ids"] for sample in batch], dim=0)
    collated["prompt_text"] = [str(sample["prompt_text"]) for sample in batch]
    collated["caption_text"] = [str(sample.get("caption_text", "")) for sample in batch]
    return collated


def create_layout_dataloaders(
    config: SDLayoutTrainConfig,
    *,
    tokenizer,
) -> Tuple[DataLoader, DataLoader, StableDiffusionLayoutDataset, StableDiffusionLayoutDataset]:
    """Build train and validation dataloaders for SD layout stage-2."""
    train_split = resolve_layout_split(
        dataset_id=config.data.dataset_id,
        split=config.data.train_split,
        root_dir=config.data.train_data_dir,
        annotations_path=config.data.train_annotations,
    )
    val_split = resolve_layout_split(
        dataset_id=config.data.dataset_id,
        split=config.data.val_split,
        root_dir=config.data.val_data_dir,
        annotations_path=config.data.val_annotations,
    )

    train_dataset = StableDiffusionLayoutDataset(
        root_dir=train_split.root_dir,
        annotations_path=train_split.annotations_path,
        tokenizer=tokenizer,
        resolution=config.data.resolution,
        normalization_mode=train_split.normalization_mode,
        prompt_mode=config.prompt.prompt_mode,
        constant_prompt=config.prompt.constant_prompt,
        thermal_scene_suffix=config.prompt.thermal_scene_suffix,
        use_captions_if_available=config.prompt.use_captions_if_available,
        max_samples=config.data.max_train_samples,
        subset_manifest=getattr(config.data, "subset_manifest", None),
    )
    val_dataset = StableDiffusionLayoutDataset(
        root_dir=val_split.root_dir,
        annotations_path=val_split.annotations_path,
        tokenizer=tokenizer,
        resolution=config.data.resolution,
        normalization_mode=val_split.normalization_mode,
        prompt_mode=config.prompt.prompt_mode,
        constant_prompt=config.prompt.constant_prompt,
        thermal_scene_suffix=config.prompt.thermal_scene_suffix,
        use_captions_if_available=config.prompt.use_captions_if_available,
        max_samples=config.data.max_val_samples,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_sd_layout_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.validation.num_validation_images,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_sd_layout_batch,
    )
    return train_loader, val_loader, train_dataset, val_dataset


def build_fixed_validation_batch(
    dataset: StableDiffusionLayoutDataset,
    *,
    max_examples: int,
) -> Optional[Dict[str, Any]]:
    """Create a deterministic small validation batch for sample logging."""
    if len(dataset) == 0 or max_examples <= 0:
        return None
    examples = [dataset[index] for index in range(min(len(dataset), int(max_examples)))]
    return collate_sd_layout_batch(examples)

#!/usr/bin/env python
# coding=utf-8
"""Data loading for Stage-1 Stable Diffusion IR adaptation."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

from src.core.data.dataset_targets import resolve_dataset_target
from src.core.normalization import (
    RAW_UINT16_PERCENTILE,
    UINT8_LINEAR,
    normalize_image_tensor,
)


DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}


@dataclass(frozen=True)
class ResolvedTrainingData:
    """Resolved training dataset location and normalization mode."""

    train_data_dir: Optional[str]
    dataset_name: Optional[str]
    dataset_config_name: Optional[str]
    normalization_mode: str


def _load_metadata_jsonl(meta_path: str) -> Dict[str, str]:
    mapping = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            filename = obj.get("file_name")
            if filename is None:
                continue
            mapping[os.path.basename(filename)] = obj.get("text", "")
    return mapping


def resolve_training_data_source(
    *,
    dataset_id: Optional[str],
    dataset_name: Optional[str],
    dataset_config_name: Optional[str],
    train_data_dir: Optional[str],
    train_split: str,
) -> ResolvedTrainingData:
    """Resolve a repo-native dataset_id or fall back to existing inputs."""
    if dataset_id is not None:
        target = resolve_dataset_target(dataset_id)
        split_dir = target.split_dir(train_split)
        return ResolvedTrainingData(
            train_data_dir=str(split_dir),
            dataset_name=None,
            dataset_config_name=None,
            normalization_mode=target.normalization_mode,
        )

    if train_data_dir is not None:
        normalized_dir = train_data_dir
        if os.path.isdir(os.path.join(train_data_dir, train_split)):
            normalized_dir = os.path.join(train_data_dir, train_split)
        return ResolvedTrainingData(
            train_data_dir=normalized_dir,
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            normalization_mode=RAW_UINT16_PERCENTILE,
        )

    return ResolvedTrainingData(
        train_data_dir=None,
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        normalization_mode=RAW_UINT16_PERCENTILE,
    )


def get_interpolation_mode(mode_name: str) -> transforms.InterpolationMode:
    mode = getattr(transforms.InterpolationMode, mode_name.upper(), None)
    if mode is None:
        raise ValueError(f"Unsupported interpolation mode: {mode_name}")
    return mode


class SquarePad:
    """Pad a PIL image to square before resizing."""

    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        diff = abs(w - h)
        pad1 = diff // 2
        pad2 = diff - pad1
        if w < h:
            padding = (pad1, 0, pad2, 0)
        else:
            padding = (0, pad1, 0, pad2)
        return TF.pad(img, padding, fill=0)


def get_transforms(
    resolution: int,
    center_crop: bool = False,
    random_flip: bool = False,
    interpolation_mode: str = "lanczos",
) -> transforms.Compose:
    interpolation = get_interpolation_mode(interpolation_mode)
    transform_list: List[Callable] = [
        SquarePad(),
        transforms.Resize((resolution, resolution), interpolation=interpolation),
    ]

    if center_crop:
        transform_list.append(transforms.CenterCrop((resolution, resolution)))

    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    return transforms.Compose(transform_list)


def _normalized_ir_to_pil_rgb(norm_tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized single-channel tensor in [-1, 1] to replicated RGB."""
    x01 = ((norm_tensor.squeeze(0).clamp(-1.0, 1.0) + 1.0) / 2.0).cpu().numpy()
    x8 = np.clip(np.rint(x01 * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(x8).convert("RGB")


def ir_npy_to_normalized_rgb(
    npy_or_path,
    *,
    normalization_mode: str,
) -> Image.Image:
    """Load a local IR `.npy`, normalize to [-1,1], then convert to RGB."""
    if isinstance(npy_or_path, (str, os.PathLike)):
        arr = np.load(npy_or_path)
    else:
        arr = np.asarray(npy_or_path)

    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Expected 1-channel .npy, got shape {arr.shape}")
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D or 3D 1-channel .npy, got {arr.ndim}D")

    tensor = torch.from_numpy(arr).unsqueeze(0).float()
    normalized = normalize_image_tensor(tensor, normalization_mode=normalization_mode)
    return _normalized_ir_to_pil_rgb(normalized)


class TextImageDataset:
    """Dataset wrapper for text-image pairs."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        image_transforms: transforms.Compose,
        image_column: str = "image",
        caption_column: str = "text",
        image_preprocessor: Optional[Callable] = None,
        prompt_text: Optional[str] = None,
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_transforms = image_transforms
        self.image_column = image_column
        self.caption_column = caption_column
        self.prompt_text = prompt_text
        self.image_preprocessor = image_preprocessor or (lambda x: x.convert("RGB"))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        image = self.image_preprocessor(example[self.image_column])
        pixel_values = self.image_transforms(image)

        caption = self.prompt_text
        if caption is None:
            caption = example.get(self.caption_column, "")
            if isinstance(caption, (list, np.ndarray)):
                caption = random.choice(caption)

        input_ids = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
        }


class RecordDataset(Dataset):
    """Small dict-record dataset used for local repo-native training data."""

    def __init__(self, records: List[Dict[str, object]]):
        self.records = list(records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        return self.records[idx]

    def shuffle(self, *, seed: Optional[int] = None) -> "RecordDataset":
        rng = random.Random(seed)
        records = list(self.records)
        rng.shuffle(records)
        return RecordDataset(records)

    def select(self, indices) -> "RecordDataset":
        return RecordDataset([self.records[int(idx)] for idx in indices])


def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def _build_local_training_dataset(
    train_data_dir: str,
    image_column: str,
    caption_column: str,
) -> Tuple[Dataset, str, str]:
    if os.path.isdir(os.path.join(train_data_dir, "images")):
        images_dir = os.path.join(train_data_dir, "images")
    else:
        images_dir = train_data_dir

    if not os.path.isdir(images_dir):
        raise ValueError(f"Expected training data directory at: {images_dir}")

    npy_paths = sorted(
        os.path.join(images_dir, fn)
        for fn in os.listdir(images_dir)
        if fn.lower().endswith(".npy")
    )
    if len(npy_paths) == 0:
        raise ValueError(f"No .npy files found in {images_dir}")

    candidate_meta_paths = [
        os.path.join(train_data_dir, "metadata.jsonl"),
        os.path.join(os.path.dirname(images_dir), "metadata.jsonl"),
    ]
    captions_map: Dict[str, str] = {}
    for meta_path in candidate_meta_paths:
        if os.path.isfile(meta_path):
            captions_map = _load_metadata_jsonl(meta_path)
            break

    captions = [captions_map.get(os.path.basename(path), "") for path in npy_paths]

    ds = RecordDataset(
        [
            {
                image_column: path,
                caption_column: caption,
            }
            for path, caption in zip(npy_paths, captions)
        ]
    )
    return ds, image_column, caption_column


def load_training_dataset(
    *,
    dataset_name: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    train_data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    image_column: str = "image",
    caption_column: str = "text",
) -> Tuple[Dataset, str, str]:
    if dataset_name is not None:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError(
                "The `datasets` package is required only when `dataset_name` "
                "points to a Hugging Face dataset. Install it or use "
                "`dataset_id`/`train_data_dir` for repo-native local data."
            ) from exc

        dataset = load_dataset(
            dataset_name,
            dataset_config_name,
            cache_dir=cache_dir,
            data_dir=train_data_dir,
        )
        column_names = dataset["train"].column_names
        dataset_columns = DATASET_NAME_MAPPING.get(dataset_name, None)

        if image_column is None:
            image_column = dataset_columns[0] if dataset_columns else column_names[0]
        elif image_column not in column_names:
            raise ValueError(f"Image column '{image_column}' not found. Available: {column_names}")

        if caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns else column_names[1]
        elif caption_column not in column_names:
            raise ValueError(
                f"Caption column '{caption_column}' not found. Available: {column_names}"
            )

        return dataset["train"], image_column, caption_column

    if train_data_dir is None:
        raise ValueError("train_data_dir must be provided when dataset_name is None")

    return _build_local_training_dataset(train_data_dir, image_column, caption_column)


def create_dataloader(
    *,
    dataset_id: Optional[str],
    dataset_name: Optional[str],
    dataset_config_name: Optional[str],
    train_data_dir: Optional[str],
    train_split: str,
    cache_dir: Optional[str],
    tokenizer,
    resolution: int,
    center_crop: bool,
    random_flip: bool,
    interpolation_mode: str,
    image_column: str,
    caption_column: str,
    batch_size: int,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    seed: Optional[int] = None,
    use_ir_preprocessing: bool = True,
    prompt_text: Optional[str] = None,
) -> Tuple[DataLoader, str]:
    """Create the training dataloader and return the resolved normalization mode."""
    resolved = resolve_training_data_source(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        train_data_dir=train_data_dir,
        train_split=train_split,
    )

    raw_dataset, img_col, cap_col = load_training_dataset(
        dataset_name=resolved.dataset_name,
        dataset_config_name=resolved.dataset_config_name,
        train_data_dir=resolved.train_data_dir,
        cache_dir=cache_dir,
        image_column=image_column,
        caption_column=caption_column,
    )

    if max_train_samples is not None:
        raw_dataset = raw_dataset.shuffle(seed=seed).select(range(max_train_samples))

    image_transforms = get_transforms(
        resolution=resolution,
        center_crop=center_crop,
        random_flip=random_flip,
        interpolation_mode=interpolation_mode,
    )

    image_preprocessor: Optional[Callable]
    if use_ir_preprocessing and resolved.train_data_dir is not None:
        image_preprocessor = lambda path: ir_npy_to_normalized_rgb(
            path,
            normalization_mode=resolved.normalization_mode,
        )
    else:
        image_preprocessor = None

    dataset = TextImageDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        image_transforms=image_transforms,
        image_column=img_col,
        caption_column=cap_col,
        image_preprocessor=image_preprocessor,
        prompt_text=prompt_text,
    )

    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    return dataloader, resolved.normalization_mode

#!/usr/bin/env python
# coding=utf-8
"""Data loading for FLUX.1-dev QLoRA fine-tuning."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.algorithms.stable_diffusion.data import (
    get_transforms,
    ir_npy_to_normalized_rgb,
    load_training_dataset,
    resolve_training_data_source,
)


class FluxImageDataset(Dataset):
    """Minimal image-only dataset for FLUX fine-tuning.

    Returns ``{"pixel_values": CHW tensor in [-1, 1]}``.  Prompt handling is
    global (one fixed prompt per dataset), so no tokenization happens here.
    """

    def __init__(
        self,
        dataset: Dataset,
        image_transforms,
        *,
        image_column: str = "image",
        image_preprocessor: Optional[Callable] = None,
    ):
        self.dataset = dataset
        self.image_transforms = image_transforms
        self.image_column = image_column
        self.image_preprocessor = image_preprocessor or (lambda x: x.convert("RGB"))

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        image: Image.Image = self.image_preprocessor(example[self.image_column])
        pixel_values = self.image_transforms(image)
        return {"pixel_values": pixel_values}


def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pixel_values = torch.stack([ex["pixel_values"] for ex in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    return {"pixel_values": pixel_values}


def create_dataloader(
    *,
    dataset_id: Optional[str],
    dataset_name: Optional[str],
    dataset_config_name: Optional[str],
    train_data_dir: Optional[str],
    train_split: str,
    cache_dir: Optional[str],
    resolution: int,
    center_crop: bool,
    random_flip: bool,
    interpolation_mode: str,
    image_column: str,
    caption_column: str,
    batch_size: int,
    num_workers: int = 0,
    max_train_samples: Optional[int] = None,
    subset_manifest: Optional[str] = None,
    seed: Optional[int] = None,
    use_ir_preprocessing: bool = True,
) -> Tuple[DataLoader, str]:
    """Create a FLUX training dataloader and return the normalization mode."""
    resolved = resolve_training_data_source(
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        train_data_dir=train_data_dir,
        train_split=train_split,
    )
    raw_dataset, img_col, _cap_col = load_training_dataset(
        dataset_name=resolved.dataset_name,
        dataset_config_name=resolved.dataset_config_name,
        train_data_dir=resolved.train_data_dir,
        cache_dir=cache_dir,
        image_column=image_column,
        caption_column=caption_column,
        subset_manifest=subset_manifest,
        manifest_path=resolved.manifest_path,
    )
    if max_train_samples is not None:
        raw_dataset = raw_dataset.shuffle(seed=seed).select(range(max_train_samples))

    image_preprocessor: Optional[Callable]
    if use_ir_preprocessing and resolved.train_data_dir is not None:
        image_preprocessor = lambda path: ir_npy_to_normalized_rgb(
            path,
            normalization_mode=resolved.normalization_mode,
        )
    else:
        image_preprocessor = None

    dataset = FluxImageDataset(
        dataset=raw_dataset,
        image_transforms=get_transforms(
            resolution=resolution,
            center_crop=center_crop,
            random_flip=random_flip,
            interpolation_mode=interpolation_mode,
        ),
        image_column=img_col,
        image_preprocessor=image_preprocessor,
    )
    return (
        DataLoader(
            dataset,
            shuffle=True,
            collate_fn=collate_fn,
            batch_size=batch_size,
            num_workers=num_workers,
        ),
        resolved.normalization_mode,
    )

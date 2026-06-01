#!/usr/bin/env python
# coding=utf-8
"""Data loading for Stage-1 Stable Diffusion XL IR adaptation."""

from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from src.algorithms.stable_diffusion.data import (
    get_transforms,
    ir_npy_to_normalized_rgb,
    load_training_dataset,
    resolve_training_data_source,
)


class TextImageDatasetXL(Dataset):
    """Dataset wrapper that emits SDXL dual-tokenizer inputs and size metadata."""

    def __init__(
        self,
        dataset: Dataset,
        tokenizer_one,
        tokenizer_two,
        image_transforms,
        *,
        resolution: int,
        image_column: str = "image",
        caption_column: str = "text",
        image_preprocessor: Optional[Callable] = None,
        prompt_text: Optional[str] = None,
    ):
        self.dataset = dataset
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.image_transforms = image_transforms
        self.resolution = int(resolution)
        self.image_column = image_column
        self.caption_column = caption_column
        self.prompt_text = prompt_text
        self.image_preprocessor = image_preprocessor or (lambda x: x.convert("RGB"))

    def __len__(self) -> int:
        return len(self.dataset)

    def _tokenize(self, tokenizer, caption: str) -> torch.Tensor:
        return tokenizer(
            caption,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | Tuple[int, int]]:
        example = self.dataset[idx]
        image: Image.Image = self.image_preprocessor(example[self.image_column])
        original_size = (int(image.height), int(image.width))
        pixel_values = self.image_transforms(image)

        caption = self.prompt_text
        if caption is None:
            caption = example.get(self.caption_column, "")
            if isinstance(caption, (list, np.ndarray)):
                caption = random.choice(caption)
        caption = str(caption or "")

        return {
            "pixel_values": pixel_values,
            "input_ids_one": self._tokenize(self.tokenizer_one, caption),
            "input_ids_two": self._tokenize(self.tokenizer_two, caption),
            "original_size": original_size,
            "crop_top_left": (0, 0),
            "target_size": (self.resolution, self.resolution),
        }


def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor | List[Tuple[int, int]]]:
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids_one = torch.stack([example["input_ids_one"] for example in examples])
    input_ids_two = torch.stack([example["input_ids_two"] for example in examples])
    return {
        "pixel_values": pixel_values,
        "input_ids_one": input_ids_one,
        "input_ids_two": input_ids_two,
        "original_sizes": [tuple(example["original_size"]) for example in examples],
        "crop_top_lefts": [tuple(example["crop_top_left"]) for example in examples],
        "target_sizes": [tuple(example["target_size"]) for example in examples],
    }


def create_dataloader(
    *,
    dataset_id: Optional[str],
    dataset_name: Optional[str],
    dataset_config_name: Optional[str],
    train_data_dir: Optional[str],
    train_split: str,
    cache_dir: Optional[str],
    tokenizer_one,
    tokenizer_two,
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
    prompt_text: Optional[str] = None,
) -> Tuple[DataLoader, str]:
    """Create an SDXL training dataloader and return the normalization mode."""
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

    dataset = TextImageDatasetXL(
        dataset=raw_dataset,
        tokenizer_one=tokenizer_one,
        tokenizer_two=tokenizer_two,
        image_transforms=get_transforms(
            resolution=resolution,
            center_crop=center_crop,
            random_flip=random_flip,
            interpolation_mode=interpolation_mode,
        ),
        resolution=resolution,
        image_column=img_col,
        caption_column=cap_col,
        image_preprocessor=image_preprocessor,
        prompt_text=prompt_text,
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


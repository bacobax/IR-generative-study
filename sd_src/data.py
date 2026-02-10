#!/usr/bin/env python
# coding=utf-8
"""
Data module for Stable Diffusion LoRA fine-tuning.

Handles dataset loading, preprocessing, transforms, and dataloaders.
"""

import os
import random
from typing import Callable, Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from datasets import load_dataset, Dataset
from torchvision import transforms
from torch.utils.data import DataLoader

from torchvision.transforms import functional as TF
from PIL import Image
from sd_src.helpers import P0001_PERCENTILE_RAW_IMAGES as A
from sd_src.helpers import P9999_PERCENTILE_RAW_IMAGES as B

import json

def _load_metadata_jsonl(meta_path: str) -> Dict[str, str]:
    """
    Reads metadata.jsonl and returns mapping:
      normalized_filename -> caption_text
    Accepts:
      file_name: "images/xxx.npy" or "xxx.npy"
    """
    mapping = {}
    with open(meta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fn = obj.get("file_name")
            txt = obj.get("text", "")
            if fn is None:
                continue
            # normalize to basename
            mapping[os.path.basename(fn)] = txt
    return mapping

def npy_ir_to_3ch_percentile_rgb(
    npy_or_path,
) -> Image.Image:
    """
    Load 1-channel uint16 thermal .npy and convert to 3-channel RGB PIL image
    using fixed percentile-window linear normalization.

    Output is an 8-bit RGB image where:
      x01 = clip((r - A)/(B-A), 0, 1)
      pixel_uint8 = round(x01 * 255)
    Then the existing transforms do:
      ToTensor(): pixel_uint8/255 = x01
      Normalize(mean=0.5,std=0.5): 2*x01 - 1 = x in [-1,1]
    """
    # Load npy if path is provided
    if isinstance(npy_or_path, (str, os.PathLike)):
        arr = np.load(npy_or_path)
    else:
        arr = npy_or_path  # allow already-loaded arrays if ever used

    # Normalize shape to (H, W)
    if arr.ndim == 3:
        if arr.shape[0] == 1:
            arr = arr[0]
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Expected 1-channel .npy, got shape {arr.shape}")
    elif arr.ndim != 2:
        raise ValueError(f"Expected 2D or 3D 1-channel .npy, got {arr.ndim}D")

    if arr.dtype != np.uint16:
        raise TypeError(f"Expected uint16 thermal .npy, got {arr.dtype}")

    # Fixed-window normalization: raw -> [0,1]
    arr_f = arr.astype(np.float32)
    x01 = (arr_f - float(A)) / (float(B) - float(A))
    x01 = np.clip(x01, 0.0, 1.0)

    # Map to uint8 (keeps rest of pipeline unchanged)
    x8 = np.round(x01 * 255.0).astype(np.uint8)

    # Convert to 3-channel RGB (SD expects 3ch)
    return Image.fromarray(x8, mode="L").convert("RGB")

# Mapping of known datasets to their column names
DATASET_NAME_MAPPING = {
    "lambdalabs/naruto-blip-captions": ("image", "text"),
}


def get_interpolation_mode(mode_name: str) -> transforms.InterpolationMode:
    """Get interpolation mode from string name."""
    mode = getattr(transforms.InterpolationMode, mode_name.upper(), None)
    if mode is None:
        raise ValueError(f"Unsupported interpolation mode: {mode_name}")
    return mode



class SquarePad:
    def __call__(self, img):
        w, h = img.size
        if w == h:
            return img
        # pad equally on both sides to make it square
        diff = abs(w - h)
        pad1 = diff // 2
        pad2 = diff - pad1
        if w < h:
            padding = (pad1, 0, pad2, 0)   # left, top, right, bottom
        else:
            padding = (0, pad1, 0, pad2)
        return TF.pad(img, padding, fill=0)  # fill=0 -> black padding


def get_transforms(
    resolution: int,
    center_crop: bool = False,
    random_flip: bool = False,
    interpolation_mode: str = "lanczos",
) -> transforms.Compose:
    """
    Create image transforms for training.
    
    Args:
        resolution: Target image resolution.
        center_crop: Whether to use center crop instead of random crop.
        random_flip: Whether to apply random horizontal flips.
        interpolation_mode: Interpolation method for resizing.
    
    Returns:
        Composed transform pipeline.
    """
    interpolation = get_interpolation_mode(interpolation_mode)

    transform_list = [
        SquarePad(),
        transforms.Resize((resolution, resolution), interpolation=interpolation),
    ]

    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    return transforms.Compose(transform_list)


class TextImageDataset:
    """
    Dataset wrapper for text-image pairs.
    
    Handles tokenization and image preprocessing.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        tokenizer,
        image_transforms: transforms.Compose,
        image_column: str = "image",
        caption_column: str = "text",
        image_preprocessor: Optional[Callable] = None,
    ):
        """
        Initialize the dataset.
        
        Args:
            dataset: HuggingFace dataset with images and captions.
            tokenizer: Text tokenizer for caption processing.
            image_transforms: Transform pipeline for images.
            image_column: Name of the image column.
            caption_column: Name of the caption column.
            image_preprocessor: Optional custom image preprocessing function.
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_transforms = image_transforms
        self.image_column = image_column
        self.caption_column = caption_column

        self.image_preprocessor = image_preprocessor or (lambda x: x.convert("RGB"))
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.dataset[idx]
        
        # Process image
        image = example[self.image_column]

        # If local dataset: image is a path to .npy
        # If HF dataset: image may be a PIL image object
        image = self.image_preprocessor(image)
        pixel_values = self.image_transforms(image)
        
        # Process caption
        caption = example[self.caption_column]
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


def collate_fn(examples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for batching examples."""
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    return {"pixel_values": pixel_values, "input_ids": input_ids}


def load_training_dataset(
    dataset_name: Optional[str] = None,
    dataset_config_name: Optional[str] = None,
    train_data_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
    image_column: str = "image",
    caption_column: str = "text",
) -> Tuple[Dataset, str, str]:

    if dataset_name is not None:
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
            raise ValueError(f"Caption column '{caption_column}' not found. Available: {column_names}")

        return dataset["train"], image_column, caption_column

    # -------- LOCAL DATASET (NPY) --------
    if train_data_dir is None:
        raise ValueError("train_data_dir must be provided when dataset_name is None")

    images_dir = os.path.join(train_data_dir, "images")
    if not os.path.isdir(images_dir):
        raise ValueError(f"Expected images folder at: {images_dir}")

    npy_paths = sorted([
        os.path.join(images_dir, fn)
        for fn in os.listdir(images_dir)
        if fn.lower().endswith(".npy")
    ])
    if len(npy_paths) == 0:
        raise ValueError(f"No .npy files found in {images_dir}")

    # captions from metadata.jsonl (same convention as HF imagefolder)
    meta_path = os.path.join(train_data_dir, "metadata.jsonl")
    captions_map = _load_metadata_jsonl(meta_path) if os.path.isfile(meta_path) else {}

    captions = []
    for p in npy_paths:
        base = os.path.basename(p)
        captions.append(captions_map.get(base, ""))

    # Build HF Dataset with image paths (string) and text
    ds = Dataset.from_dict({
        image_column: npy_paths,     # <--- store paths, not PIL images
        caption_column: captions,
    })

    return ds, image_column, caption_column 

def create_dataloader(
    dataset_name: Optional[str],
    dataset_config_name: Optional[str],
    train_data_dir: Optional[str],
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
    accelerator=None,
    use_ir_preprocessing: bool = True,
) -> DataLoader:
    """
    Create the training dataloader.
    
    Args:
        dataset_name: Name of dataset from HuggingFace hub.
        dataset_config_name: Dataset configuration name.
        train_data_dir: Local directory with training data.
        cache_dir: Cache directory for downloads.
        tokenizer: Text tokenizer.
        resolution: Image resolution.
        center_crop: Use center crop.
        random_flip: Apply random horizontal flips.
        interpolation_mode: Interpolation method.
        image_column: Image column name.
        caption_column: Caption column name.
        batch_size: Batch size per device.
        num_workers: Number of dataloader workers.
        max_train_samples: Max samples for debugging.
        seed: Random seed.
        accelerator: Accelerator for distributed training.
        use_ir_preprocessing: Whether to use IR image preprocessing.
    
    Returns:
        Training dataloader.
    """
    # Load raw dataset
    raw_dataset, img_col, cap_col = load_training_dataset(
        dataset_name=dataset_name,
        dataset_config_name=dataset_config_name,
        train_data_dir=train_data_dir,
        cache_dir=cache_dir,
        image_column=image_column,
        caption_column=caption_column,
    )
    
    # Apply sample limit
    if max_train_samples is not None:
        raw_dataset = raw_dataset.shuffle(seed=seed).select(range(max_train_samples))
    
    # Create transforms
    image_transforms = get_transforms(
        resolution=resolution,
        center_crop=center_crop,
        random_flip=random_flip,
        interpolation_mode=interpolation_mode,
    )
    

    print(f"Using IR preprocessing: {use_ir_preprocessing}")

    # Select image preprocessor
    image_preprocessor = (
        npy_ir_to_3ch_percentile_rgb
        if use_ir_preprocessing
        else None
    )
    
    # Create dataset wrapper
    dataset = TextImageDataset(
        dataset=raw_dataset,
        tokenizer=tokenizer,
        image_transforms=image_transforms,
        image_column=img_col,
        caption_column=cap_col,
        image_preprocessor=image_preprocessor,
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    
    return dataloader


def tokenize_captions(
    examples: Dict[str, Any],
    tokenizer,
    caption_column: str,
    is_train: bool = True,
) -> torch.Tensor:
    """
    Tokenize captions from examples.
    
    Args:
        examples: Dictionary containing caption data.
        tokenizer: Text tokenizer.
        caption_column: Name of the caption column.
        is_train: Whether in training mode (affects random choice).
    
    Returns:
        Tokenized input IDs.
    """
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(f"Caption should be string or list, got {type(caption)}")
    
    inputs = tokenizer(
        captions,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    return inputs.input_ids

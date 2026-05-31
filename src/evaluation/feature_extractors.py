"""Feature extractors and image loading for generative evaluation."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Mapping, Sequence

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm

from src.algorithms.stable_diffusion.data import ir_npy_to_normalized_rgb
from src.core.normalization import UINT8_LINEAR, raw_array_to_png_uint8


def load_image_rgb(
    path: str | Path,
    *,
    normalization_mode: str = UINT8_LINEAR,
) -> Image.Image:
    """Load an image path as RGB, converting single-channel arrays when needed."""
    image_path = Path(path)
    suffix = image_path.suffix.lower()
    if suffix == ".npy":
        return ir_npy_to_normalized_rgb(image_path, normalization_mode=normalization_mode)
    if suffix in {".tif", ".tiff"}:
        with Image.open(image_path) as image:
            arr = np.asarray(image)
        preview = raw_array_to_png_uint8(arr, normalization_mode=normalization_mode)
        return Image.fromarray(preview).convert("RGB")
    if suffix in {".png", ".jpg", ".jpeg"}:
        with Image.open(image_path) as image:
            return image.convert("RGB")
    raise ValueError(f"Unsupported image suffix for feature extraction: {image_path}")


@dataclass
class FeatureExtractor:
    """Small protocol-like base class for image feature extractors."""

    name: str
    device: torch.device
    config: Mapping[str, object]

    @torch.no_grad()
    def extract_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        raise NotImplementedError


class DINOv2FeatureExtractor(FeatureExtractor):
    """DINOv2 features from Hugging Face transformers."""

    def __init__(self, *, config: Mapping[str, object], device: torch.device) -> None:
        super().__init__(name="dinov2", device=device, config=dict(config))
        from transformers import AutoImageProcessor, AutoModel

        model_name = str(config.get("model_name", "facebook/dinov2-base"))
        self.pooling = str(config.get("pooling", "mean"))
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def extract_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=list(images), return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self.device)
        outputs = self.model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state
        if self.pooling in {"mean", "mean_patch"}:
            return hidden[:, 1:].mean(dim=1).detach().cpu().to(torch.float32)
        if self.pooling == "cls":
            return hidden[:, 0].detach().cpu().to(torch.float32)
        raise ValueError(f"Unsupported DINOv2 pooling mode: {self.pooling!r}.")


class InceptionFeatureExtractor(FeatureExtractor):
    """Torchvision Inception-v3 pooled features."""

    def __init__(self, *, config: Mapping[str, object], device: torch.device) -> None:
        super().__init__(name="inception", device=device, config=dict(config))
        from torchvision import models, transforms

        weights = models.Inception_V3_Weights.IMAGENET1K_V1
        self.resize_to = int(config.get("resize_to", 299))
        self.normalize = bool(config.get("normalize", True))
        transform_items: List[object] = [
            transforms.Resize((self.resize_to, self.resize_to)),
            transforms.ToTensor(),
        ]
        if self.normalize:
            transform_items.append(
                transforms.Normalize(
                    mean=weights.transforms().mean,
                    std=weights.transforms().std,
                )
            )
        self.transform = transforms.Compose(transform_items)
        self.model = models.inception_v3(weights=weights, transform_input=False)
        self.model.fc = torch.nn.Identity()
        self.model.to(device).eval()

    @torch.no_grad()
    def extract_batch(self, images: Sequence[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.transform(image) for image in images], dim=0).to(self.device)
        features = self.model(batch)
        if isinstance(features, tuple):
            features = features[0]
        return features.detach().cpu().to(torch.float32)


def build_feature_extractor(
    name: str,
    cfg: Mapping[str, object],
    device: str | torch.device,
) -> FeatureExtractor:
    """Build a supported feature extractor by name."""
    active_device = torch.device(device)
    normalized_name = str(name).lower()
    if normalized_name == "dinov2":
        extractor_cfg = cfg.get("dinov2", {}) if isinstance(cfg, Mapping) else {}
        return DINOv2FeatureExtractor(config=extractor_cfg, device=active_device)
    if normalized_name == "inception":
        extractor_cfg = cfg.get("inception", {}) if isinstance(cfg, Mapping) else {}
        return InceptionFeatureExtractor(config=extractor_cfg, device=active_device)
    raise ValueError("Unsupported feature_extractor={!r}. Expected 'dinov2' or 'inception'.".format(name))


def _load_cached_features(cache_path: Path) -> np.ndarray:
    with np.load(cache_path, allow_pickle=False) as data:
        if "features" not in data:
            raise ValueError(f"Feature cache {cache_path} does not contain a 'features' array.")
        return np.asarray(data["features"], dtype=np.float32)


def extract_features(
    paths: Sequence[str | Path],
    extractor: FeatureExtractor,
    *,
    batch_size: int,
    cache_path: str | Path,
    force: bool = False,
    normalization_mode: str = UINT8_LINEAR,
    metadata: Mapping[str, object] | None = None,
) -> np.ndarray:
    """Extract or load cached features for image paths."""
    cache = Path(cache_path)
    path_strings = [str(Path(path)) for path in paths]
    if cache.is_file() and not force:
        try:
            cached = _load_cached_features(cache)
            if cached.shape[0] == len(path_strings):
                return cached
        except (OSError, ValueError, EOFError):
            pass
        cache.unlink(missing_ok=True)

    if not paths:
        raise ValueError("No image paths were provided for feature extraction.")
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}.")

    features: List[torch.Tensor] = []
    for start in tqdm(range(0, len(path_strings), batch_size), desc=f"Extracting {extractor.name} features"):
        batch_paths = path_strings[start : start + batch_size]
        images = [
            load_image_rgb(path, normalization_mode=normalization_mode)
            for path in batch_paths
        ]
        features.append(extractor.extract_batch(images))

    feature_array = torch.cat(features, dim=0).numpy().astype(np.float32, copy=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "feature_extractor": extractor.name,
        "feature_config": dict(extractor.config),
        "normalization_mode": normalization_mode,
    }
    if metadata:
        payload.update(dict(metadata))
    tmp_path = cache.with_name(f".{cache.name}.{os.getpid()}.tmp")
    with tmp_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            features=feature_array,
            paths=np.asarray(path_strings),
            metadata=json.dumps(payload, sort_keys=True),
            feature_shape=np.asarray(feature_array.shape, dtype=np.int64),
        )
    os.replace(tmp_path, cache)
    return feature_array

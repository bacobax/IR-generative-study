"""Reusable dataset classes for loading single-channel local images."""

import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src.core.data.annotations import (
    build_category_id_to_name,
    get_boxes_and_labels_for_image,
    index_annotations,
    load_coco_annotations,
)
from src.core.data.subset_manifest import (
    filter_files_by_subset_manifest,
    filter_manifest_records_by_subset_manifest,
)
from src.core.data.transforms import horizontal_flip
from src.core.paths import repo_root
from src.core.normalization import RAW_UINT16_PERCENTILE, resize_and_normalize

SINGLE_CHANNEL_IMAGE_EXTENSIONS = {".npy", ".tif", ".tiff"}


def load_single_channel_array(path: str | os.PathLike) -> np.ndarray:
    """Load a local one-channel image array from ``.npy`` or TIFF."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        return np.load(path)
    if path.suffix.lower() in {".tif", ".tiff"}:
        with Image.open(path) as image:
            return np.asarray(image)
    raise ValueError(f"Unsupported single-channel image extension: {path.suffix}")


def load_single_channel_tensor(path: str | os.PathLike) -> torch.Tensor:
    """Load a local image as a float tensor with shape ``(1, H, W)``."""
    arr = load_single_channel_array(path)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 3 and arr.shape[-1] == 1:
        arr = np.moveaxis(arr, -1, 0)
    elif arr.ndim != 3:
        raise ValueError(f"Expected 2D or 1-channel image, got shape {arr.shape}")
    if arr.ndim == 3 and arr.shape[0] != 1:
        raise ValueError(f"Expected one channel, got shape {arr.shape}")
    return torch.from_numpy(np.asarray(arr).copy()).float()


def _resolve_manifest_image_path(raw_path: str, *, root_dir: str | os.PathLike) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidates = [repo_root() / path, Path(root_dir) / path, Path(root_dir) / path.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _load_manifest_records(
    manifest_path: str | os.PathLike,
    *,
    root_dir: str | os.PathLike,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(manifest_path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            raw_path = record.get(
                "image_path",
                record.get("path", record.get("file_name", record.get("filename"))),
            )
            if not isinstance(raw_path, str) or not raw_path:
                raise ValueError(
                    f"Manifest {manifest_path} line {line_number} is missing image_path"
                )
            image_path = _resolve_manifest_image_path(raw_path, root_dir=root_dir)
            record = dict(record)
            record["image_path"] = str(image_path)
            record.setdefault("file_name", image_path.name)
            records.append(record)
    if not records:
        raise RuntimeError(f"No image records found in manifest {manifest_path}")
    return records


class NPYImageDataset(Dataset):
    """Load single-channel ``.npy`` images and return tensors.

    Parameters
    ----------
    root_dir : str
        Directory containing ``.npy`` files.
    transform : callable, optional
        Applied to the raw float tensor after loading.
    """

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        subset_manifest: Optional[str] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        files = sorted(
            f for f in os.listdir(root_dir) if f.endswith(".npy")
        )
        self.files = filter_files_by_subset_manifest(
            files,
            subset_manifest,
            split_dir=root_dir,
            context="NPYImageDataset",
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


class SingleChannelImageDataset(Dataset):
    """Load one-channel ``.npy`` or TIFF files and return tensors."""

    def __init__(
        self,
        root_dir: str,
        transform: Optional[Callable] = None,
        subset_manifest: Optional[str] = None,
        manifest_path: Optional[str] = None,
        return_metadata: bool = False,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.return_metadata = return_metadata
        self.records: list[dict[str, Any]]

        if manifest_path is not None:
            self.records = _load_manifest_records(manifest_path, root_dir=root_dir)
            self.records = filter_manifest_records_by_subset_manifest(
                self.records,
                subset_manifest,
                split_dir=root_dir,
                context="SingleChannelImageDataset",
            )
        else:
            files = sorted(
                f
                for f in os.listdir(root_dir)
                if Path(f).suffix.lower() in SINGLE_CHANNEL_IMAGE_EXTENSIONS
            )
            files = filter_files_by_subset_manifest(
                files,
                subset_manifest,
                split_dir=root_dir,
                context="SingleChannelImageDataset",
            )
            self.records = [
                {"image_path": os.path.join(root_dir, file_name), "file_name": file_name}
                for file_name in files
            ]

        if not self.records:
            raise RuntimeError(f"No single-channel image files found in {root_dir}")

    @property
    def files(self) -> list[str]:
        return [str(record["file_name"]) for record in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        record = self.records[idx]
        x = load_single_channel_tensor(record["image_path"])
        if self.transform is not None:
            x = self.transform(x)
        if not self.return_metadata:
            return x
        metadata = {
            key: value
            for key, value in record.items()
            if key not in {"image_path"}
        }
        metadata["image_path"] = str(record["image_path"])
        return {"pixel_values": x, "metadata": metadata}


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


class AnnotationLayoutDataset(Dataset):
    """Load images together with aligned COCO bbox annotations.

    This dataset is intended for bbox-driven inspection or layout-aware
    training paths. It supports deterministic resize + normalize, plus an
    optional horizontal flip schedule that mirrors both image and boxes.
    """

    def __init__(
        self,
        root_dir: str,
        annotations_path: str,
        *,
        image_size: int = 256,
        normalization_mode: str = RAW_UINT16_PERCENTILE,
        include_label_names: bool = True,
        horizontal_flip_schedule=None,
        augment_schedule=None,
        subset_manifest: Optional[str] = None,
    ):
        self.root_dir = root_dir
        self.image_size = int(image_size)
        self.normalization_mode = normalization_mode
        self.include_label_names = include_label_names
        self.horizontal_flip_schedule = horizontal_flip_schedule
        # Bbox-aware multi-op augmentation (hflip/rot90/crop/photometric).
        # Supersedes ``horizontal_flip_schedule`` when set; applied on the
        # resized+normalized image with boxes in image_size pixel coords.
        self.augment_schedule = augment_schedule

        files = sorted(
            f for f in os.listdir(root_dir) if f.endswith(".npy")
        )
        self.files = filter_files_by_subset_manifest(
            files,
            subset_manifest,
            split_dir=root_dir,
            context="AnnotationLayoutDataset",
        )
        if not self.files:
            raise RuntimeError(f"No .npy files found in {root_dir}")

        coco = load_coco_annotations(annotations_path)
        self.category_id_to_name = build_category_id_to_name(coco)
        self.category_ids = sorted(self.category_id_to_name)
        self.num_categories = (
            max(self.category_ids) + 1 if self.category_ids else 0
        )
        self.images_by_id, self.anns_by_image_id, self.fname_to_imgid = (
            index_annotations(coco)
        )

    def __len__(self) -> int:
        return len(self.files)

    def set_epoch(self, epoch: int) -> None:
        if self.horizontal_flip_schedule is not None and hasattr(self.horizontal_flip_schedule, "set_epoch"):
            self.horizontal_flip_schedule.set_epoch(epoch)
        if self.augment_schedule is not None and hasattr(self.augment_schedule, "set_epoch"):
            self.augment_schedule.set_epoch(epoch)

    @staticmethod
    def _scale_boxes_xyxy(
        boxes_xyxy: torch.Tensor,
        *,
        src_width: float,
        src_height: float,
        dst_size: int,
    ) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy
        scaled = boxes_xyxy.clone()
        scaled[:, [0, 2]] *= float(dst_size) / max(float(src_width), 1.0)
        scaled[:, [1, 3]] *= float(dst_size) / max(float(src_height), 1.0)
        return scaled

    @staticmethod
    def _horizontal_flip_boxes_xyxy(
        boxes_xyxy: torch.Tensor,
        *,
        image_width: float,
    ) -> torch.Tensor:
        if boxes_xyxy.numel() == 0:
            return boxes_xyxy
        flipped = boxes_xyxy.clone()
        flipped[:, 0] = float(image_width) - boxes_xyxy[:, 2]
        flipped[:, 2] = float(image_width) - boxes_xyxy[:, 0]
        return flipped

    def __getitem__(self, idx: int) -> dict:
        fname = self.files[idx]
        path = os.path.join(self.root_dir, fname)

        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[None, ...]
        raw_tensor = torch.from_numpy(arr.copy()).float()

        horizontal_flipped = False
        # Legacy single-op hflip is skipped when the richer augment_schedule is
        # active (the latter handles hflip jointly with the other ops below).
        if (
            self.augment_schedule is None
            and self.horizontal_flip_schedule is not None
            and self.horizontal_flip_schedule.should_flip()
        ):
            raw_tensor = horizontal_flip(raw_tensor)
            horizontal_flipped = True

        pixel_values = resize_and_normalize(
            raw_tensor,
            image_size=self.image_size,
            normalization_mode=self.normalization_mode,
        )

        image_id = self.fname_to_imgid.get(fname)
        image_info = self.images_by_id.get(image_id, {})
        width = float(image_info.get("width", arr.shape[-1]))
        height = float(image_info.get("height", arr.shape[-2]))

        boxes_xyxy_raw, labels, label_names = get_boxes_and_labels_for_image(
            self.anns_by_image_id,
            image_id,
            category_id_to_name=self.category_id_to_name,
            include_label_names=self.include_label_names,
        )

        boxes_xyxy_original = torch.tensor(
            boxes_xyxy_raw,
            dtype=torch.float32,
        )
        if boxes_xyxy_original.numel() == 0:
            boxes_xyxy_original = boxes_xyxy_original.reshape(0, 4)

        boxes_xyxy = self._scale_boxes_xyxy(
            boxes_xyxy_original,
            src_width=width,
            src_height=height,
            dst_size=self.image_size,
        )
        if horizontal_flipped:
            boxes_xyxy = self._horizontal_flip_boxes_xyxy(
                boxes_xyxy,
                image_width=self.image_size,
            )

        labels_tensor = torch.tensor(labels, dtype=torch.long)
        if labels_tensor.numel() == 0:
            labels_tensor = labels_tensor.reshape(0)

        # Bbox-aware augmentation (hflip/rot90/crop/photometric) jointly on the
        # resized image and scaled boxes. Crop may drop out-of-view boxes; labels
        # are filtered alongside boxes by the augment.
        if self.augment_schedule is not None:
            pixel_values, boxes_xyxy, labels_tensor = self.augment_schedule(
                pixel_values, boxes_xyxy, labels_tensor
            )

        sample = {
            "pixel_values": pixel_values,
            "boxes_xyxy": boxes_xyxy,
            "labels": labels_tensor,
            "image_id": image_id,
            "file_name": fname,
            "n_objects": int(labels_tensor.numel()),
            "boxes_xyxy_original": boxes_xyxy_original,
            "horizontal_flipped": horizontal_flipped,
        }
        if self.include_label_names:
            sample["label_names"] = label_names or []
        return sample


class TextImageDataset(Dataset):
    """Load ``.npy`` images paired with text captions.

    Returns a dict with ``"pixel_values"`` (normalised image tensor) and
    ``"text"`` (string caption).

    Text captions can come from:

    * A JSON file mapping image stems to captions
      (``{"image_0001": "a thermal image of a cat", ...}``).
    * ``.txt`` companion files alongside each ``.npy`` file.

    If neither is available and *fallback_text* is set, that string is
    used for every sample.

    Parameters
    ----------
    root_dir : str
        Directory containing ``.npy`` image files.
    text_annotations : str, optional
        Path to a JSON file mapping ``{stem: text}``.
    fallback_text : str, optional
        Default text when no caption is found.  If ``None`` and a file
        has no caption, ``RuntimeError`` is raised.
    transform : callable, optional
        Applied to the raw float tensor after loading.
    """

    def __init__(
        self,
        root_dir: str,
        text_annotations: Optional[str] = None,
        *,
        fallback_text: Optional[str] = None,
        transform: Optional[Callable] = None,
    ):
        self.root_dir = root_dir
        self.fallback_text = fallback_text
        self.transform = transform
        self.files = sorted(
            f for f in os.listdir(root_dir) if f.endswith(".npy")
        )
        if not self.files:
            raise RuntimeError(f"No .npy files found in {root_dir}")

        # Build stem → text mapping
        self._text_map: Dict[str, str] = {}
        if text_annotations is not None:
            with open(text_annotations, "r") as fh:
                self._text_map = json.load(fh)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        fname = self.files[idx]
        stem = Path(fname).stem

        # Image
        x = np.load(os.path.join(self.root_dir, fname))
        if x.ndim == 2:
            x = x[None, ...]
        x = torch.from_numpy(x).float()
        if self.transform is not None:
            x = self.transform(x)

        # Text — JSON map → companion .txt → fallback
        text = self._text_map.get(stem)
        if text is None:
            txt_path = os.path.join(self.root_dir, f"{stem}.txt")
            if os.path.isfile(txt_path):
                with open(txt_path, "r") as f:
                    text = f.read().strip()
        if text is None:
            if self.fallback_text is not None:
                text = self.fallback_text
            else:
                raise RuntimeError(
                    f"No text caption for '{stem}' and no fallback_text set."
                )

        return {"pixel_values": x, "text": text}

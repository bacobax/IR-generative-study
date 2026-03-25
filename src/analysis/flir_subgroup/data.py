"""Dataset discovery and image IO helpers for FLIR subgroup analysis."""

from __future__ import annotations

import io
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image

from src.analysis.flir_subgroup.constants import IMAGE_EXTENSIONS, PREFERRED_SPLITS


def load_json(path: Path) -> dict:
    """Load a JSON file."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl_preview(path: Path, n_rows: int = 3) -> pd.DataFrame:
    """Return a small preview of a JSONL file."""

    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            rows.append(json.loads(line))
            if idx + 1 >= n_rows:
                break
    return pd.DataFrame(rows)


def looks_like_coco(payload: dict) -> bool:
    """Return ``True`` when a JSON payload looks like COCO annotations."""

    return isinstance(payload, dict) and {"images", "annotations", "categories"}.issubset(payload.keys())


def find_coco_annotation_files(split_dir: Path) -> List[Path]:
    """Search a split directory for COCO-style annotation files."""

    candidates: List[Path] = []
    for path in sorted(split_dir.rglob("*.json")):
        try:
            payload = load_json(path)
        except Exception:
            continue
        if looks_like_coco(payload):
            candidates.append(path)
    return candidates


def choose_best_annotation_file(split_dir: Path) -> Path:
    """Choose the best COCO annotation file under a split directory."""

    candidates = find_coco_annotation_files(split_dir)
    if not candidates:
        raise FileNotFoundError(f"No COCO-style annotation JSON found under: {split_dir}")
    candidates = sorted(
        candidates,
        key=lambda path: (
            "annotation" not in path.name.lower(),
            len(path.relative_to(split_dir).parts),
            len(str(path)),
        ),
    )
    return candidates[0]


def list_image_files(split_dir: Path) -> List[Path]:
    """List all supported image files under a split directory."""

    return sorted(path for path in split_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def normalize_rel_path(path: str | Path) -> str:
    """Normalize a relative path to POSIX form."""

    return Path(path).as_posix()


@dataclass
class ImageLookup:
    """Index of image files discovered for a split."""

    by_rel_path: Dict[str, Path]
    by_name: Dict[str, Path]
    by_stem: Dict[str, Path]


def build_image_lookup(image_files: Sequence[Path], split_dir: Path) -> ImageLookup:
    """Create lookup tables for discovered images."""

    by_rel_path: Dict[str, Path] = {}
    by_name: Dict[str, Path] = {}
    by_stem: Dict[str, Path] = {}
    for path in image_files:
        rel_path = normalize_rel_path(path.relative_to(split_dir))
        by_rel_path.setdefault(rel_path, path)
        by_name.setdefault(path.name, path)
        by_stem.setdefault(path.stem, path)
    return ImageLookup(by_rel_path=by_rel_path, by_name=by_name, by_stem=by_stem)


def resolve_image_path(file_name: str | None, split_dir: Path, lookup: ImageLookup) -> Optional[Path]:
    """Resolve an annotation ``file_name`` to an on-disk path."""

    if not file_name:
        return None

    candidates = [normalize_rel_path(file_name), normalize_rel_path(Path(file_name).name)]
    for candidate in candidates:
        if candidate in lookup.by_rel_path:
            return lookup.by_rel_path[candidate]
        if candidate in lookup.by_name:
            return lookup.by_name[candidate]

    stem = Path(file_name).stem
    return lookup.by_stem.get(stem)


def discover_split_dirs(data_root: Path, preferred_splits: Sequence[str] = PREFERRED_SPLITS) -> List[Path]:
    """Discover candidate dataset split directories."""

    if not data_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {data_root}")

    ordered: List[Path] = []
    seen: set[Path] = set()
    for split in preferred_splits:
        split_dir = data_root / split
        if split_dir.is_dir():
            ordered.append(split_dir)
            seen.add(split_dir)

    for child in sorted(data_root.iterdir()):
        if child.is_dir() and child not in seen:
            try:
                has_annotations = bool(find_coco_annotation_files(child))
            except Exception:
                has_annotations = False
            if has_annotations:
                ordered.append(child)

    return ordered


def inspect_dataset_root(data_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Inspect a dataset root and summarize discovered splits and metadata."""

    split_dirs = discover_split_dirs(data_root)
    if not split_dirs:
        raise FileNotFoundError(f"No candidate split directories found under: {data_root}")

    split_rows: List[dict] = []
    for split_dir in split_dirs:
        image_files = list_image_files(split_dir)
        ann_path = choose_best_annotation_file(split_dir)
        coco = load_json(ann_path)
        lookup = build_image_lookup(image_files, split_dir)

        referenced_paths = [img.get("file_name") for img in coco.get("images", [])]
        resolved_paths = [resolve_image_path(name, split_dir, lookup) for name in referenced_paths]
        missing_count = sum(path is None for path in resolved_paths)
        category_names = [cat.get("name", cat.get("id")) for cat in coco.get("categories", [])]

        split_rows.append(
            {
                "split": split_dir.name,
                "split_dir": str(split_dir),
                "annotation_path": str(ann_path),
                "captions_path": str(split_dir / "captions.json") if (split_dir / "captions.json").exists() else None,
                "n_images_json": len(coco.get("images", [])),
                "n_annotations_json": len(coco.get("annotations", [])),
                "n_categories": len(coco.get("categories", [])),
                "category_names": ", ".join(map(str, category_names)),
                "n_image_files_disk": len(image_files),
                "image_extensions": ", ".join(sorted({path.suffix.lower() for path in image_files})),
                "n_missing_referenced_images": missing_count,
            }
        )

    metadata_rows: List[dict] = []
    for meta_path in sorted(data_root.glob("metadata*.jsonl")):
        preview = load_jsonl_preview(meta_path, n_rows=1)
        metadata_rows.append(
            {
                "path": str(meta_path),
                "n_preview_rows": len(preview),
                "preview_columns": ", ".join(preview.columns.tolist()),
            }
        )

    split_df = pd.DataFrame(split_rows).sort_values("split").reset_index(drop=True)
    metadata_df = pd.DataFrame(metadata_rows)
    return split_df, metadata_df


def load_captions_if_present(split_dir: Path) -> Dict[str, str]:
    """Load optional ``captions.json`` from a split directory."""

    captions_path = split_dir / "captions.json"
    if not captions_path.exists():
        return {}

    payload = load_json(captions_path)
    if isinstance(payload, dict):
        return {str(key): str(value) for key, value in payload.items()}
    raise ValueError(f"Unsupported captions schema in: {captions_path}")


def coerce_bbox_xywh(raw_bbox: Sequence[float]) -> Tuple[float, float, float, float]:
    """Normalize a COCO ``bbox`` field."""

    if len(raw_bbox) != 4:
        raise ValueError(f"Expected bbox with 4 values, got {raw_bbox!r}")
    x, y, w, h = map(float, raw_bbox)
    return x, y, w, h


def build_split_tables(split_record: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, str]]:
    """Build image-level and instance-level tables for one split."""

    split = str(split_record["split"])
    split_dir = Path(split_record["split_dir"])
    ann_path = Path(split_record["annotation_path"])
    coco = load_json(ann_path)

    image_files = list_image_files(split_dir)
    lookup = build_image_lookup(image_files, split_dir)
    captions = load_captions_if_present(split_dir)
    category_name_by_id = {int(cat["id"]): str(cat.get("name", cat["id"])) for cat in coco.get("categories", [])}

    image_rows: List[dict] = []
    image_meta_by_id: Dict[str, dict] = {}
    for image in coco.get("images", []):
        image_id = str(image["id"])
        file_name = str(image.get("file_name", image_id))
        resolved_path = resolve_image_path(file_name, split_dir, lookup)
        width = float(image.get("width", np.nan))
        height = float(image.get("height", np.nan))
        image_area = width * height if np.isfinite(width) and np.isfinite(height) else np.nan
        image_key = f"{split}::{image_id}"
        image_meta = {
            "split": split,
            "image_id": image_id,
            "image_key": image_key,
            "file_name": file_name,
            "file_path": str(resolved_path) if resolved_path is not None else None,
            "image_exists": resolved_path is not None and resolved_path.exists(),
            "image_width": width,
            "image_height": height,
            "image_area": image_area,
            "caption": captions.get(image_id),
        }
        image_meta_by_id[image_id] = image_meta
        image_rows.append(image_meta)

    instance_rows: List[dict] = []
    for annotation in coco.get("annotations", []):
        image_id = str(annotation["image_id"])
        if image_id not in image_meta_by_id:
            continue

        x, y, w, h = coerce_bbox_xywh(annotation["bbox"])
        meta = image_meta_by_id[image_id]
        image_width = float(meta["image_width"])
        image_height = float(meta["image_height"])
        image_area = float(meta["image_area"])
        bbox_area = float(max(w, 0.0) * max(h, 0.0))
        bbox_area_norm = bbox_area / image_area if image_area > 0 else np.nan
        bbox_center_x = x + 0.5 * w
        bbox_center_y = y + 0.5 * h
        bbox_center_x_norm = bbox_center_x / image_width if image_width > 0 else np.nan
        bbox_center_y_norm = bbox_center_y / image_height if image_height > 0 else np.nan

        instance_rows.append(
            {
                "split": split,
                "image_id": image_id,
                "image_key": meta["image_key"],
                "file_name": meta["file_name"],
                "file_path": meta["file_path"],
                "class_id": int(annotation["category_id"]),
                "class_label": category_name_by_id.get(int(annotation["category_id"]), str(annotation["category_id"])),
                "ann_id": annotation["id"],
                "bbox_x": x,
                "bbox_y": y,
                "bbox_w": w,
                "bbox_h": h,
                "bbox_area": bbox_area,
                "bbox_area_norm": bbox_area_norm,
                "bbox_center_x": bbox_center_x,
                "bbox_center_y": bbox_center_y,
                "bbox_center_x_norm": bbox_center_x_norm,
                "bbox_center_y_norm": bbox_center_y_norm,
                "image_width": image_width,
                "image_height": image_height,
                "image_area": image_area,
                "iscrowd": int(annotation.get("iscrowd", 0)),
            }
        )

    image_df = pd.DataFrame(image_rows)
    instance_df = pd.DataFrame(instance_rows)
    if image_df.empty:
        raise ValueError(f"No image records were loaded from: {ann_path}")
    if instance_df.empty:
        warnings.warn(f"No annotation rows were loaded from: {ann_path}", stacklevel=2)

    return image_df, instance_df, category_name_by_id


def load_dataset_tables(layout_df: pd.DataFrame, selected_splits: Sequence[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build combined tables for the selected splits."""

    missing_splits = sorted(set(selected_splits) - set(layout_df["split"].tolist()))
    if missing_splits:
        raise ValueError(f"Requested splits are unavailable: {missing_splits}")

    image_tables: List[pd.DataFrame] = []
    instance_tables: List[pd.DataFrame] = []
    category_rows: List[dict] = []

    for split in selected_splits:
        split_record = layout_df.loc[layout_df["split"] == split]
        if split_record.empty:
            raise ValueError(f"Split not found in layout table: {split}")
        image_df, instance_df, category_name_by_id = build_split_tables(split_record.iloc[0])
        image_tables.append(image_df)
        instance_tables.append(instance_df)
        for category_id, category_name in category_name_by_id.items():
            category_rows.append({"split": split, "class_id": int(category_id), "class_label": str(category_name)})

    full_image_df = pd.concat(image_tables, ignore_index=True)
    full_instance_df = pd.concat(instance_tables, ignore_index=True)
    category_df = (
        pd.DataFrame(category_rows)
        .drop_duplicates()
        .sort_values(["split", "class_id"])
        .reset_index(drop=True)
    )

    ann_counts = full_instance_df.groupby("image_key").size().rename("n_annotations")
    full_image_df = full_image_df.merge(ann_counts, on="image_key", how="left")
    full_image_df["n_annotations"] = full_image_df["n_annotations"].fillna(0).astype(int)

    return full_image_df, full_instance_df, category_df


def load_image_array(path: Path) -> np.ndarray:
    """Load an image from disk, supporting ``.npy`` and raster formats."""

    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
    else:
        with Image.open(path) as image:
            arr = np.asarray(image)

    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def _normalize_numeric_image(arr: np.ndarray) -> np.ndarray:
    """Normalize an arbitrary numeric image to ``uint8`` for preview."""

    arr = np.asarray(arr).astype(np.float32)
    if arr.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        if arr.dtype == np.uint8:
            return arr.astype(np.uint8)
        low = float(np.quantile(arr, 0.01))
        high = float(np.quantile(arr, 0.99))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(arr.min())
            high = float(arr.max())
        if high <= low:
            return np.zeros(arr.shape, dtype=np.uint8)
        scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
        return (scaled * 255.0).astype(np.uint8)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    low = float(np.quantile(arr, 0.01))
    high = float(np.quantile(arr, 0.99))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        low = float(arr.min())
        high = float(arr.max())
    if high <= low:
        return np.zeros(arr.shape[:2], dtype=np.uint8)

    scaled = np.clip((arr - low) / (high - low), 0.0, 1.0)
    return (scaled * 255.0).astype(np.uint8)


def normalize_for_display(arr: np.ndarray) -> np.ndarray:
    """Convert an image array to a preview-friendly ``uint8`` array."""

    return _normalize_numeric_image(arr)


def render_preview_png_bytes(path: Path) -> bytes:
    """Render a dataset image to PNG bytes."""

    arr = normalize_for_display(load_image_array(path))
    image = Image.fromarray(arr, mode="L" if arr.ndim == 2 else None)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()

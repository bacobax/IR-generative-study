#!/usr/bin/env python3
"""Build the FLIR private-proxy dataset end-to-end.

This standalone entry-point replaces the old two-step workflow:

1. `docs/notebooks/flir_private_proxy_alignment.ipynb`
2. `scripts/standalone/build_flir_private_proxy_v18_dataset.py`

It supports two modes:

1. Analysis / reduced-train mode
   - derive private-train matching constraints
   - score FLIR train images
   - build a reduced FLIR train subset
   - save analysis plots, tables, summaries, mappings, and run metadata
   - export the final v18-style dataset
   - align val/test annotations so they do not contain categories absent from
     the final exported train split

2. Full-dataset passthrough mode (`--use-full-dataset`)
   - skip all analysis and subset construction
   - export requested FLIR splits directly to the target v18-style layout
   - keep val/test semantically unchanged apart from the output formatting

Examples
--------
Analysis mode:

```bash
python scripts/standalone/build_flir_private_proxy_alignment_dataset.py \
  --flir-root data/raw/flir \
  --output-root data/raw/flir_private_proxy_alignment_v18 \
  --analysis-output-root artifacts/analysis/flir_private_proxy_alignment
```

Full passthrough mode:

```bash
python scripts/standalone/build_flir_private_proxy_alignment_dataset.py \
  --use-full-dataset \
  --flir-root data/raw/flir \
  --output-root data/raw/flir_private_proxy_alignment_v18
```
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy import stats
except Exception:  # pragma: no cover - optional dependency fallback
    stats = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is expected but keep CLI safe
    def tqdm(iterable=None, **_: object):  # type: ignore[override]
        return iterable

pd = None
plt = None
sns = None


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.standalone.build_flir_private_proxy_v18_dataset import (  # noqa: E402
    SPLIT_TO_SOURCE_DIR,
    build_category_remap,
    caption_from_count,
    dedupe_images,
    resolve_source_image_path,
    write_image_npy,
)
from src.core.paths import analysis_root, repo_root, v18_root  # noqa: E402


PERSON_LABEL = "person"
FLIR_SPLITS = ("train", "val", "test")
HEATMAP_BINS_X = 16
HEATMAP_BINS_Y = 12
COUNT_BIN_CAP = 10

FLIR_CANONICAL_LABELS = {
    "person": "person",
    "bike": "bike",
    "car": "car",
    "motor": "motorcycle",
    "bus": "bus",
    "train": "train",
    "truck": "truck",
    "light": "traffic light",
    "hydrant": "fire hydrant",
    "sign": "street sign",
    "dog": "dog",
    "skateboard": "skateboard",
    "stroller": "stroller",
    "scooter": "scooter",
    "other vehicle": "other vehicle",
    "deer": "deer",
}

V18_STEM_RE = re.compile(
    r"^sequence-"
    r"(?P<date>\d{8})-"
    r"(?P<time>\d{6})-"
    r"(?P<subsec>\d+)-"
    r"(?P<camera>[A-Za-z0-9]+)-"
    r"(?P<scene_hash>[A-Za-z0-9]+)-"
    r"(?P<frame>\d+)$"
)
FLIR_STEM_RE = re.compile(
    r"^video-(?P<video_id>[A-Za-z0-9]+)-frame-(?P<frame_index>\d+)-(?P<frame_token>[A-Za-z0-9]+)$"
)


@dataclass
class ExportSummary:
    split: str
    image_count: int
    annotation_count: int
    category_count: int
    sample_dtype: Optional[str]
    sample_shape: Optional[Tuple[int, ...]]
    source_annotations: str


def default_flir_root() -> Path:
    return repo_root() / "data" / "raw" / "flir"


def default_output_root() -> Path:
    return repo_root() / "data" / "raw" / "flir_private_proxy_alignment_v18"


def default_analysis_output_root() -> Path:
    return analysis_root() / "flir_private_proxy_alignment"


def ensure_plotting_imports() -> None:
    global plt, sns
    if plt is not None and sns is not None:
        return
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as _plt
        import seaborn as _sns
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError(
            "Analysis mode requires matplotlib and seaborn. "
            "Install the notebook/analysis dependencies or run with "
            "`--use-full-dataset`."
        ) from exc
    plt = _plt
    sns = _sns


def ensure_tabular_imports() -> None:
    global pd
    if pd is not None:
        return
    try:
        import pandas as _pd
    except ImportError as exc:  # pragma: no cover - depends on runtime env
        raise ImportError(
            "This script requires pandas. Install the repository data-analysis "
            "dependencies before running it."
        ) from exc
    pd = _pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build the FLIR private-proxy dataset end-to-end, either by "
            "constructing a reduced train subset with saved analysis artifacts "
            "or by exporting the full FLIR splits directly."
        )
    )
    parser.add_argument(
        "--flir-root",
        type=Path,
        default=default_flir_root(),
        help="Root directory of the raw FLIR dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="Destination root for the exported v18-style dataset.",
    )
    parser.add_argument(
        "--analysis-output-root",
        type=Path,
        default=default_analysis_output_root(),
        help="Where analysis artifacts are saved when analysis mode is active.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=list(FLIR_SPLITS),
        choices=list(FLIR_SPLITS),
        help="Which splits to export.",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional cap for quick smoke runs after selection/alignment.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing exported .npy files instead of reusing them.",
    )
    parser.add_argument(
        "--no-train-captions",
        action="store_true",
        help="Do not write train/captions.json.",
    )
    parser.add_argument(
        "--use-full-dataset",
        action="store_true",
        help="Skip analysis/subsetting and export full FLIR splits directly.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed used for deterministic analysis and subset selection.",
    )
    return parser.parse_args()


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def load_coco_annotations(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def robust_entropy(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    total = arr.sum()
    if total <= 0:
        return 0.0
    p = arr / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def gini_coefficient(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.allclose(arr.sum(), 0.0):
        return 0.0
    arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) / (n * arr.sum())) - (n + 1) / n)


def js_divergence(p: Sequence[float], q: Sequence[float], eps: float = 1e-12) -> float:
    p_arr = np.asarray(p, dtype=float)
    q_arr = np.asarray(q, dtype=float)
    if p_arr.sum() <= 0 and q_arr.sum() <= 0:
        return 0.0
    p_arr = p_arr / max(p_arr.sum(), eps)
    q_arr = q_arr / max(q_arr.sum(), eps)
    m_arr = 0.5 * (p_arr + q_arr)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log2((a[mask] + eps) / (b[mask] + eps))))

    return 0.5 * _kl(p_arr, m_arr) + 0.5 * _kl(q_arr, m_arr)


def wasserstein_1d(a: Sequence[float], b: Sequence[float]) -> float:
    a_arr = np.asarray(list(a), dtype=float)
    b_arr = np.asarray(list(b), dtype=float)
    if a_arr.size == 0 or b_arr.size == 0:
        return float("nan")
    if stats is not None:
        return float(stats.wasserstein_distance(a_arr, b_arr))
    grid = np.linspace(0.0, 1.0, 257)
    aq = np.quantile(a_arr, grid)
    bq = np.quantile(b_arr, grid)
    return float(np.mean(np.abs(aq - bq)))


def normalize_histogram(values: Sequence[float], bins: Sequence[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return np.zeros(len(bins) - 1, dtype=float)
    hist, _ = np.histogram(arr, bins=bins)
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def safe_log(values: Sequence[float], floor: float = 1e-9) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return np.log(np.clip(arr, floor, None))


def make_2d_histogram(
    x: Sequence[float],
    y: Sequence[float],
    x_bins: np.ndarray,
    y_bins: np.ndarray,
) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((len(y_bins) - 1, len(x_bins) - 1), dtype=float)
    hist, _, _ = np.histogram2d(y, x, bins=[y_bins, x_bins])
    hist = hist.astype(float)
    if hist.sum() > 0:
        hist /= hist.sum()
    return hist


def support_lookup_1d(values: Sequence[float], reference_hist: np.ndarray, bins: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.array([], dtype=float)
    indices = np.clip(np.digitize(arr, bins) - 1, 0, len(reference_hist) - 1)
    return reference_hist[indices]


def support_lookup_2d(
    points: np.ndarray,
    reference_hist: np.ndarray,
    x_bins: np.ndarray,
    y_bins: np.ndarray,
) -> np.ndarray:
    if len(points) == 0:
        return np.array([], dtype=float)
    x_idx = np.clip(np.digitize(points[:, 0], x_bins) - 1, 0, reference_hist.shape[1] - 1)
    y_idx = np.clip(np.digitize(points[:, 1], y_bins) - 1, 0, reference_hist.shape[0] - 1)
    return reference_hist[y_idx, x_idx]


def proportional_quotas(counts: pd.Series, total_target: int) -> Dict[str, int]:
    total_available = int(counts.sum())
    if total_available <= 0:
        return {str(key): 0 for key in counts.index}
    raw = counts.astype(float) / total_available * total_target
    base = np.floor(raw).astype(int)
    remainder = raw - base
    deficit = int(total_target - base.sum())
    order = remainder.sort_values(ascending=False).index.tolist()
    quotas = base.to_dict()
    for key in order[:deficit]:
        quotas[key] += 1
    return {str(key): int(value) for key, value in quotas.items()}


def parse_v18_stem(stem: str) -> Dict[str, object]:
    match = V18_STEM_RE.match(stem)
    sequence_id = stem.rsplit("-", 1)[0]
    if match is None:
        return {
            "camera_id": None,
            "scene_id": sequence_id,
            "sequence_id": sequence_id,
            "frame_index": None,
            "timestamp": None,
        }
    payload = match.groupdict()
    timestamp = pd.to_datetime(
        f"{payload['date']}{payload['time']}{payload['subsec'][:6].ljust(6, '0')}",
        format="%Y%m%d%H%M%S%f",
        errors="coerce",
    )
    return {
        "camera_id": payload["camera"],
        "scene_id": payload["scene_hash"],
        "sequence_id": sequence_id,
        "frame_index": int(payload["frame"]),
        "timestamp": timestamp,
    }


def parse_flir_stem(stem: str) -> Dict[str, object]:
    match = FLIR_STEM_RE.match(stem)
    if match is None:
        return {
            "camera_id": None,
            "scene_id": None,
            "sequence_id": None,
            "frame_index": None,
            "video_id": None,
        }
    payload = match.groupdict()
    return {
        "camera_id": None,
        "scene_id": payload["video_id"],
        "sequence_id": payload["video_id"],
        "frame_index": int(payload["frame_index"]),
        "video_id": payload["video_id"],
    }


def canonicalize_annotations(
    annotations_df: pd.DataFrame,
    images_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    *,
    source_dataset: str,
) -> pd.DataFrame:
    if annotations_df.empty:
        return annotations_df.copy()

    merged = annotations_df.merge(
        images_df[
            [
                "image_id",
                "split",
                "source_dataset",
                "file_name",
                "file_path",
                "width",
                "height",
                "camera_id",
                "scene_id",
                "sequence_id",
                "scene_proxy_unit",
                "frame_index",
                "timestamp",
                "video_id",
            ]
        ],
        on="image_id",
        how="left",
    )
    merged = merged.merge(
        categories_df[["category_id", "raw_label", "canonical_label"]],
        on="category_id",
        how="left",
    )
    merged["source_dataset"] = source_dataset
    merged["x"] = merged["bbox"].apply(lambda box: float(box[0]))
    merged["y"] = merged["bbox"].apply(lambda box: float(box[1]))
    merged["w"] = merged["bbox"].apply(lambda box: float(box[2]))
    merged["h"] = merged["bbox"].apply(lambda box: float(box[3]))
    merged["x1"] = merged["x"]
    merged["y1"] = merged["y"]
    merged["x2"] = merged["x"] + merged["w"]
    merged["y2"] = merged["y"] + merged["h"]
    merged["center_x_norm"] = (merged["x"] + 0.5 * merged["w"]) / merged["width"].clip(lower=1)
    merged["center_y_norm"] = (merged["y"] + 0.5 * merged["h"]) / merged["height"].clip(lower=1)
    merged["bbox_w_norm"] = merged["w"] / merged["width"].clip(lower=1)
    merged["bbox_h_norm"] = merged["h"] / merged["height"].clip(lower=1)
    merged["area_ratio"] = (merged["w"] * merged["h"]) / (merged["width"] * merged["height"]).clip(lower=1)
    merged["aspect_ratio"] = merged["w"] / merged["h"].clip(lower=1e-9)
    merged["log_bbox_h_norm"] = safe_log(merged["bbox_h_norm"].clip(lower=1e-9))
    merged["is_person"] = merged["canonical_label"].eq(PERSON_LABEL)
    merged["annotation_id"] = merged["annotation_id"].astype(str)
    return merged


def ensure_split_scoped_flir_ids(
    images_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    images_df = images_df.copy()
    annotations_df = annotations_df.copy()

    image_ids = images_df["image_id"].astype(str)
    if not image_ids.duplicated().any() and image_ids.str.contains("::").all():
        return images_df, annotations_df

    if "source_image_id" not in images_df.columns:
        images_df["source_image_id"] = images_df["image_id"]
    images_df["source_image_id"] = images_df["source_image_id"].astype(str)
    images_df["image_id"] = images_df["split"].astype(str) + "::" + images_df["source_image_id"]

    if "source_annotation_id" not in annotations_df.columns:
        annotations_df["source_annotation_id"] = annotations_df["annotation_id"]
    if "source_image_id" not in annotations_df.columns:
        annotations_df["source_image_id"] = annotations_df["image_id"]
    annotations_df["source_annotation_id"] = annotations_df["source_annotation_id"].astype(str)
    annotations_df["source_image_id"] = annotations_df["source_image_id"].astype(str)
    annotations_df["annotation_id"] = (
        annotations_df["split"].astype(str) + "::" + annotations_df["source_annotation_id"]
    )
    annotations_df["image_id"] = annotations_df["split"].astype(str) + "::" + annotations_df["source_image_id"]
    return images_df, annotations_df


def build_flir_category_table(coco_payload: dict, annotations_df: pd.DataFrame) -> pd.DataFrame:
    categories_df = pd.DataFrame(coco_payload.get("categories", [])).rename(
        columns={"id": "category_id", "name": "raw_label"}
    )
    categories_df["canonical_label"] = categories_df["raw_label"].map(
        lambda name: FLIR_CANONICAL_LABELS.get(name, name)
    )
    support = annotations_df["category_id"].value_counts().rename("instance_count")
    categories_df = categories_df.merge(
        support,
        left_on="category_id",
        right_index=True,
        how="left",
    )
    categories_df["instance_count"] = categories_df["instance_count"].fillna(0).astype(int)
    categories_df["is_active"] = categories_df["instance_count"] > 0
    return categories_df


def assign_flir_scene_proxy_units(images_df: pd.DataFrame, chunk_size: int) -> pd.DataFrame:
    chunk_size = max(1, int(chunk_size))
    images_df = images_df.sort_values(["split", "video_id", "frame_index", "stem"]).copy()
    images_df["video_order"] = images_df.groupby(["split", "video_id"]).cumcount()
    images_df["scene_chunk_index"] = images_df["video_order"] // chunk_size
    images_df["scene_proxy_unit"] = (
        images_df["video_id"].astype(str)
        + "::chunk-"
        + images_df["scene_chunk_index"].astype(str)
    )
    return images_df


def validate_flir_split_tables(
    split: str,
    images_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    categories_df: pd.DataFrame,
    *,
    flir_root: Path,
) -> None:
    if images_df["image_id"].duplicated().any():
        duplicated = images_df.loc[images_df["image_id"].duplicated(), "image_id"].astype(str).unique()
        raise ValueError(f"Duplicate FLIR image ids detected in split={split!r}: {duplicated[:10].tolist()}")
    if annotations_df["annotation_id"].duplicated().any():
        duplicated = (
            annotations_df.loc[annotations_df["annotation_id"].duplicated(), "annotation_id"]
            .astype(str)
            .unique()
        )
        raise ValueError(
            f"Duplicate FLIR annotation ids detected in split={split!r}: {duplicated[:10].tolist()}"
        )
    missing_images = sorted(set(annotations_df["image_id"]) - set(images_df["image_id"]))
    if missing_images:
        raise ValueError(
            f"Annotations reference missing images in split={split!r}: {missing_images[:10]}"
        )
    missing_categories = sorted(set(annotations_df["category_id"]) - set(categories_df["category_id"]))
    if missing_categories:
        raise ValueError(
            f"Annotations reference missing categories in split={split!r}: {missing_categories[:10]}"
        )
    missing_files = []
    for row in images_df.itertuples(index=False):
        path = resolve_source_image_path(flir_root, split, str(row.file_name))
        if not path.exists():
            missing_files.append(str(path))
            if len(missing_files) >= 5:
                break
    if missing_files:
        raise FileNotFoundError(
            f"Missing raw FLIR image files for split={split!r}: {missing_files}"
        )


def load_v18_train() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    split = "train"
    split_dir = v18_root() / split
    coco = load_coco_annotations(split_dir / "annotations.json")

    images_df = pd.DataFrame(coco.get("images", [])).rename(columns={"id": "image_id"})
    images_df["split"] = split
    images_df["source_dataset"] = "private"
    images_df["file_path"] = images_df["file_name"].map(lambda name: str(split_dir / name))
    images_df["stem"] = images_df["file_name"].map(lambda name: Path(name).stem)
    meta_df = images_df["stem"].map(parse_v18_stem).apply(pd.Series)
    images_df = pd.concat([images_df, meta_df], axis=1)
    images_df["scene_proxy_unit"] = images_df["sequence_id"]
    images_df["video_id"] = None

    categories_df = pd.DataFrame(coco.get("categories", [])).rename(
        columns={"id": "category_id", "name": "raw_label"}
    )
    categories_df["canonical_label"] = categories_df["raw_label"]

    annotations_df = pd.DataFrame(coco.get("annotations", [])).rename(columns={"id": "annotation_id"})
    return images_df, categories_df, canonicalize_annotations(
        annotations_df=annotations_df,
        images_df=images_df,
        categories_df=categories_df,
        source_dataset="private",
    )


def load_flir_split(
    split: str,
    *,
    flir_root: Path,
    scene_chunk_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    split_dir = flir_root / SPLIT_TO_SOURCE_DIR[split]
    coco_path = split_dir / "coco.json"
    coco = load_coco_annotations(coco_path)

    images_df = pd.DataFrame(coco.get("images", [])).rename(columns={"id": "image_id"})
    images_df["source_image_id"] = images_df["image_id"]
    images_df["image_id"] = images_df["source_image_id"].map(lambda raw_id: f"{split}::{raw_id}")
    images_df["split"] = split
    images_df["source_dataset"] = "flir_original"
    images_df["file_path"] = images_df["file_name"].map(lambda name: str(split_dir / name))
    images_df["stem"] = images_df["file_name"].map(lambda name: Path(name).stem)
    parse_df = images_df["stem"].map(parse_flir_stem).apply(pd.Series)
    extra_df = (
        images_df["extra_info"]
        .apply(lambda payload: payload if isinstance(payload, dict) else {})
        .apply(pd.Series)
    )
    extra_df = extra_df.add_prefix("extra_")
    images_df = pd.concat([images_df, parse_df, extra_df], axis=1)
    if "extra_video_id" in images_df.columns:
        images_df["video_id"] = images_df["video_id"].fillna(images_df["extra_video_id"])
    if "extra_scene" in images_df.columns:
        images_df["scene_id"] = images_df["scene_id"].fillna(images_df["extra_scene"])
    images_df["timestamp"] = None
    images_df = assign_flir_scene_proxy_units(images_df, chunk_size=scene_chunk_size)

    raw_annotations_df = pd.DataFrame(coco.get("annotations", [])).rename(columns={"id": "annotation_id"})
    raw_annotations_df["source_annotation_id"] = raw_annotations_df["annotation_id"]
    raw_annotations_df["annotation_id"] = raw_annotations_df["source_annotation_id"].map(
        lambda raw_id: f"{split}::{raw_id}"
    )
    raw_annotations_df["source_image_id"] = raw_annotations_df["image_id"]
    raw_annotations_df["image_id"] = raw_annotations_df["source_image_id"].map(
        lambda raw_id: f"{split}::{raw_id}"
    )
    raw_annotations_df["split"] = split
    categories_df = build_flir_category_table(coco, raw_annotations_df)
    annotations_df = canonicalize_annotations(
        annotations_df=raw_annotations_df,
        images_df=images_df,
        categories_df=categories_df,
        source_dataset="flir_original",
    )
    validate_flir_split_tables(
        split,
        images_df=images_df,
        annotations_df=annotations_df,
        categories_df=categories_df,
        flir_root=flir_root,
    )
    return images_df, categories_df, annotations_df, coco


def build_image_metrics(
    images_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    person_annotations_df: pd.DataFrame,
) -> pd.DataFrame:
    image_metrics = images_df[
        [
            "image_id",
            "split",
            "source_dataset",
            "file_name",
            "file_path",
            "width",
            "height",
            "camera_id",
            "scene_id",
            "sequence_id",
            "scene_proxy_unit",
            "frame_index",
            "timestamp",
            "video_id",
        ]
    ].copy()

    object_count = annotations_df.groupby("image_id").size().rename("object_count")
    person_count = person_annotations_df.groupby("image_id").size().rename("person_count")
    person_occupancy = (
        person_annotations_df.groupby("image_id")["area_ratio"].sum().rename("person_occupancy_ratio")
    )

    image_metrics = image_metrics.merge(object_count, on="image_id", how="left")
    image_metrics = image_metrics.merge(person_count, on="image_id", how="left")
    image_metrics = image_metrics.merge(person_occupancy, on="image_id", how="left")
    image_metrics[["object_count", "person_count"]] = (
        image_metrics[["object_count", "person_count"]].fillna(0).astype(int)
    )
    image_metrics["person_occupancy_ratio"] = image_metrics["person_occupancy_ratio"].fillna(0.0)
    image_metrics["is_empty"] = image_metrics["object_count"].eq(0)
    return image_metrics


def build_class_summary(images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    instance_counts = (
        annotations_df["canonical_label"]
        .value_counts()
        .rename_axis("canonical_label")
        .reset_index(name="instance_count")
    )
    image_presence = (
        annotations_df.groupby(["image_id", "canonical_label"]).size().reset_index(name="n")
        .groupby("canonical_label").size()
        .rename("image_presence")
        .reset_index()
    )
    summary = instance_counts.merge(image_presence, on="canonical_label", how="left")
    summary["image_presence"] = summary["image_presence"].fillna(0).astype(int)
    summary["instance_rank"] = np.arange(1, len(summary) + 1)
    total_instances = summary["instance_count"].sum()
    summary["instance_share"] = summary["instance_count"] / max(total_instances, 1)
    summary["cumulative_share"] = summary["instance_share"].cumsum()

    aggregate = pd.DataFrame(
        [
            {
                "n_active_classes": int(len(summary)),
                "class_entropy": robust_entropy(summary["instance_count"]),
                "class_gini": gini_coefficient(summary["instance_count"]),
            }
        ]
    )
    return {"table": summary, "aggregate": aggregate}


def build_scene_summary(images_df: pd.DataFrame) -> pd.DataFrame:
    counts = images_df["scene_proxy_unit"].fillna("missing").value_counts()
    if counts.empty:
        return pd.DataFrame(
            [
                {
                    "n_scenes": 0,
                    "scene_entropy": 0.0,
                    "top_1_share": 0.0,
                    "top_5_share": 0.0,
                }
            ]
        )
    ordered = counts.sort_values(ascending=False)
    return pd.DataFrame(
        [
            {
                "n_scenes": int(counts.size),
                "scene_entropy": robust_entropy(counts.values),
                "top_1_share": float(ordered.head(1).sum() / counts.sum()),
                "top_5_share": float(ordered.head(5).sum() / counts.sum()),
            }
        ]
    )


def profile_dataset(dataset_name: str, images_df: pd.DataFrame, annotations_df: pd.DataFrame) -> Dict[str, object]:
    person_annotations = annotations_df[annotations_df["is_person"]].copy()
    image_metrics = build_image_metrics(images_df, annotations_df, person_annotations)
    class_summary = build_class_summary(images_df, annotations_df)
    scene_summary = build_scene_summary(images_df)
    inventory = pd.DataFrame(
        [
            {
                "dataset": dataset_name,
                "images": int(len(images_df)),
                "annotations": int(len(annotations_df)),
                "person_annotations": int(len(person_annotations)),
                "mean_annotations_per_image": float(len(annotations_df) / max(len(images_df), 1)),
                "mean_persons_per_image": float(image_metrics["person_count"].mean()) if not image_metrics.empty else 0.0,
                "scene_units": int(images_df["scene_proxy_unit"].nunique()) if "scene_proxy_unit" in images_df.columns else 0,
            }
        ]
    )
    return {
        "dataset_name": dataset_name,
        "images": images_df,
        "annotations": annotations_df,
        "person_annotations": person_annotations,
        "image_metrics": image_metrics,
        "class_summary": class_summary,
        "scene_summary": scene_summary,
        "inventory": inventory,
    }


def compare_profiles(reference: Dict[str, object], candidate: Dict[str, object], comparison_name: str) -> pd.DataFrame:
    ref_image_metrics = reference["image_metrics"]
    cand_image_metrics = candidate["image_metrics"]
    ref_person = reference["person_annotations"]
    cand_person = candidate["person_annotations"]

    count_bins = np.arange(
        -0.5,
        max(
            ref_image_metrics["person_count"].max() if not ref_image_metrics.empty else 0,
            cand_image_metrics["person_count"].max() if not cand_image_metrics.empty else 0,
            COUNT_BIN_CAP,
        )
        + 1.5,
        1.0,
    )
    center_x_bins = np.linspace(0.0, 1.0, HEATMAP_BINS_X + 1)
    center_y_bins = np.linspace(0.0, 1.0, HEATMAP_BINS_Y + 1)
    log_h_bins = np.linspace(
        float(
            min(
                ref_person["log_bbox_h_norm"].quantile(0.01) if not ref_person.empty else -8.0,
                cand_person["log_bbox_h_norm"].quantile(0.01) if not cand_person.empty else -8.0,
            )
        ),
        float(
            max(
                ref_person["log_bbox_h_norm"].quantile(0.99) if not ref_person.empty else -1.0,
                cand_person["log_bbox_h_norm"].quantile(0.99) if not cand_person.empty else -1.0,
            )
        ),
        HEATMAP_BINS_Y + 1,
    )

    center_hist_ref = make_2d_histogram(
        ref_person["center_x_norm"],
        ref_person["center_y_norm"],
        center_x_bins,
        center_y_bins,
    )
    center_hist_cand = make_2d_histogram(
        cand_person["center_x_norm"],
        cand_person["center_y_norm"],
        center_x_bins,
        center_y_bins,
    )
    yh_hist_ref = make_2d_histogram(
        ref_person["center_y_norm"],
        ref_person["log_bbox_h_norm"],
        center_y_bins,
        log_h_bins,
    )
    yh_hist_cand = make_2d_histogram(
        cand_person["center_y_norm"],
        cand_person["log_bbox_h_norm"],
        center_y_bins,
        log_h_bins,
    )

    rows = [
        {
            "comparison": comparison_name,
            "statistic": "person_count_per_image",
            "distance": js_divergence(
                normalize_histogram(ref_image_metrics["person_count"], count_bins),
                normalize_histogram(cand_image_metrics["person_count"], count_bins),
            ),
        },
        {
            "comparison": comparison_name,
            "statistic": "person_area_ratio",
            "distance": wasserstein_1d(ref_person["area_ratio"], cand_person["area_ratio"]),
        },
        {
            "comparison": comparison_name,
            "statistic": "person_aspect_ratio",
            "distance": wasserstein_1d(ref_person["aspect_ratio"], cand_person["aspect_ratio"]),
        },
        {
            "comparison": comparison_name,
            "statistic": "person_center_distribution",
            "distance": js_divergence(center_hist_ref.flatten(), center_hist_cand.flatten()),
        },
        {
            "comparison": comparison_name,
            "statistic": "person_y_logh_distribution",
            "distance": js_divergence(yh_hist_ref.flatten(), yh_hist_cand.flatten()),
        },
        {
            "comparison": comparison_name,
            "statistic": "scene_top_5_share",
            "distance": float(
                abs(
                    reference["scene_summary"].iloc[0]["top_5_share"]
                    - candidate["scene_summary"].iloc[0]["top_5_share"]
                )
            ),
        },
        {
            "comparison": comparison_name,
            "statistic": "scene_entropy",
            "distance": float(
                abs(
                    reference["scene_summary"].iloc[0]["scene_entropy"]
                    - candidate["scene_summary"].iloc[0]["scene_entropy"]
                )
            ),
        },
    ]

    ref_class_table = reference["class_summary"]["table"]
    cand_class_table = candidate["class_summary"]["table"]
    shared_labels = sorted(set(ref_class_table["canonical_label"]) | set(cand_class_table["canonical_label"]))
    if shared_labels:
        ref_counts = (
            ref_class_table.set_index("canonical_label")["instance_count"]
            .reindex(shared_labels, fill_value=0)
            .to_numpy()
        )
        cand_counts = (
            cand_class_table.set_index("canonical_label")["instance_count"]
            .reindex(shared_labels, fill_value=0)
            .to_numpy()
        )
        rows.extend(
            [
                {
                    "comparison": comparison_name,
                    "statistic": "class_frequency",
                    "distance": js_divergence(ref_counts, cand_counts),
                },
                {
                    "comparison": comparison_name,
                    "statistic": "class_entropy",
                    "distance": float(
                        abs(
                            reference["class_summary"]["aggregate"].iloc[0]["class_entropy"]
                            - candidate["class_summary"]["aggregate"].iloc[0]["class_entropy"]
                        )
                    ),
                },
                {
                    "comparison": comparison_name,
                    "statistic": "class_gini",
                    "distance": float(
                        abs(
                            reference["class_summary"]["aggregate"].iloc[0]["class_gini"]
                            - candidate["class_summary"]["aggregate"].iloc[0]["class_gini"]
                        )
                    ),
                },
            ]
        )
    return pd.DataFrame(rows).sort_values("statistic").reset_index(drop=True)


def derive_private_constraints(private_profile: Dict[str, object]) -> Dict[str, object]:
    person_annotations = private_profile["person_annotations"]
    image_metrics = private_profile["image_metrics"]
    scene_sizes = private_profile["images"]["scene_proxy_unit"].value_counts()
    chunk_size = int(scene_sizes.median()) if not scene_sizes.empty else 64

    count_values = image_metrics["person_count"].to_numpy()
    count_bins = np.arange(-0.5, max(int(count_values.max()) + 1, COUNT_BIN_CAP) + 1.5, 1.0)
    area_bins = np.linspace(
        float(person_annotations["area_ratio"].quantile(0.01)) if not person_annotations.empty else 0.0,
        float(person_annotations["area_ratio"].quantile(0.99)) if not person_annotations.empty else 0.1,
        24,
    )
    aspect_bins = np.linspace(
        float(person_annotations["aspect_ratio"].quantile(0.01)) if not person_annotations.empty else 0.0,
        float(person_annotations["aspect_ratio"].quantile(0.99)) if not person_annotations.empty else 4.0,
        24,
    )
    y_bins = np.linspace(0.0, 1.0, HEATMAP_BINS_Y + 1)
    x_bins = np.linspace(0.0, 1.0, HEATMAP_BINS_X + 1)
    h_bins = np.linspace(
        float(person_annotations["log_bbox_h_norm"].quantile(0.01)) if not person_annotations.empty else -8.0,
        float(person_annotations["log_bbox_h_norm"].quantile(0.99)) if not person_annotations.empty else -1.0,
        HEATMAP_BINS_Y + 1,
    )
    return {
        "target_total_images": int(len(private_profile["images"])),
        "scene_chunk_size": chunk_size,
        "person_count_bins": count_bins,
        "person_count_hist": normalize_histogram(count_values, count_bins),
        "area_bins": area_bins,
        "area_hist": normalize_histogram(person_annotations["area_ratio"], area_bins),
        "aspect_bins": aspect_bins,
        "aspect_hist": normalize_histogram(person_annotations["aspect_ratio"], aspect_bins),
        "center_x_bins": x_bins,
        "center_y_bins": y_bins,
        "center_hist": make_2d_histogram(
            person_annotations["center_x_norm"],
            person_annotations["center_y_norm"],
            x_bins,
            y_bins,
        ),
        "y_bins": y_bins,
        "log_h_bins": h_bins,
        "y_logh_hist": make_2d_histogram(
            person_annotations["center_y_norm"],
            person_annotations["log_bbox_h_norm"],
            y_bins,
            h_bins,
        ),
        "person_count_max": int(image_metrics["person_count"].quantile(0.99)),
        "scene_top_5_share": float(private_profile["scene_summary"].iloc[0]["top_5_share"]),
        "scene_entropy": float(private_profile["scene_summary"].iloc[0]["scene_entropy"]),
        "min_aux_class_instances": 25,
        "min_aux_class_images": 10,
    }


def score_flir_images_against_private(
    flir_images_df: pd.DataFrame,
    flir_annotations_df: pd.DataFrame,
    private_constraints: Dict[str, object],
) -> pd.DataFrame:
    person_df = flir_annotations_df[flir_annotations_df["is_person"]].copy()
    support_area = support_lookup_1d(
        person_df["area_ratio"],
        private_constraints["area_hist"],
        private_constraints["area_bins"],
    )
    support_aspect = support_lookup_1d(
        person_df["aspect_ratio"],
        private_constraints["aspect_hist"],
        private_constraints["aspect_bins"],
    )
    center_points = person_df[["center_x_norm", "center_y_norm"]].to_numpy(dtype=float)
    support_center = support_lookup_2d(
        center_points,
        private_constraints["center_hist"],
        private_constraints["center_x_bins"],
        private_constraints["center_y_bins"],
    )
    y_logh_points = person_df[["center_y_norm", "log_bbox_h_norm"]].to_numpy(dtype=float)
    support_y_logh = support_lookup_2d(
        y_logh_points,
        private_constraints["y_logh_hist"],
        private_constraints["y_bins"],
        private_constraints["log_h_bins"],
    )

    person_df["support_score_box"] = np.vstack(
        [support_area, support_aspect, support_center, support_y_logh]
    ).mean(axis=0)
    image_scores = (
        person_df.groupby("image_id")
        .agg(
            support_score=("support_score_box", "mean"),
            support_rate=("support_score_box", lambda s: float(np.mean(s > 0.0))),
            mean_person_area=("area_ratio", "mean"),
        )
        .reset_index()
    )

    image_counts = flir_annotations_df.groupby("image_id").size().rename("object_count")
    person_counts = person_df.groupby("image_id").size().rename("person_count")
    image_frame = flir_images_df[["image_id", "split", "scene_proxy_unit", "video_id"]].copy()
    image_frame = image_frame.merge(person_counts.rename("person_count").reset_index(), on="image_id", how="left")
    image_frame["person_count"] = image_frame["person_count"].fillna(0).astype(int)
    image_frame = image_frame.merge(image_scores, on="image_id", how="left")
    image_frame = image_frame.merge(image_counts, on="image_id", how="left")
    image_frame["object_count"] = image_frame["object_count"].fillna(0).astype(int)
    image_frame[["support_score", "support_rate", "mean_person_area"]] = image_frame[
        ["support_score", "support_rate", "mean_person_area"]
    ].fillna(0.0)

    count_support_hist = private_constraints["person_count_hist"]
    count_bins = private_constraints["person_count_bins"]
    count_indices = np.clip(
        np.digitize(image_frame["person_count"], count_bins) - 1,
        0,
        len(count_support_hist) - 1,
    )
    image_frame["count_support"] = count_support_hist[count_indices]

    class_counts = (
        flir_annotations_df[~flir_annotations_df["is_person"]]
        .groupby(["image_id", "canonical_label"])
        .size()
        .reset_index(name="class_count")
    )
    rare_class_inverse_support = (
        flir_annotations_df[~flir_annotations_df["is_person"]]["canonical_label"]
        .value_counts()
        .rename("global_class_count")
        .rename_axis("canonical_label")
        .reset_index()
    )
    class_counts = class_counts.merge(rare_class_inverse_support, on="canonical_label", how="left")
    class_counts["rarity_score"] = class_counts["class_count"] / class_counts["global_class_count"].clip(lower=1)
    rarity_bonus = class_counts.groupby("image_id")["rarity_score"].sum().rename("rarity_bonus")
    image_frame = image_frame.merge(rarity_bonus, on="image_id", how="left")
    image_frame["rarity_bonus"] = image_frame["rarity_bonus"].fillna(0.0)

    image_frame["hard_support"] = (
        image_frame["person_count"].le(private_constraints["person_count_max"])
        & (image_frame["person_count"].eq(0) | image_frame["support_rate"].ge(0.50))
    )
    image_frame["match_score"] = (
        0.35 * image_frame["count_support"]
        + 0.45 * image_frame["support_score"]
        + 0.20 * image_frame["rarity_bonus"]
    )
    return image_frame


def select_reduced_train_subset(
    train_images_df: pd.DataFrame,
    train_annotations_df: pd.DataFrame,
    private_constraints: Dict[str, object],
) -> Dict[str, object]:
    image_scores = score_flir_images_against_private(
        train_images_df,
        train_annotations_df,
        private_constraints,
    )

    starting_images = len(train_images_df)
    candidate_images = image_scores[
        image_scores["hard_support"] | image_scores["rarity_bonus"].gt(0.0)
    ].copy()
    after_coarse_filter = len(candidate_images)
    target_total_images = min(
        int(private_constraints["target_total_images"]),
        int(len(train_images_df)),
    )

    group_features = candidate_images.groupby(["split", "scene_proxy_unit"], as_index=False).agg(
        scene_group_images=("image_id", "size"),
        scene_group_score=("match_score", "mean"),
        scene_group_support=("support_score", "mean"),
        scene_group_person_rate=("person_count", lambda s: float(np.mean(np.asarray(s) > 0))),
        scene_group_rarity=("rarity_bonus", "sum"),
    )
    group_features["scene_selection_score"] = (
        0.45 * group_features["scene_group_score"]
        + 0.25 * group_features["scene_group_support"]
        + 0.15 * group_features["scene_group_person_rate"]
        + 0.15 * group_features["scene_group_rarity"]
    )
    group_features = group_features.sort_values(
        ["scene_selection_score", "scene_group_rarity"],
        ascending=[False, False],
    ).reset_index(drop=True)

    selected_scene_units: List[str] = []
    running_image_count = 0
    for row in tqdm(
        group_features.itertuples(index=False),
        total=len(group_features),
        desc="Selecting train scene units",
    ):
        if running_image_count >= target_total_images:
            break
        selected_scene_units.append(str(row.scene_proxy_unit))
        running_image_count += int(row.scene_group_images)

    reduced_images = train_images_df[train_images_df["scene_proxy_unit"].isin(selected_scene_units)].copy()
    reduced_annotations = train_annotations_df[
        train_annotations_df["image_id"].isin(reduced_images["image_id"])
    ].copy()
    after_scene_screening = len(reduced_images)

    aux_annotations = reduced_annotations[~reduced_annotations["is_person"]].copy()
    aux_image_presence = (
        aux_annotations.groupby(["canonical_label", "image_id"]).size().reset_index(name="n")
        .groupby("canonical_label").size().rename("image_support")
    )
    aux_instance_support = aux_annotations["canonical_label"].value_counts().rename("instance_support")
    aux_support_table = (
        pd.concat([aux_instance_support, aux_image_presence], axis=1)
        .fillna(0)
        .reset_index()
        .rename(columns={"index": "canonical_label"})
    )
    missing_classes = aux_support_table[
        (aux_support_table["instance_support"] < private_constraints["min_aux_class_instances"])
        | (aux_support_table["image_support"] < private_constraints["min_aux_class_images"])
    ]["canonical_label"].tolist()

    if missing_classes:
        remaining_annotations = train_annotations_df[
            train_annotations_df["image_id"].isin(candidate_images["image_id"])
        ]
        for missing_class in tqdm(missing_classes, desc="Backfilling rare classes"):
            class_scene_units = (
                remaining_annotations[remaining_annotations["canonical_label"] == missing_class]
                ["scene_proxy_unit"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            for scene_unit in class_scene_units:
                if scene_unit in selected_scene_units:
                    continue
                selected_scene_units.append(scene_unit)
                break

    reduced_images = train_images_df[train_images_df["scene_proxy_unit"].isin(selected_scene_units)].copy()
    reduced_annotations = train_annotations_df[
        train_annotations_df["image_id"].isin(reduced_images["image_id"])
    ].copy()
    reduced_images = reduced_images.merge(
        image_scores[["image_id", "match_score", "support_score", "support_rate", "hard_support"]],
        on="image_id",
        how="left",
    )

    stage_table = pd.DataFrame(
        [
            {"stage": "starting_flir_train", "images": starting_images},
            {"stage": "after_coarse_filter", "images": after_coarse_filter},
            {"stage": "after_scene_screening", "images": after_scene_screening},
            {"stage": "final_reduced_train", "images": int(len(reduced_images))},
        ]
    )
    return {
        "images": reduced_images,
        "annotations": reduced_annotations,
        "stage_table": stage_table,
        "selected_scene_units": selected_scene_units,
        "image_scores": image_scores,
        "selection_mode": "reduced_train_subset",
    }


def sort_images_for_selection(images_df: pd.DataFrame) -> pd.DataFrame:
    sort_columns = [column for column in ["video_id", "frame_index", "file_name", "image_id"] if column in images_df.columns]
    if not sort_columns:
        return images_df.copy()
    return images_df.sort_values(sort_columns, na_position="last").reset_index(drop=True)


def cap_split_tables(
    images_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    max_images: Optional[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    images_df = sort_images_for_selection(images_df)
    if max_images is None:
        return images_df, annotations_df[annotations_df["image_id"].isin(images_df["image_id"])].copy()
    images_df = images_df.head(int(max_images)).copy()
    annotations_df = annotations_df[annotations_df["image_id"].isin(images_df["image_id"])].copy()
    return images_df, annotations_df


def align_eval_split_to_train_categories(
    images_df: pd.DataFrame,
    annotations_df: pd.DataFrame,
    allowed_category_ids: set[int],
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    before_images = int(len(images_df))
    before_annotations = int(len(annotations_df))

    filtered_annotations = annotations_df[
        annotations_df["category_id"].astype(int).isin(allowed_category_ids)
    ].copy()
    kept_image_ids = set(filtered_annotations["image_id"])
    filtered_images = images_df[images_df["image_id"].isin(kept_image_ids)].copy()
    filtered_annotations = filtered_annotations[
        filtered_annotations["image_id"].isin(filtered_images["image_id"])
    ].copy()

    stats_payload = {
        "images_before": before_images,
        "images_after": int(len(filtered_images)),
        "images_removed": int(before_images - len(filtered_images)),
        "annotations_before": before_annotations,
        "annotations_after": int(len(filtered_annotations)),
        "annotations_removed": int(before_annotations - len(filtered_annotations)),
    }
    return filtered_images, filtered_annotations, stats_payload


def records_for_export(images_df: pd.DataFrame, annotations_df: pd.DataFrame, split: str) -> Tuple[List[dict], List[dict]]:
    image_records = []
    for row in images_df.itertuples(index=False):
        image_records.append(
            {
                "id": row.image_id,
                "file_name": row.file_name,
                "width": int(row.width),
                "height": int(row.height),
                "source_split": split,
            }
        )
    annotation_records = []
    for row in annotations_df.itertuples(index=False):
        annotation_records.append(
            {
                "id": row.annotation_id,
                "image_id": row.image_id,
                "category_id": int(row.category_id),
                "bbox": [float(x) for x in row.bbox],
                "area": float(row.area),
                "iscrowd": int(bool(row.iscrowd)),
            }
        )
    return image_records, annotation_records


def export_split_to_v18(
    *,
    split: str,
    image_records: List[dict],
    annotation_records: List[dict],
    source_categories: List[dict],
    flir_root: Path,
    output_root: Path,
    overwrite: bool,
    write_train_captions: bool,
    source_annotations_ref: str,
) -> ExportSummary:
    unique_images = dedupe_images(image_records)
    selected_source_ids = {img["id"] for img in unique_images}
    source_annotations = [ann for ann in annotation_records if ann["image_id"] in selected_source_ids]

    output_categories, category_remap, person_output_category_id = build_category_remap(
        source_categories,
        person_only=False,
    )
    output_split_dir = output_root / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    output_images: List[dict] = []
    output_annotations: List[dict] = []
    source_id_to_output_id: Dict[object, str] = {}
    seen_output_filenames: set[str] = set()

    for image in tqdm(unique_images, desc=f"Exporting {split} images", total=len(unique_images)):
        source_id = image["id"]
        output_id = Path(str(image["file_name"])).stem
        output_file_name = f"{output_id}.npy"
        if output_file_name in seen_output_filenames:
            raise ValueError(f"Duplicate output filename within split={split!r}: {output_file_name}")
        seen_output_filenames.add(output_file_name)
        source_id_to_output_id[source_id] = output_id

        source_split = str(image.get("source_split", split))
        source_path = resolve_source_image_path(flir_root, source_split, str(image["file_name"]))
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source image: {source_path}")

        output_path = output_split_dir / output_file_name
        arr = write_image_npy(source_path, output_path, overwrite=overwrite)
        height, width = arr.shape[:2]
        output_images.append(
            {
                "id": output_id,
                "width": int(width),
                "height": int(height),
                "file_name": output_file_name,
            }
        )

    ann_id = 0
    for ann in tqdm(source_annotations, desc=f"Exporting {split} annotations", total=len(source_annotations)):
        source_category_id = int(ann["category_id"])
        if source_category_id not in category_remap:
            continue
        bbox = [float(x) for x in ann["bbox"]]
        area = float(ann.get("area", bbox[2] * bbox[3]))
        output_annotations.append(
            {
                "id": ann_id,
                "image_id": source_id_to_output_id[ann["image_id"]],
                "category_id": category_remap[source_category_id],
                "bbox": bbox,
                "area": area,
                "iscrowd": int(bool(ann.get("iscrowd", 0))),
            }
        )
        ann_id += 1

    output_coco = {
        "info": {
            "description": "Reduced FLIR thermal proxy subset converted to a v18-style layout",
            "source_annotations": source_annotations_ref,
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "person_only": False,
        },
        "licenses": [],
        "categories": output_categories,
        "images": output_images,
        "annotations": output_annotations,
    }
    save_json(output_split_dir / "annotations.json", output_coco)

    if split == "train" and write_train_captions:
        counts_by_image_id = defaultdict(int)
        if person_output_category_id is not None:
            for ann in output_annotations:
                if ann["category_id"] == person_output_category_id:
                    counts_by_image_id[ann["image_id"]] += 1
        captions = {
            image["id"]: caption_from_count(counts_by_image_id.get(image["id"], 0))
            for image in output_images
        }
        save_json(output_split_dir / "captions.json", captions)

    sample_dtype = None
    sample_shape = None
    if output_images:
        sample_arr = np.load(output_split_dir / output_images[0]["file_name"])
        sample_dtype = str(sample_arr.dtype)
        sample_shape = tuple(int(x) for x in sample_arr.shape)

    return ExportSummary(
        split=split,
        image_count=len(output_images),
        annotation_count=len(output_annotations),
        category_count=len(output_categories),
        sample_dtype=sample_dtype,
        sample_shape=sample_shape,
        source_annotations=source_annotations_ref,
    )


def save_plot(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_class_distribution(
    *,
    original_profile: Dict[str, object],
    reduced_profile: Dict[str, object],
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    original_table = original_profile["class_summary"]["table"].head(20)
    reduced_table = reduced_profile["class_summary"]["table"].head(20)
    sns.barplot(data=original_table, x="instance_count", y="canonical_label", ax=axes[0], color="#457b9d")
    axes[0].set_title("Original FLIR train class counts")
    sns.barplot(data=reduced_table, x="instance_count", y="canonical_label", ax=axes[1], color="#2a9d8f")
    axes[1].set_title("Reduced FLIR train class counts")
    fig.tight_layout()
    return fig


def plot_person_histograms(
    *,
    private_profile: Dict[str, object],
    original_profile: Dict[str, object],
    reduced_profile: Dict[str, object],
) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    profiles = [
        ("Private train", private_profile, "#355070"),
        ("Original FLIR train", original_profile, "#457b9d"),
        ("Reduced FLIR train", reduced_profile, "#2a9d8f"),
    ]
    for label, profile, color in profiles:
        sns.histplot(profile["image_metrics"]["person_count"], bins=20, ax=axes[0, 0], label=label, color=color, alpha=0.35)
        sns.histplot(profile["person_annotations"]["area_ratio"], bins=30, ax=axes[0, 1], label=label, color=color, alpha=0.35)
        sns.histplot(profile["person_annotations"]["aspect_ratio"], bins=30, ax=axes[1, 0], label=label, color=color, alpha=0.35)
        sns.histplot(profile["person_annotations"]["center_y_norm"], bins=30, ax=axes[1, 1], label=label, color=color, alpha=0.35)
    axes[0, 0].set_title("Persons per image")
    axes[0, 1].set_title("Person area ratio")
    axes[1, 0].set_title("Person aspect ratio")
    axes[1, 1].set_title("Person y-center")
    for ax in axes.ravel():
        ax.legend()
    fig.tight_layout()
    return fig


def plot_scene_concentration(
    *,
    original_profile: Dict[str, object],
    reduced_profile: Dict[str, object],
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, profile, color in [
        ("Original FLIR train", original_profile, "#457b9d"),
        ("Reduced FLIR train", reduced_profile, "#2a9d8f"),
    ]:
        counts = (
            profile["images"]["scene_proxy_unit"]
            .fillna("missing")
            .value_counts()
            .sort_values(ascending=False)
        )
        cumulative = counts.cumsum() / counts.sum() if not counts.empty else pd.Series(dtype=float)
        ax.plot(np.arange(1, len(cumulative) + 1), cumulative.values, label=label, color=color, lw=2)
    ax.set_title("Scene concentration")
    ax.set_xlabel("Top-ranked scene units")
    ax.set_ylabel("Cumulative image share")
    ax.set_ylim(0.0, 1.01)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_reduction_funnel(stage_table: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(data=stage_table, x="stage", y="images", ax=ax, color="#6c757d")
    ax.set_title("Reduction funnel")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    return fig


def plot_comparison_heatmap(*, comparison_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 5))
    pivot = comparison_df.pivot(index="statistic", columns="comparison", values="distance")
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="mako", ax=ax)
    ax.set_title("Alignment comparison distances")
    fig.tight_layout()
    return fig


def build_summary_payload(
    *,
    private_profile: Dict[str, object],
    original_profile: Dict[str, object],
    reduced_profile: Dict[str, object],
    stage_table: pd.DataFrame,
    comparison_private_vs_original: pd.DataFrame,
    comparison_private_vs_reduced: pd.DataFrame,
    comparison_original_vs_reduced: pd.DataFrame,
    alignment_rows: List[dict],
) -> Dict[str, object]:
    def _inventory(profile: Dict[str, object]) -> Dict[str, object]:
        return profile["inventory"].iloc[0].to_dict()

    return {
        "private_train_inventory": _inventory(private_profile),
        "original_flir_train_inventory": _inventory(original_profile),
        "reduced_flir_train_inventory": _inventory(reduced_profile),
        "reduction_stage_table": stage_table.to_dict(orient="records"),
        "private_vs_original": comparison_private_vs_original.to_dict(orient="records"),
        "private_vs_reduced": comparison_private_vs_reduced.to_dict(orient="records"),
        "original_vs_reduced": comparison_original_vs_reduced.to_dict(orient="records"),
        "eval_split_alignment": alignment_rows,
    }


def save_analysis_artifacts(
    *,
    analysis_output_root: Path,
    private_profile: Dict[str, object],
    original_profile: Dict[str, object],
    reduced_profile: Dict[str, object],
    selection_bundle: Dict[str, object],
    comparison_private_vs_original: pd.DataFrame,
    comparison_private_vs_reduced: pd.DataFrame,
    comparison_original_vs_reduced: pd.DataFrame,
    alignment_rows: List[dict],
    train_export_images: pd.DataFrame,
    train_export_annotations: pd.DataFrame,
    source_categories: List[dict],
    args: argparse.Namespace,
) -> Dict[str, Path]:
    plots_dir = analysis_output_root / "plots"
    tables_dir = analysis_output_root / "tables"
    mappings_dir = analysis_output_root / "mappings"
    metadata_dir = analysis_output_root / "metadata"
    source_coco_dir = analysis_output_root / "source_coco"

    stage_table = selection_bundle["stage_table"].copy()
    train_manifest = sort_images_for_selection(
        selection_bundle["images"][
            [
                "image_id",
                "split",
                "file_name",
                "file_path",
                "width",
                "height",
                "video_id",
                "scene_proxy_unit",
                "frame_index",
                "match_score",
                "support_score",
                "support_rate",
                "hard_support",
            ]
        ]
    )

    plot_jobs = [
        (
            plots_dir / "class_distribution.png",
            lambda: plot_class_distribution(
                original_profile=original_profile,
                reduced_profile=reduced_profile,
            ),
        ),
        (
            plots_dir / "person_histograms.png",
            lambda: plot_person_histograms(
                private_profile=private_profile,
                original_profile=original_profile,
                reduced_profile=reduced_profile,
            ),
        ),
        (
            plots_dir / "scene_concentration.png",
            lambda: plot_scene_concentration(
                original_profile=original_profile,
                reduced_profile=reduced_profile,
            ),
        ),
        (
            plots_dir / "reduction_funnel.png",
            lambda: plot_reduction_funnel(stage_table),
        ),
        (
            plots_dir / "comparison_heatmap.png",
            lambda: plot_comparison_heatmap(
                comparison_df=pd.concat(
                    [
                        comparison_private_vs_original,
                        comparison_private_vs_reduced,
                        comparison_original_vs_reduced,
                    ],
                    ignore_index=True,
                ),
            ),
        ),
    ]
    for path, builder in tqdm(plot_jobs, desc="Saving analysis plots"):
        save_plot(builder(), path)

    save_csv(tables_dir / "private_train_inventory.csv", private_profile["inventory"])
    save_csv(tables_dir / "flir_train_inventory.csv", original_profile["inventory"])
    save_csv(tables_dir / "reduced_train_inventory.csv", reduced_profile["inventory"])
    save_csv(tables_dir / "reduction_stage_table.csv", stage_table)
    save_csv(tables_dir / "private_vs_flir_train.csv", comparison_private_vs_original)
    save_csv(tables_dir / "private_vs_reduced_train.csv", comparison_private_vs_reduced)
    save_csv(tables_dir / "flir_train_vs_reduced_train.csv", comparison_original_vs_reduced)
    save_csv(tables_dir / "train_subset_manifest.csv", train_manifest)
    save_csv(tables_dir / "train_image_scores.csv", selection_bundle["image_scores"])
    save_csv(tables_dir / "train_class_summary_original.csv", original_profile["class_summary"]["table"])
    save_csv(tables_dir / "train_class_summary_reduced.csv", reduced_profile["class_summary"]["table"])
    save_csv(tables_dir / "eval_split_alignment.csv", pd.DataFrame(alignment_rows))

    train_source_images, train_source_annotations = records_for_export(
        train_export_images,
        train_export_annotations,
        "train",
    )
    train_source_coco = {
        "info": {
            "description": "Reduced FLIR train source selection for end-to-end private-proxy export",
            "export_split": "train",
        },
        "images": train_source_images,
        "annotations": train_source_annotations,
        "categories": source_categories,
    }
    train_source_coco_path = source_coco_dir / "reduced_flir_train_coco.json"
    save_json(train_source_coco_path, train_source_coco)

    mapping_payload = {
        "selected_scene_units": [str(x) for x in selection_bundle["selected_scene_units"]],
        "train_images_after_cap": int(len(train_export_images)),
        "train_annotations_after_cap": int(len(train_export_annotations)),
        "train_category_ids": sorted(train_export_annotations["category_id"].astype(int).unique().tolist()),
    }
    save_json(mappings_dir / "train_subset_mapping.json", mapping_payload)

    summary_payload = build_summary_payload(
        private_profile=private_profile,
        original_profile=original_profile,
        reduced_profile=reduced_profile,
        stage_table=stage_table,
        comparison_private_vs_original=comparison_private_vs_original,
        comparison_private_vs_reduced=comparison_private_vs_reduced,
        comparison_original_vs_reduced=comparison_original_vs_reduced,
        alignment_rows=alignment_rows,
    )
    save_json(metadata_dir / "analysis_summary.json", summary_payload)
    save_json(
        metadata_dir / "run_metadata.json",
        {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "analysis",
            "seed": int(args.seed),
            "flir_root": str(args.flir_root),
            "output_root": str(args.output_root),
            "analysis_output_root": str(args.analysis_output_root),
            "splits": list(args.splits),
            "max_images_per_split": args.max_images_per_split,
            "overwrite": bool(args.overwrite),
            "write_train_captions": not args.no_train_captions,
            "use_full_dataset": False,
        },
    )

    return {
        "train_source_coco": train_source_coco_path,
        "run_metadata": metadata_dir / "run_metadata.json",
    }


def print_mode_summary(
    *,
    mode_name: str,
    export_summaries: List[ExportSummary],
    alignment_rows: Optional[List[dict]] = None,
) -> None:
    print(f"Mode: {mode_name}")
    for summary in export_summaries:
        print(
            f"[{summary.split}] images={summary.image_count} "
            f"annotations={summary.annotation_count} "
            f"categories={summary.category_count} "
            f"sample_dtype={summary.sample_dtype} "
            f"sample_shape={summary.sample_shape}"
        )
    if alignment_rows:
        for row in alignment_rows:
            print(
                f"[alignment:{row['split']}] "
                f"images_removed={row['images_removed']} "
                f"annotations_removed={row['annotations_removed']}"
            )


def run_full_passthrough(args: argparse.Namespace) -> None:
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    write_train_captions = not args.no_train_captions

    export_summaries: List[ExportSummary] = []
    for split in args.splits:
        images_df, categories_df, annotations_df, coco = load_flir_split(
            split,
            flir_root=args.flir_root,
            scene_chunk_size=1,
        )
        images_df, annotations_df = cap_split_tables(images_df, annotations_df, args.max_images_per_split)
        image_records, annotation_records = records_for_export(images_df, annotations_df, split)
        summary = export_split_to_v18(
            split=split,
            image_records=image_records,
            annotation_records=annotation_records,
            source_categories=coco.get("categories", []),
            flir_root=args.flir_root,
            output_root=output_root,
            overwrite=args.overwrite,
            write_train_captions=write_train_captions,
            source_annotations_ref=str(args.flir_root / SPLIT_TO_SOURCE_DIR[split] / "coco.json"),
        )
        export_summaries.append(summary)
    print_mode_summary(mode_name="full_passthrough", export_summaries=export_summaries)


def run_analysis_mode(args: argparse.Namespace) -> None:
    if "train" not in args.splits:
        raise ValueError("Analysis mode requires 'train' in --splits so the reduced train subset can be built.")

    args.analysis_output_root.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_train_captions = not args.no_train_captions

    private_images, _, private_annotations = load_v18_train()
    private_profile = profile_dataset("private_train", private_images, private_annotations)
    private_constraints = derive_private_constraints(private_profile)

    split_tables: Dict[str, Dict[str, object]] = {}
    split_categories: Dict[str, List[dict]] = {}
    for split in tqdm(args.splits, desc="Loading FLIR splits"):
        images_df, categories_df, annotations_df, coco = load_flir_split(
            split,
            flir_root=args.flir_root,
            scene_chunk_size=private_constraints["scene_chunk_size"],
        )
        split_tables[split] = {
            "images": images_df,
            "annotations": annotations_df,
            "categories_df": categories_df,
            "coco": coco,
        }
        split_categories[split] = coco.get("categories", [])

    train_images = split_tables["train"]["images"]
    train_annotations = split_tables["train"]["annotations"]
    original_profile = profile_dataset("flir_train_original", train_images, train_annotations)

    selection_bundle = select_reduced_train_subset(
        train_images_df=train_images,
        train_annotations_df=train_annotations,
        private_constraints=private_constraints,
    )
    train_export_images, train_export_annotations = cap_split_tables(
        selection_bundle["images"],
        selection_bundle["annotations"],
        args.max_images_per_split,
    )
    reduced_profile = profile_dataset(
        "flir_train_reduced",
        train_export_images,
        train_export_annotations,
    )

    comparison_private_vs_original = compare_profiles(
        private_profile,
        original_profile,
        "Private vs original FLIR train",
    )
    comparison_private_vs_reduced = compare_profiles(
        private_profile,
        reduced_profile,
        "Private vs reduced FLIR train",
    )
    comparison_original_vs_reduced = compare_profiles(
        original_profile,
        reduced_profile,
        "Original FLIR train vs reduced FLIR train",
    )

    train_category_ids = set(train_export_annotations["category_id"].astype(int).unique().tolist())
    alignment_rows: List[dict] = []
    aligned_eval_tables: Dict[str, Tuple[pd.DataFrame, pd.DataFrame]] = {}
    export_summaries: List[ExportSummary] = []

    for split in [name for name in args.splits if name in {"val", "test"}]:
        images_df = split_tables[split]["images"]
        annotations_df = split_tables[split]["annotations"]
        aligned_images, aligned_annotations, alignment_stats = align_eval_split_to_train_categories(
            images_df,
            annotations_df,
            train_category_ids,
        )
        aligned_images, aligned_annotations = cap_split_tables(
            aligned_images,
            aligned_annotations,
            args.max_images_per_split,
        )
        alignment_stats["split"] = split
        alignment_stats["train_category_count"] = int(len(train_category_ids))
        alignment_rows.append(alignment_stats)
        aligned_eval_tables[split] = (aligned_images, aligned_annotations)

    analysis_artifacts = save_analysis_artifacts(
        analysis_output_root=args.analysis_output_root,
        private_profile=private_profile,
        original_profile=original_profile,
        reduced_profile=reduced_profile,
        selection_bundle=selection_bundle,
        comparison_private_vs_original=comparison_private_vs_original,
        comparison_private_vs_reduced=comparison_private_vs_reduced,
        comparison_original_vs_reduced=comparison_original_vs_reduced,
        alignment_rows=alignment_rows,
        train_export_images=train_export_images,
        train_export_annotations=train_export_annotations,
        source_categories=split_categories["train"],
        args=args,
    )

    train_image_records, train_annotation_records = records_for_export(
        train_export_images,
        train_export_annotations,
        "train",
    )
    export_summaries.append(
        export_split_to_v18(
            split="train",
            image_records=train_image_records,
            annotation_records=train_annotation_records,
            source_categories=split_categories["train"],
            flir_root=args.flir_root,
            output_root=args.output_root,
            overwrite=args.overwrite,
            write_train_captions=write_train_captions,
            source_annotations_ref=str(analysis_artifacts["train_source_coco"]),
        )
    )

    for split in [name for name in args.splits if name in {"val", "test"}]:
        aligned_images, aligned_annotations = aligned_eval_tables[split]
        image_records, annotation_records = records_for_export(aligned_images, aligned_annotations, split)
        export_summaries.append(
            export_split_to_v18(
                split=split,
                image_records=image_records,
                annotation_records=annotation_records,
                source_categories=split_categories[split],
                flir_root=args.flir_root,
                output_root=args.output_root,
                overwrite=args.overwrite,
                write_train_captions=write_train_captions,
                source_annotations_ref=str(args.flir_root / SPLIT_TO_SOURCE_DIR[split] / "coco.json"),
            )
        )

    save_csv(args.analysis_output_root / "tables" / "eval_split_alignment.csv", pd.DataFrame(alignment_rows))
    save_json(
        args.analysis_output_root / "metadata" / "analysis_summary.json",
        build_summary_payload(
            private_profile=private_profile,
            original_profile=original_profile,
            reduced_profile=reduced_profile,
            stage_table=selection_bundle["stage_table"],
            comparison_private_vs_original=comparison_private_vs_original,
            comparison_private_vs_reduced=comparison_private_vs_reduced,
            comparison_original_vs_reduced=comparison_original_vs_reduced,
            alignment_rows=alignment_rows,
        ),
    )

    print_mode_summary(
        mode_name="analysis_reduced_train",
        export_summaries=export_summaries,
        alignment_rows=alignment_rows,
    )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"FLIR root:            {args.flir_root}")
    print(f"Output root:          {args.output_root}")
    print(f"Analysis output root: {args.analysis_output_root}")
    print(f"Splits:               {args.splits}")
    print(f"Max images:           {args.max_images_per_split}")
    print(f"Overwrite:            {args.overwrite}")
    print(f"Use full dataset:     {args.use_full_dataset}")
    print(f"Seed:                 {args.seed}")

    if args.use_full_dataset:
        ensure_tabular_imports()
        run_full_passthrough(args)
    else:
        ensure_tabular_imports()
        ensure_plotting_imports()
        sns.set_theme(style="whitegrid", context="talk")
        plt.rcParams["figure.figsize"] = (10, 6)
        plt.rcParams["figure.dpi"] = 120
        run_analysis_mode(args)
    print("Done.")


if __name__ == "__main__":
    main()

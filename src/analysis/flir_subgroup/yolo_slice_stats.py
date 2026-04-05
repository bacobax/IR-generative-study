"""Slice-stat helpers for YOLO-format detection datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.core.configs.config_loader import load_yaml


SIZE_BIN_ORDER = ("small", "medium", "large")
POSITION_ROW_LABELS = ("top", "middle", "bottom")
POSITION_COL_LABELS = ("left", "center", "right")
POSITION_BIN_ORDER = (
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)


@dataclass(frozen=True)
class YoloSliceThresholds:
    """Global size thresholds used to assign size bins."""

    q33: float
    q67: float

    def to_dict(self) -> dict[str, float]:
        return {"q33": self.q33, "q67": self.q67}


@dataclass(frozen=True)
class YoloSliceDataset:
    """Structured slice tables for one YOLO train split."""

    dataset_yaml: str
    train_image_dir: str
    train_label_dir: str
    names: dict[int, str]
    instance_df: pd.DataFrame
    image_df: pd.DataFrame
    thresholds: YoloSliceThresholds


def resolve_yolo_train_dirs(dataset_yaml_path: str | Path) -> tuple[Path, Path, dict[int, str]]:
    """Resolve the train image and label directories from a dataset YAML."""

    yaml_path = Path(dataset_yaml_path).resolve()
    payload = load_yaml(yaml_path)
    train_dir = Path(str(payload["train"])).resolve()
    if train_dir.parent.name == "images":
        label_dir = train_dir.parent.parent / "labels" / train_dir.name
    elif train_dir.name == "images":
        label_dir = train_dir.parent / "labels"
    else:
        label_dir = train_dir.parents[1] / "labels" / train_dir.name
    raw_names = payload.get("names", {})
    names = {int(idx): str(label) for idx, label in raw_names.items()}
    return train_dir, label_dir.resolve(), names


def assign_bins_from_thresholds(values: pd.Series, q33: float, q67: float) -> pd.Categorical:
    """Assign global size bins from explicit tertile thresholds."""

    labels = np.select(
        [values.astype(float) <= q33, values.astype(float) <= q67],
        ["small", "medium"],
        default="large",
    )
    return pd.Categorical(labels, categories=list(SIZE_BIN_ORDER), ordered=True)


def add_position_bin_columns(instance_df: pd.DataFrame) -> pd.DataFrame:
    """Assign 3x3 notebook-style position bins from normalized bbox centers."""

    df = instance_df.copy()
    x_idx = np.clip(np.floor(df["bbox_center_x_norm"].astype(float).clip(0.0, 1.0) * 3.0).astype(int), 0, 2)
    y_idx = np.clip(np.floor(df["bbox_center_y_norm"].astype(float).clip(0.0, 1.0) * 3.0).astype(int), 0, 2)
    df["position_row_bin"] = pd.Categorical(
        [POSITION_ROW_LABELS[idx] for idx in y_idx],
        categories=list(POSITION_ROW_LABELS),
        ordered=True,
    )
    df["position_col_bin"] = pd.Categorical(
        [POSITION_COL_LABELS[idx] for idx in x_idx],
        categories=list(POSITION_COL_LABELS),
        ordered=True,
    )
    df["position_bin"] = pd.Categorical(
        [POSITION_BIN_ORDER[row * 3 + col] for row, col in zip(y_idx, x_idx)],
        categories=list(POSITION_BIN_ORDER),
        ordered=True,
    )
    return df


def load_yolo_slice_dataset(dataset_yaml_path: str | Path) -> YoloSliceDataset:
    """Load train labels from a YOLO dataset YAML and assign notebook-style slices."""

    train_dir, label_dir, names = resolve_yolo_train_dirs(dataset_yaml_path)
    if not train_dir.exists():
        raise FileNotFoundError(f"YOLO train image directory not found: {train_dir}")
    if not label_dir.exists():
        raise FileNotFoundError(f"YOLO train label directory not found: {label_dir}")

    image_paths = sorted(path for path in train_dir.glob("*") if path.is_file())
    image_rows: list[dict[str, Any]] = []
    instance_rows: list[dict[str, Any]] = []

    for image_index, image_path in enumerate(image_paths):
        stem = image_path.stem
        label_path = label_dir / f"{stem}.txt"
        image_rows.append(
            {
                "image_index": image_index,
                "image_stem": stem,
                "image_path": str(image_path.resolve()),
                "label_path": str(label_path.resolve()),
            }
        )
        if not label_path.exists():
            continue
        text = label_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        for ann_index, line in enumerate(text.splitlines()):
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Expected 5 YOLO columns in {label_path}, got: {line!r}")
            class_idx = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])
            instance_rows.append(
                {
                    "image_index": image_index,
                    "image_stem": stem,
                    "image_path": str(image_path.resolve()),
                    "label_path": str(label_path.resolve()),
                    "instance_index": ann_index,
                    "class_idx": class_idx,
                    "class_label": names.get(class_idx, str(class_idx)),
                    "bbox_center_x_norm": x_center,
                    "bbox_center_y_norm": y_center,
                    "bbox_w_norm": width,
                    "bbox_h_norm": height,
                    "bbox_area_ratio": width * height,
                }
            )

    image_df = pd.DataFrame(image_rows)
    instance_df = pd.DataFrame(instance_rows)
    if image_df.empty:
        raise ValueError(f"No training images found under: {train_dir}")

    if instance_df.empty:
        thresholds = YoloSliceThresholds(q33=0.0, q67=0.0)
        return YoloSliceDataset(
            dataset_yaml=str(Path(dataset_yaml_path).resolve()),
            train_image_dir=str(train_dir),
            train_label_dir=str(label_dir),
            names=names,
            instance_df=instance_df,
            image_df=image_df,
            thresholds=thresholds,
        )

    q33, q67 = instance_df["bbox_area_ratio"].astype(float).quantile([1.0 / 3.0, 2.0 / 3.0]).tolist()
    thresholds = YoloSliceThresholds(q33=float(q33), q67=float(q67))
    instance_df = add_position_bin_columns(instance_df)
    instance_df["size_bin"] = assign_bins_from_thresholds(instance_df["bbox_area_ratio"], thresholds.q33, thresholds.q67)
    instance_df["slice_key"] = list(
        zip(
            instance_df["class_label"].astype(str),
            instance_df["size_bin"].astype(str),
            instance_df["position_bin"].astype(str),
        )
    )
    return YoloSliceDataset(
        dataset_yaml=str(Path(dataset_yaml_path).resolve()),
        train_image_dir=str(train_dir),
        train_label_dir=str(label_dir),
        names=names,
        instance_df=instance_df,
        image_df=image_df,
        thresholds=thresholds,
    )


def build_slice_counts_table(instance_df: pd.DataFrame, class_labels: list[str]) -> pd.DataFrame:
    """Build a dense `(class, size, position)` slice count table."""

    slice_index = pd.MultiIndex.from_product(
        [class_labels, list(SIZE_BIN_ORDER), list(POSITION_BIN_ORDER)],
        names=["class_label", "size_bin", "position_bin"],
    )
    if instance_df.empty:
        counts = pd.DataFrame(index=slice_index).reset_index()
        counts["count"] = 0
        return counts
    counts = (
        instance_df.groupby(["class_label", "size_bin", "position_bin"], observed=False)
        .size()
        .rename("count")
        .reindex(slice_index, fill_value=0)
        .reset_index()
    )
    return counts

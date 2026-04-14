"""YOLO-format export helpers for FLIR subset experiments."""

from __future__ import annotations

import json
import shutil
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
import yaml
from PIL import Image

from src.analysis.flir_subgroup.data import load_image_array, normalize_for_display


@dataclass
class YOLOPartitionExportSummary:
    """Summary of one exported YOLO partition."""

    dataset_name: str
    split_name: str
    image_dir: str
    label_dir: str
    n_images_requested: int
    n_images: int
    n_label_files: int
    n_annotations_requested: int
    n_annotations_source: int
    n_annotations_exported: int
    n_empty_images: int
    n_missing_source_images: int
    n_annotations_skipped_missing_source: int


def _ensure_clean_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _build_class_mapping(category_table: pd.DataFrame) -> tuple[list[str], dict[int, int]]:
    category_df = category_table.copy().sort_values("class_id").reset_index(drop=True)
    class_labels = category_df["class_label"].astype(str).tolist()
    class_id_to_yolo = {
        int(row.class_id): idx
        for idx, row in enumerate(category_df.itertuples(index=False))
    }
    return class_labels, class_id_to_yolo


def _write_rgb_png(src_path: Path, dst_path: Path) -> None:
    arr = normalize_for_display(load_image_array(src_path))
    image = Image.fromarray(arr, mode="L" if arr.ndim == 2 else None).convert("RGB")
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(dst_path, format="PNG")


def _yolo_line_from_row(row, *, class_id_to_yolo: dict[int, int]) -> str:
    class_idx = class_id_to_yolo[int(row.class_id)]
    x_center = float(row.bbox_x + 0.5 * row.bbox_w) / float(row.image_width)
    y_center = float(row.bbox_y + 0.5 * row.bbox_h) / float(row.image_height)
    width = float(row.bbox_w) / float(row.image_width)
    height = float(row.bbox_h) / float(row.image_height)
    return f"{class_idx} {x_center:.8f} {y_center:.8f} {width:.8f} {height:.8f}"


def _resolve_source_path(image_row) -> Optional[Path]:
    """Return the best-effort source path from an image row."""

    for field_name in ("file_path", "image_path", "source_path"):
        raw_value = getattr(image_row, field_name, None)
        if raw_value is None or pd.isna(raw_value):
            continue
        value = str(raw_value).strip()
        if not value or value.lower() == "nan":
            continue
        return Path(value)
    return None


def export_partition_to_yolo(
    *,
    dataset_name: str,
    split_name: str,
    image_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    category_table: pd.DataFrame,
    output_root: Path,
    overwrite: bool = False,
    skip_missing_source_images: bool = True,
) -> YOLOPartitionExportSummary:
    """Export one partition to standard YOLO image/label folders."""

    image_df = image_df.copy().sort_values("image_id").reset_index(drop=True)
    instance_df = instance_df.copy().sort_values(["image_id", "ann_id"]).reset_index(drop=True)
    class_labels, class_id_to_yolo = _build_class_mapping(category_table)

    dataset_root = output_root / dataset_name
    image_dir = dataset_root / "images" / split_name
    label_dir = dataset_root / "labels" / split_name
    _ensure_clean_dir(image_dir, overwrite=overwrite)
    _ensure_clean_dir(label_dir, overwrite=overwrite)

    exportable_rows = []
    missing_image_ids: list[str] = []
    for image_row in image_df.itertuples(index=False):
        image_id = str(image_row.image_id)
        source_path = _resolve_source_path(image_row)
        if source_path is None or not source_path.exists():
            if not skip_missing_source_images:
                missing_value = source_path if source_path is not None else getattr(image_row, "file_path", None)
                raise FileNotFoundError(
                    f"Missing source image for YOLO export: dataset={dataset_name!r}, "
                    f"split={split_name!r}, image_id={image_id!r}, file_path={missing_value!r}"
                )
            missing_image_ids.append(image_id)
            continue
        exportable_rows.append((image_row, source_path))

    if missing_image_ids:
        preview = ", ".join(missing_image_ids[:5])
        if len(missing_image_ids) > 5:
            preview = f"{preview}, ..."
        warnings.warn(
            f"Skipping {len(missing_image_ids)} images with missing source files for "
            f"{dataset_name}/{split_name}: {preview}",
            stacklevel=2,
        )

    exportable_image_ids = {str(image_row.image_id) for image_row, _ in exportable_rows}
    exportable_instance_df = instance_df.loc[instance_df["image_id"].astype(str).isin(exportable_image_ids)].copy()

    annotations_by_image = {
        str(image_id): group_df
        for image_id, group_df in exportable_instance_df.groupby("image_id", sort=False)
    }

    exported_annotations = 0
    empty_images = 0
    for image_row, source_path in exportable_rows:
        image_id = str(image_row.image_id)
        stem = Path(str(image_row.file_name)).stem
        image_out_path = image_dir / f"{stem}.png"
        label_out_path = label_dir / f"{stem}.txt"
        _write_rgb_png(source_path, image_out_path)

        image_annotations = annotations_by_image.get(image_id)
        label_lines: list[str] = []
        if image_annotations is not None:
            for ann_row in image_annotations.itertuples(index=False):
                label_lines.append(_yolo_line_from_row(ann_row, class_id_to_yolo=class_id_to_yolo))
            exported_annotations += len(label_lines)
        else:
            empty_images += 1
        label_out_path.write_text("\n".join(label_lines) + ("\n" if label_lines else ""), encoding="utf-8")

    label_files = sorted(label_dir.glob("*.txt"))
    summary = YOLOPartitionExportSummary(
        dataset_name=dataset_name,
        split_name=split_name,
        image_dir=str(image_dir),
        label_dir=str(label_dir),
        n_images_requested=int(len(image_df)),
        n_images=len(exportable_rows),
        n_label_files=len(label_files),
        n_annotations_requested=int(len(instance_df)),
        n_annotations_source=int(len(exportable_instance_df)),
        n_annotations_exported=exported_annotations,
        n_empty_images=empty_images,
        n_missing_source_images=len(missing_image_ids),
        n_annotations_skipped_missing_source=int(len(instance_df) - len(exportable_instance_df)),
    )

    _validate_partition_export(summary=summary, class_labels=class_labels)
    return summary


def _validate_partition_export(
    *,
    summary: YOLOPartitionExportSummary,
    class_labels: Iterable[str],
) -> None:
    if summary.n_images != summary.n_label_files:
        raise ValueError(
            f"YOLO export mismatch for {summary.dataset_name}/{summary.split_name}: "
            f"{summary.n_images} images vs {summary.n_label_files} label files"
        )
    if summary.n_annotations_source != summary.n_annotations_exported:
        raise ValueError(
            f"YOLO annotation mismatch for {summary.dataset_name}/{summary.split_name}: "
            f"{summary.n_annotations_source} source vs {summary.n_annotations_exported} exported"
        )
    if not list(class_labels):
        raise ValueError("Cannot export YOLO dataset with an empty class list.")


def write_dataset_yaml(
    *,
    yaml_path: Path,
    train_dir: Path,
    val_dir: Path,
    test_dir: Path,
    class_labels: list[str],
) -> Path:
    """Write a Ultralytics-compatible dataset YAML."""

    payload = {
        "path": str(yaml_path.parent.resolve()),
        "train": str(train_dir.resolve()),
        "val": str(val_dir.resolve()),
        "test": str(test_dir.resolve()),
        "names": {idx: label for idx, label in enumerate(class_labels)},
        "nc": len(class_labels),
    }
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path


def write_export_manifest(
    *,
    output_root: Path,
    summaries: list[YOLOPartitionExportSummary],
) -> Path:
    """Persist the export summaries as JSON."""

    manifest_path = output_root / "export_manifest.json"
    manifest_path.write_text(
        json.dumps([asdict(item) for item in summaries], indent=2),
        encoding="utf-8",
    )
    return manifest_path


def export_experiment_a_yolo_datasets(
    *,
    output_root: Path,
    category_table: pd.DataFrame,
    balanced_image_df: pd.DataFrame,
    balanced_instance_df: pd.DataFrame,
    unbalanced_image_df: pd.DataFrame,
    unbalanced_instance_df: pd.DataFrame,
    full_train_image_df: Optional[pd.DataFrame] = None,
    full_train_instance_df: Optional[pd.DataFrame] = None,
    val_image_df: pd.DataFrame,
    val_instance_df: pd.DataFrame,
    test_image_df: pd.DataFrame,
    test_instance_df: pd.DataFrame,
    overwrite: bool = False,
    skip_missing_source_images: bool = True,
) -> dict[str, object]:
    """Export the balanced/unbalanced/full-train/val/test partitions for Experiment A."""

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    class_labels, _ = _build_class_mapping(category_table)

    summaries = [
        export_partition_to_yolo(
            dataset_name="balanced",
            split_name="train",
            image_df=balanced_image_df,
            instance_df=balanced_instance_df,
            category_table=category_table,
            output_root=output_root,
            overwrite=overwrite,
            skip_missing_source_images=skip_missing_source_images,
        ),
        export_partition_to_yolo(
            dataset_name="unbalanced",
            split_name="train",
            image_df=unbalanced_image_df,
            instance_df=unbalanced_instance_df,
            category_table=category_table,
            output_root=output_root,
            overwrite=overwrite,
            skip_missing_source_images=skip_missing_source_images,
        ),
    ]
    if full_train_image_df is not None and full_train_instance_df is not None:
        summaries.append(
            export_partition_to_yolo(
                dataset_name="full_train",
                split_name="train",
                image_df=full_train_image_df,
                instance_df=full_train_instance_df,
                category_table=category_table,
                output_root=output_root,
                overwrite=overwrite,
                skip_missing_source_images=skip_missing_source_images,
            )
        )
    summaries.extend(
        [
            export_partition_to_yolo(
                dataset_name="val",
                split_name="val",
                image_df=val_image_df,
                instance_df=val_instance_df,
                category_table=category_table,
                output_root=output_root,
                overwrite=overwrite,
                skip_missing_source_images=skip_missing_source_images,
            ),
            export_partition_to_yolo(
                dataset_name="test",
                split_name="test",
                image_df=test_image_df,
                instance_df=test_instance_df,
                category_table=category_table,
                output_root=output_root,
                overwrite=overwrite,
                skip_missing_source_images=skip_missing_source_images,
            ),
        ]
    )

    val_image_dir = output_root / "val" / "images" / "val"
    test_image_dir = output_root / "test" / "images" / "test"
    balanced_yaml = write_dataset_yaml(
        yaml_path=output_root / "balanced.yaml",
        train_dir=output_root / "balanced" / "images" / "train",
        val_dir=val_image_dir,
        test_dir=test_image_dir,
        class_labels=class_labels,
    )
    unbalanced_yaml = write_dataset_yaml(
        yaml_path=output_root / "unbalanced.yaml",
        train_dir=output_root / "unbalanced" / "images" / "train",
        val_dir=val_image_dir,
        test_dir=test_image_dir,
        class_labels=class_labels,
    )
    full_train_yaml = None
    if full_train_image_df is not None and full_train_instance_df is not None:
        full_train_yaml = write_dataset_yaml(
            yaml_path=output_root / "full_train.yaml",
            train_dir=output_root / "full_train" / "images" / "train",
            val_dir=val_image_dir,
            test_dir=test_image_dir,
            class_labels=class_labels,
        )
    test_yaml = write_dataset_yaml(
        yaml_path=output_root / "test.yaml",
        train_dir=test_image_dir,
        val_dir=val_image_dir,
        test_dir=test_image_dir,
        class_labels=class_labels,
    )
    manifest_path = write_export_manifest(output_root=output_root, summaries=summaries)

    summary_df = pd.DataFrame([asdict(item) for item in summaries])
    return {
        "summary_df": summary_df,
        "balanced_yaml": balanced_yaml,
        "unbalanced_yaml": unbalanced_yaml,
        "full_train_yaml": full_train_yaml,
        "test_yaml": test_yaml,
        "manifest_path": manifest_path,
        "class_labels": class_labels,
    }

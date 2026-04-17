"""Config-driven Ultralytics YOLO training/evaluation entrypoint.

Usage::

    python -m src.cli.train_yolo --action train --config configs/yolo/exp_a/flir/exp_balanced.yaml
    python -m src.cli.train_yolo --action eval --config configs/yolo/exp_a/flir/exp_balanced.yaml
    python -m src.cli.train_yolo --action run_exp_a --config configs/yolo/exp_a/flir/run_exp_a.yaml
    python -m src.cli.train_yolo --action run_exp_a_all --config configs/yolo/exp_a/v18/run_exp_a_all.yaml
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import fields, is_dataclass
from functools import partial
import json
import os
import random
import re
import shutil
from pathlib import Path
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn, optim

from src.algorithms.training.yolo_slice_baselines import (
    build_weighted_train_dataloader,
    make_slice_aware_yolo_dataset_class,
    prepare_yolo_slice_baseline,
)
from src.core.configs.config_loader import load_yaml, merge_config_and_cli
from src.core.configs.yolo_experiment_config import YOLOExperimentConfig
from src.core.paths import repo_root, yolo_analysis_root


def _parse_bool_arg(value: str) -> bool:
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ultralytics YOLO experiment runner")
    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file. CLI overrides config values.")
    parser.add_argument("--action", type=str, default="train",
                        choices=["train", "eval", "run_exp_a", "run_exp_a_all"])
    parser.add_argument("--dataset_yaml", type=str, default=str(Path("data/derived/yolo-test-ds/balanced.yaml")))
    parser.add_argument("--balanced_dataset_yaml", type=str, default=str(Path("data/derived/yolo-test-ds/balanced.yaml")))
    parser.add_argument("--unbalanced_dataset_yaml", type=str, default=str(Path("data/derived/yolo-test-ds/unbalanced.yaml")))
    parser.add_argument("--full_train_dataset_yaml", type=str, default=str(Path("data/derived/yolo-test-ds/full_train.yaml")))
    parser.add_argument("--test_dataset_yaml", type=str, default=None)
    parser.add_argument("--weights", type=str, default="yolov8n.pt")
    parser.add_argument("--task", type=str, default="detect")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="auto")
    parser.add_argument("--momentum", type=float, default=0.937)
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--deterministic", action="store_true", default=False)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cos_lr", action="store_true", default=False)
    parser.add_argument("--freeze_backbone_epochs", type=int, default=0)
    parser.add_argument("--backbone_lr_multiplier", type=float, default=1.0)
    parser.add_argument("--baseline_mode", type=str, default=None,
                        choices=["none", "baseline_a", "baseline_b"])
    parser.add_argument("--rarity_alpha", type=float, default=None)
    parser.add_argument("--rarity_eps", type=float, default=None)
    parser.add_argument("--image_score_top_k", type=int, default=None)
    parser.add_argument("--normalize_weights", type=_parse_bool_arg, default=None)
    parser.add_argument("--clip_weight_min", type=float, default=None)
    parser.add_argument("--clip_weight_max", type=float, default=None)
    parser.add_argument("--sampler_replacement", type=_parse_bool_arg, default=None)
    parser.add_argument("--targeted_aug_probability", type=float, default=None)
    parser.add_argument("--targeted_aug_rarity_quantile", type=float, default=None)
    parser.add_argument("--translate_fraction", type=float, default=None)
    parser.add_argument("--scale_min", type=float, default=None)
    parser.add_argument("--scale_max", type=float, default=None)
    parser.add_argument("--crop_scale_min", type=float, default=None)
    parser.add_argument("--crop_scale_max", type=float, default=None)
    parser.add_argument("--crop_min_rare_box_area_retained", type=float, default=None)
    parser.add_argument("--crop_max_attempts", type=int, default=None)
    parser.add_argument("--allow_horizontal_flip", type=_parse_bool_arg, default=None)
    parser.add_argument("--baseline_seed", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default="exp_balanced")
    parser.add_argument("--balanced_experiment_name", type=str, default="exp_balanced")
    parser.add_argument("--unbalanced_experiment_name", type=str, default="exp_unbalanced")
    parser.add_argument("--full_train_experiment_name", type=str, default="exp_full_train")
    parser.add_argument("--comparison_filename", type=str, default="comparison_summary.csv")
    parser.add_argument("--runs_root", type=str, default=None)
    parser.add_argument("--checkpoints_root", type=str, default=None)
    parser.add_argument("--analysis_root", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--save_json", action="store_true", default=False)
    parser.add_argument("--save_hybrid", action="store_true", default=False)
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser


_FLAT_TO_NESTED = {
    "dataset_yaml": "data.dataset_yaml",
    "balanced_dataset_yaml": "data.balanced_dataset_yaml",
    "unbalanced_dataset_yaml": "data.unbalanced_dataset_yaml",
    "full_train_dataset_yaml": "data.full_train_dataset_yaml",
    "test_dataset_yaml": "data.test_dataset_yaml",
    "batch_size": "data.batch_size",
    "workers": "data.workers",
    "image_size": "data.image_size",
    "weights": "model.weights",
    "task": "model.task",
    "epochs": "training.epochs",
    "lr0": "training.lr0",
    "optimizer": "training.optimizer",
    "momentum": "training.momentum",
    "weight_decay": "training.weight_decay",
    "seed": "training.seed",
    "deterministic": "training.deterministic",
    "patience": "training.patience",
    "cos_lr": "training.cos_lr",
    "freeze_backbone_epochs": "training.freeze_backbone_epochs",
    "backbone_lr_multiplier": "training.backbone_lr_multiplier",
    "baseline_mode": "baseline.mode",
    "rarity_alpha": "baseline.rarity_alpha",
    "rarity_eps": "baseline.rarity_eps",
    "image_score_top_k": "baseline.image_score_top_k",
    "normalize_weights": "baseline.normalize_weights",
    "clip_weight_min": "baseline.clip_weight_min",
    "clip_weight_max": "baseline.clip_weight_max",
    "sampler_replacement": "baseline.sampler_replacement",
    "targeted_aug_probability": "baseline.targeted_aug_probability",
    "targeted_aug_rarity_quantile": "baseline.targeted_aug_rarity_quantile",
    "translate_fraction": "baseline.translate_fraction",
    "scale_min": "baseline.scale_min",
    "scale_max": "baseline.scale_max",
    "crop_scale_min": "baseline.crop_scale_min",
    "crop_scale_max": "baseline.crop_scale_max",
    "crop_min_rare_box_area_retained": "baseline.crop_min_rare_box_area_retained",
    "crop_max_attempts": "baseline.crop_max_attempts",
    "allow_horizontal_flip": "baseline.allow_horizontal_flip",
    "baseline_seed": "baseline.seed",
    "split": "evaluation.split",
    "save_json": "evaluation.save_json",
    "save_hybrid": "evaluation.save_hybrid",
    "conf": "evaluation.conf",
    "iou": "evaluation.iou",
    "experiment_name": "output.experiment_name",
    "runs_root": "output.runs_root",
    "checkpoints_root": "output.checkpoints_root",
    "analysis_root": "output.analysis_root",
    "balanced_experiment_name": "launcher.balanced_experiment_name",
    "unbalanced_experiment_name": "launcher.unbalanced_experiment_name",
    "full_train_experiment_name": "launcher.full_train_experiment_name",
    "comparison_filename": "launcher.comparison_filename",
    "device": "device",
}


def _require_ultralytics():
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError(
            "Ultralytics is required for YOLO experiments. Install it with "
            "`pip install ultralytics`."
        ) from exc
    return YOLO


def _set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_csv_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def _config_field_schema(config_cls: type[Any]) -> dict[str, Any]:
    schema: dict[str, Any] = {}
    for field in fields(config_cls):
        nested_type: Any = None
        field_type = field.type
        if isinstance(field_type, type) and is_dataclass(field_type):
            nested_type = field_type
        elif getattr(field, "default_factory", None) is not None:
            try:
                default_value = field.default_factory()  # type: ignore[misc]
            except TypeError:
                default_value = None
            if default_value is not None and is_dataclass(default_value):
                nested_type = type(default_value)
        schema[field.name] = _config_field_schema(nested_type) if nested_type is not None else None
    return schema


def _collect_unknown_config_keys(payload: dict[str, Any], schema: dict[str, Any], *, prefix: str = "") -> list[str]:
    unknown: list[str] = []
    for key, value in payload.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if key not in schema:
            unknown.append(dotted)
            continue
        child_schema = schema[key]
        if isinstance(value, dict) and isinstance(child_schema, dict):
            unknown.extend(_collect_unknown_config_keys(value, child_schema, prefix=dotted))
    return unknown


def _validate_config_yaml_keys(yaml_path: str | None) -> dict[str, Any]:
    if not yaml_path:
        return {}
    yaml_data = load_yaml(yaml_path)
    unknown = _collect_unknown_config_keys(yaml_data, _config_field_schema(YOLOExperimentConfig))
    if unknown:
        unknown_list = ", ".join(sorted(unknown))
        raise ValueError(
            f"Unknown keys in YOLO config {yaml_path}: {unknown_list}. "
            "Add them to YOLOExperimentConfig or remove the typo instead of relying on ignored YAML entries."
        )
    return yaml_data


def _copy_if_exists(src: Path, dst: Path) -> Optional[Path]:
    if not src.exists():
        return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst


def _extract_metric(metrics_dict: dict[str, Any], keys: list[str], default: Optional[float] = None) -> Optional[float]:
    for key in keys:
        if key in metrics_dict and metrics_dict[key] is not None:
            try:
                return float(metrics_dict[key])
            except Exception:
                continue
    return default


def _summarize_validation_results(results, *, experiment_name: str, dataset_yaml: str, split: str, weights_path: str) -> dict[str, Any]:
    metrics_dict = dict(getattr(results, "results_dict", {}) or {})
    box_metrics = getattr(results, "box", None)

    map50 = _extract_metric(metrics_dict, ["metrics/mAP50(B)", "metrics/mAP50-95(B)"])
    precision = _extract_metric(metrics_dict, ["metrics/precision(B)"])
    recall = _extract_metric(metrics_dict, ["metrics/recall(B)"])
    map5095 = _extract_metric(metrics_dict, ["metrics/mAP50-95(B)"])
    map75 = float(getattr(box_metrics, "map75", 0.0)) if box_metrics is not None else None

    if box_metrics is not None:
        if map5095 is None:
            map5095 = float(getattr(box_metrics, "map", 0.0))
        if map50 is None:
            map50 = float(getattr(box_metrics, "map50", 0.0))
        if precision is None:
            precision = float(getattr(box_metrics, "mp", 0.0))
        if recall is None:
            recall = float(getattr(box_metrics, "mr", 0.0))

    summary = {
        "experiment_name": experiment_name,
        "dataset_yaml": str(Path(dataset_yaml).resolve()),
        "weights_path": str(Path(weights_path).resolve()),
        "split": split,
        "map": map5095,
        "map50": map50,
        "map75": map75,
        "precision": precision,
        "recall": recall,
        "raw_results": metrics_dict,
    }
    return summary


def _per_class_metrics_dataframe(results) -> pd.DataFrame:
    box_metrics = getattr(results, "box", None)
    names = getattr(results, "names", None) or {}
    raw_maps = getattr(box_metrics, "maps", None) if box_metrics is not None else None
    if raw_maps is None:
        return pd.DataFrame()
    maps = np.asarray(raw_maps, dtype=float).reshape(-1)
    if maps.size == 0:
        return pd.DataFrame()

    rows = []
    for idx, class_map in enumerate(maps):
        rows.append(
            {
                "class_idx": idx,
                "class_label": names[idx] if isinstance(names, dict) and idx in names else str(idx),
                "map": float(class_map),
            }
        )
    return pd.DataFrame(rows)


def _eval_class_support_dataframe(results) -> pd.DataFrame:
    names = getattr(results, "names", None) or {}
    nt_per_class = getattr(results, "nt_per_class", None)
    if nt_per_class is None:
        return pd.DataFrame()

    support = np.asarray(nt_per_class, dtype=int).reshape(-1)
    rows = []
    for idx, count in enumerate(support):
        rows.append(
            {
                "class_idx": idx,
                "class_label": names[idx] if isinstance(names, dict) and idx in names else str(idx),
                "n_eval_instances": int(count),
            }
        )
    return pd.DataFrame(rows)


def _render_confusion_matrix_png(
    *,
    matrix: np.ndarray,
    names: list[str],
    normalize: bool,
    output_path: Path,
) -> None:
    array = matrix.astype(float)
    if normalize:
        array = array / (array.sum(0, keepdims=True) + 1e-9)
    array[array < 0.005] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=(12, 9))
    ticklabels = [*names, "background"]
    nc = len(ticklabels)
    xy_ticks = np.arange(nc)
    tick_fontsize = max(6, 15 - 0.1 * nc)
    label_fontsize = max(6, 12 - 0.1 * nc)
    title_fontsize = max(6, 12 - 0.1 * nc)
    bottom = max(0.1, 0.25 - 0.001 * nc)

    with np.errstate(all="ignore"):
        vmax = 1.0 if normalize else np.nanmax(array)
    im = ax.imshow(array, cmap="Blues", vmin=0.0, vmax=vmax if np.isfinite(vmax) else None, interpolation="none")
    ax.xaxis.set_label_position("bottom")
    if nc < 30:
        color_threshold = 0.45 * (1.0 if normalize else (np.nanmax(array) if np.isfinite(np.nanmax(array)) else 1.0))
        for i, row in enumerate(array):
            for j, val in enumerate(row):
                if np.isnan(val):
                    continue
                ax.text(
                    j,
                    i,
                    f"{val:.2f}" if normalize else f"{int(val)}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color="white" if val > color_threshold else "black",
                )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.05)
    title = "Confusion Matrix" + " Normalized" * normalize
    ax.set_xlabel("True", fontsize=label_fontsize, labelpad=10)
    ax.set_ylabel("Predicted", fontsize=label_fontsize, labelpad=10)
    ax.set_title(title, fontsize=title_fontsize, pad=20)
    ax.set_xticks(xy_ticks)
    ax.set_yticks(xy_ticks)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    ax.tick_params(axis="y", left=True, right=False, labelleft=True, labelright=False)
    ax.set_xticklabels(ticklabels, fontsize=tick_fontsize, rotation=90, ha="center")
    ax.set_yticklabels(ticklabels, fontsize=tick_fontsize)
    for spine_name in {"left", "right", "bottom", "top", "outline"}:
        if spine_name != "outline":
            ax.spines[spine_name].set_visible(False)
        cbar.ax.spines[spine_name].set_visible(False)
    fig.subplots_adjust(left=0, right=0.84, top=0.94, bottom=bottom)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=250)
    plt.close(fig)


def _save_filtered_confusion_matrix_plots(results, *, analysis_dir: Path) -> dict[str, Any]:
    confusion_matrix = getattr(results, "confusion_matrix", None)
    names = getattr(results, "names", None) or {}
    if confusion_matrix is None or not hasattr(confusion_matrix, "matrix"):
        return {"filtered": False, "reason": "missing_confusion_matrix"}

    matrix = np.asarray(confusion_matrix.matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        return {"filtered": False, "reason": "unexpected_confusion_matrix_shape"}

    nc = len(names)
    if matrix.shape[0] != nc + 1:
        return {"filtered": False, "reason": "name_count_mismatch"}

    class_block = matrix[:nc, :nc]
    background_row = matrix[nc, :nc]
    background_col = matrix[:nc, nc]
    non_empty_mask = (class_block.sum(axis=0) + class_block.sum(axis=1) + background_row + background_col) > 0
    keep_class_indices = np.flatnonzero(non_empty_mask)
    if len(keep_class_indices) == nc:
        return {
            "filtered": False,
            "reason": "all_classes_non_empty",
            "kept_class_indices": keep_class_indices.tolist(),
        }
    if len(keep_class_indices) == 0:
        return {"filtered": False, "reason": "no_non_empty_classes"}

    kept_names = [
        names[int(class_idx)] if isinstance(names, dict) and int(class_idx) in names else str(int(class_idx))
        for class_idx in keep_class_indices
    ]
    keep_with_background = np.concatenate([keep_class_indices, [nc]])
    filtered_matrix = matrix[np.ix_(keep_with_background, keep_with_background)]

    _render_confusion_matrix_png(
        matrix=filtered_matrix,
        names=kept_names,
        normalize=False,
        output_path=analysis_dir / "confusion_matrix.png",
    )
    _render_confusion_matrix_png(
        matrix=filtered_matrix,
        names=kept_names,
        normalize=True,
        output_path=analysis_dir / "confusion_matrix_normalized.png",
    )

    support_df = _eval_class_support_dataframe(results)
    if not support_df.empty:
        support_df["is_displayed_in_confusion_matrix"] = support_df["class_idx"].isin(keep_class_indices)
        support_df.to_csv(analysis_dir / "eval_class_support.csv", index=False)

    filtered_summary = {
        "filtered": True,
        "kept_class_indices": keep_class_indices.astype(int).tolist(),
        "kept_class_labels": kept_names,
        "dropped_class_indices": [idx for idx in range(nc) if idx not in set(keep_class_indices.tolist())],
        "output_files": [
            str((analysis_dir / "confusion_matrix.png").resolve()),
            str((analysis_dir / "confusion_matrix_normalized.png").resolve()),
        ],
    }
    _write_json(analysis_dir / "confusion_matrix_filtering.json", filtered_summary)
    return filtered_summary


def _save_resolved_config(cfg: YOLOExperimentConfig, path: Path) -> None:
    payload = {
        "data": vars(cfg.data),
        "model": vars(cfg.model),
        "training": vars(cfg.training),
        "baseline": vars(cfg.baseline),
        "evaluation": vars(cfg.evaluation),
        "output": vars(cfg.output),
        "launcher": vars(cfg.launcher),
        "device": cfg.device,
    }
    _write_json(path, payload)


def _resolve_repo_path(path_str: str | None) -> Optional[Path]:
    if path_str is None:
        return None
    path = Path(path_str)
    if not path.is_absolute():
        path = repo_root() / path
    return path.resolve()


def _require_dataset_yaml(path_str: str) -> str:
    path = _resolve_repo_path(path_str)
    if path is None or not path.exists():
        raise FileNotFoundError(
            f"YOLO dataset YAML not found: {path_str}. "
            "Run the YOLO export section in docs/notebooks/flir_slice_subset_study.ipynb first."
        )
    return str(path)


def _require_config_file(path_str: str, *, label: str = "config") -> str:
    path = _resolve_repo_path(path_str)
    if path is None or not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path_str}")
    return str(path)


def _prepare_output_dirs(cfg: YOLOExperimentConfig) -> tuple[Path, Path, Path]:
    run_dir = cfg.output.run_dir()
    checkpoint_dir = cfg.output.checkpoint_dir()
    analysis_dir = cfg.output.analysis_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    return run_dir, checkpoint_dir, analysis_dir


def _normalize_ultralytics_device(device: str | None) -> str:
    """Convert torch-style device strings into Ultralytics trainer values."""
    if device is None:
        return "0" if torch.cuda.is_available() else "cpu"

    normalized = str(device).strip()
    if not normalized:
        return "0" if torch.cuda.is_available() else "cpu"

    lowered = normalized.lower()
    if lowered == "cuda":
        return "0"
    if lowered in {"cpu", "mps"}:
        return lowered
    if lowered.startswith("cuda:"):
        return lowered.split(":", 1)[1]
    if "," in lowered:
        parts = []
        for part in lowered.split(","):
            part = part.strip()
            if part.startswith("cuda:"):
                part = part.split(":", 1)[1]
            parts.append(part)
        return ",".join(parts)
    return normalized


def _localize_device_for_process(device: str | None) -> str | None:
    """Pin the current process to the requested physical CUDA device(s).

    Ultralytics often reports device indices relative to visible devices.
    For example, requesting ``cuda:3`` will set ``CUDA_VISIBLE_DEVICES=3``
    and return ``0`` so the process uses physical GPU 3 as local GPU 0.
    """
    if device is None:
        return None

    normalized = _normalize_ultralytics_device(device)
    if normalized in {"cpu", "mps"}:
        return normalized

    if not torch.cuda.is_available():
        return normalized

    if "," in normalized:
        visible_devices = [part.strip() for part in normalized.split(",") if part.strip()]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(visible_devices)
        return ",".join(str(idx) for idx in range(len(visible_devices)))

    if normalized.isdigit():
        os.environ["CUDA_VISIBLE_DEVICES"] = normalized
        return "0"

    return normalized


def _activate_torch_device(device: str | None) -> None:
    """Set torch's current CUDA device when applicable."""
    if device is None or not torch.cuda.is_available():
        return
    if device.isdigit():
        torch.cuda.set_device(int(device))


def _normalized_freeze_layers_spec(freeze_layers: Any) -> int | list[int] | None:
    if freeze_layers is None:
        return None
    if isinstance(freeze_layers, int):
        return freeze_layers
    if isinstance(freeze_layers, list):
        if not all(isinstance(layer_idx, int) for layer_idx in freeze_layers):
            raise TypeError("training.freeze_backbone_layers must contain only integers when provided as a list.")
        return freeze_layers
    raise TypeError("training.freeze_backbone_layers must be null, an int, or a list of ints.")


def _derive_freeze_layers_from_prefixes(prefixes: list[str]) -> list[int]:
    layer_indices: set[int] = set()
    for prefix in prefixes:
        match = re.match(r"^model\.(\d+)\.", prefix)
        if match:
            layer_indices.add(int(match.group(1)))
    return sorted(layer_indices)


def _validate_training_schedule(cfg: YOLOExperimentConfig) -> None:
    total_epochs = int(cfg.training.epochs)
    freeze_epochs = int(cfg.training.freeze_backbone_epochs)
    backbone_lr_multiplier = float(cfg.training.backbone_lr_multiplier)
    prefixes = list(cfg.training.backbone_param_prefixes)

    if total_epochs <= 0:
        raise ValueError("training.epochs must be positive.")
    if freeze_epochs < 0:
        raise ValueError("training.freeze_backbone_epochs must be >= 0.")
    if freeze_epochs > total_epochs:
        raise ValueError("training.freeze_backbone_epochs cannot exceed training.epochs.")
    if backbone_lr_multiplier <= 0:
        raise ValueError("training.backbone_lr_multiplier must be > 0.")

    freeze_layers = _normalized_freeze_layers_spec(cfg.training.freeze_backbone_layers)
    if freeze_epochs > 0 and freeze_layers is None:
        derived_layers = _derive_freeze_layers_from_prefixes(prefixes)
        if not derived_layers:
            raise ValueError(
                "training.freeze_backbone_epochs requires training.freeze_backbone_layers, "
                "or backbone_param_prefixes that can be converted into Ultralytics layer indices."
            )
        cfg.training.freeze_backbone_layers = derived_layers
    elif freeze_layers is not None:
        cfg.training.freeze_backbone_layers = freeze_layers

    if backbone_lr_multiplier != 1.0 and not prefixes:
        raise ValueError(
            "training.backbone_lr_multiplier != 1.0 requires explicit training.backbone_param_prefixes so the "
            "optimizer can separate backbone and head parameters without guessing."
        )


def _validate_baseline_config(cfg: YOLOExperimentConfig) -> None:
    mode = str(cfg.baseline.mode)
    if mode not in {"none", "baseline_a", "baseline_b"}:
        raise ValueError("baseline.mode must be one of: none, baseline_a, baseline_b.")
    if float(cfg.baseline.rarity_alpha) <= 0:
        raise ValueError("baseline.rarity_alpha must be > 0.")
    if float(cfg.baseline.rarity_eps) <= 0:
        raise ValueError("baseline.rarity_eps must be > 0.")
    if int(cfg.baseline.image_score_top_k) <= 0:
        raise ValueError("baseline.image_score_top_k must be >= 1.")
    if cfg.baseline.clip_weight_min is not None and float(cfg.baseline.clip_weight_min) <= 0:
        raise ValueError("baseline.clip_weight_min must be > 0 when provided.")
    if cfg.baseline.clip_weight_max is not None and float(cfg.baseline.clip_weight_max) <= 0:
        raise ValueError("baseline.clip_weight_max must be > 0 when provided.")
    if (
        cfg.baseline.clip_weight_min is not None
        and cfg.baseline.clip_weight_max is not None
        and float(cfg.baseline.clip_weight_min) > float(cfg.baseline.clip_weight_max)
    ):
        raise ValueError("baseline.clip_weight_min cannot exceed baseline.clip_weight_max.")
    if not 0.0 <= float(cfg.baseline.targeted_aug_probability) <= 1.0:
        raise ValueError("baseline.targeted_aug_probability must be in [0, 1].")
    if not 0.0 <= float(cfg.baseline.targeted_aug_rarity_quantile) <= 1.0:
        raise ValueError("baseline.targeted_aug_rarity_quantile must be in [0, 1].")
    if float(cfg.baseline.translate_fraction) < 0.0:
        raise ValueError("baseline.translate_fraction must be >= 0.")
    if float(cfg.baseline.scale_min) <= 0.0 or float(cfg.baseline.scale_max) <= 0.0:
        raise ValueError("baseline.scale_min and baseline.scale_max must be > 0.")
    if float(cfg.baseline.scale_min) > float(cfg.baseline.scale_max):
        raise ValueError("baseline.scale_min cannot exceed baseline.scale_max.")
    if float(cfg.baseline.crop_scale_min) <= 0.0 or float(cfg.baseline.crop_scale_max) <= 0.0:
        raise ValueError("baseline.crop_scale_min and baseline.crop_scale_max must be > 0.")
    if float(cfg.baseline.crop_scale_min) > float(cfg.baseline.crop_scale_max):
        raise ValueError("baseline.crop_scale_min cannot exceed baseline.crop_scale_max.")
    if float(cfg.baseline.crop_scale_max) > 1.0:
        raise ValueError("baseline.crop_scale_max must be <= 1.0.")
    if not 0.0 <= float(cfg.baseline.crop_min_rare_box_area_retained) <= 1.0:
        raise ValueError("baseline.crop_min_rare_box_area_retained must be in [0, 1].")
    if int(cfg.baseline.crop_max_attempts) <= 0:
        raise ValueError("baseline.crop_max_attempts must be >= 1.")


def _build_train_stages(cfg: YOLOExperimentConfig) -> list[dict[str, Any]]:
    _validate_training_schedule(cfg)
    total_epochs = int(cfg.training.epochs)
    freeze_epochs = int(cfg.training.freeze_backbone_epochs)
    freeze_layers = _normalized_freeze_layers_spec(cfg.training.freeze_backbone_layers)
    stages: list[dict[str, Any]] = []
    if freeze_epochs > 0:
        stages.append(
            {
                "stage_index": 1,
                "stage_name": "backbone_frozen",
                "epochs": freeze_epochs,
                "freeze": freeze_layers,
                "run_name": f"{cfg.output.experiment_name}_phase1_frozen",
            }
        )
    remaining_epochs = total_epochs - freeze_epochs
    if remaining_epochs > 0:
        stages.append(
            {
                "stage_index": len(stages) + 1,
                "stage_name": "full_finetune",
                "epochs": remaining_epochs,
                "freeze": None,
                "run_name": cfg.output.experiment_name,
            }
        )
    return stages


def _require_backbone_prefix_matches(torch_model: Any, prefixes: list[str]) -> list[str]:
    if not prefixes:
        return []
    param_names = [name for name, _ in torch_model.named_parameters()]
    matched = [prefix for prefix in prefixes if any(name.startswith(prefix) for name in param_names)]
    missing = [prefix for prefix in prefixes if prefix not in matched]
    if missing:
        raise ValueError(
            "Some training.backbone_param_prefixes did not match model parameters: "
            + ", ".join(missing)
        )
    return matched


def _make_differential_lr_detection_trainer(
    *,
    backbone_prefixes: list[str],
    backbone_lr_multiplier: float,
    cfg: YOLOExperimentConfig,
    prepared_baseline=None,
):
    from ultralytics.engine.trainer import MuSGD
    from ultralytics.data import build_dataloader, build_yolo_dataset
    from ultralytics.data.augment import Compose
    from ultralytics.data.build import InfiniteDataLoader, seed_worker
    from ultralytics.data.dataset import YOLODataset
    from ultralytics.models.yolo.detect import DetectionTrainer
    from ultralytics.utils import LOGGER, colorstr, RANK
    from ultralytics.utils.torch_utils import torch_distributed_zero_first, unwrap_model

    slice_aware_dataset_cls = make_slice_aware_yolo_dataset_class(YOLODataset, Compose)

    class DifferentialLRDetectionTrainer(DetectionTrainer):
        def build_dataset(self, img_path: str, mode: str = "train", batch: int | None = None):
            gs = max(int(unwrap_model(self.model).stride.max()), 32)
            if mode == "train" and prepared_baseline is not None and str(cfg.baseline.mode) in {"baseline_a", "baseline_b"}:
                return slice_aware_dataset_cls(
                    img_path=img_path,
                    imgsz=self.args.imgsz,
                    batch_size=batch,
                    augment=mode == "train",
                    hyp=self.args,
                    rect=self.args.rect or mode == "val",
                    cache=self.args.cache or None,
                    single_cls=self.args.single_cls or False,
                    stride=gs,
                    pad=0.0 if mode == "train" else 0.5,
                    prefix=colorstr(f"{mode}: "),
                    task=self.args.task,
                    classes=self.args.classes,
                    data=self.data,
                    fraction=self.args.fraction if mode == "train" else 1.0,
                    baseline_mode=str(cfg.baseline.mode),
                    prepared_baseline=prepared_baseline,
                    baseline_cfg=cfg.baseline,
                )
            return build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, rect=mode == "val", stride=gs)

        def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
            assert mode in {"train", "val"}, f"Mode must be 'train' or 'val', not {mode}."
            with torch_distributed_zero_first(rank):
                dataset = self.build_dataset(dataset_path, mode, batch_size)
            shuffle = mode == "train"
            if getattr(dataset, "rect", False) and shuffle and not np.all(dataset.batch_shapes == dataset.batch_shapes[0]):
                LOGGER.warning("'rect=True' is incompatible with DataLoader shuffle, setting shuffle=False")
                shuffle = False
            if mode == "train" and prepared_baseline is not None and str(cfg.baseline.mode) in {"baseline_a", "baseline_b"}:
                return build_weighted_train_dataloader(
                    dataset=dataset,
                    batch_size=batch_size,
                    workers=self.args.workers,
                    replacement=bool(cfg.baseline.sampler_replacement),
                    drop_last=bool(self.args.compile and mode == "train"),
                    pin_memory=True,
                    infinite_dataloader_cls=InfiniteDataLoader,
                    seed_worker_fn=seed_worker,
                    rank=rank,
                    rank_seed_offset=int(cfg.baseline.seed) + int(RANK),
                )
            return build_dataloader(
                dataset,
                batch=batch_size,
                workers=self.args.workers if mode == "train" else self.args.workers * 2,
                shuffle=shuffle,
                rank=rank,
                drop_last=self.args.compile and mode == "train",
            )

        def build_optimizer(self, model, name="auto", lr=0.001, momentum=0.9, decay=1e-5, iterations=1e5):
            if not backbone_prefixes or backbone_lr_multiplier == 1.0:
                return super().build_optimizer(
                    model=model,
                    name=name,
                    lr=lr,
                    momentum=momentum,
                    decay=decay,
                    iterations=iterations,
                )

            bn_types = tuple(module for module_name, module in nn.__dict__.items() if "Norm" in module_name)
            optimizers = {"Adam", "Adamax", "AdamW", "NAdam", "RAdam", "RMSProp", "SGD", "MuSGD", "auto"}
            if name == "auto":
                LOGGER.info(
                    f"{colorstr('optimizer:')} 'optimizer=auto' found, "
                    f"ignoring 'lr0={self.args.lr0}' and 'momentum={self.args.momentum}' and "
                    "determining best 'optimizer', 'lr0' and 'momentum' automatically... "
                )
                nc = self.data.get("nc", 10)
                lr_fit = round(0.002 * 5 / (4 + nc), 6)
                name, lr, momentum = ("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)
                self.args.warmup_bias_lr = 0.0

            name = {candidate.lower(): candidate for candidate in optimizers}.get(str(name).lower())
            if name is None:
                raise NotImplementedError(
                    f"Optimizer '{name}' not found in list of available optimizers {optimizers}."
                )
            if name == "MuSGD":
                raise ValueError(
                    "Differential backbone/head learning rates are not supported with optimizer=MuSGD. "
                    "Set training.optimizer explicitly to SGD, Adam, or AdamW."
                )

            if name in {"Adam", "Adamax", "AdamW", "NAdam", "RAdam"}:
                optim_args = dict(lr=lr, betas=(momentum, 0.999))
            elif name == "RMSProp":
                optim_args = dict(lr=lr, momentum=momentum)
            elif name == "SGD":
                optim_args = dict(lr=lr, momentum=momentum, nesterov=True)
            else:
                raise NotImplementedError(
                    f"Optimizer '{name}' not found in list of available optimizers {optimizers}."
                )

            group_buckets: dict[tuple[str, str], list[torch.nn.Parameter]] = {
                ("backbone", "weight"): [],
                ("backbone", "bn"): [],
                ("backbone", "bias"): [],
                ("head", "weight"): [],
                ("head", "bn"): [],
                ("head", "bias"): [],
            }

            for module_name, module in unwrap_model(model).named_modules():
                for param_name, param in module.named_parameters(recurse=False):
                    fullname = f"{module_name}.{param_name}" if module_name else param_name
                    part = "backbone" if any(fullname.startswith(prefix) for prefix in backbone_prefixes) else "head"
                    if "bias" in fullname:
                        bucket = "bias"
                    elif isinstance(module, bn_types) or "logit_scale" in fullname:
                        bucket = "bn"
                    else:
                        bucket = "weight"
                    group_buckets[(part, bucket)].append(param)

            if not any(group_buckets[("backbone", bucket)] for bucket in ("weight", "bn", "bias")):
                raise ValueError(
                    "training.backbone_param_prefixes matched zero optimizer parameters. "
                    "Update the prefixes to match the model parameter names."
                )

            param_groups: list[dict[str, Any]] = []
            group_counts: list[str] = []
            for part, lr_scale in (("backbone", backbone_lr_multiplier), ("head", 1.0)):
                group_lr = lr * lr_scale
                for bucket, bucket_decay in (("weight", decay), ("bn", 0.0), ("bias", 0.0)):
                    params = group_buckets[(part, bucket)]
                    if not params:
                        continue
                    group_counts.append(f"{part}:{bucket}={len(params)}@lr={group_lr:g}")
                    param_groups.append(
                        {
                            "params": params,
                            "lr": group_lr,
                            "weight_decay": bucket_decay,
                        }
                    )

            optimizer_ctor = getattr(optim, name, partial(MuSGD, muon=0.2, sgd=1.0))
            optimizer = optimizer_ctor(params=param_groups, **optim_args)
            LOGGER.info(
                f"{colorstr('optimizer:')} {type(optimizer).__name__} with differential LR groups "
                f"(backbone x{backbone_lr_multiplier:g}, head x1.0): " + ", ".join(group_counts)
            )
            return optimizer

    return DifferentialLRDetectionTrainer


def _stage_train_kwargs(
    cfg: YOLOExperimentConfig,
    *,
    dataset_yaml: str,
    device: str,
    stage: dict[str, Any],
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "data": dataset_yaml,
        "epochs": int(stage["epochs"]),
        "imgsz": cfg.data.image_size,
        "batch": cfg.data.batch_size,
        "workers": cfg.data.workers,
        "lr0": cfg.training.lr0,
        "optimizer": cfg.training.optimizer,
        "momentum": cfg.training.momentum,
        "weight_decay": cfg.training.weight_decay,
        "seed": cfg.training.seed,
        "deterministic": cfg.training.deterministic,
        "patience": cfg.training.patience,
        "cos_lr": cfg.training.cos_lr,
        "device": device,
        "project": str(Path(cfg.output.runs_root).resolve()),
        "name": stage["run_name"],
        "exist_ok": True,
        "val": True,
        "plots": True,
    }
    if stage["freeze"] is not None:
        kwargs["freeze"] = stage["freeze"]
    return kwargs


def run_train(cfg: YOLOExperimentConfig) -> dict[str, Any]:
    YOLO = _require_ultralytics()
    _set_seed(cfg.training.seed, cfg.training.deterministic)
    run_dir, checkpoint_dir, analysis_dir = _prepare_output_dirs(cfg)
    _save_resolved_config(cfg, analysis_dir / "resolved_config.json")
    _validate_baseline_config(cfg)

    dataset_yaml = _require_dataset_yaml(cfg.data.dataset_yaml)
    device = _normalize_ultralytics_device(cfg.resolved_device())
    stages = _build_train_stages(cfg)
    prepared_baseline = None
    baseline_artifact_paths: dict[str, Optional[str]] = {
        "slice_counts_csv": None,
        "slice_summary_json": None,
        "image_sampling_weights_csv": None,
        "sampling_weight_summary_json": None,
    }
    if str(cfg.baseline.mode) in {"baseline_a", "baseline_b"}:
        prepared_baseline = prepare_yolo_slice_baseline(
            dataset_yaml=dataset_yaml,
            analysis_dir=analysis_dir,
            baseline_cfg=cfg.baseline,
        )
        baseline_artifact_paths = {
            "slice_counts_csv": str((analysis_dir / "slice_counts.csv").resolve()),
            "slice_summary_json": str((analysis_dir / "slice_summary.json").resolve()),
            "image_sampling_weights_csv": str((analysis_dir / "image_sampling_weights.csv").resolve()),
            "sampling_weight_summary_json": str((analysis_dir / "sampling_weight_summary.json").resolve()),
        }
    _write_json(
        analysis_dir / "train_stage_plan.json",
        {
            "experiment_name": cfg.output.experiment_name,
            "dataset_yaml": dataset_yaml,
            "baseline_mode": cfg.baseline.mode,
            "baseline_artifact_paths": baseline_artifact_paths,
            "stages": stages,
        },
    )

    stage_summaries: list[dict[str, Any]] = []
    initial_model = YOLO(cfg.model.weights, task=cfg.model.task)
    _require_backbone_prefix_matches(initial_model.model, list(cfg.training.backbone_param_prefixes))

    current_weights = cfg.model.weights
    final_run_dir = run_dir
    final_best_dst: Optional[Path] = None
    final_last_dst: Optional[Path] = None

    for stage in stages:
        model = YOLO(current_weights, task=cfg.model.task)
        _require_backbone_prefix_matches(model.model, list(cfg.training.backbone_param_prefixes))
        trainer_cls = None
        if (
            cfg.training.backbone_lr_multiplier != 1.0
            and list(cfg.training.backbone_param_prefixes)
        ) or prepared_baseline is not None:
            trainer_cls = _make_differential_lr_detection_trainer(
                backbone_prefixes=list(cfg.training.backbone_param_prefixes),
                backbone_lr_multiplier=float(cfg.training.backbone_lr_multiplier),
                cfg=cfg,
                prepared_baseline=prepared_baseline,
            )

        train_kwargs = _stage_train_kwargs(cfg, dataset_yaml=dataset_yaml, device=device, stage=stage)
        _write_json(analysis_dir / f"train_stage_{stage['stage_index']}_requested_args.json", train_kwargs)
        model.train(trainer=trainer_cls, **train_kwargs)

        trainer = getattr(model, "trainer", None)
        resolved_run_dir = Path(getattr(trainer, "save_dir", run_dir))
        final_run_dir = resolved_run_dir
        best_src = resolved_run_dir / "weights" / "best.pt"
        last_src = resolved_run_dir / "weights" / "last.pt"
        stage_best_dst = _copy_if_exists(best_src, checkpoint_dir / f"stage_{stage['stage_index']}_best.pt")
        stage_last_dst = _copy_if_exists(last_src, checkpoint_dir / f"stage_{stage['stage_index']}_last.pt")

        if trainer is not None:
            _write_json(
                analysis_dir / f"train_stage_{stage['stage_index']}_trainer_args.json",
                dict(vars(trainer.args)),
            )

        stage_summary = {
            "stage_index": stage["stage_index"],
            "stage_name": stage["stage_name"],
            "epochs": stage["epochs"],
            "freeze": stage["freeze"],
            "run_name": stage["run_name"],
            "run_dir": str(resolved_run_dir.resolve()),
            "best_weights_path": str(stage_best_dst.resolve()) if stage_best_dst is not None else None,
            "last_weights_path": str(stage_last_dst.resolve()) if stage_last_dst is not None else None,
        }
        stage_summaries.append(stage_summary)

        if stage_best_dst is not None:
            current_weights = str(stage_best_dst)
        elif stage_last_dst is not None:
            current_weights = str(stage_last_dst)

        final_best_dst = stage_best_dst or final_best_dst
        final_last_dst = stage_last_dst or final_last_dst

    if final_best_dst is not None:
        final_best_dst = _copy_if_exists(final_best_dst, checkpoint_dir / "best.pt")
    if final_last_dst is not None:
        final_last_dst = _copy_if_exists(final_last_dst, checkpoint_dir / "last.pt")

    summary = {
        "experiment_name": cfg.output.experiment_name,
        "dataset_yaml": dataset_yaml,
        "baseline_mode": cfg.baseline.mode,
        "baseline_settings": vars(cfg.baseline),
        "baseline_artifact_paths": baseline_artifact_paths,
        "run_dir": str(final_run_dir.resolve()),
        "tensorboard_log_dir": str(final_run_dir.resolve()),
        "best_weights_path": str(final_best_dst.resolve()) if final_best_dst is not None else None,
        "last_weights_path": str(final_last_dst.resolve()) if final_last_dst is not None else None,
        "stages": stage_summaries,
    }
    _write_json(analysis_dir / "train_summary.json", summary)
    _write_csv_row(analysis_dir / "train_summary.csv", summary)
    return summary


def run_eval(cfg: YOLOExperimentConfig, *, weights_path: Optional[str] = None) -> dict[str, Any]:
    YOLO = _require_ultralytics()
    _set_seed(cfg.training.seed, cfg.training.deterministic)
    _, checkpoint_dir, analysis_dir = _prepare_output_dirs(cfg)
    _save_resolved_config(cfg, analysis_dir / "resolved_config.json")

    resolved_weights = weights_path or str(checkpoint_dir / "best.pt")
    data_yaml = _require_dataset_yaml(cfg.data.test_dataset_yaml or cfg.data.dataset_yaml)
    device = _normalize_ultralytics_device(cfg.resolved_device())

    model = YOLO(resolved_weights, task=cfg.model.task)
    val_kwargs: dict[str, Any] = {
        "data": data_yaml,
        "split": cfg.evaluation.split,
        "imgsz": cfg.data.image_size,
        "batch": cfg.data.batch_size,
        "workers": cfg.data.workers,
        "device": device,
        "project": str(analysis_dir.resolve()),
        "name": f"{cfg.output.experiment_name}_{cfg.evaluation.split}",
        "exist_ok": True,
        "save_json": cfg.evaluation.save_json,
        "save_hybrid": cfg.evaluation.save_hybrid,
    }
    if cfg.evaluation.conf is not None:
        val_kwargs["conf"] = cfg.evaluation.conf
    if cfg.evaluation.iou is not None:
        val_kwargs["iou"] = cfg.evaluation.iou

    results = model.val(**val_kwargs)
    summary = _summarize_validation_results(
        results,
        experiment_name=cfg.output.experiment_name,
        dataset_yaml=data_yaml,
        split=cfg.evaluation.split,
        weights_path=resolved_weights,
    )
    summary["baseline_mode"] = cfg.baseline.mode
    summary["confusion_matrix_filtering"] = _save_filtered_confusion_matrix_plots(
        results,
        analysis_dir=analysis_dir,
    )
    _write_json(analysis_dir / "eval_summary.json", summary)
    _write_csv_row(analysis_dir / "eval_summary.csv", {
        key: value
        for key, value in summary.items()
        if key not in {"raw_results", "confusion_matrix_filtering"}
    })

    per_class_df = _per_class_metrics_dataframe(results)
    if not per_class_df.empty:
        per_class_df.to_csv(analysis_dir / "per_class_metrics.csv", index=False)
    return summary


def _reuse_existing_train_summary(cfg: YOLOExperimentConfig) -> Optional[dict[str, Any]]:
    checkpoint_dir = cfg.output.checkpoint_dir()
    analysis_dir = cfg.output.analysis_dir()
    best_weights = checkpoint_dir / "best.pt"
    summary_path = analysis_dir / "train_summary.json"
    if best_weights.exists() and summary_path.exists():
        return _read_json(summary_path)
    return None


def _reuse_existing_eval_summary(cfg: YOLOExperimentConfig) -> Optional[dict[str, Any]]:
    summary_path = cfg.output.analysis_dir() / "eval_summary.json"
    if summary_path.exists():
        return _read_json(summary_path)
    return None


def _load_cli_config(
    *,
    config_path: str,
    parser: argparse.ArgumentParser,
    device: str | None,
) -> YOLOExperimentConfig:
    _require_config_file(config_path, label="YOLO config")
    _validate_config_yaml_keys(config_path)
    cli_args = ["--device", device] if device is not None else []
    parsed_args = parser.parse_args(cli_args)
    cfg = merge_config_and_cli(
        YOLOExperimentConfig,
        config_path,
        parser,
        parsed_args,
        flat_to_nested=_FLAT_TO_NESTED,
    )
    cfg.device = device
    return cfg


def _ordered_experiment_entries(cfg: YOLOExperimentConfig) -> list[tuple[str, str]]:
    config_paths = list(cfg.launcher.ordered_config_paths)
    labels = list(cfg.launcher.ordered_labels)

    if not config_paths:
        raise ValueError(
            "launcher.ordered_config_paths must be set for --action run_exp_a_all. "
            "Point it to the per-experiment YAML files, not dataset YAML files."
        )

    if labels and len(labels) != len(config_paths):
        raise ValueError("launcher.ordered_labels must have the same length as launcher.ordered_config_paths.")
    if not labels:
        labels = [Path(path).stem for path in config_paths]

    return list(zip(labels, config_paths))


def run_experiment_a_all(
    cfg: YOLOExperimentConfig,
    *,
    parser: argparse.ArgumentParser,
) -> dict[str, Any]:
    ordered_entries = _ordered_experiment_entries(cfg)
    comparison_dir = Path(cfg.output.analysis_root) / "experiment_a"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    ordered_runs: list[dict[str, Any]] = []
    for label, config_path in ordered_entries:
        resolved_config_path = _require_config_file(config_path, label=f"{label} experiment config")
        child_cfg = _load_cli_config(
            config_path=resolved_config_path,
            parser=parser,
            device=cfg.device,
        )
        train_summary = _reuse_existing_train_summary(child_cfg) or run_train(child_cfg)
        eval_summary = _reuse_existing_eval_summary(child_cfg) or run_eval(
            child_cfg,
            weights_path=train_summary["best_weights_path"],
        )
        ordered_runs.append(
            {
                "label": label,
                "config_path": str(Path(resolved_config_path).resolve()),
                "experiment_name": child_cfg.output.experiment_name,
                "dataset_yaml": child_cfg.data.dataset_yaml,
                "baseline_mode": child_cfg.baseline.mode,
                "weights_path": train_summary["best_weights_path"],
                "tensorboard_log_dir": train_summary["tensorboard_log_dir"],
                "slice_counts_csv": train_summary.get("baseline_artifact_paths", {}).get("slice_counts_csv"),
                "image_sampling_weights_csv": train_summary.get("baseline_artifact_paths", {}).get("image_sampling_weights_csv"),
                "map": eval_summary["map"],
                "map50": eval_summary["map50"],
                "map75": eval_summary["map75"],
                "precision": eval_summary["precision"],
                "recall": eval_summary["recall"],
            }
        )

    comparison_df = pd.DataFrame(ordered_runs)
    comparison_csv = comparison_dir / "comparison_summary_all.csv"
    comparison_json = comparison_dir / "comparison_summary_all.json"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(comparison_csv, index=False)
    comparison_json.write_text(comparison_df.to_json(orient="records", indent=2), encoding="utf-8")
    return {
        "ordered_runs": ordered_runs,
        "comparison_csv": str(comparison_csv.resolve()),
        "comparison_json": str(comparison_json.resolve()),
    }


def run_experiment_a(cfg: YOLOExperimentConfig) -> dict[str, Any]:
    analysis_root = Path(cfg.output.analysis_root)
    comparison_dir = analysis_root / "experiment_a"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    balanced_cfg = copy.deepcopy(cfg)
    balanced_cfg.data.dataset_yaml = cfg.data.balanced_dataset_yaml
    balanced_cfg.output.experiment_name = cfg.launcher.balanced_experiment_name

    unbalanced_cfg = copy.deepcopy(cfg)
    unbalanced_cfg.data.dataset_yaml = cfg.data.unbalanced_dataset_yaml
    unbalanced_cfg.output.experiment_name = cfg.launcher.unbalanced_experiment_name

    balanced_train = _reuse_existing_train_summary(balanced_cfg) or run_train(balanced_cfg)
    balanced_eval = _reuse_existing_eval_summary(balanced_cfg) or run_eval(
        balanced_cfg,
        weights_path=balanced_train["best_weights_path"],
    )

    unbalanced_train = _reuse_existing_train_summary(unbalanced_cfg) or run_train(unbalanced_cfg)
    unbalanced_eval = _reuse_existing_eval_summary(unbalanced_cfg) or run_eval(
        unbalanced_cfg,
        weights_path=unbalanced_train["best_weights_path"],
    )

    comparison_rows = [
        {
            "experiment_name": balanced_cfg.output.experiment_name,
            "dataset_yaml": balanced_cfg.data.dataset_yaml,
            "baseline_mode": balanced_cfg.baseline.mode,
            "weights_path": balanced_train["best_weights_path"],
            "tensorboard_log_dir": balanced_train["tensorboard_log_dir"],
            "map": balanced_eval["map"],
            "map50": balanced_eval["map50"],
            "map75": balanced_eval["map75"],
            "precision": balanced_eval["precision"],
            "recall": balanced_eval["recall"],
        },
        {
            "experiment_name": unbalanced_cfg.output.experiment_name,
            "dataset_yaml": unbalanced_cfg.data.dataset_yaml,
            "baseline_mode": unbalanced_cfg.baseline.mode,
            "weights_path": unbalanced_train["best_weights_path"],
            "tensorboard_log_dir": unbalanced_train["tensorboard_log_dir"],
            "map": unbalanced_eval["map"],
            "map50": unbalanced_eval["map50"],
            "map75": unbalanced_eval["map75"],
            "precision": unbalanced_eval["precision"],
            "recall": unbalanced_eval["recall"],
        },
    ]
    comparison_df = pd.DataFrame(comparison_rows)
    comparison_csv = comparison_dir / cfg.launcher.comparison_filename
    comparison_json = comparison_dir / "comparison_summary.json"
    comparison_df.to_csv(comparison_csv, index=False)
    comparison_json.write_text(comparison_df.to_json(orient="records", indent=2), encoding="utf-8")
    return {
        "comparison_csv": str(comparison_csv.resolve()),
        "comparison_json": str(comparison_json.resolve()),
    }


def main(argv: Optional[list[str]] = None) -> None:
    import sys

    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    args = parser.parse_args(effective_argv)
    _validate_config_yaml_keys(args.config)
    cfg = merge_config_and_cli(
        YOLOExperimentConfig,
        args.config,
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
        cli_argv=effective_argv,
    )
    cfg.device = _localize_device_for_process(cfg.device)
    _activate_torch_device(cfg.device)

    if args.action == "train":
        run_train(cfg)
    elif args.action == "eval":
        run_eval(cfg)
    elif args.action == "run_exp_a":
        run_experiment_a(cfg)
    elif args.action == "run_exp_a_all":
        run_experiment_a_all(cfg, parser=parser)
    else:
        raise ValueError(f"Unsupported action: {args.action}")


if __name__ == "__main__":
    main()

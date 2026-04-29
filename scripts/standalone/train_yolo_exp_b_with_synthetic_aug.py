#!/usr/bin/env python3
"""Run YOLO Experiment B using a precomputed synthetic dataset as augmentation."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.core.paths import repo_root


REPO_ROOT = repo_root()
DEFAULT_CONFIG = REPO_ROOT / "configs/yolo/exp_b/flir_yolov8n/exp_precomputed_regiondiff_fm_ot_hflip.yaml"
DEFAULT_SYNTHETIC_ROOT = REPO_ROOT / "artifacts/generated/yolo/exp_b/precomputed_candidates"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO with one generated Experiment-B synthetic dataset as augmentation."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="YOLO Experiment-B config to use as the base.")
    parser.add_argument(
        "--synthetic-dataset-dir",
        default="",
        help=(
            "Generated dataset folder containing images/ and filtered annotations.json. "
            "If omitted, the config's experiment_b.precomputed_dataset_dir is used."
        ),
    )
    parser.add_argument(
        "--generator-name",
        default="",
        help=(
            "Convenience name under artifacts/generated/yolo/exp_b/precomputed_candidates, "
            "for example stay_fm_hflip or regiondiff_fm_ot_hflip."
        ),
    )
    parser.add_argument("--output-experiment-name", default="", help="Optional override for output.experiment_name.")
    parser.add_argument("--augmented-yolo-root", default="", help="Optional override for experiment_b.augmented_yolo_root.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--python", default=sys.executable, help="Python interpreter used to launch src.cli.train_yolo.")
    parser.add_argument(
        "--no-validate-synthetic-dir",
        action="store_true",
        help="Skip checking that the generated dataset has images/ and filtered annotations.json before training.",
    )
    return parser.parse_args()


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved.resolve()


def _precomputed_dir_from_config(config_path: str | Path) -> str | None:
    path = _resolve_path(config_path)
    if not path.is_file():
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    experiment_b = payload.get("experiment_b", {}) if isinstance(payload, dict) else {}
    value = experiment_b.get("precomputed_dataset_dir") if isinstance(experiment_b, dict) else None
    return None if value in (None, "") else str(_resolve_path(str(value)))


def _resolve_synthetic_dir(args: argparse.Namespace) -> str | None:
    if args.synthetic_dataset_dir:
        path = _resolve_path(args.synthetic_dataset_dir)
    elif args.generator_name:
        path = _resolve_path(DEFAULT_SYNTHETIC_ROOT / args.generator_name)
    else:
        return _precomputed_dir_from_config(args.config)
    return str(path)


def _validate_synthetic_dir(path_str: str) -> None:
    path = Path(path_str)
    missing: list[str] = []
    if not path.is_dir():
        missing.append(str(path))
    if not (path / "images").is_dir():
        missing.append(str(path / "images"))
    if not (path / "annotations.json").is_file():
        missing.append(str(path / "annotations.json"))
    if missing:
        raise FileNotFoundError(
            "Synthetic augmentation dataset is not ready. Missing: " + ", ".join(missing)
        )


def main() -> None:
    args = parse_args()
    synthetic_dir = _resolve_synthetic_dir(args)
    if synthetic_dir is not None and not args.no_validate_synthetic_dir:
        _validate_synthetic_dir(synthetic_dir)

    cmd = [
        args.python,
        "-m",
        "src.cli.train_yolo",
        "--action",
        "run_exp_b",
        "--config",
        args.config,
        "--experiment_b_mode",
        "precomputed_aug",
        "--experiment_b_filter_enabled",
        "false",
    ]
    if synthetic_dir is not None:
        cmd.extend(["--experiment_b_precomputed_dataset_dir", synthetic_dir])
    if args.output_experiment_name:
        cmd.extend(["--experiment_name", args.output_experiment_name])
    if args.augmented_yolo_root:
        cmd.extend(["--experiment_b_augmented_yolo_root", args.augmented_yolo_root])
    if args.device:
        cmd.extend(["--device", args.device])

    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()

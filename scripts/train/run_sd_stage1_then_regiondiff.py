#!/usr/bin/env python3
"""Run SD Stage-1 training followed by RegionDiff Stage-2 training.

The launcher is intentionally thin around the existing training CLIs.  Its job
is orchestration: skip completed stages, resume interrupted stages from their
latest checkpoints, and keep the Slurm wrappers free of training logic.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.diffusers_compat import disable_diffusers_optional_scipy

disable_diffusers_optional_scipy(lightweight_diffusers_imports=False)

from src.algorithms.stable_diffusion.config import parse_args as parse_stage1_config
from src.algorithms.stable_diffusion.layout_models import (
    STAGE2_MANIFEST_NAME,
    STAGE2_UNET_WEIGHTS,
)
from src.algorithms.stable_diffusion.models import (
    LORA_WEIGHT_FILENAMES,
    STAGE1_MANIFEST_NAME,
    UNET_EXPORT_DIRNAME,
    get_canonical_output_dir,
)
from src.core.configs.sd_layout_config import parse_args as parse_stage2_config


CHAIN_MARKER_NAME = "stage_chain_complete.json"


def _resolve_path(path: str | Path, *, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate
    return root / candidate


def _latest_checkpoint(output_dir: Path) -> Path | None:
    if not output_dir.is_dir():
        return None
    checkpoints: list[Path] = []
    for path in output_dir.iterdir():
        if not path.is_dir() or not path.name.startswith("checkpoint-"):
            continue
        try:
            int(path.name.split("-", 1)[1])
        except ValueError:
            continue
        checkpoints.append(path)
    return max(checkpoints, key=lambda path: int(path.name.split("-", 1)[1]), default=None)


def _stage1_final_paths(stage1_config, output_dir: Path) -> list[Path]:
    paths = [output_dir / STAGE1_MANIFEST_NAME]
    if stage1_config.baseline_mode == "sd_ir_lora":
        paths.append(output_dir / LORA_WEIGHT_FILENAMES[0])
    elif stage1_config.baseline_mode == "sd_ir_unet":
        paths.append(output_dir / UNET_EXPORT_DIRNAME / "diffusion_pytorch_model.safetensors")
    else:
        raise ValueError(f"Unsupported Stage-1 baseline_mode={stage1_config.baseline_mode!r}")
    return paths


def _stage2_final_paths(output_dir: Path) -> list[Path]:
    return [
        output_dir / STAGE2_UNET_WEIGHTS,
        output_dir / STAGE2_MANIFEST_NAME,
    ]


def _all_exist(paths: Sequence[Path]) -> bool:
    return all(path.is_file() for path in paths)


def _command_with_resume(command: list[str], *, output_dir: Path) -> list[str]:
    if _latest_checkpoint(output_dir) is None:
        return command
    return [*command, "--resume_from_checkpoint", "latest"]


def _run(command: Sequence[str], *, root: Path, dry_run: bool) -> None:
    print(f"[chain] {' '.join(command)}", flush=True)
    if dry_run:
        return
    subprocess.run(list(command), cwd=str(root), check=True)


def _write_chain_marker(
    *,
    output_dir: Path,
    stage1_config_path: Path,
    stage2_config_path: Path,
    stage1_output_dir: Path,
    root: Path,
    dry_run: bool,
) -> None:
    if dry_run:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "completed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stage1_config": str(stage1_config_path),
        "stage2_config": str(stage2_config_path),
        "stage1_output_dir": str(stage1_output_dir),
        "stage2_output_dir": str(output_dir),
    }
    marker = output_dir / CHAIN_MARKER_NAME
    marker.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[chain] Wrote completion marker: {marker.relative_to(root) if marker.is_relative_to(root) else marker}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SD Stage-1 training, then RegionDiff Stage-2 training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--stage1-config", required=True, help="Stage-1 SD training YAML.")
    parser.add_argument("--stage2-config", required=True, help="RegionDiff Stage-2 training YAML.")
    parser.add_argument(
        "--mixed-precision",
        default=None,
        choices=("no", "fp16", "bf16"),
        help="Optional mixed precision override passed to both training CLIs.",
    )
    parser.add_argument(
        "--project-root",
        default=str(ROOT),
        help="Repository root used as subprocess working directory.",
    )
    parser.add_argument(
        "--stage1-num-workers",
        type=int,
        default=None,
        help="Optional Stage-1 dataloader worker override.",
    )
    parser.add_argument(
        "--stage2-num-workers",
        type=int,
        default=None,
        help="Optional Stage-2 dataloader worker override.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands that would run without launching training.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    stage1_config_path = _resolve_path(args.stage1_config, root=root)
    stage2_config_path = _resolve_path(args.stage2_config, root=root)

    if not stage1_config_path.is_file():
        raise FileNotFoundError(f"Stage-1 config not found: {stage1_config_path}")
    if not stage2_config_path.is_file():
        raise FileNotFoundError(f"Stage-2 config not found: {stage2_config_path}")

    stage1_config = parse_stage1_config(["--config", str(stage1_config_path)])
    stage1_config.output_dir = get_canonical_output_dir(stage1_config)
    stage2_config = parse_stage2_config(["--config", str(stage2_config_path)])

    stage1_output_dir = _resolve_path(stage1_config.output_dir, root=root)
    stage2_output_dir = _resolve_path(stage2_config.output.output_dir, root=root)
    stage1_final_paths = _stage1_final_paths(stage1_config, stage1_output_dir)
    stage2_final_paths = _stage2_final_paths(stage2_output_dir)

    print(f"[chain] Project root: {root}")
    print(f"[chain] Stage-1 output: {stage1_output_dir}")
    print(f"[chain] Stage-2 output: {stage2_output_dir}")

    stage1_command = [
        sys.executable,
        "-m",
        "src.cli.adapt_stable_diffusion",
        "--stage",
        "stage1",
        "--config",
        str(stage1_config_path),
    ]
    stage2_command = [
        sys.executable,
        "-m",
        "src.cli.adapt_stable_diffusion",
        "--stage",
        "regiondiff-stage2",
        "--config",
        str(stage2_config_path),
    ]
    if args.mixed_precision is not None:
        stage1_command.extend(["--mixed_precision", args.mixed_precision])
        stage2_command.extend(["--mixed_precision", args.mixed_precision])
    if args.stage1_num_workers is not None:
        stage1_command.extend(["--dataloader_num_workers", str(args.stage1_num_workers)])
    if args.stage2_num_workers is not None:
        stage2_command.extend(["--num_workers", str(args.stage2_num_workers)])

    if _all_exist(stage1_final_paths):
        print("[chain] Stage-1 final artifacts found; skipping Stage-1.")
    else:
        stage1_command = _command_with_resume(stage1_command, output_dir=stage1_output_dir)
        _run(stage1_command, root=root, dry_run=args.dry_run)
        if not args.dry_run and not _all_exist(stage1_final_paths):
            missing = [str(path) for path in stage1_final_paths if not path.is_file()]
            raise RuntimeError(f"Stage-1 finished but expected artifact(s) are missing: {missing}")

    if not args.dry_run and not _all_exist(stage1_final_paths):
        missing = [str(path) for path in stage1_final_paths if not path.is_file()]
        raise RuntimeError(f"Cannot start Stage-2; Stage-1 artifact(s) are missing: {missing}")

    if _all_exist(stage2_final_paths):
        print("[chain] Stage-2 final artifacts found; skipping Stage-2.")
    else:
        stage2_command = _command_with_resume(stage2_command, output_dir=stage2_output_dir)
        _run(stage2_command, root=root, dry_run=args.dry_run)
        if not args.dry_run and not _all_exist(stage2_final_paths):
            missing = [str(path) for path in stage2_final_paths if not path.is_file()]
            raise RuntimeError(f"Stage-2 finished but expected artifact(s) are missing: {missing}")

    _write_chain_marker(
        output_dir=stage2_output_dir,
        stage1_config_path=stage1_config_path,
        stage2_config_path=stage2_config_path,
        stage1_output_dir=stage1_output_dir,
        root=root,
        dry_run=args.dry_run,
    )
    print("[chain] Stage-1 -> RegionDiff chain complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

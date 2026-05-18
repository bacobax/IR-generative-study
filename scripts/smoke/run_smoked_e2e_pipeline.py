#!/usr/bin/env python3
"""Run the tiny smoked end-to-end training/generation/YOLO pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[2]
SMOKE_ROOT = ROOT / "artifacts" / "smoked_e2e"
SUMMARY_DIR = SMOKE_ROOT / "smoked_summary"
SUMMARY_JSON = SUMMARY_DIR / "smoked_e2e_summary.json"
SUMMARY_LOG = SUMMARY_DIR / "smoked_e2e.log"
SUMMARY_TXT = SUMMARY_DIR / "smoked_e2e_summary.txt"
SUMMARY_CSV = SUMMARY_DIR / "smoked_e2e_summary.csv"


@dataclass(frozen=True)
class SmokeStage:
    name: str
    key: str
    command: list[str]
    expected_paths: list[str] = field(default_factory=list)
    deps: list[str] = field(default_factory=list)
    output_dir: str = ""
    checkpoint_path: str = ""
    config_path: str = ""


def _py() -> str:
    return sys.executable


def _stage_definitions(device: str) -> list[SmokeStage]:
    return [
        SmokeStage(
            name="smoked_fm_uncond",
            key="fm_uncond",
            config_path="configs/fm/train/presets/smoked_e2e/uncond_latent_fm.yaml",
            output_dir="artifacts/smoked_e2e/smoked_fm_uncond",
            checkpoint_path="artifacts/smoked_e2e/smoked_fm_uncond/UNET/unet_fm_epoch_1.pt",
            command=[_py(), "-m", "src.cli.train_flow_matching", "--config", "configs/fm/train/presets/smoked_e2e/uncond_latent_fm.yaml", "--device", device],
            expected_paths=["artifacts/smoked_e2e/smoked_fm_uncond/UNET/unet_fm_epoch_1.pt"],
        ),
        SmokeStage(
            name="smoked_dm_uncond",
            key="dm_uncond",
            config_path="configs/sd_uncond/train/presets/smoked_e2e/uncond_latent_dm.yaml",
            output_dir="artifacts/smoked_e2e/smoked_dm_uncond",
            checkpoint_path="artifacts/smoked_e2e/smoked_dm_uncond/UNET/unet_sd_uncond_epoch_1.pt",
            command=[_py(), "-m", "src.cli.train_latent_diffusion", "--config", "configs/sd_uncond/train/presets/smoked_e2e/uncond_latent_dm.yaml", "--device", device],
            expected_paths=["artifacts/smoked_e2e/smoked_dm_uncond/UNET/unet_sd_uncond_epoch_1.pt"],
        ),
        SmokeStage(
            name="smoked_sd15_finetune",
            key="sd15_finetune",
            config_path="configs/sd/train/presets/smoked_e2e/sd15_finetune.yaml",
            output_dir="artifacts/smoked_e2e/smoked_sd15_finetune",
            checkpoint_path="artifacts/smoked_e2e/smoked_sd15_finetune/checkpoint-1",
            command=[_py(), "-m", "src.cli.adapt_stable_diffusion", "--config", "configs/sd/train/presets/smoked_e2e/sd15_finetune.yaml"],
            expected_paths=["artifacts/smoked_e2e/smoked_sd15_finetune/checkpoint-1", "artifacts/smoked_e2e/smoked_sd15_finetune/unet"],
        ),
        SmokeStage(
            name="smoked_sd15_lora",
            key="sd15_lora",
            config_path="configs/sd/train/presets/smoked_e2e/sd15_lora.yaml",
            output_dir="artifacts/smoked_e2e/smoked_sd15_lora",
            checkpoint_path="artifacts/smoked_e2e/smoked_sd15_lora/checkpoint-1",
            command=[_py(), "-m", "src.cli.adapt_stable_diffusion", "--config", "configs/sd/train/presets/smoked_e2e/sd15_lora.yaml"],
            expected_paths=["artifacts/smoked_e2e/smoked_sd15_lora/checkpoint-1", "artifacts/smoked_e2e/smoked_sd15_lora/pytorch_lora_weights.safetensors"],
        ),
        SmokeStage(
            name="smoked_regiondiff_fm",
            key="regiondiff_fm",
            deps=["fm_uncond"],
            config_path="configs/fm/train/presets/smoked_e2e/regiondiff_from_fm.yaml",
            output_dir="artifacts/smoked_e2e/smoked_regiondiff_fm",
            checkpoint_path="artifacts/smoked_e2e/smoked_regiondiff_fm/UNET/unet_fm_epoch_1.pt",
            command=[_py(), "-m", "src.cli.train_flow_matching", "--config", "configs/fm/train/presets/smoked_e2e/regiondiff_from_fm.yaml", "--device", device],
            expected_paths=["artifacts/smoked_e2e/smoked_regiondiff_fm/UNET/unet_fm_epoch_1.pt", "artifacts/smoked_e2e/smoked_regiondiff_fm/regiondiff_config.json"],
        ),
        SmokeStage(
            name="smoked_regiondiff_dm",
            key="regiondiff_dm",
            deps=["dm_uncond"],
            config_path="configs/sd_uncond/train/presets/smoked_e2e/regiondiff_from_dm.yaml",
            output_dir="artifacts/smoked_e2e/smoked_regiondiff_dm",
            checkpoint_path="artifacts/smoked_e2e/smoked_regiondiff_dm/UNET/unet_sd_uncond_epoch_1.pt",
            command=[_py(), "-m", "src.cli.train_latent_diffusion", "--config", "configs/sd_uncond/train/presets/smoked_e2e/regiondiff_from_dm.yaml", "--device", device],
            expected_paths=["artifacts/smoked_e2e/smoked_regiondiff_dm/UNET/unet_sd_uncond_epoch_1.pt", "artifacts/smoked_e2e/smoked_regiondiff_dm/regiondiff_config.json"],
        ),
        SmokeStage(
            name="smoked_regiondiff_sd15_finetune",
            key="regiondiff_sd15_finetune",
            deps=["sd15_finetune"],
            config_path="configs/sd_layout/train/presets/smoked_e2e/regiondiff_from_sd15_finetune.yaml",
            output_dir="artifacts/smoked_e2e/smoked_regiondiff_sd15_finetune",
            checkpoint_path="artifacts/smoked_e2e/smoked_regiondiff_sd15_finetune/regiondiff_unet.safetensors",
            command=[_py(), "-m", "src.cli.adapt_stable_diffusion", "--config", "configs/sd_layout/train/presets/smoked_e2e/regiondiff_from_sd15_finetune.yaml"],
            expected_paths=["artifacts/smoked_e2e/smoked_regiondiff_sd15_finetune/regiondiff_unet.safetensors", "artifacts/smoked_e2e/smoked_regiondiff_sd15_finetune/stage2_layout_manifest.json"],
        ),
        SmokeStage(
            name="smoked_regiondiff_sd15_lora",
            key="regiondiff_sd15_lora",
            deps=["sd15_lora"],
            config_path="configs/sd_layout/train/presets/smoked_e2e/regiondiff_from_sd15_lora.yaml",
            output_dir="artifacts/smoked_e2e/smoked_regiondiff_sd15_lora",
            checkpoint_path="artifacts/smoked_e2e/smoked_regiondiff_sd15_lora/regiondiff_unet.safetensors",
            command=[_py(), "-m", "src.cli.adapt_stable_diffusion", "--config", "configs/sd_layout/train/presets/smoked_e2e/regiondiff_from_sd15_lora.yaml"],
            expected_paths=["artifacts/smoked_e2e/smoked_regiondiff_sd15_lora/regiondiff_unet.safetensors", "artifacts/smoked_e2e/smoked_regiondiff_sd15_lora/stage2_layout_manifest.json"],
        ),
        _synthetic_stage("fm", "regiondiff_fm", device),
        _synthetic_stage("dm", "regiondiff_dm", device),
        _synthetic_stage("sd15_finetune", "regiondiff_sd15_finetune", device),
        _synthetic_stage("sd15_lora", "regiondiff_sd15_lora", device),
        _yolo_stage("real_full_train", "yolo_real_full_train", "real_full_train.yaml", deps=[]),
        _yolo_stage("fm_aug", "yolo_fm_aug", "precomputed_fm.yaml", deps=["synthetic_fm"]),
        _yolo_stage("dm_aug", "yolo_dm_aug", "precomputed_dm.yaml", deps=["synthetic_dm"]),
        _yolo_stage("sd15_finetune_aug", "yolo_sd15_finetune_aug", "precomputed_sd15_finetune.yaml", deps=["synthetic_sd15_finetune"]),
        _yolo_stage("sd15_lora_aug", "yolo_sd15_lora_aug", "precomputed_sd15_lora.yaml", deps=["synthetic_sd15_lora"]),
    ]


def _synthetic_stage(kind: str, dep: str, device: str) -> SmokeStage:
    artifact = {
        "fm": "artifacts/smoked_e2e/smoked_regiondiff_fm",
        "dm": "artifacts/smoked_e2e/smoked_regiondiff_dm",
        "sd15_finetune": "artifacts/smoked_e2e/smoked_regiondiff_sd15_finetune",
        "sd15_lora": "artifacts/smoked_e2e/smoked_regiondiff_sd15_lora",
    }[kind]
    output = f"artifacts/smoked_e2e/smoked_synthetic/{kind}"
    return SmokeStage(
        name=f"smoked_synthetic_{kind}",
        key=f"synthetic_{kind}",
        deps=[dep],
        output_dir=output,
        command=[
            _py(),
            "scripts/smoke/generate_smoked_regiondiff_dataset.py",
            "--model-kind",
            kind,
            "--artifact-dir",
            artifact,
            "--output-dir",
            output,
            "--max-samples",
            "2",
            "--batch-size",
            "1",
            "--steps",
            "2",
            "--device",
            device,
            "--precision",
            "fp32",
        ],
        expected_paths=[f"{output}/annotations.json", f"{output}/metadata/summary.json"],
    )


def _yolo_stage(label: str, key: str, config_name: str, *, deps: list[str]) -> SmokeStage:
    config_path = f"configs/yolo/exp_b/smoked_e2e/{config_name}"
    output = f"artifacts/smoked_e2e/smoked_yolo/{label}"
    return SmokeStage(
        name=f"smoked_yolo_{label}",
        key=key,
        deps=deps,
        config_path=config_path,
        output_dir=output,
        command=[_py(), "-m", "src.cli.train_yolo", "--action", "run_exp_b", "--config", config_path],
        expected_paths=[f"{output}/analysis/smoked_yolo_{label}/experiment_b_summary.json"],
    )


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _write_log(message: str) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    with SUMMARY_LOG.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _console(message: str) -> None:
    try:
        print(message, flush=True)
    except BrokenPipeError:
        pass


def _tail_text(text: str, *, max_lines: int = 40) -> str:
    lines = [line for line in text.rstrip().splitlines() if line.strip()]
    return "\n".join(lines[-max_lines:])


def _missing_paths(stage: SmokeStage) -> list[str]:
    return [path for path in stage.expected_paths if not (ROOT / path).exists()]


def _stage_row(stage: SmokeStage, *, status: str, started: float, ended: float, returncode: int | None, message: str) -> dict[str, object]:
    return {
        "stage_name": stage.name,
        "status_key": stage.key,
        "status": status,
        "returncode": returncode,
        "duration_sec": round(max(0.0, ended - started), 3),
        "dependency_keys": ",".join(stage.deps),
        "config_path": stage.config_path,
        "output_dir": stage.output_dir,
        "checkpoint_path": stage.checkpoint_path,
        "expected_paths": list(stage.expected_paths),
        "message": message,
        "command": " ".join(stage.command),
    }


def run_stage(stage: SmokeStage, *, dry_run: bool, completed: dict[str, dict[str, object]]) -> dict[str, object]:
    started = time.time()
    blocked = [] if dry_run else [dep for dep in stage.deps if completed.get(dep, {}).get("status") != "success"]
    if blocked:
        message = f"Skipped because dependencies did not succeed: {', '.join(blocked)}"
        row = _stage_row(stage, status="skipped", started=started, ended=time.time(), returncode=None, message=message)
        _write_log(f"[SKIP] {stage.name}: {message}")
        _console(f"[SKIP] {stage.name}: {message}")
        return row

    if dry_run:
        message = "Dry run; command was not executed."
        row = _stage_row(stage, status="dry_run", started=started, ended=time.time(), returncode=None, message=message)
        _write_log(f"[DRY] {stage.name}: {' '.join(stage.command)}")
        _console(f"[DRY] {stage.name}")
        return row

    _write_log(f"[START] {stage.name}: {' '.join(stage.command)}")
    _console(f"[START] {stage.name}")
    result = subprocess.run(
        stage.command,
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        _write_log(result.stdout)
    missing = _missing_paths(stage) if result.returncode == 0 else list(stage.expected_paths)
    if result.returncode != 0:
        status = "failed"
        message = f"Command exited with {result.returncode}."
    elif missing:
        status = "failed"
        message = f"Missing expected artifact(s): {', '.join(missing)}"
    else:
        status = "success"
        message = "Stage completed and expected artifacts exist."
    _write_log(f"[{status.upper()}] {stage.name}: {message}")
    _console(f"[{status.upper()}] {stage.name}: {message}")
    if status == "failed" and result.stdout:
        tail = _tail_text(result.stdout)
        if tail:
            _console(f"[{stage.name}] last log lines:\n{tail}")
    _console(f"[{stage.name}] full log: {_rel(SUMMARY_LOG)}")
    return _stage_row(
        stage,
        status=status,
        started=started,
        ended=time.time(),
        returncode=result.returncode,
        message=message,
    )


def write_summaries(rows: Sequence[dict[str, object]], *, dry_run: bool, device: str) -> None:
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dry_run": bool(dry_run),
        "device": device,
        "summary_files": {
            "json": _rel(SUMMARY_JSON),
            "log": _rel(SUMMARY_LOG),
            "txt": _rel(SUMMARY_TXT),
            "csv": _rel(SUMMARY_CSV),
        },
        "stages": list(rows),
    }
    SUMMARY_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "Smoked E2E Pipeline Summary",
        f"created_at: {payload['created_at']}",
        f"dry_run: {dry_run}",
        f"device: {device}",
        "",
    ]
    for row in rows:
        lines.append(f"{row['status_key']}: {row['status']} - {row['message']}")
    SUMMARY_TXT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    fieldnames = [
        "stage_name",
        "status_key",
        "status",
        "returncode",
        "duration_sec",
        "dependency_keys",
        "config_path",
        "output_dir",
        "checkpoint_path",
        "message",
        "command",
    ]
    with SUMMARY_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the smoked end-to-end training pipeline.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_LOG.write_text("", encoding="utf-8")
    rows: list[dict[str, object]] = []
    completed: dict[str, dict[str, object]] = {}

    for stage in _stage_definitions(args.device):
        row = run_stage(stage, dry_run=bool(args.dry_run), completed=completed)
        rows.append(row)
        completed[stage.key] = row
        write_summaries(rows, dry_run=bool(args.dry_run), device=str(args.device))
        if args.stop_on_failure and row["status"] == "failed":
            break

    return 1 if any(row["status"] == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Recover interrupted publication checkpoint-selection evaluation folders."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import select_best_checkpoint_and_compute_metrics as pipeline  # noqa: E402


CHECKPOINT_DIR_PREFIXES = ("step_", "epoch_")
KNOWN_STAGE_NAMES = {"selection", "final", "final_extra", "final_combined", "stage1", "stage2", "stage3"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Publication checkpoint-selection output root to scan.")
    parser.add_argument("--config", required=True, help="Publication checkpoint-selection YAML config.")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Report actions without deleting files.")
    parser.add_argument("--execute", action="store_true", help="Execute safe cleanup actions.")
    parser.add_argument(
        "--delete-invalid-analysis",
        action="store_true",
        help="Delete invalid checkpoint analysis folders when the model checkpoint still exists.",
    )
    parser.add_argument("--only-run", default=None, help="Optional run id substring filter.")
    parser.add_argument("--only-checkpoint", default=None, help="Optional checkpoint id filter.")
    parser.add_argument("--log-file", default=None, help="Optional text log path.")
    parser.add_argument(
        "--allow-heavy-metrics",
        action="store_true",
        help="Allow metric recomputation that may be too heavy for a login node.",
    )
    return parser.parse_args(argv)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _jsonable(value: Any) -> Any:
    return pipeline._jsonable(value)


def _write_json(path: Path, payload: Any) -> None:
    pipeline.save_json(path, payload)


def _log(lines: list[str], message: str) -> None:
    print(message, flush=True)
    lines.append(message)


def _load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in config: {path}")
    if pipeline.pipeline_mode(data) != "clean_fid_selection_publication":
        raise ValueError("Recovery script only supports clean_fid_selection_publication configs.")
    return pipeline._publication_flat_generation_config(data)


def _is_checkpoint_analysis_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name in {"best", "final"} or path.name.startswith(CHECKPOINT_DIR_PREFIXES):
        return True
    return False


def _has_analysis_state(path: Path) -> bool:
    if not path.is_dir():
        return False
    if any(path.glob("*_manifest.json")) or any(path.glob("*_metrics.json")):
        return True
    if (path / "generated_npy_images").is_dir() or (path / "features").is_dir():
        return True
    for child in path.iterdir():
        if child.is_dir() and child.name in KNOWN_STAGE_NAMES:
            if _has_analysis_state(child):
                return True
    return False


def _stage_dirs_for_checkpoint(checkpoint_dir: Path) -> list[tuple[str, Path]]:
    stages: list[tuple[str, Path]] = []
    if _has_analysis_state(checkpoint_dir) and (checkpoint_dir / "generated_npy_images").is_dir():
        stages.append(("selection", checkpoint_dir))
    for child in sorted(checkpoint_dir.iterdir() if checkpoint_dir.is_dir() else []):
        if child.is_dir() and child.name in KNOWN_STAGE_NAMES and _has_analysis_state(child):
            stage_name = "final" if child.name in {"final", "final_combined", "stage3"} else "selection"
            stages.append((stage_name, child))
    return stages


def _image_files_under(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [
        file
        for file in sorted(path.rglob("*"))
        if file.is_file() and file.suffix.lower() in pipeline.GENERATED_IMAGE_EXTENSIONS
        and not pipeline._is_protected_analysis_preview_path(file)
    ]


def _feature_files_under(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [file for file in sorted(path.rglob("*.npz")) if file.is_file() and not file.name.endswith(".tmp")]


def _metric_files_under(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return [file for file in sorted(path.rglob("*metrics*.json")) if file.is_file()]


def _stage_expected_keys(config: Mapping[str, Any], *, stage_name: str) -> list[str]:
    return pipeline._publication_expected_metric_keys(
        config,
        include_clean_fid=pipeline._metric_enabled(config, "compute_clean_fid", True),
        include_fd_dinov2=pipeline._metric_enabled(config, "compute_fd_dinov2", False),
        include_kid=pipeline._metric_enabled(config, "compute_kid", True),
        include_mmd=pipeline._metric_enabled(config, "compute_mmd", True),
        include_intra_lpips=stage_name == "final" and pipeline._metric_enabled(config, "compute_intra_lpips", False),
    )


def _validate_stage(
    *,
    run: pipeline.RunResolution,
    checkpoint: pipeline.CheckpointCandidate,
    config: Mapping[str, Any],
    stage_name: str,
    stage_dir: Path,
    expected_num_images: int,
) -> tuple[str, str]:
    metrics_path = pipeline._publication_stage_metrics_path(stage_dir, stage_name)
    payload = pipeline.load_json_if_valid(metrics_path)
    if not isinstance(payload, Mapping):
        return "invalid", f"missing stage metrics: {metrics_path}"
    try:
        pipeline._verify_publication_stage_outputs(
            run=run,
            checkpoint=checkpoint,
            stage_name=stage_name,
            stage_dir=stage_dir,
            expected_num_images=expected_num_images,
            expected_metric_keys=_stage_expected_keys(config, stage_name=stage_name),
            metrics_path=metrics_path,
            metrics_payload=payload,
            require_images_present=False,
        )
    except Exception as exc:
        return "invalid", f"{type(exc).__name__}: {exc}"
    return "valid", "verified metrics/features"


def _delete_invalid_analysis_dir(path: Path, *, execute: bool) -> dict[str, Any]:
    files = [
        file
        for file in path.rglob("*")
        if file.is_file() and not pipeline._is_protected_analysis_preview_path(file)
    ] if path.is_dir() else []
    protected_files = [
        file
        for file in path.rglob("*")
        if file.is_file() and pipeline._is_protected_analysis_preview_path(file)
    ] if path.is_dir() else []
    total_bytes = sum(int(file.stat().st_size) for file in files if file.exists())
    if execute and path.is_dir():
        for file in files:
            file.unlink(missing_ok=True)
        for directory in sorted(
            [item for item in path.rglob("*") if item.is_dir()],
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            try:
                directory.rmdir()
            except OSError:
                pass
        try:
            path.rmdir()
        except OSError:
            pass
    return {
        "path": str(path),
        "files_deleted": len(files) if execute else 0,
        "bytes_freed": int(total_bytes) if execute else 0,
        "planned_files": len(files),
        "planned_bytes": int(total_bytes),
        "protected_preview_files": [str(file) for file in protected_files],
        "executed": bool(execute),
    }


def recover(config: Mapping[str, Any], root: Path, args: argparse.Namespace) -> dict[str, Any]:
    execute = bool(args.execute)
    delete_invalid = bool(args.delete_invalid_analysis and execute)
    seeds = pipeline.make_publication_seeds(config)
    run_entries = {str(entry.get("run_identifier")): entry for entry in config.get("runs") or []}
    log_lines: list[str] = []
    run_reports = []

    for run_output_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        run_id = run_output_dir.name
        if args.only_run and args.only_run not in run_id:
            continue
        run_entry = run_entries.get(run_id)
        if run_entry is None:
            run_reports.append({"run_identifier": run_id, "status": "skipped", "reason": "run not present in config"})
            continue
        run = pipeline.resolve_run(run_entry, config)
        discovery = pipeline.discover_candidate_checkpoints(
            run.run_dir,
            model_type=run.model_type,
            checkpoint_min_epoch=int(config.get("checkpoint_min_epoch", 50)),
            checkpoint_min_step=config.get("checkpoint_min_step"),
        )
        by_id = {candidate.checkpoint_identifier: candidate for candidate in discovery.candidates}
        checkpoint_reports = []
        for checkpoint_dir in sorted(path for path in run_output_dir.iterdir() if _is_checkpoint_analysis_dir(path)):
            checkpoint_id = checkpoint_dir.name
            if args.only_checkpoint and args.only_checkpoint != checkpoint_id:
                continue
            if not _has_analysis_state(checkpoint_dir):
                continue
            checkpoint = by_id.get(checkpoint_id)
            image_files = _image_files_under(checkpoint_dir)
            feature_files = _feature_files_under(checkpoint_dir)
            metric_files = _metric_files_under(checkpoint_dir)
            row = {
                "checkpoint_identifier": checkpoint_id,
                "analysis_dir": str(checkpoint_dir),
                "model_checkpoint_exists": checkpoint is not None and Path(checkpoint.checkpoint_path).exists(),
                "detected_image_count": len(image_files),
                "detected_image_bytes": sum(int(path.stat().st_size) for path in image_files if path.exists()),
                "detected_feature_files": [str(path) for path in feature_files],
                "detected_metric_files": [str(path) for path in metric_files],
                "stages": [],
                "action": "none",
                "reason": "",
                "deletion": None,
            }
            if checkpoint is None:
                row["action"] = "skip"
                row["reason"] = "model checkpoint is not discoverable; refusing cleanup"
                checkpoint_reports.append(row)
                continue

            invalid_reasons = []
            for stage_name, stage_dir in _stage_dirs_for_checkpoint(checkpoint_dir):
                expected_count = len(seeds["final"] if stage_name == "final" else seeds["selection"])
                status, reason = _validate_stage(
                    run=run,
                    checkpoint=checkpoint,
                    config=config,
                    stage_name=stage_name,
                    stage_dir=stage_dir,
                    expected_num_images=expected_count,
                )
                row["stages"].append({"stage": stage_name, "path": str(stage_dir), "status": status, "reason": reason})
                if status != "valid":
                    invalid_reasons.append(f"{stage_name}: {reason}")

            if invalid_reasons:
                row["action"] = "delete_invalid_analysis" if args.delete_invalid_analysis else "mark_invalid"
                row["reason"] = "; ".join(invalid_reasons)
                if args.delete_invalid_analysis:
                    row["deletion"] = _delete_invalid_analysis_dir(checkpoint_dir, execute=delete_invalid)
                    _log(
                        log_lines,
                        f"{'DELETED' if delete_invalid else 'DRY-RUN'} invalid analysis {checkpoint_dir}: {row['reason']}",
                    )
            elif image_files:
                row["action"] = "delete_verified_images" if execute else "dry_run_delete_verified_images"
                row["reason"] = "verified metrics/features exist"
                deletions = []
                if execute:
                    for image_dir in sorted({path.parent for path in image_files}):
                        deletions.append(
                            pipeline._safe_delete_explicit_generated_files(
                                checkpoint_identifier=checkpoint_id,
                                image_dir=image_dir,
                                paths=[path for path in image_files if path.parent == image_dir],
                                dry_run=False,
                                reason="recovery verified image cleanup",
                            )
                        )
                    row["deletion"] = deletions
                _log(log_lines, f"{'DELETE' if execute else 'DRY-RUN'} verified images for {checkpoint_dir}")
            else:
                row["action"] = "already_clean"
                row["reason"] = "no generated images detected"
            checkpoint_reports.append(row)
        run_reports.append(
            {
                "run_identifier": run_id,
                "run_output_dir": str(run_output_dir),
                "status": "scanned",
                "checkpoints": checkpoint_reports,
            }
        )

    report = {
        "recovery": True,
        "root": str(root),
        "config": str(args.config),
        "dry_run": not execute,
        "execute": execute,
        "delete_invalid_analysis": bool(args.delete_invalid_analysis),
        "allow_heavy_metrics": bool(args.allow_heavy_metrics),
        "runs": run_reports,
        "timestamp": utc_timestamp(),
    }
    report_path = root / f"recovery_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    _write_json(report_path, report)
    summary_path = report_path.with_suffix(".txt")
    summary_lines = [
        f"Recovery report: {report_path}",
        f"Root: {root}",
        f"Mode: {'execute' if execute else 'dry-run'}",
    ]
    summary_lines.extend(log_lines)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    if args.log_file:
        Path(args.log_file).write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    report["report_path"] = str(report_path)
    report["summary_path"] = str(summary_path)
    _write_json(report_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = _load_config(Path(args.config))
    root = Path(args.root).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"Recovery root not found: {root}")
    print(json.dumps(_jsonable(recover(config, root, args)), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

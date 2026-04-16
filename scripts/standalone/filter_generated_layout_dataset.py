#!/usr/bin/env python3
"""Audit generated layout datasets with the foreground/background classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.algorithms.inference.rare_layout_dataset_tools import (
    audit_generated_layout_dataset,
    auto_find_latest_filter_run,
    export_audit_results,
    load_filter_from_run_or_checkpoint,
    resolve_filter_output_dir,
)
from src.core.configs.config_loader import load_yaml
from src.core.paths import checkpoints_root


def _auto_find_latest_classifier_run() -> str:
    candidate_roots = [
        checkpoints_root() / "foreground_background_filter" / "runs",
        checkpoints_root() / "multiclass_foreground_background_filter" / "runs",
    ]
    candidates = []
    for root in candidate_roots:
        if not root.is_dir():
            continue
        try:
            run_dir = auto_find_latest_filter_run(root)
        except FileNotFoundError:
            continue
        summary_path = run_dir / "metrics" / "summary.json"
        if summary_path.is_file():
            candidates.append((summary_path.stat().st_mtime, str(run_dir)))
    if not candidates:
        raise FileNotFoundError("No binary or multiclass classifier runs were found under artifacts/checkpoints.")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a generated layout dataset with the foreground/background classifier")
    parser.add_argument("--config", type=str, required=True, help="YAML preset providing all run parameters.")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parsed = parser.parse_args()
    config_data = load_yaml(parsed.config)
    if not isinstance(config_data, dict):
        raise ValueError(f"Expected a mapping in config file {parsed.config!r}.")
    config_data = dict(config_data)
    config_data["config"] = str(parsed.config)
    config_data["device"] = str(parsed.device)
    return argparse.Namespace(**config_data)


def main() -> None:
    args = parse_args()
    if not getattr(args, "generated_dataset_dir", ""):
        raise ValueError("--generated_dataset_dir is required.")

    filter_run_dir = str(getattr(args, "filter_run_dir", "") or "")
    filter_checkpoint = str(getattr(args, "filter_checkpoint", "") or "")
    output_dir_arg = str(getattr(args, "output_dir", "") or "")
    threshold_arg = getattr(args, "threshold", float("nan"))
    max_discarded_valid_threshold = float(getattr(args, "max_discarded_valid_threshold", 1.0))
    score_alpha = float(getattr(args, "score_alpha", 1.0))
    score_beta = float(getattr(args, "score_beta", 1.0))
    min_valid_object_fraction_sweep = str(getattr(args, "min_valid_object_fraction_sweep", "") or "")
    batch_size = int(getattr(args, "batch_size", 64))

    if not filter_run_dir and not filter_checkpoint:
        filter_run_dir = _auto_find_latest_classifier_run()

    threshold_override = None if threshold_arg != threshold_arg else float(threshold_arg)
    model, summary, threshold, input_size, context_ratio, resolved_run_dir = load_filter_from_run_or_checkpoint(
        device=args.device,
        run_dir=(filter_run_dir or None),
        checkpoint_path=(filter_checkpoint or None),
        threshold_override=threshold_override,
    )

    instance_rows, image_rows, stats = audit_generated_layout_dataset(
        dataset_dir=args.generated_dataset_dir,
        filter_model=model,
        threshold=threshold,
        filter_input_size=input_size,
        context_ratio=context_ratio,
        device=args.device,
        crop_batch_size=batch_size,
        show_progress=True,
        classifier_summary=summary,
    )

    output_dir = resolve_filter_output_dir(
        dataset_dir=args.generated_dataset_dir,
        output_dir=(Path(output_dir_arg) if output_dir_arg else None),
        filter_run_dir=(resolved_run_dir if resolved_run_dir is not None else (Path(filter_run_dir) if filter_run_dir else None)),
        checkpoint_path=(Path(filter_checkpoint) if filter_checkpoint else None),
    )
    summary_payload = {
        "generated_dataset_dir": str(Path(args.generated_dataset_dir)),
        "filter_run_dir": None if resolved_run_dir is None else str(resolved_run_dir),
        "checkpoint_summary": summary,
        "threshold": float(threshold) if isinstance(threshold, (int, float)) else dict(threshold),
        "input_size": int(input_size),
        "context_ratio": float(context_ratio),
        "max_discarded_valid_threshold": max_discarded_valid_threshold,
        "score_alpha": score_alpha,
        "score_beta": score_beta,
        "stats": stats,
        "min_valid_object_fraction_sweep": min_valid_object_fraction_sweep,
    }
    export_audit_results(
        output_dir=output_dir,
        summary=summary_payload,
        instance_rows=instance_rows,
        image_rows=image_rows,
    )
    print(json.dumps({"output_dir": str(output_dir), **summary_payload}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

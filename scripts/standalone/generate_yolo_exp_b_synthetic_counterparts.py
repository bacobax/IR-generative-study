#!/usr/bin/env python3
"""Generate 1:1 synthetic YOLO Experiment-B counterpart datasets.

This is the production entrypoint for precomputing synthetic augmentation
candidates from annotated real YOLO images. Each configured generator writes a
separate dataset folder with images, filtered and unfiltered COCO annotations,
filter audit manifests, sanity overlays, provenance, and distribution metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.algorithms.inference.regiondiff_smoke_generation import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_YOLO_DATASET_YAML,
    generate_production_synthetic_datasets,
    load_generation_config,
    validate_generator_checkpoint_readability,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic counterpart datasets for YOLO Experiment B."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--yolo-dataset-yaml", default=DEFAULT_YOLO_DATASET_YAML)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--generators",
        default="",
        help="Comma-separated generator names from the config. Empty runs all configured generators.",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing generator folders and skip already-written images/sample_*.npy files.",
    )
    parser.add_argument("--max-tries", "--max_tries", dest="max_tries", type=int, default=None)
    parser.add_argument("--invalid-ratio-threshold", type=float, default=None)
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument(
        "--metrics-only",
        action="store_true",
        help="Compute metrics for existing generated datasets without generating, filtering, or rendering sanity images.",
    )
    parser.add_argument(
        "--skip-invalid-generators",
        action="store_true",
        help="Deprecated: invalid generator checkpoints are skipped by default.",
    )
    parser.add_argument(
        "--fail-on-invalid-generators",
        action="store_true",
        help="Fail before generation if any selected generator checkpoint is missing or corrupt.",
    )
    return parser.parse_args()


def _requested_generator_names(args: argparse.Namespace) -> list[str]:
    return [item.strip() for item in args.generators.split(",") if item.strip()]


def _preflight_configured_checkpoints(
    config: dict,
    *,
    requested_names: list[str],
    fail_on_invalid_generators: bool,
) -> dict:
    selected = []
    invalid_messages = []
    requested = set(requested_names)
    for generator in config.get("generators", []):
        name = str(generator.get("name", ""))
        if requested and name not in requested:
            continue
        ok, detail = validate_generator_checkpoint_readability(str(generator.get("checkpoint_path", "")))
        if ok:
            selected.append(generator)
        else:
            invalid_messages.append(f"{name or '<unnamed>'}: {detail}")

    if invalid_messages and fail_on_invalid_generators:
        joined = "\n  - ".join(invalid_messages)
        raise RuntimeError(
            "One or more configured generator checkpoints are not usable:\n"
            f"  - {joined}\n\n"
            "Fix/resync the checkpoint, restrict the run with --generators, or omit "
            "--fail-on-invalid-generators to generate the valid configured datasets."
        )
    if invalid_messages:
        print("[warn] Skipping invalid generator checkpoint(s):")
        for message in invalid_messages:
            print(f"  - {message}")
    filtered = dict(config)
    filtered["generators"] = selected
    if not selected:
        joined = "\n  - ".join(invalid_messages) if invalid_messages else "none"
        raise RuntimeError(
            "No valid generator checkpoints remain after preflight. Invalid checkpoint(s):\n"
            f"  - {joined}"
        )
    return filtered


def main() -> None:
    args = parse_args()
    config = load_generation_config(args.config)
    if args.metrics_only:
        config["resume"] = True
        config["overwrite"] = False
    if args.max_tries is not None or args.invalid_ratio_threshold is not None:
        retry_cfg = dict(config.get("retry", {}))
        if args.max_tries is not None:
            retry_cfg["max_tries"] = int(args.max_tries)
            retry_cfg["enabled"] = int(args.max_tries) > 1
        if args.invalid_ratio_threshold is not None:
            retry_cfg["invalid_instance_ratio_threshold"] = float(args.invalid_ratio_threshold)
        config["retry"] = retry_cfg
    if args.resume:
        config["resume"] = True
        config["overwrite"] = False
    generator_names = _requested_generator_names(args)
    if not args.metrics_only:
        config = _preflight_configured_checkpoints(
            config,
            requested_names=generator_names,
            fail_on_invalid_generators=args.fail_on_invalid_generators,
        )
    summary = generate_production_synthetic_datasets(
        config=config,
        yolo_dataset_yaml=args.yolo_dataset_yaml,
        output_root=args.output_root,
        max_samples=args.max_samples,
        generator_names=generator_names or None,
        device=args.device,
        skip_filter=args.skip_filter,
        skip_metrics=args.skip_metrics,
        metrics_only=args.metrics_only,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

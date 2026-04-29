#!/usr/bin/env python3
"""Generate production precomputed synthetic datasets for Experiment B."""

from __future__ import annotations

import argparse
import json

from src.algorithms.inference.regiondiff_smoke_generation import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_YOLO_DATASET_YAML,
    generate_production_synthetic_datasets,
    generate_regiondiff_candidate_dataset,
    load_generation_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Experiment-B precomputed synthetic datasets.")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--yolo-dataset-yaml", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--generators", default="", help="Comma-separated generator names to run.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--skip-metrics", action="store_true")

    # Compatibility with the historical smoke CLI. If supplied, run the tiny
    # legacy placeholder export instead of the production multi-generator path.
    parser.add_argument("--model-kind", default=None, choices=["fm", "dm", "sd15_finetune", "sd15_lora"])
    parser.add_argument("--artifact-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--t-scale", type=float, default=1000.0)
    parser.add_argument("--train-target", default="v", choices=["v", "x0"])
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--precision", default="fp32", choices=["fp16", "bf16", "fp32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.model_kind is not None:
        if not args.output_dir:
            raise ValueError("--output-dir is required with the legacy --model-kind interface.")
        summary = generate_regiondiff_candidate_dataset(
            model_kind=args.model_kind,
            artifact_dir=args.artifact_dir or "",
            yolo_dataset_yaml=args.yolo_dataset_yaml or DEFAULT_YOLO_DATASET_YAML,
            output_dir=args.output_dir,
            max_samples=2 if args.max_samples is None else args.max_samples,
            batch_size=args.batch_size,
            image_size=args.image_size,
            steps=args.steps,
            seed=args.seed,
            device=args.device or "cpu",
            t_scale=args.t_scale,
            train_target=args.train_target,
            guidance_scale=args.guidance_scale,
            precision=args.precision,
        )
    else:
        generator_names = [item.strip() for item in args.generators.split(",") if item.strip()]
        config = load_generation_config(args.config)
        summary = generate_production_synthetic_datasets(
            config=config,
            yolo_dataset_yaml=args.yolo_dataset_yaml or config.get("yolo_dataset_yaml") or DEFAULT_YOLO_DATASET_YAML,
            output_root=args.output_root or config.get("output_root") or DEFAULT_OUTPUT_ROOT,
            max_samples=args.max_samples,
            generator_names=generator_names or None,
            device=args.device,
            skip_filter=args.skip_filter,
            skip_metrics=args.skip_metrics,
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


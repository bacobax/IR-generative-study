#!/usr/bin/env python3
"""Generate a tiny precomputed dataset from a RegionDiff smoke artifact."""

from __future__ import annotations

import argparse
import json

from src.algorithms.inference.regiondiff_smoke_generation import generate_regiondiff_candidate_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a smoked RegionDiff candidate dataset.")
    parser.add_argument("--model-kind", required=True, choices=["fm", "dm", "sd15_finetune", "sd15_lora"])
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--yolo-dataset-yaml", default="data/derived/yolo-test-ds/full_train.yaml")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--t-scale", type=float, default=1000.0)
    parser.add_argument("--train-target", default="v", choices=["v", "x0"])
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--precision", default="fp32", choices=["fp16", "bf16", "fp32"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_regiondiff_candidate_dataset(
        model_kind=args.model_kind,
        artifact_dir=args.artifact_dir,
        yolo_dataset_yaml=args.yolo_dataset_yaml,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        image_size=args.image_size,
        steps=args.steps,
        seed=args.seed,
        device=args.device,
        t_scale=args.t_scale,
        train_target=args.train_target,
        guidance_scale=args.guidance_scale,
        precision=args.precision,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate a rare-layout synthetic dataset from a trained FM layout model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from src.algorithms.inference.rare_layout_dataset_tools import (
    append_generated_layout_batch,
    build_layout_dataset,
    build_layout_rarity_records,
    compute_size_bin_thresholds,
    finalize_generated_layout_export,
    initialize_generated_layout_export,
    load_sampler_from_pipeline,
    resolve_generation_output_dir,
    sample_layout_batch,
    select_layout_records,
)
from src.core.configs.config_loader import apply_yaml_defaults
from src.core.data.layout_batching import collate_layout_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a rare-layout synthetic dataset")
    parser.add_argument("--config", type=str, default=None, help="YAML preset. CLI flags override preset values.")
    parser.add_argument(
        "--pipeline_dir",
        type=str,
        default="artifacts/checkpoints/flow_matching/serious_runs/stay_layout_latent_flir_sd15_512",
    )
    parser.add_argument(
        "--preset_path",
        type=str,
        default="configs/fm/train/presets/stay_layout_latent_flir_sd15_512.yaml",
    )
    parser.add_argument("--checkpoint_name", type=str, default="unet_fm_best.pt")
    parser.add_argument("--dataset_id", type=str, default="")
    parser.add_argument("--dataset_root", type=str, default="", help="Optional split-root override for custom datasets or tests.")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])
    parser.add_argument("--n_samples", type=int, default=256)
    parser.add_argument("--min_objects", type=int, default=1)
    parser.add_argument("--selection_mode", type=str, default="rare_first", choices=["rare_first", "random"])
    parser.add_argument("--rarity_aggregation", type=str, default="sum", choices=["sum", "mean"])
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")

    preliminary, _ = parser.parse_known_args()
    apply_yaml_defaults(parser, preliminary.config)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    output_dir = resolve_generation_output_dir(args.output_dir, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    preset, _layout_meta, _unet, _vae, sampler = load_sampler_from_pipeline(
        args.pipeline_dir,
        args.preset_path,
        args.checkpoint_name,
        args.device,
    )

    dataset = build_layout_dataset(
        preset,
        split=args.split,
        dataset_root=(Path(args.dataset_root) if args.dataset_root else None),
        dataset_id=(args.dataset_id or None),
    )
    dataset_id = args.dataset_id or preset["data"]["dataset_id"]

    size_thresholds = compute_size_bin_thresholds(dataset, show_progress=True)
    rarity_records, _instance_counter = build_layout_rarity_records(
        dataset,
        thresholds=size_thresholds,
        aggregation=args.rarity_aggregation,
        show_progress=True,
    )
    selected_records = select_layout_records(
        rarity_records,
        n_samples=args.n_samples,
        min_objects=args.min_objects,
        selection_mode=args.selection_mode,
        seed=args.seed,
    )

    if len(selected_records) < int(args.n_samples):
        print(
            f"[warn] Requested {args.n_samples} samples but only {len(selected_records)} matching layouts were available."
        )

    export_state = initialize_generated_layout_export(
        output_dir=output_dir,
        selected_records=selected_records,
    )
    for start_idx in tqdm(range(0, len(selected_records), max(1, int(args.batch_size))), desc="Generating rare layouts"):
        chunk = selected_records[start_idx:start_idx + max(1, int(args.batch_size))]
        batch = collate_layout_batch([record.sample for record in chunk])
        chunk_images = sample_layout_batch(
            sampler,
            batch,
            steps=int(args.steps),
            seed=int(args.seed) + int(start_idx),
        )
        append_generated_layout_batch(
            export_state,
            records=chunk,
            generated_images=list(chunk_images),
        )
        del chunk_images
        del batch
        if torch.cuda.is_available() and str(args.device).startswith("cuda"):
            torch.cuda.empty_cache()

    summary = finalize_generated_layout_export(
        export_state,
        split=args.split,
        dataset_id=dataset_id,
        selection_mode=args.selection_mode,
        rarity_aggregation=args.rarity_aggregation,
        size_bin_thresholds=size_thresholds,
        pipeline_dir=args.pipeline_dir,
        preset_path=args.preset_path,
        checkpoint_name=args.checkpoint_name,
        steps=int(args.steps),
        seed=int(args.seed),
        n_requested_samples=len(selected_records),
    )

    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

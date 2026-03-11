"""Modular CLI entrypoint for flow-matching sampling / inference.

This module is the **source of truth** for launching stand-alone FM
image generation.  The root-level ``generate_datasets.py`` still
handles the multi-mode (SD1.5 / FM / FM-guided) dispatch, but FM
sampling logic is now encapsulated here.

Usage::

    python -m src.cli.sample \
        --pipeline_dir ./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled \
        --steps 100 --batch_size 8 --max_samples 64 \
        --output_dir ./artifacts/generated/main/fm_cli

    # or:
    python -m src.cli.sample --help
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.core.configs.fm_config import FMSampleConfig
from src.core.configs.config_loader import load_yaml, merge_config_and_cli
from src.core.normalization import fm_output_to_uint16, uint16_to_png_uint8
from src.core.registry import REGISTRIES

# Ensure default components are registered
import src.models.fm_unet  # noqa: F401
import src.algorithms.inference.flow_matching_sampler  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser for FM sampling."""
    parser = argparse.ArgumentParser(description="Flow-Matching Sampling CLI")

    # Config file (optional)
    parser.add_argument(
        "--config", type=str, default=None,
        help="Path to YAML config file. CLI flags override config values.",
    )

    # Model / pipeline
    parser.add_argument(
        "--pipeline_dir", type=str,
        default="./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled/",
        help="Directory containing UNET/ and VAE/ sub-folders",
    )
    parser.add_argument(
        "--vae_weights", type=str, default=None,
        help="Optional path to override VAE weights",
    )

    # Sampling parameters
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of Euler steps")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for generation")
    parser.add_argument("--max_samples", type=int, default=64,
                        help="Total number of samples to generate")
    parser.add_argument("--t_scale", type=float, default=1000.0,
                        help="Time-scaling factor for the UNet")
    parser.add_argument("--train_target", type=str, default="v",
                        choices=["v", "x0"],
                        help="Prediction target used during training")

    # Device / output
    parser.add_argument("--device", type=str, default=None,
                        help="Device (default: auto-detect)")
    parser.add_argument("--output_dir", type=str, default="./artifacts/generated/main/fm_cli",
                        help="Output directory for generated samples")

    return parser


# Mapping from flat CLI arg names → FMSampleConfig field names.
# FMSampleConfig is flat (no nesting), so the mapping is identity for
# config fields and marks extra args (max_samples, output_dir) as non-config.
_FLAT_TO_NESTED: dict = {}
# (empty → all flat keys are used as-is, which is correct for FMSampleConfig)


# ═══════════════════════════════════════════════════════════════════════════
# Sampling pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_sampling(cfg: FMSampleConfig, *, max_samples: int, output_dir: str) -> None:
    """Execute FM sampling from a structured config.

    Parameters
    ----------
    cfg : FMSampleConfig
        Sampling configuration (model path, steps, device, etc.).
    max_samples : int
        Total number of images to produce.
    output_dir : str
        Directory where ``.npy`` and ``.png`` outputs are saved.
    """
    # ── Resolve sampler class through registry ──
    SamplerCls = REGISTRIES.sampler.get(cfg.sampler_name)
    sampler = SamplerCls.from_config(cfg)

    os.makedirs(output_dir, exist_ok=True)

    generated = 0
    batch_size = cfg.batch_size

    print(f"[FM-sample] Generating {max_samples} samples "
          f"(batch_size={batch_size}, steps={cfg.steps}) ...")

    with tqdm(total=max_samples, desc="[FM-sample]", unit="img") as pbar:
        while generated < max_samples:
            bs = min(batch_size, max_samples - generated)
            z = sampler.sample_euler(steps=cfg.steps, batch_size=bs)
            x_gen = sampler.decode(z)  # (bs, 1, H, W), [-1, 1]

            for j in range(bs):
                raw_uint16 = fm_output_to_uint16(x_gen[j])
                npy_path = os.path.join(output_dir, f"sample_{generated:05d}.npy")
                np.save(npy_path, raw_uint16)

                png_path = os.path.join(output_dir, f"sample_{generated:05d}.png")
                vis = uint16_to_png_uint8(raw_uint16)
                Image.fromarray(vis, mode="L").save(png_path)

                generated += 1
                pbar.update(1)

    print(f"[FM-sample] Done. {max_samples} samples in {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[list] = None) -> None:
    """Parse CLI flags and launch FM sampling.

    Parameters
    ----------
    argv : list[str], optional
        Explicit argument list (for testing). ``None`` → ``sys.argv[1:]``.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Merge: defaults → YAML → explicit CLI overrides
    cfg = merge_config_and_cli(
        FMSampleConfig, args.config, parser, args,
        flat_to_nested=_FLAT_TO_NESTED,
    )

    # max_samples and output_dir are not part of FMSampleConfig;
    # resolve them from YAML fallback then CLI.
    yaml_data = load_yaml(args.config) if args.config else {}
    max_samples = args.max_samples
    output_dir = args.output_dir
    # If the user didn't explicitly set these on CLI, check YAML
    cli_defaults = vars(parser.parse_args([]))
    if args.max_samples == cli_defaults["max_samples"] and "max_samples" in yaml_data:
        max_samples = yaml_data["max_samples"]
    if args.output_dir == cli_defaults["output_dir"] and "output_dir" in yaml_data:
        output_dir = yaml_data["output_dir"]

    run_sampling(cfg, max_samples=max_samples, output_dir=output_dir)


if __name__ == "__main__":
    main()

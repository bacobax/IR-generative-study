"""CLI entrypoint for text-conditioned FM sampling with CFG.

Usage::

    python -m src.cli.sample_text_fm \\
        --config configs/fm/sample/presets/text_cfg.yaml \\
        --prompt "a thermal image of a building" \\
        --guidance_scale 7.5

    # Unconditional (no CFG):
    python -m src.cli.sample_text_fm --config ... --guidance_scale 1.0
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.core.configs.text_fm_config import TextFMSampleConfig
from src.core.configs.config_loader import load_yaml, merge_config_and_cli
from src.core.normalization import fm_output_to_uint16, uint16_to_png_uint8
from src.core.registry import REGISTRIES

# Ensure components are registered
import src.models.fm_text_unet  # noqa: F401
import src.algorithms.inference.cfg_flow_matching_sampler  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# Argument parsing
# ═══════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Text-Conditioned FM Sampling with CFG")

    parser.add_argument("--config", type=str, default=None,
                        help="YAML config file. CLI overrides config values.")

    # Pipeline
    parser.add_argument("--pipeline_dir", type=str,
                        default="./artifacts/checkpoints/flow_matching/text_fm/")
    parser.add_argument("--vae_weights", type=str, default=None)

    # Sampling
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_samples", type=int, default=64)
    parser.add_argument("--guidance_scale", type=float, default=7.5,
                        help="CFG scale. 1.0=conditional only, 0.0=unconditional")
    parser.add_argument("--prompt", type=str, default="",
                        help="Text prompt (applied to all samples)")
    parser.add_argument("--t_scale", type=float, default=1000.0)
    parser.add_argument("--train_target", type=str, default="v",
                        choices=["v", "x0"])

    # Text encoder (fallback if conditioner.json not in pipeline_dir)
    parser.add_argument("--text_encoder", type=str,
                        default="openai/clip-vit-large-patch14")
    parser.add_argument("--max_text_length", type=int, default=77)

    # Device / output
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str,
                        default="./artifacts/generated/main/text_fm_cfg")

    return parser


_FLAT_TO_NESTED: dict = {}


# ═══════════════════════════════════════════════════════════════════════════
# Sampling pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_sampling(
    cfg: TextFMSampleConfig,
    *,
    max_samples: int,
    output_dir: str,
) -> None:
    """Execute CFG sampling from a structured config."""
    SamplerCls = REGISTRIES.sampler.get(cfg.sampler_name)
    sampler = SamplerCls.from_config(cfg)

    os.makedirs(output_dir, exist_ok=True)

    prompt = cfg.prompt
    guidance_scale = cfg.guidance_scale
    batch_size = cfg.batch_size
    generated = 0

    print(
        f"[TextFM-sample] Generating {max_samples} samples "
        f"(batch={batch_size}, steps={cfg.steps}, guidance={guidance_scale})"
    )
    if prompt:
        print(f"  prompt: {prompt!r}")

    with tqdm(total=max_samples, desc="[TextFM-sample]", unit="img") as pbar:
        while generated < max_samples:
            bs = min(batch_size, max_samples - generated)
            prompts = [prompt] * bs

            if guidance_scale != 1.0 and prompt:
                z = sampler.sample_euler_cfg(
                    prompts,
                    steps=cfg.steps,
                    guidance_scale=guidance_scale,
                )
            else:
                # No CFG: use conditional if prompt given, else unconditional
                if prompt:
                    cond_kw = sampler.conditioner.prepare_conditional(
                        prompts, sampler.device,
                    )
                else:
                    cond_kw = sampler.conditioner.prepare_for_sampling(
                        bs, sampler.device,
                    )
                z = sampler.sample_euler(steps=cfg.steps, batch_size=bs)

            x_gen = sampler.decode(z)

            for j in range(bs):
                raw_uint16 = fm_output_to_uint16(x_gen[j])
                np.save(
                    os.path.join(output_dir, f"sample_{generated:05d}.npy"),
                    raw_uint16,
                )
                vis = uint16_to_png_uint8(raw_uint16)
                Image.fromarray(vis, mode="L").save(
                    os.path.join(output_dir, f"sample_{generated:05d}.png"),
                )
                generated += 1
                pbar.update(1)

    print(f"[TextFM-sample] Done. {max_samples} samples in {output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry
# ═══════════════════════════════════════════════════════════════════════════

def main(argv: Optional[list] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = merge_config_and_cli(
        TextFMSampleConfig, args.config, parser, args,
        flat_to_nested=_FLAT_TO_NESTED,
    )

    # Resolve extra args from YAML or CLI
    yaml_data = load_yaml(args.config) if args.config else {}
    cli_defaults = vars(parser.parse_args([]))
    max_samples = args.max_samples
    output_dir = args.output_dir
    if args.max_samples == cli_defaults["max_samples"] and "max_samples" in yaml_data:
        max_samples = yaml_data["max_samples"]
    if args.output_dir == cli_defaults["output_dir"] and "output_dir" in yaml_data:
        output_dir = yaml_data["output_dir"]

    run_sampling(cfg, max_samples=max_samples, output_dir=output_dir)


if __name__ == "__main__":
    main()

"""Modular CLI entrypoint for synthetic dataset generation.

This module is the **source of truth** for generating synthetic IR datasets
using either SD 1.5 with LoRA or Stable Flow Matching.

The root-level ``generate_datasets.py`` is a thin compatibility wrapper that
forwards to :func:`main` here.

Usage::

    # SD 1.5 LoRA
    python -m src.cli.generate \\
        --mode sd15 \\
        --stage1_dir ./artifacts/checkpoints/stable_diffusion/lora_runs/... \\
        --max_samples 100 --output_dir ./artifacts/generated/main/sd15

    # Stable Flow Matching
    python -m src.cli.generate \\
        --mode fm \\
        --fm_pipeline_dir ./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled \\
        --max_samples 100 --output_dir ./artifacts/generated/main/fm
"""

from __future__ import annotations

import argparse
import json
import os
from contextlib import nullcontext
from typing import Dict, List

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.core.normalization import fm_output_to_uint16, raw_array_to_png_uint8, sd_output_to_npy, uint16_to_png_uint8
from src.core.configs.fm_config import FMSampleConfig
from src.core.configs.config_loader import apply_yaml_defaults
from src.core.registry import REGISTRIES
from src.core.normalization import RAW_UINT16_PERCENTILE
from src.algorithms.stable_diffusion.models import load_stage1_pipeline

# Ensure default FM components are registered
import src.algorithms.inference.flow_matching_sampler  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# Metadata reader
# ═══════════════════════════════════════════════════════════════════════════

def load_metadata(jsonl_path: str, max_samples: int) -> List[Dict]:
    entries: List[Dict] = []
    if not os.path.isfile(jsonl_path):
        return [{"text": "", "file_name": ""} for _ in range(max_samples)]
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
            if len(entries) >= max_samples:
                break
    return entries


# ═══════════════════════════════════════════════════════════════════════════
# SD 1.5 generator
# ═══════════════════════════════════════════════════════════════════════════

def generate_sd15(args, entries: List[Dict]):
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    weight_dtype = dtype_map[args.precision]
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    normalization_mode = RAW_UINT16_PERCENTILE
    artifact_prompt_text = None
    if args.stage1_dir is not None:
        print(f"[SD1.5] Loading stage-1 artifact from {args.stage1_dir}")
        pipe, manifest = load_stage1_pipeline(
            stage1_dir=args.stage1_dir,
            base_model=args.base_model,
            torch_dtype=weight_dtype,
        )
        normalization_mode = manifest.get("normalization_mode", RAW_UINT16_PERCENTILE)
        artifact_prompt_text = manifest.get("prompt_text")
    else:
        from diffusers import StableDiffusionPipeline

        print(f"[SD1.5] Loading legacy LoRA weights from {args.lora_dir}")
        pipe = StableDiffusionPipeline.from_pretrained(
            args.base_model,
            torch_dtype=weight_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe.load_lora_weights(args.lora_dir)

    pipe.to(device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=weight_dtype)
        if device.startswith("cuda") and weight_dtype != torch.float32
        else nullcontext()
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[SD1.5] Generating {len(entries)} samples ...")
    generic_prompt_text = artifact_prompt_text or "overhead infrared surveillance image with any people or objects"
    for idx, entry in enumerate(entries):
        prompt = generic_prompt_text if args.generic_prompt else entry.get("text") or generic_prompt_text
        base_seed = args.seed + idx

        image = None
        last_flagged = False
        for attempt in range(args.max_tries):
            seed = base_seed + attempt
            generator = torch.Generator(device=device).manual_seed(seed)

            with autocast_ctx:
                result = pipe(
                    prompt,
                    negative_prompt=args.negative_prompt,
                    num_inference_steps=args.sd_steps,
                    guidance_scale=args.guidance,
                    generator=generator,
                )

            image = result.images[0]
            nsfw = getattr(result, "nsfw_content_detected", None)
            if nsfw is None:
                last_flagged = False
                break

            if isinstance(nsfw, list):
                last_flagged = any(bool(x) for x in nsfw)
            else:
                last_flagged = bool(nsfw)

            if not last_flagged:
                break

        if last_flagged:
            print(f"  [SD1.5] NSFW detected for sample {idx:05d}; saved last retry seed={seed}")

        raw_arr = sd_output_to_npy(image, normalization_mode=normalization_mode)
        out_path = os.path.join(args.output_dir, f"sample_{idx:05d}.npy")
        np.save(out_path, raw_arr)

        png_path = os.path.join(args.output_dir, f"sample_{idx:05d}.png")
        vis = raw_array_to_png_uint8(raw_arr, normalization_mode=normalization_mode)
        Image.fromarray(vis, mode="L").save(png_path)

        if (idx + 1) % 50 == 0 or idx == len(entries) - 1:
            print(f"  [{idx + 1}/{len(entries)}] saved {out_path}")

    meta_out = os.path.join(args.output_dir, "metadata.jsonl")
    with open(meta_out, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(entries):
            record = {
                "file_name": f"sample_{idx:05d}.npy",
                "text": entry.get("text", ""),
                "source_file": entry.get("file_name", ""),
            }
            f.write(json.dumps(record) + "\n")
    print(f"[SD1.5] Done. {len(entries)} samples in {args.output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# Helpers: build sampler from CLI args via config + registry
# ═══════════════════════════════════════════════════════════════════════════

def _build_sampler(args):
    """Build a FlowMatchingSampler from CLI args using modular components."""
    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = FMSampleConfig(
        pipeline_dir=args.fm_pipeline_dir,
        vae_weights=args.fm_vae_weights,
        t_scale=args.fm_t_scale,
        device=device,
    )
    SamplerCls = REGISTRIES.sampler.get(cfg.sampler_name)
    return SamplerCls.from_config(cfg), device


# ═══════════════════════════════════════════════════════════════════════════
# Stable Flow-Matching generator (plain Euler)
# ═══════════════════════════════════════════════════════════════════════════

def generate_fm(args, entries: List[Dict]):
    print(f"[FM] Building sampler from {args.fm_pipeline_dir}")
    sampler, device = _build_sampler(args)

    os.makedirs(args.output_dir, exist_ok=True)

    n_total = len(entries)
    generated = 0
    batch_size = args.fm_batch_size

    print(f"[FM] Generating {n_total} samples (batch_size={batch_size}, steps={args.fm_steps}) ...")
    with tqdm(total=n_total, desc="[FM] samples", unit="img") as pbar:
        while generated < n_total:
            bs = min(batch_size, n_total - generated)
            z = sampler.sample_euler(steps=args.fm_steps, batch_size=bs)
            x_gen = sampler.decode(z)

            for j in range(bs):
                raw_uint16 = fm_output_to_uint16(x_gen[j])
                out_path = os.path.join(args.output_dir, f"sample_{generated:05d}.npy")
                np.save(out_path, raw_uint16)

                png_path = os.path.join(args.output_dir, f"sample_{generated:05d}.png")
                vis = uint16_to_png_uint8(raw_uint16)
                Image.fromarray(vis, mode="L").save(png_path)

                generated += 1
                pbar.update(1)

    meta_out = os.path.join(args.output_dir, "metadata.jsonl")
    with open(meta_out, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(entries):
            record = {
                "file_name": f"sample_{idx:05d}.npy",
                "text": entry.get("text", ""),
                "source_file": entry.get("file_name", ""),
            }
            f.write(json.dumps(record) + "\n")
    print(f"[FM] Done. {n_total} samples in {args.output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic IR datasets using SD1.5-LoRA or Stable Flow Matching."
    )

    # Config file (optional — values override argparse defaults, CLI overrides config)
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file. CLI flags override config values.")

    # --- general ---
    p.add_argument("--mode", type=str, default=None, choices=["sd15", "fm"],
                   help="Which generator to use: 'sd15' or 'fm'.")
    p.add_argument("--metadata", type=str, default="./data/raw/v18/metadata.jsonl",
                   help="Path to metadata.jsonl with prompts.")
    p.add_argument("--max_samples", type=int, default=100,
                   help="Number of entries to read from metadata.")
    p.add_argument("--output_dir", type=str, default="./artifacts/generated/main",
                   help="Directory to save generated .npy files.")
    p.add_argument("--seed", type=int, default=42)

    # --- SD 1.5 specific ---
    sd = p.add_argument_group("SD 1.5 options")
    sd.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5")
    sd.add_argument("--stage1_dir", type=str, default=None)
    sd.add_argument("--lora_dir", type=str, default=None)
    sd.add_argument("--lora_rank", type=int, default=4)
    sd.add_argument("--sd_steps", type=int, default=30)
    sd.add_argument("--guidance", type=float, default=7.5)
    sd.add_argument("--negative_prompt", type=str, default=None)
    sd.add_argument("--generic_prompt", action="store_true")
    sd.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16", "fp32"])
    sd.add_argument("--max_tries", type=int, default=5)
    sd.add_argument("--lora_alpha_scale", type=float, default=1.0)

    # --- Flow Matching specific ---
    fm = p.add_argument_group("Flow Matching options")
    fm.add_argument("--fm_pipeline_dir", type=str, default=None)
    fm.add_argument("--fm_vae_weights", type=str, default=None)
    fm.add_argument("--fm_t_scale", type=float, default=1000.0)
    fm.add_argument("--fm_steps", type=int, default=50)
    fm.add_argument("--fm_batch_size", type=int, default=8)

    p.add_argument("--device", type=str, default=None)

    # Two-pass parse: first grab --config, apply YAML defaults, then re-parse
    preliminary, _ = p.parse_known_args()
    apply_yaml_defaults(p, preliminary.config)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    entries = load_metadata(args.metadata, args.max_samples)
    print(f"Loaded {len(entries)} entries from {args.metadata}")

    if args.mode == "sd15":
        if args.stage1_dir is None and args.lora_dir is None:
            raise ValueError("--stage1_dir or --lora_dir is required for mode=sd15")
        generate_sd15(args, entries)

    elif args.mode == "fm":
        if args.fm_pipeline_dir is None:
            raise ValueError("--fm_pipeline_dir is required for mode=fm")
        generate_fm(args, entries)


if __name__ == "__main__":
    main()

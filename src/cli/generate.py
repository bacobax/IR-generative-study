"""Modular CLI entrypoint for synthetic dataset generation.

This module is the **source of truth** for generating synthetic IR datasets
using either SD 1.5 with LoRA or Stable Flow Matching (plain or guided).

The root-level ``generate_datasets.py`` is a thin compatibility wrapper that
forwards to :func:`main` here.

Usage::

    # SD 1.5 LoRA
    python -m src.cli.generate \\
        --mode sd15 \\
        --lora_dir ./artifacts/checkpoints/stable_diffusion/lora_runs/.../checkpoint-32000 \\
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

from src.core.normalization import (
    sd_output_to_uint16,
    fm_output_to_uint16,
    uint16_to_png_uint8,
)
from src.core.configs.fm_config import FMSampleConfig
from src.core.configs.config_loader import apply_yaml_defaults
from src.core.registry import REGISTRIES

# Ensure default FM components are registered
import src.algorithms.inference.flow_matching_sampler  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════
# Metadata reader
# ═══════════════════════════════════════════════════════════════════════════

def load_metadata(jsonl_path: str, max_samples: int) -> List[Dict]:
    entries: List[Dict] = []
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
    from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict
    from diffusers.utils import convert_unet_state_dict_to_peft

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    weight_dtype = dtype_map[args.precision]
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"[SD1.5] Loading base model: {args.base_model}")
    tokenizer = CLIPTokenizer.from_pretrained(args.base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(
        args.base_model, subfolder="text_encoder", torch_dtype=weight_dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        args.base_model, subfolder="vae", torch_dtype=weight_dtype,
    )
    unet = UNet2DConditionModel.from_pretrained(
        args.base_model, subfolder="unet", torch_dtype=weight_dtype,
    )

    # --- Apply LoRA ---
    print(f"[SD1.5] Loading LoRA weights from {args.lora_dir}")
    lora_state_dict, _ = StableDiffusionPipeline.lora_state_dict(args.lora_dir)
    unet_state_dict = {
        k.replace("unet.", ""): v for k, v in lora_state_dict.items() if k.startswith("unet.")
    }
    unet_state_dict = convert_unet_state_dict_to_peft(unet_state_dict)

    unet_lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * args.lora_alpha_scale,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"],
    )
    print(f"[SD1.5] Applying LoRA with rank={args.lora_rank} and alpha={unet_lora_config.lora_alpha}")
    unet.add_adapter(unet_lora_config)
    set_peft_model_state_dict(unet, unet_state_dict, adapter_name="default")

    unet = unet.to(device)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.base_model,
        unet=unet,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        torch_dtype=weight_dtype,
    )
    pipe.to(device)

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=weight_dtype)
        if device.startswith("cuda") and weight_dtype != torch.float32
        else nullcontext()
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[SD1.5] Generating {len(entries)} samples ...")
    generic_prompt_text = "overhead infrared surveillance image with any people or objects"
    for idx, entry in enumerate(entries):
        prompt = generic_prompt_text if args.generic_prompt else entry.get("text", generic_prompt_text)
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

        raw_uint16 = sd_output_to_uint16(image)
        out_path = os.path.join(args.output_dir, f"sample_{idx:05d}.npy")
        np.save(out_path, raw_uint16)

        png_path = os.path.join(args.output_dir, f"sample_{idx:05d}.png")
        vis = uint16_to_png_uint8(raw_uint16)
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
# Stable Flow-Matching generator with guidance
# ═══════════════════════════════════════════════════════════════════════════

def generate_fm_guided(args, entries: List[Dict]):
    """Guided FM generation supporting all sampling methods."""
    import sys as _sys
    from src.core.paths import legacy_code_root
    _sys.path.insert(0, str(legacy_code_root()))
    from fm_src.guidance.score_predictor_guidance import (  # noqa: E402
        ScoreGuidanceConfig,
        ScorePredictorGuidance,
    )

    method = args.fm_guidance_method

    print(f"[FM-guided:{method}] Building sampler from {args.fm_pipeline_dir}")
    sampler, device = _build_sampler(args)

    guidance = None
    if method != "euler":
        if not args.fm_surprise_ckpt:
            raise ValueError(
                f"--fm_surprise_ckpt is required for fm_guidance_method='{method}'"
            )
        cfg = ScoreGuidanceConfig(
            ckpt_path=args.fm_surprise_ckpt,
            vae_config_path=args.fm_predictor_vae_config,
            vae_weights_path=args.fm_predictor_vae_weights,
            dino_name=args.fm_dino_name,
            hidden_dim=args.fm_hidden_dim,
            energy_mode=args.fm_energy_mode,
            sign=args.fm_sign,
            w_surprise=args.fm_w_surprise,
            w_gmm=args.fm_w_gmm,
            lambda_start=args.fm_lambda_start,
            lambda_end=args.fm_lambda_end,
            lambda_schedule=args.fm_lambda_schedule,
            grad_clip_norm=args.fm_grad_clip_norm,
            normalize_grad=args.fm_normalize_grad,
            guidance_on=args.fm_guidance_on,
            use_ddim_hat=args.fm_use_ddim_hat,
            use_amp=args.fm_use_amp,
            num_refine_steps=args.fm_num_refine_steps,
            refine_step_size=args.fm_refine_step_size,
        )
        guidance = ScorePredictorGuidance.from_checkpoint(cfg, device=device)
        print(
            f"[FM-guided:{method}] Guidance loaded.  "
            f"energy={args.fm_energy_mode}  sign={args.fm_sign}"
        )

    os.makedirs(args.output_dir, exist_ok=True)

    n_total = len(entries)
    generated = 0
    batch_size = args.fm_batch_size

    print(
        f"[FM-guided:{method}] Generating {n_total} samples "
        f"(batch_size={batch_size}, steps={args.fm_steps}) ..."
    )

    with tqdm(total=n_total, desc=f"[FM-guided:{method}] samples", unit="img") as pbar:
        while generated < n_total:
            bs = min(batch_size, n_total - generated)

            if method == "euler":
                z = sampler.sample_euler(steps=args.fm_steps, batch_size=bs)
            elif method == "euler_guided":
                z = sampler.sample_euler_guided(
                    steps=args.fm_steps,
                    batch_size=bs,
                    guidance=guidance,
                    guidance_scale=args.fm_guidance_scale,
                    return_logs=False,
                )
            elif method == "rerank":
                z = sampler.sample_euler_with_candidates(
                    steps=args.fm_steps,
                    n_candidates=args.fm_n_candidates,
                    keep_top_k=1,
                    batch_size=bs,
                    guidance=guidance,
                    guidance_scale=0.0,
                )
            elif method == "beam":
                z = sampler.sample_euler_beam(
                    steps=args.fm_steps,
                    batch_size=bs,
                    beam_size=args.fm_beam_size,
                    branch_factor=args.fm_branch_factor,
                    sigma_perturb=args.fm_sigma_perturb,
                    guidance=guidance,
                    guidance_scale=0.0,
                    return_all_beams=False,
                )
            elif method == "refine":
                z = sampler.sample_euler(steps=args.fm_steps, batch_size=bs)
                z = sampler.refine_latents_energy(
                    z=z,
                    guidance=guidance,
                    num_steps=args.fm_num_refine_steps,
                    step_size=args.fm_refine_step_size,
                )
            else:
                raise ValueError(f"Unknown fm_guidance_method: {method!r}")

            with torch.no_grad():
                x_gen = sampler.decode(z)

            for j in range(x_gen.shape[0]):
                raw_uint16 = fm_output_to_uint16(x_gen[j])
                out_path = os.path.join(args.output_dir, f"sample_{generated:05d}.npy")
                np.save(out_path, raw_uint16)

                png_path = os.path.join(args.output_dir, f"sample_{generated:05d}.png")
                vis = uint16_to_png_uint8(raw_uint16)
                Image.fromarray(vis, mode="L").save(png_path)

                generated += 1
                pbar.update(1)
                if generated >= n_total:
                    break

    if guidance is not None:
        scores = guidance.log_scores(z)
        print(
            f"[FM-guided:{method}] Last-batch mean scores: "
            f"surprise={scores['mean_surprise']:.4f}  "
            f"gmm={scores['mean_gmm']:.4f}"
        )

    meta_out = os.path.join(args.output_dir, "metadata.jsonl")
    with open(meta_out, "w", encoding="utf-8") as f:
        for idx, entry in enumerate(entries):
            record = {
                "file_name": f"sample_{idx:05d}.npy",
                "text": entry.get("text", ""),
                "source_file": entry.get("file_name", ""),
                "fm_guidance_method": method,
                "fm_energy_mode": args.fm_energy_mode if guidance else "none",
                "fm_sign": args.fm_sign if guidance else "none",
            }
            f.write(json.dumps(record) + "\n")
    print(f"[FM-guided:{method}] Done. {n_total} samples in {args.output_dir}")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
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

    # --- FM guidance ---
    fg = p.add_argument_group("FM Guidance options (mode=fm only)")
    fg.add_argument("--fm_guidance_method", type=str, default="euler",
                    choices=["euler", "euler_guided", "rerank", "beam", "refine"])
    fg.add_argument("--fm_surprise_ckpt", type=str, default=None)
    fg.add_argument("--fm_predictor_vae_config", type=str, default=None)
    fg.add_argument("--fm_predictor_vae_weights", type=str, default=None)
    fg.add_argument("--fm_dino_name", type=str, default=None)
    fg.add_argument("--fm_hidden_dim", type=int, default=None)
    fg.add_argument("--fm_energy_mode", type=str, default="surprise",
                    choices=["surprise", "gmm", "combo"])
    fg.add_argument("--fm_sign", type=str, default="minimize",
                    choices=["minimize", "maximize"])
    fg.add_argument("--fm_w_surprise", type=float, default=1.0)
    fg.add_argument("--fm_w_gmm", type=float, default=1.0)
    fg.add_argument("--fm_guidance_scale", type=float, default=1.0)
    fg.add_argument("--fm_lambda_start", type=float, default=1.0)
    fg.add_argument("--fm_lambda_end", type=float, default=1.0)
    fg.add_argument("--fm_lambda_schedule", type=str, default="constant",
                    choices=["constant", "linear", "cosine", "step"])
    fg.add_argument("--fm_grad_clip_norm", type=float, default=None)
    fg.add_argument("--fm_normalize_grad", action="store_true")
    fg.add_argument("--fm_guidance_on", type=str, default="latent",
                    choices=["latent", "decoded"])
    fg.add_argument("--fm_use_ddim_hat", action="store_true")
    fg.add_argument("--fm_use_amp", action="store_true")
    fg.add_argument("--fm_n_candidates", type=int, default=8)
    fg.add_argument("--fm_beam_size", type=int, default=4)
    fg.add_argument("--fm_branch_factor", type=int, default=2)
    fg.add_argument("--fm_sigma_perturb", type=float, default=0.05)
    fg.add_argument("--fm_num_refine_steps", type=int, default=10)
    fg.add_argument("--fm_refine_step_size", type=float, default=0.01)

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
        if args.lora_dir is None:
            raise ValueError("--lora_dir is required for mode=sd15")
        generate_sd15(args, entries)

    elif args.mode == "fm":
        if args.fm_pipeline_dir is None:
            raise ValueError("--fm_pipeline_dir is required for mode=fm")
        if args.fm_guidance_method == "euler":
            generate_fm(args, entries)
        else:
            generate_fm_guided(args, entries)


if __name__ == "__main__":
    main()

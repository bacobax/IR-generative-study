#!/usr/bin/env python3
"""
Generate synthetic datasets using either:
  1. Pretrained Stable-Diffusion 1.5 with LoRA weights (prompt-conditioned)
  2. Pretrained Stable Flow-Matching pipeline      (unconditional)

Both generators output 1-channel float32 .npy arrays in [-1, 1].

Usage examples:

  # SD 1.5 LoRA
  python generate_datasets.py \
      --mode sd15 \
      --lora_dir ./stable_diffusion_15_out/out_ir_lora_sd15r4_p_norm/checkpoint-32000 \
      --max_samples 100 \
      --output_dir ./generated/sd15

  # Stable Flow Matching
  python generate_datasets.py \
      --mode fm \
      --fm_pipeline_dir ./serious_runs/stable_training_t_scaled \
      --fm_vae_weights ./fm_src/vae_best.pt \
      --max_samples 100 \
      --output_dir ./generated/fm
"""

import argparse
import json
import os
from contextlib import nullcontext
from typing import List, Dict

import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers shared with the rest of the codebase
# ---------------------------------------------------------------------------
# Raw-to-[-1,1] constants (from sd_src/helpers.py)
A_RAW = 11667.0  # p0.001 percentile
B_RAW = 13944.0  # p99.999 percentile
S_RAW = B_RAW - A_RAW


def sd_output_to_raw_m1t1(image_pil) -> np.ndarray:
    """
    SD 1.5 output → 1-ch float32 in [-1, 1].
    Mirrors the conversion in sd_src/scripts/sd_demo.py.
    """
    raw = np.asarray(image_pil).astype(np.float32) / 255.0
    # 3-ch RGB → grayscale (mean)
    if raw.ndim == 3:
        raw = raw.mean(axis=-1)
    # map [0,1] back through the uint16 IR range → [-1, 1]
    rawu16 = A_RAW + raw * S_RAW
    rawm1t1 = 2.0 * np.clip((rawu16 - A_RAW) / S_RAW, 0.0, 1.0) - 1.0
    return rawm1t1.astype(np.float32)


def fm_output_to_raw_m1t1(tensor: torch.Tensor) -> np.ndarray:
    """
    Flow-matching output tensor (1, H, W) already in [-1, 1].
    Clamp and return as float32 numpy.
    """
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]  # (1, H, W) → (H, W)
    return np.clip(arr, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Metadata reader
# ---------------------------------------------------------------------------
def load_metadata(jsonl_path: str, max_samples: int) -> List[Dict]:
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
            if len(entries) >= max_samples:
                break
    return entries


# ---------------------------------------------------------------------------
# SD 1.5 generator
# ---------------------------------------------------------------------------
def generate_sd15(args, entries: List[Dict]):
    from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from peft import LoraConfig
    from peft.utils import set_peft_model_state_dict
    from diffusers.utils import convert_unet_state_dict_to_peft

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    weight_dtype = dtype_map[args.precision]
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
        lora_alpha=args.lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0", "proj_in", "proj_out"],
    )
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
        if device == "cuda" and weight_dtype != torch.float32
        else nullcontext()
    )

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[SD1.5] Generating {len(entries)} samples ...")
    for idx, entry in enumerate(entries):
        prompt = entry.get("text", "overhead infrared surveillance image, circular field of view")
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

        raw_m1t1 = sd_output_to_raw_m1t1(image)
        out_path = os.path.join(args.output_dir, f"sample_{idx:05d}.npy")
        np.save(out_path, raw_m1t1)

        png_path = os.path.join(args.output_dir, f"sample_{idx:05d}.png")
        # grayscale from [-1,1] → [0,255]
        vis = ((raw_m1t1 + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
        Image.fromarray(vis, mode="L").save(png_path)

        if (idx + 1) % 50 == 0 or idx == len(entries) - 1:
            print(f"  [{idx + 1}/{len(entries)}] saved {out_path}")

    # Save a companion metadata file so generated samples are traceable
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


# ---------------------------------------------------------------------------
# Stable Flow-Matching generator
# ---------------------------------------------------------------------------
def generate_fm(args, entries: List[Dict]):
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipelines"))
    from flow_matching_pipeline import StableFlowMatchingPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[FM] Building pipeline from {args.fm_pipeline_dir}")
    pipe = StableFlowMatchingPipeline(
        device=device,
        t_scale=args.fm_t_scale,
        model_dir=args.fm_pipeline_dir,
    )

    # Build models from saved configs
    pipe.load_from_pipeline_folder_auto(
        args.fm_pipeline_dir,
        strict=True,
        map_location=device,
        set_eval=True,
    )

    # If separate VAE weights are provided, load them on top
    if args.fm_vae_weights is not None:
        print(f"[FM] Loading VAE weights from {args.fm_vae_weights}")
        pipe.load_vae_weights(args.fm_vae_weights, strict=True, map_location=device)
        pipe.vae.eval()

    pipe.freeze_vae()

    os.makedirs(args.output_dir, exist_ok=True)

    n_total = len(entries)
    generated = 0
    batch_size = args.fm_batch_size

    print(f"[FM] Generating {n_total} samples (batch_size={batch_size}, steps={args.fm_steps}) ...")
    while generated < n_total:
        bs = min(batch_size, n_total - generated)
        z = pipe.sample_euler(steps=args.fm_steps, batch_size=bs)
        x_gen = pipe.decode_fm_output(z)  # (bs, 1, H, W), [-1, 1]

        for j in range(bs):
            raw_m1t1 = fm_output_to_raw_m1t1(x_gen[j])
            out_path = os.path.join(args.output_dir, f"sample_{generated:05d}.npy")
            np.save(out_path, raw_m1t1)

            png_path = os.path.join(args.output_dir, f"sample_{generated:05d}.png")
            vis = ((raw_m1t1 + 1.0) / 2.0 * 255.0).clip(0, 255).astype(np.uint8)
            Image.fromarray(vis, mode="L").save(png_path)

            generated += 1

        if generated % 50 == 0 or generated == n_total:
            print(f"  [{generated}/{n_total}] saved")

    # Companion metadata (FM is unconditional, but we keep the entries for reference)
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Generate synthetic IR datasets using SD1.5-LoRA or Stable Flow Matching."
    )

    # --- general ---
    p.add_argument("--mode", type=str, required=True, choices=["sd15", "fm"],
                   help="Which generator to use: 'sd15' or 'fm'.")
    p.add_argument("--metadata", type=str, default="./v18/metadata.jsonl",
                   help="Path to metadata.jsonl with prompts.")
    p.add_argument("--max_samples", type=int, default=100,
                   help="Number of entries to read from metadata (= number of images to generate).")
    p.add_argument("--output_dir", type=str, default="./generated",
                   help="Directory to save generated .npy files.")
    p.add_argument("--seed", type=int, default=42)

    # --- SD 1.5 specific ---
    sd = p.add_argument_group("SD 1.5 options")
    sd.add_argument("--base_model", type=str, default="runwayml/stable-diffusion-v1-5",
                    help="HuggingFace model id for the base SD model.")
    sd.add_argument("--lora_dir", type=str, default=None,
                    help="Folder containing pytorch_lora_weights.safetensors.")
    sd.add_argument("--lora_rank", type=int, default=4,
                    help="LoRA rank used during training.")
    sd.add_argument("--sd_steps", type=int, default=30,
                    help="Number of diffusion sampling steps.")
    sd.add_argument("--guidance", type=float, default=7.5,
                    help="Classifier-free guidance scale.")
    sd.add_argument("--negative_prompt", type=str, default=None)
    sd.add_argument("--precision", type=str, default="fp16",
                    choices=["fp16", "bf16", "fp32"])
    sd.add_argument("--max_tries", type=int, default=5,
                    help="Max retries per prompt when NSFW is detected.")

    # --- Flow Matching specific ---
    fm = p.add_argument_group("Flow Matching options")
    fm.add_argument("--fm_pipeline_dir", type=str, default=None,
                    help="Root folder of a StableFlowMatchingPipeline save (contains UNET/ and VAE/).")
    fm.add_argument("--fm_vae_weights", type=str, default=None,
                    help="Optional: explicit VAE .pt weights to load (overrides auto-detected ones).")
    fm.add_argument("--fm_t_scale", type=float, default=1000.0,
                    help="Time-scaling factor used during FM training.")
    fm.add_argument("--fm_steps", type=int, default=50,
                    help="Number of Euler sampling steps.")
    fm.add_argument("--fm_batch_size", type=int, default=8,
                    help="Batch size for FM sampling.")

    return p.parse_args()


def main():
    args = parse_args()

    # Load metadata entries
    entries = load_metadata(args.metadata, args.max_samples)
    print(f"Loaded {len(entries)} entries from {args.metadata}")

    if args.mode == "sd15":
        if args.lora_dir is None:
            raise ValueError("--lora_dir is required for mode=sd15")
        generate_sd15(args, entries)

    elif args.mode == "fm":
        if args.fm_pipeline_dir is None:
            raise ValueError("--fm_pipeline_dir is required for mode=fm")
        generate_fm(args, entries)


if __name__ == "__main__":
    main()

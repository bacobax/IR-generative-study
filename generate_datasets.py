#!/usr/bin/env python3
"""
Generate synthetic datasets using either:
1. Pretrained Stable-Diffusion 1.5 with LoRA weights (prompt-conditioned)
2. Pretrained Stable Flow-Matching pipeline      (unconditional)

Both generators output 1-channel uint16 .npy arrays and uint8 .png previews.

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
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Helpers shared with the rest of the codebase
# ---------------------------------------------------------------------------
# Raw-to-[-1,1] constants (from sd_src/helpers.py)
A_RAW = 11667.0  # p0.001 percentile
B_RAW = 13944.0  # p99.999 percentile
S_RAW = B_RAW - A_RAW


def sd_output_to_uint16(image_pil) -> np.ndarray:
    """
    SD 1.5 output → 1-ch uint16.
    Reverse the percentile normalisation used during SD training:
    training mapped raw uint16 from [A, B] → [-1, 1], so the inverse is
    uint16 = A + grey_01 * S,  where grey_01 is the [0,1] grayscale output.
    """
    raw = np.asarray(image_pil).astype(np.float32) / 255.0
    if raw.ndim == 3:
        raw = raw.mean(axis=-1)  # RGB → grayscale
    uint16_val = A_RAW + np.clip(raw, 0.0, 1.0) * S_RAW
    return np.clip(uint16_val, 0, 65535).astype(np.uint16)


# def from_norm_to_uint16(recon: torch.Tensor) -> torch.Tensor:
#     # Reverse: [-1, 1] -> uint16 [0, 65535] for saving
#     return ((recon + 1) / 2) * S + A


def fm_output_to_uint16(tensor: torch.Tensor) -> np.ndarray:
    """
    Flow-matching output tensor (1, H, W) in [-1, 1] → uint16 [0, 65535].
    Reverse of the full linear normalisation: ((x + 1) / 2) * 65535.
    """
    arr = tensor.detach().cpu().float().numpy()
    if arr.ndim == 3:
        arr = arr[0]  # (1, H, W) → (H, W)
    # Clamp tiny out-of-range values, then round to nearest uint16.
    uint16_val = ((np.clip(arr, -1.0, 1.0) + 1.0) / 2.0) * S_RAW + A_RAW
    return np.rint(uint16_val).astype(np.uint16)


def uint16_to_png_uint8(arr_uint16: np.ndarray) -> np.ndarray:
    """Convert uint16 image to uint8 for PNG visualization only.

    Uses per-image percentile stretch (p1–p99) to avoid low-contrast previews.
    """
    arr = arr_uint16.astype(np.float32)
    p1 = float(np.percentile(arr, 1.0))
    p99 = float(np.percentile(arr, 99.0))
    if p99 <= p1:
        return np.zeros_like(arr_uint16, dtype=np.uint8)
    norm = (arr - p1) / (p99 - p1)
    return (norm * 255.0).clip(0, 255).astype(np.uint8)


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
        lora_alpha=args.lora_rank*args.lora_alpha_scale,
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
        # PNG visualization only (normalized to [0, 255])
        vis = uint16_to_png_uint8(raw_uint16)
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
# Stable Flow-Matching generator  (plain Euler – unchanged baseline)
# ---------------------------------------------------------------------------
def generate_fm(args, entries: List[Dict]):
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "pipelines"))
    from fm_src.pipelines.flow_matching_pipeline import StableFlowMatchingPipeline

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

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
    with tqdm(total=n_total, desc="[FM] samples", unit="img") as pbar:
        while generated < n_total:
            bs = min(batch_size, n_total - generated)
            z = pipe.sample_euler(steps=args.fm_steps, batch_size=bs)
            x_gen = pipe.decode_fm_output(z)  # (bs, 1, H, W), [-1, 1]

            for j in range(bs):
                raw_uint16 = fm_output_to_uint16(x_gen[j])
                out_path = os.path.join(args.output_dir, f"sample_{generated:05d}.npy")
                np.save(out_path, raw_uint16)

                png_path = os.path.join(args.output_dir, f"sample_{generated:05d}.png")
                # PNG visualization only (normalized to [0, 255])
                vis = uint16_to_png_uint8(raw_uint16)
                Image.fromarray(vis, mode="L").save(png_path)

                generated += 1
                pbar.update(1)

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
# Stable Flow-Matching generator with guidance
# ---------------------------------------------------------------------------
def generate_fm_guided(args, entries: List[Dict]):
    """Guided FM generation supporting all sampling methods.

    When ``args.fm_guidance_method == 'euler'`` this is equivalent to the
    baseline :func:`generate_fm` but unified for consistency.
    All other methods require ``args.fm_surprise_ckpt`` to be set.
    """
    import sys

    sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
    from fm_src.pipelines.flow_matching_pipeline import StableFlowMatchingPipeline
    from fm_src.guidance.score_predictor_guidance import (
        ScoreGuidanceConfig,
        ScorePredictorGuidance,
    )

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    method = args.fm_guidance_method

    # ---- Build and load pipeline -------------------------------------------
    print(f"[FM-guided:{method}] Building pipeline from {args.fm_pipeline_dir}")
    pipe = StableFlowMatchingPipeline(
        device=device,
        t_scale=args.fm_t_scale,
        model_dir=args.fm_pipeline_dir,
    )
    pipe.load_from_pipeline_folder_auto(
        args.fm_pipeline_dir,
        strict=True,
        map_location=device,
        set_eval=True,
    )
    if args.fm_vae_weights is not None:
        print(f"[FM-guided:{method}] Loading VAE weights from {args.fm_vae_weights}")
        pipe.load_vae_weights(args.fm_vae_weights, strict=True, map_location=device)
        pipe.vae.eval()
    pipe.freeze_vae()

    # ---- Build guidance (only needed for non-euler methods) ----------------
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
        print(f"[FM-guided:{method}] Guidance loaded.  "
            f"energy={args.fm_energy_mode}  sign={args.fm_sign}")

    # ---- Generation loop ---------------------------------------------------
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

            # ---- Dispatch to the requested sampling method ----
            if method == "euler":
                z = pipe.sample_euler(steps=args.fm_steps, batch_size=bs)

            elif method == "euler_guided":
                z = pipe.sample_euler_guided(
                    steps=args.fm_steps,
                    batch_size=bs,
                    guidance=guidance,
                    guidance_scale=args.fm_guidance_scale,
                    return_logs=False,
                )

            elif method == "rerank":
                z = pipe.sample_euler_with_candidates(
                    steps=args.fm_steps,
                    n_candidates=args.fm_n_candidates,
                    keep_top_k=1,          # keep best per slot; expand batch externally
                    batch_size=bs,
                    guidance=guidance,
                    guidance_scale=0.0,    # score at end, no inline guidance
                )

            elif method == "beam":
                z = pipe.sample_euler_beam(
                    steps=args.fm_steps,
                    batch_size=bs,
                    beam_size=args.fm_beam_size,
                    branch_factor=args.fm_branch_factor,
                    sigma_perturb=args.fm_sigma_perturb,
                    guidance=guidance,
                    guidance_scale=0.0,    # use guidance only for beam pruning
                    return_all_beams=False,
                )

            elif method == "refine":
                z = pipe.sample_euler(
                    steps=args.fm_steps,
                    batch_size=bs,
                )
                z = pipe.refine_latents_energy(
                    z=z,
                    guidance=guidance,
                    num_steps=args.fm_num_refine_steps,
                    step_size=args.fm_refine_step_size,
                )

            else:
                raise ValueError(f"Unknown fm_guidance_method: {method!r}")

            # ---- Decode + save -------------------------------------------------
            with torch.no_grad():
                x_gen = pipe.decode_fm_output(z)   # (bs, 1, H, W), [-1, 1]

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

    # ---- Log mean predicted scores (if guidance available) -----------------
    if guidance is not None:
        # Score the last batch as a proxy
        scores = guidance.log_scores(z)
        print(
            f"[FM-guided:{method}] Last-batch mean scores: "
            f"surprise={scores['mean_surprise']:.4f}  "
            f"gmm={scores['mean_gmm']:.4f}"
        )

    # ---- Companion metadata ------------------------------------------------
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
    sd.add_argument("--generic_prompt", action="store_true",
                    help="Ignore per-sample captions and use the fixed generic prompt: "
                        "'overhead infrared surveillance image, circular field of view'.")
    sd.add_argument("--precision", type=str, default="fp16",
                    choices=["fp16", "bf16", "fp32"])
    sd.add_argument("--max_tries", type=int, default=5,
                    help="Max retries per prompt when NSFW is detected.")

    sd.add_argument("--lora_alpha_scale", type=float, default=1.0,
                    help="Scaling factor for LoRA alpha (default=1.0 means alpha = rank). Adjusting this can control the strength of the LoRA effect during generation.")

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

    # --- FM guidance (all methods) ---
    fg = p.add_argument_group("FM Guidance options (mode=fm only)")
    fg.add_argument(
        "--fm_guidance_method", type=str, default="euler",
        choices=["euler", "euler_guided", "rerank", "beam", "refine"],
        help="FM sampling method.  'euler' = plain baseline (default, no predictor needed).",
    )
    fg.add_argument("--fm_surprise_ckpt", type=str, default=None,
                    help="Path to best_model.pt from train_surprise_predictor.py. "
                        "Required for all fm_guidance_method values except 'euler'.")
    fg.add_argument("--fm_predictor_vae_config", type=str, default=None,
                    help="VAE config JSON for the predictor. "
                        "Auto-read from ckpt['args'] if not given.")
    fg.add_argument("--fm_predictor_vae_weights", type=str, default=None,
                    help="VAE weights .pt for the predictor. "
                        "Auto-read from ckpt['args'] if not given.")
    fg.add_argument("--fm_dino_name", type=str, default=None,
                    help="DINOv2 variant, e.g. 'dinov2_vits14'. "
                        "Auto-read from ckpt['args'] if not given.")
    fg.add_argument("--fm_hidden_dim", type=int, default=None,
                    help="Predictor MLP hidden dim. Auto-read from ckpt if not given.")
    fg.add_argument(
        "--fm_energy_mode", type=str, default="surprise",
        choices=["surprise", "gmm", "combo"],
        help="Energy to optimise: 'surprise', 'gmm', or 'combo' (weighted sum).",
    )
    fg.add_argument(
        "--fm_sign", type=str, default="minimize",
        choices=["minimize", "maximize"],
        help="Guidance direction: 'minimize' drives toward lower energy, "
            "'maximize' drives toward higher energy.",
    )
    fg.add_argument("--fm_w_surprise", type=float, default=1.0,
                    help="[combo] Weight for the surprise energy term.")
    fg.add_argument("--fm_w_gmm", type=float, default=1.0,
                    help="[combo] Weight for the gmm energy term.")
    fg.add_argument("--fm_guidance_scale", type=float, default=1.0,
                    help="[euler_guided] Global guidance scale multiplier.")
    fg.add_argument("--fm_lambda_start", type=float, default=1.0,
                    help="λ value at t=0.")
    fg.add_argument("--fm_lambda_end", type=float, default=1.0,
                    help="λ value at t=1.")
    fg.add_argument(
        "--fm_lambda_schedule", type=str, default="constant",
        choices=["constant", "linear", "cosine", "step"],
        help="Schedule for λ(t).",
    )
    fg.add_argument("--fm_grad_clip_norm", type=float, default=None,
                    help="Per-sample guidance gradient norm cap. None = no clipping.")
    fg.add_argument("--fm_normalize_grad", action="store_true",
                    help="Normalise guidance gradient to unit norm before scaling.")
    fg.add_argument(
        "--fm_guidance_on", type=str, default="latent",
        choices=["latent", "decoded"],
        help="Compute guidance on the latent or decoded image.",
    )
    fg.add_argument("--fm_use_ddim_hat", action="store_true",
                    help="Evaluate guidance energy on DDIM-style clean estimate z_hat(z_t, t, v) "
                        "while differentiating w.r.t. z_t.")
    fg.add_argument("--fm_use_amp", action="store_true",
                    help="Use AMP autocast in the predictor forward pass.")
    # rerank
    fg.add_argument("--fm_n_candidates", type=int, default=8,
                    help="[rerank] Number of candidate trajectories to generate.")
    # beam
    fg.add_argument("--fm_beam_size", type=int, default=4,
                    help="[beam] Number of surviving beams per step.")
    fg.add_argument("--fm_branch_factor", type=int, default=2,
                    help="[beam] Branching fan-out at each step.")
    fg.add_argument("--fm_sigma_perturb", type=float, default=0.05,
                    help="[beam] Std-dev of Gaussian noise added when branching.")
    # refine
    fg.add_argument("--fm_num_refine_steps", type=int, default=10,
                    help="[refine] Number of post-sampling gradient refinement steps.")
    fg.add_argument("--fm_refine_step_size", type=float, default=0.01,
                    help="[refine] Step size per refinement step.")

    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string for torch (e.g., 'cuda', 'cuda:1', or 'cpu'). Defaults to auto.",
    )

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
        if args.fm_guidance_method == "euler":
            # Backward-compatible plain Euler path.
            generate_fm(args, entries)
        else:
            generate_fm_guided(args, entries)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/sample_guided_fm.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
CLI for guided sampling from a trained StableFlowMatchingPipeline.

Supports the following sampling methods:
  euler          – plain Euler ODE (baseline, no guidance)
  euler_guided   – Euler ODE with predictor energy-based guidance
  rerank         – generate N candidates, keep top-K by predictor energy
  beam           – beam / branching sampling pruned by predictor energy
  refine         – plain Euler followed by latent gradient refinement

Example usage
-------------
# Baseline (no guidance):
python scripts/sample_guided_fm.py \\
    --fm_pipeline_dir runs/my_run_001 \\
    --method euler \\
    --batch_size 8 --steps 50 \\
    --output_dir generated/guided_test

# Guided sampling:
python scripts/sample_guided_fm.py \\
    --fm_pipeline_dir runs/my_run_001 \\
    --surprise_ckpt runs/surprise_predictor/best_model.pt \\
    --method euler_guided \\
    --energy_mode surprise --sign minimize \\
    --guidance_scale 2.0 --lambda_schedule cosine \\
    --lambda_start 3.0 --lambda_end 0.5 \\
    --grad_clip_norm 1.0 \\
    --batch_size 8 --steps 50 \\
    --output_dir generated/guided_test

# Reranking:
python scripts/sample_guided_fm.py \\
    --fm_pipeline_dir runs/my_run_001 \\
    --surprise_ckpt runs/surprise_predictor/best_model.pt \\
    --method rerank --n_candidates 16 --keep_top_k 4 \\
    --output_dir generated/rerank_test

# Beam search:
python scripts/sample_guided_fm.py \\
    --fm_pipeline_dir runs/my_run_001 \\
    --surprise_ckpt runs/surprise_predictor/best_model.pt \\
    --method beam --beam_size 4 --branch_factor 2 \\
    --output_dir generated/beam_test

# Refinement:
python scripts/sample_guided_fm.py \\
    --fm_pipeline_dir runs/my_run_001 \\
    --surprise_ckpt runs/surprise_predictor/best_model.pt \\
    --method refine \\
    --num_refine_steps 20 --refine_step_size 0.005 \\
    --output_dir generated/refine_test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

import torch
import torchvision.utils as vutils

# ---------------------------------------------------------------------------
# Ensure repo root is importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fm_src.pipelines.flow_matching_pipeline import StableFlowMatchingPipeline  # noqa: E402
from fm_src.guidance.score_predictor_guidance import (  # noqa: E402
    ScoreGuidanceConfig,
    ScorePredictorGuidance,
    run_sanity_check,
)
from src.core.configs.config_loader import apply_yaml_defaults  # noqa: E402


# =============================================================================
# Helpers
# =============================================================================


def _save_grid(
    z: torch.Tensor,
    pipeline: StableFlowMatchingPipeline,
    out_path: str,
    nrow: int = 4,
) -> None:
    """Decode latents, clamp to [0,1], and save as a grid image."""
    with torch.no_grad():
        x = pipeline.decode_fm_output(z)
    x_vis = pipeline.from_norm_to_display(x).clamp(0, 1)
    vutils.save_image(x_vis, out_path, nrow=nrow)
    print(f"  Saved grid → {out_path}")


def _save_individual(
    z: torch.Tensor,
    pipeline: StableFlowMatchingPipeline,
    out_dir: str,
    prefix: str = "sample",
) -> None:
    """Decode latents and save each image individually."""
    with torch.no_grad():
        x = pipeline.decode_fm_output(z)
    x_vis = pipeline.from_norm_to_display(x).clamp(0, 1)
    for i, img in enumerate(x_vis):
        path = os.path.join(out_dir, f"{prefix}_{i:04d}.png")
        vutils.save_image(img, path)
    print(f"  Saved {x_vis.shape[0]} individual images to {out_dir}/")


def _collect_scores(
    z: torch.Tensor,
    guidance: Optional[ScorePredictorGuidance],
) -> Optional[list]:
    """Compute per-sample predicted scores if guidance is available."""
    if guidance is None:
        return None
    with torch.no_grad():
        preds = guidance.predict(z)
    scores = []
    for i in range(z.shape[0]):
        scores.append({
            "sample_idx": i,
            "pred_surprise": preds["surprise"][i].item(),
            "pred_gmm":      preds["gmm"][i].item(),
        })
    return scores


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Guided sampling from a trained StableFlowMatchingPipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Pipeline loading ---------------------------------------------------
    g = p.add_argument_group("Pipeline")
    g.add_argument(
        "--fm_pipeline_dir", type=str, required=True,
        help="Root folder of a saved StableFlowMatchingPipeline "
             "(must contain VAE/ and UNET/ sub-directories).",
    )
    g.add_argument(
        "--t_scale", type=float, default=None,
        help="Time scaling factor for UNet input.  Defaults to the t_scale "
             "stored in the pipeline (1.0 if not saved).",
    )
    g.add_argument(
        "--train_target", type=str, default="v", choices=["v", "x0"],
        help="UNet prediction target used during training.",
    )
    g.add_argument(
        "--device", type=str, default=None,
        help="Device override (e.g. 'cpu', 'cuda:1').  Defaults to CUDA if available.",
    )

    # ---- Guidance predictor -------------------------------------------------
    g2 = p.add_argument_group("Guidance predictor")
    g2.add_argument(
        "--surprise_ckpt", type=str, default=None,
        help="Path to best_model.pt from train_surprise_predictor.py.  "
             "Required for all methods except 'euler'.",
    )
    g2.add_argument(
        "--vae_config", type=str, default=None,
        help="VAE config JSON path for predictor.  "
             "Auto-read from ckpt['args'] if not given.",
    )
    g2.add_argument(
        "--vae_weights", type=str, default=None,
        help="VAE weights .pt path for predictor.  "
             "Auto-read from ckpt['args'] if not given.",
    )
    g2.add_argument(
        "--dino_name", type=str, default=None,
        help="DINOv2 model name (e.g. dinov2_vits14).  "
             "Auto-read from ckpt['args'] if not given.",
    )
    g2.add_argument(
        "--hidden_dim", type=int, default=None,
        help="Predictor MLP hidden dim.  "
             "Auto-read from ckpt['args'] if not given.",
    )

    # ---- Sampling method ----------------------------------------------------
    g3 = p.add_argument_group("Sampling method")
    g3.add_argument(
        "--method", type=str, default="euler",
        choices=["euler", "euler_guided", "rerank", "beam", "refine"],
        help="Sampling method.",
    )
    g3.add_argument("--steps",      type=int,   default=50,  help="ODE steps.")
    g3.add_argument("--batch_size", type=int,   default=4,   help="Number of output samples.")

    # rerank
    g3.add_argument("--n_candidates", type=int,   default=8,
                    help="[rerank] Number of candidate trajectories.")
    g3.add_argument("--keep_top_k",   type=int,   default=1,
                    help="[rerank] Number of best candidates to keep.")

    # beam
    g3.add_argument("--beam_size",      type=int,   default=4,
                    help="[beam] Number of surviving beams.")
    g3.add_argument("--branch_factor",  type=int,   default=2,
                    help="[beam] Branching fan-out at each step.")
    g3.add_argument("--sigma_perturb",  type=float, default=0.05,
                    help="[beam] Std-dev of perturbation noise at each branch.")
    g3.add_argument("--return_all_beams", action="store_true",
                    help="[beam] Return all beams, not just the best one.")

    # refine
    g3.add_argument("--num_refine_steps", type=int,   default=10,
                    help="[refine] Post-sampling gradient refinement steps.")
    g3.add_argument("--refine_step_size", type=float, default=0.01,
                    help="[refine] Step size per refinement step.")
    g3.add_argument("--clamp_latent_norm", type=float, default=None,
                    help="[refine] Clamp per-sample latent norm after each step.")

    # ---- Guidance config ----------------------------------------------------
    g4 = p.add_argument_group("Guidance config")
    g4.add_argument(
        "--energy_mode", type=str, default="surprise",
        choices=["surprise", "gmm", "combo"],
        help="Which predictor output to use as energy.",
    )
    g4.add_argument(
        "--sign", type=str, default="minimize",
        choices=["minimize", "maximize"],
        help="Guidance direction.",
    )
    g4.add_argument("--w_surprise",       type=float, default=1.0,
                    help="[combo] Weight for surprise energy term.")
    g4.add_argument("--w_gmm",            type=float, default=1.0,
                    help="[combo] Weight for gmm energy term.")
    g4.add_argument("--guidance_scale",   type=float, default=1.0,
                    help="Global guidance scale multiplier.")
    g4.add_argument("--lambda_start",     type=float, default=1.0)
    g4.add_argument("--lambda_end",       type=float, default=1.0)
    g4.add_argument(
        "--lambda_schedule", type=str, default="constant",
        choices=["constant", "linear", "cosine", "step"],
    )
    g4.add_argument("--grad_clip_norm",   type=float, default=None,
                    help="Per-sample guidance gradient norm cap.")
    g4.add_argument("--normalize_grad",   action="store_true",
                    help="Normalise guidance gradient to unit norm before scaling.")
    g4.add_argument(
        "--guidance_on", type=str, default="latent",
        choices=["latent", "decoded"],
        help="Whether to compute guidance on the latent or decoded image.",
    )
    g4.add_argument(
        "--use_ddim_hat", action="store_true",
        help="Evaluate guidance energy on a DDIM-style clean estimate z_hat(z_t, t, v), "
             "while taking gradients w.r.t. z_t.",
    )
    g4.add_argument("--use_amp", action="store_true",
                    help="Use AMP autocast in the predictor forward.")

    # ---- Output / logs ------------------------------------------------------
    g5 = p.add_argument_group("Output")
    g5.add_argument(
        "--output_dir", type=str, default="generated/guided_fm",
        help="Directory to save images and JSON scores.",
    )
    g5.add_argument("--save_individual",  action="store_true",
                    help="Save individual images in addition to the grid.")
    g5.add_argument("--grid_nrow",        type=int, default=4,
                    help="Number of images per row in the saved grid.")
    g5.add_argument("--return_logs",      action="store_true",
                    help="[euler_guided] Log per-step scores and grad norms.")
    g5.add_argument("--run_sanity_check", action="store_true",
                    help="Run a quick sanity check on the guidance module before sampling.")

    return p


def main() -> None:
    p = build_parser()
    p.add_argument("--config", type=str, default=None,
                   help="Path to YAML config file. CLI flags override config values.")
    preliminary, _ = p.parse_known_args()
    apply_yaml_defaults(p, preliminary.config)
    args = p.parse_args()

    # ---- Device -------------------------------------------------------------
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[sample_guided_fm] Device: {device}")

    # ---- Load pipeline ------------------------------------------------------
    print(f"[sample_guided_fm] Loading pipeline from: {args.fm_pipeline_dir}")
    pipeline = StableFlowMatchingPipeline(
        device=device,
        t_scale=args.t_scale,
        train_target=args.train_target,
    )
    pipeline.load_from_pipeline_folder_auto(
        args.fm_pipeline_dir,
        strict=True,
        map_location=device,
        set_eval=True,
    )
    pipeline.freeze_vae()
    print("  Pipeline loaded  ✓")

    # ---- Build guidance (optional for 'euler') ------------------------------
    guidance: Optional[ScorePredictorGuidance] = None
    needs_guidance = args.method != "euler"

    if needs_guidance:
        if args.surprise_ckpt is None:
            p.error(f"--surprise_ckpt is required for method '{args.method}'.")

        cfg = ScoreGuidanceConfig(
            ckpt_path=args.surprise_ckpt,
            vae_config_path=args.vae_config,
            vae_weights_path=args.vae_weights,
            dino_name=args.dino_name,
            hidden_dim=args.hidden_dim,
            energy_mode=args.energy_mode,
            sign=args.sign,
            w_surprise=args.w_surprise,
            w_gmm=args.w_gmm,
            lambda_start=args.lambda_start,
            lambda_end=args.lambda_end,
            lambda_schedule=args.lambda_schedule,
            grad_clip_norm=args.grad_clip_norm,
            normalize_grad=args.normalize_grad,
            guidance_on=args.guidance_on,
            use_ddim_hat=args.use_ddim_hat,
            use_amp=args.use_amp,
            num_refine_steps=args.num_refine_steps,
            refine_step_size=args.refine_step_size,
        )
        guidance = ScorePredictorGuidance.from_checkpoint(cfg, device=device)
        print("  Guidance predictor loaded  ✓")

        if args.run_sanity_check:
            run_sanity_check(guidance)

    # ---- Output dir ---------------------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Sampling -----------------------------------------------------------
    print(f"[sample_guided_fm] Sampling with method='{args.method}' ...")
    trajectory_logs = None

    if args.method == "euler":
        z = pipeline.sample_euler(
            steps=args.steps,
            batch_size=args.batch_size,
        )

    elif args.method == "euler_guided":
        result = pipeline.sample_euler_guided(
            steps=args.steps,
            batch_size=args.batch_size,
            guidance=guidance,
            guidance_scale=args.guidance_scale,
            return_logs=args.return_logs,
        )
        if args.return_logs:
            z, trajectory_logs = result
        else:
            z = result

    elif args.method == "rerank":
        z = pipeline.sample_euler_with_candidates(
            steps=args.steps,
            n_candidates=args.n_candidates,
            keep_top_k=args.keep_top_k,
            batch_size=args.batch_size,
            guidance=guidance,
            guidance_scale=0.0,  # rerank: no guidance during ODE; scoring only at end
        )

    elif args.method == "beam":
        z = pipeline.sample_euler_beam(
            steps=args.steps,
            batch_size=args.batch_size,
            beam_size=args.beam_size,
            branch_factor=args.branch_factor,
            sigma_perturb=args.sigma_perturb,
            guidance=guidance,
            guidance_scale=0.0,  # beam: use guidance only for pruning by default
            return_all_beams=args.return_all_beams,
        )

    elif args.method == "refine":
        # Step 1: plain Euler sampling
        z = pipeline.sample_euler(
            steps=args.steps,
            batch_size=args.batch_size,
        )
        # Step 2: latent refinement
        z = pipeline.refine_latents_energy(
            z=z,
            guidance=guidance,
            num_steps=args.num_refine_steps,
            step_size=args.refine_step_size,
            clamp_latent_norm=args.clamp_latent_norm,
        )

    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(f"  Generated z: shape={list(z.shape)}")

    # ---- Save images --------------------------------------------------------
    grid_path = os.path.join(args.output_dir, f"{args.method}_grid.png")
    _save_grid(z, pipeline, grid_path, nrow=args.grid_nrow)

    if args.save_individual:
        _save_individual(z, pipeline, args.output_dir, prefix=args.method)

    # ---- Compute and save predicted scores ----------------------------------
    scores_list = _collect_scores(z, guidance)

    output_info: dict = {
        "method": args.method,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "device": device,
        "fm_pipeline_dir": args.fm_pipeline_dir,
        "guidance_scale": args.guidance_scale if guidance is not None else 0.0,
    }

    if scores_list is not None:
        output_info["per_sample_scores"] = scores_list
        mean_surp = sum(s["pred_surprise"] for s in scores_list) / len(scores_list)
        mean_gmm  = sum(s["pred_gmm"]      for s in scores_list) / len(scores_list)
        output_info["mean_pred_surprise"] = mean_surp
        output_info["mean_pred_gmm"]      = mean_gmm
        print(
            f"  Mean predicted surprise={mean_surp:.4f}  "
            f"Mean predicted gmm={mean_gmm:.4f}"
        )

    if trajectory_logs:
        output_info["trajectory_logs"] = trajectory_logs

    results_path = os.path.join(args.output_dir, f"{args.method}_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(output_info, f, indent=2)
    print(f"  Saved results JSON → {results_path}")

    print("[sample_guided_fm] Done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
train_surprise_predictor.py

Multi-task regression training:  VAE latent -> decoded image -> DINOv2 features
-> shared trunk -> two heads predicting (surprise, gmm_score).

Forward path:
    z (latent .npy)
    -> trainable VAE decoder (post_quant_conv + decoder)
    -> decoded image
    -> frozen DINOv2 feature extractor (same preprocessing as dataset builder)
    -> mean-pooled patch tokens
    -> shared MLP trunk
    -> head_surprise  -> pred_surprise
    -> head_gmm       -> pred_gmm

Gradient flow:
    loss -> heads -> trunk -> pooled features -> (through frozen-but-differentiable
    DINO) -> decoded image -> trainable decoder.
    DINO parameters are frozen (requires_grad=False) but forward is NOT wrapped
    in torch.no_grad(), so gradients propagate through the computation graph
    back into the decoder.
"""

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# ── repo root on path ─────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from fm_src.pipelines.flow_matching_pipeline import StableFlowMatchingPipeline  # noqa: E402
from src.core.constants import IMAGENET_MEAN, IMAGENET_STD  # noqa: E402

# ── optional scipy ─────────────────────────────────────────────────────────────
try:
    from scipy.stats import spearmanr as _scipy_spearmanr, pearsonr as _scipy_pearsonr
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False


# =============================================================================
# Correlation helpers
# =============================================================================

def _rankdata(x: np.ndarray) -> np.ndarray:
    """Simple rank-transform (average method) without scipy."""
    order = x.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks


def spearman_corr(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 3:
        return 0.0
    if _SCIPY_OK:
        r, _ = _scipy_spearmanr(pred, target)
        return float(r) if np.isfinite(r) else 0.0
    rp = _rankdata(pred)
    rt = _rankdata(target)
    return float(np.corrcoef(rp, rt)[0, 1])


def pearson_corr(pred: np.ndarray, target: np.ndarray) -> float:
    if len(pred) < 3:
        return 0.0
    if _SCIPY_OK:
        r, _ = _scipy_pearsonr(pred, target)
        return float(r) if np.isfinite(r) else 0.0
    return float(np.corrcoef(pred, target)[0, 1])


# =============================================================================
# ImageNet stats (from src.core.constants)
# =============================================================================

IMNET_MEAN = list(IMAGENET_MEAN)
IMNET_STD  = list(IMAGENET_STD)


# =============================================================================
# Dataset
# =============================================================================

class SurprisePredDataset(Dataset):
    """
    Loads (latent, surprise_target, gmm_target) triples from a dataset folder
    built by build_surprise_pred_dataset.py.

    Structure expected:
        DS_ROOT / vae_model_name / annotations.json
        DS_ROOT / vae_model_name / latents / {stem}.npy

    Args:
        ds_root:         path to DS_ROOT
        vae_model_name:  sub-folder name
        target_mode:     "minmax" or "raw"
        stem_list:       restrict to these stems (for train/val/test splits)
        z_score_stats:   dict with keys "surprise_mean", "surprise_std",
                         "gmm_mean", "gmm_std". Required when target_mode="raw".
    """

    def __init__(
        self,
        ds_root: str,
        vae_model_name: str,
        target_mode: str = "minmax",
        stem_list: Optional[List[str]] = None,
        z_score_stats: Optional[Dict[str, float]] = None,
    ):
        vae_dir = Path(ds_root) / vae_model_name
        ann_path = vae_dir / "annotations.json"
        self.latents_dir = vae_dir / "latents"
        self.target_mode = target_mode
        self.z_score_stats = z_score_stats

        ann = json.loads(ann_path.read_text())
        images = ann["images"]

        # filter valid entries
        self.entries: List[Dict[str, Any]] = []
        for img in images:
            stem = img["stem"]
            if stem_list is not None and stem not in set(stem_list):
                continue

            # pick target keys
            if target_mode == "minmax":
                surp = img.get("surprise_minmax")
                gmm  = img.get("gmm_score_minmax")
            else:
                surp = img.get("surprise_raw")
                gmm  = img.get("gmm_score_raw")

            if surp is None or gmm is None:
                continue

            lat_path = self.latents_dir / f"{stem}.npy"
            if not lat_path.exists():
                continue

            self.entries.append({
                "stem": stem,
                "surprise": float(surp),
                "gmm": float(gmm),
            })

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        e = self.entries[idx]
        z = np.load(str(self.latents_dir / f"{e['stem']}.npy"))
        z = torch.from_numpy(z).float()  # (C, H', W')

        surp = e["surprise"]
        gmm  = e["gmm"]

        # z-score normalisation for raw mode
        if self.target_mode == "raw" and self.z_score_stats is not None:
            surp = (surp - self.z_score_stats["surprise_mean"]) / max(self.z_score_stats["surprise_std"], 1e-8)
            gmm  = (gmm  - self.z_score_stats["gmm_mean"])      / max(self.z_score_stats["gmm_std"],     1e-8)

        return {
            "z": z,
            "surprise": torch.tensor(surp, dtype=torch.float32),
            "gmm": torch.tensor(gmm, dtype=torch.float32),
            "stem": e["stem"],
        }


def _collate_surprise(batch):
    z = torch.stack([b["z"] for b in batch])
    surprise = torch.stack([b["surprise"] for b in batch])
    gmm = torch.stack([b["gmm"] for b in batch])
    stems = [b["stem"] for b in batch]
    return {"z": z, "surprise": surprise, "gmm": gmm, "stem": stems}


def compute_z_score_stats(entries: List[Dict[str, Any]]) -> Dict[str, float]:
    """Compute mean/std of surprise and gmm from a list of entry dicts."""
    surps = np.array([e["surprise"] for e in entries], dtype=np.float64)
    gmms  = np.array([e["gmm"]     for e in entries], dtype=np.float64)
    return {
        "surprise_mean": float(surps.mean()),
        "surprise_std":  float(surps.std()),
        "gmm_mean":      float(gmms.mean()),
        "gmm_std":       float(gmms.std()),
    }


def split_stems(stems: List[str], seed: int, val_frac: float, test_frac: float):
    rng = random.Random(seed)
    s = stems.copy()
    rng.shuffle(s)
    n = len(s)
    n_test = int(round(test_frac * n))
    n_val  = int(round(val_frac * n))
    return s[n_test + n_val:], s[n_test:n_test + n_val], s[:n_test]


# =============================================================================
# Model
# =============================================================================

class SurprisePredictor(nn.Module):
    """
    Multi-task regression: z → decoder → DINOv2 → trunk → (surprise, gmm).

    Architecture:
        1. VAE decoder  (trainable) – post_quant_conv + decoder sub-modules
        2. DINOv2       (frozen, but differentiable forward)
        3. Mean pool    → LayerNorm → (B, D)
        4. Shared trunk → MLP
        5. head_surprise → 1
           head_gmm      → 1
    """

    def __init__(
        self,
        vae_config_path: str,
        vae_weights_path: str,
        dino_name: str = "dinov2_vits14",
        hidden_dim: int = 256,
        device: str = "cpu",
    ):
        super().__init__()

        # ---- 1. VAE decoder ------------------------------------------------
        # Build a full VAE via the pipeline, then extract decoder components.
        pipeline = StableFlowMatchingPipeline(device="cpu")
        pipeline.build_from_configs(vae_json=vae_config_path, save_configs=False)
        pipeline.load_pretrained(vae_path=vae_weights_path, set_eval=False)
        vae = pipeline.vae

        # Trainable decoder components (deep-copy to own them independently)
        self.post_quant_conv = copy.deepcopy(vae.post_quant_conv)
        self.decoder = copy.deepcopy(vae.decoder)

        # Freeze encoder (we don't carry it, but if the VAE is ever kept
        # around by reference, make it non-trainable).
        del pipeline, vae

        # ---- 2. DINOv2 (frozen) -------------------------------------------
        self.dino = torch.hub.load("facebookresearch/dinov2", dino_name)
        for p in self.dino.parameters():
            p.requires_grad = False
        self.dino.eval()

        # Infer feature dimension from the DINO model
        dino_dim = self.dino.embed_dim  # e.g. 384 for vits14

        # ---- 3. Pre-processing constants (registered as buffers) -----------
        self.register_buffer(
            "imnet_mean",
            torch.tensor(IMNET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "imnet_std",
            torch.tensor(IMNET_STD).view(1, 3, 1, 1),
        )

        # ---- 4. Pooling + LayerNorm ---------------------------------------
        self.pool_norm = nn.LayerNorm(dino_dim)

        # ---- 5. Shared trunk -----------------------------------------------
        self.trunk = nn.Sequential(
            nn.Linear(dino_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # ---- 6. Heads -------------------------------------------------------
        self.head_surprise = nn.Linear(hidden_dim, 1)
        self.head_gmm = nn.Linear(hidden_dim, 1)

        self._shape_printed = False

    # --------------------------------------------------------------------- #
    # Preprocessing (torch-only, differentiable where needed)
    # --------------------------------------------------------------------- #

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """VAE decode: post_quant_conv + decoder.  Trainable."""
        h = self.post_quant_conv(z)
        return self.decoder(h)

    @staticmethod
    def _per_image_minmax(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Per-image min/max normalisation → [-1, 1].  (B,C,H,W) → (B,C,H,W)."""
        B = x.shape[0]
        flat = x.view(B, -1)
        lo = flat.min(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        hi = flat.max(dim=1, keepdim=True)[0].view(B, 1, 1, 1)
        x = (x - lo) / (hi - lo + eps)        # [0, 1]
        return 2.0 * x - 1.0                   # [-1, 1]

    def _to_dino_input(self, x_b1hw: torch.Tensor) -> torch.Tensor:
        """
        (B,1,H,W) in [-1,1] → (B,3,224,224) with ImageNet normalisation.
        Same as build_surprise_pred_dataset.to_dino_input.
        """
        x = (x_b1hw + 1.0) * 0.5                                       # [0,1]
        x = x.expand(-1, 3, -1, -1)                                    # 3-ch
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = (x - self.imnet_mean) / self.imnet_std
        return x

    def _extract_patch_tokens(self, x_3ch: torch.Tensor) -> torch.Tensor:
        """
        DINOv2 forward — frozen params but differentiable graph.
        Returns (B, N, D).
        """
        out = self.dino.forward_features(x_3ch)
        return out["x_norm_patchtokens"]

    # --------------------------------------------------------------------- #
    # Full forward
    # --------------------------------------------------------------------- #

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1. Decode
        decoded = self._decode(z)                           # (B, C_out, H, W)

        # Ensure single-channel (VAE may output 1 or more channels)
        if decoded.shape[1] != 1:
            decoded = decoded[:, :1]                        # keep first channel

        # 2. Resize to 256 + per-image minmax → [-1, 1]
        decoded_256 = F.interpolate(
            decoded, size=(256, 256), mode="bilinear", align_corners=False
        )
        decoded_norm = self._per_image_minmax(decoded_256)  # (B,1,256,256) in [-1,1]

        # 3. DINO input + features (NO torch.no_grad)
        dino_in = self._to_dino_input(decoded_norm)         # (B,3,224,224)
        tokens = self._extract_patch_tokens(dino_in)        # (B, N, D)

        # 4. Pool + normalise
        pooled = tokens.mean(dim=1)                         # (B, D)
        pooled = self.pool_norm(pooled)

        # Shape sanity print (once)
        if not self._shape_printed:
            print(f"[shape check]  z={list(z.shape)}  decoded={list(decoded.shape)}  "
                  f"dino_tokens={list(tokens.shape)}  pooled={list(pooled.shape)}")
            self._shape_printed = True

        # 5. Trunk
        h = self.trunk(pooled)                              # (B, hidden_dim)

        # 6. Heads
        pred_surprise = self.head_surprise(h).squeeze(-1)   # (B,)
        pred_gmm      = self.head_gmm(h).squeeze(-1)       # (B,)

        return pred_surprise, pred_gmm


# =============================================================================
# Early stopping helper
# =============================================================================

def _is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best + min_delta
    return current < best - min_delta


# =============================================================================
# Training + evaluation
# =============================================================================

def train_one_epoch(
    model: SurprisePredictor,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    loss_w: Tuple[float, float],
    scaler: Optional[torch.amp.GradScaler],
    use_amp: bool,
) -> Dict[str, float]:
    model.train()
    # Keep DINO in eval mode (BatchNorm / dropout)
    model.dino.eval()

    total_loss = 0.0
    total_ls = 0.0
    total_lg = 0.0
    n = 0

    for batch in tqdm(loader, desc="  train batches", leave=False):
        z = batch["z"].to(device)
        t_surp = batch["surprise"].to(device)
        t_gmm  = batch["gmm"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            ps, pg = model(z)
            l_surp = F.mse_loss(ps, t_surp)
            l_gmm  = F.mse_loss(pg, t_gmm)
            loss   = loss_w[0] * l_surp + loss_w[1] * l_gmm

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        bs = z.shape[0]
        total_loss += loss.item()  * bs
        total_ls   += l_surp.item() * bs
        total_lg   += l_gmm.item()  * bs
        n += bs

    return {
        "loss_total":    total_loss / max(n, 1),
        "loss_surprise": total_ls   / max(n, 1),
        "loss_gmm":      total_lg   / max(n, 1),
    }


@torch.no_grad()
def evaluate(
    model: SurprisePredictor,
    loader: DataLoader,
    device: str,
    loss_w: Tuple[float, float],
    use_amp: bool,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_ls = 0.0
    total_lg = 0.0
    n = 0
    all_ps: List[float] = []
    all_pg: List[float] = []
    all_ts: List[float] = []
    all_tg: List[float] = []

    for batch in tqdm(loader, desc="  eval batches", leave=False):
        z = batch["z"].to(device)
        t_surp = batch["surprise"].to(device)
        t_gmm  = batch["gmm"].to(device)

        with torch.amp.autocast("cuda", enabled=use_amp):
            ps, pg = model(z)
            l_surp = F.mse_loss(ps, t_surp)
            l_gmm  = F.mse_loss(pg, t_gmm)
            loss   = loss_w[0] * l_surp + loss_w[1] * l_gmm

        bs = z.shape[0]
        total_loss += loss.item()  * bs
        total_ls   += l_surp.item() * bs
        total_lg   += l_gmm.item()  * bs
        n += bs

        all_ps.extend(ps.cpu().tolist())
        all_pg.extend(pg.cpu().tolist())
        all_ts.extend(t_surp.cpu().tolist())
        all_tg.extend(t_gmm.cpu().tolist())

    ps_arr = np.array(all_ps)
    pg_arr = np.array(all_pg)
    ts_arr = np.array(all_ts)
    tg_arr = np.array(all_tg)

    return {
        "loss_total":        total_loss / max(n, 1),
        "loss_surprise":     total_ls   / max(n, 1),
        "loss_gmm":          total_lg   / max(n, 1),
        "mse_surprise":      float(np.mean((ps_arr - ts_arr)**2)),
        "mse_gmm":           float(np.mean((pg_arr - tg_arr)**2)),
        "mae_surprise":      float(np.mean(np.abs(ps_arr - ts_arr))),
        "mae_gmm":           float(np.mean(np.abs(pg_arr - tg_arr))),
        "spearman_surprise": spearman_corr(ps_arr, ts_arr),
        "spearman_gmm":      spearman_corr(pg_arr, tg_arr),
        "pearson_surprise":  pearson_corr(ps_arr, ts_arr),
        "pearson_gmm":       pearson_corr(pg_arr, tg_arr),
    }


def log_scalar_dict(writer: SummaryWriter, prefix: str, d: Dict[str, float], step: int) -> None:
    for k, v in d.items():
        writer.add_scalar(f"{prefix}/{k}", v, step)


@torch.no_grad()
def log_decoded_images(
    writer: SummaryWriter,
    model: SurprisePredictor,
    val_ds: SurprisePredDataset,
    device: str,
    step: int,
    n_images: int = 4,
) -> None:
    """Decode a few val latents and log to TensorBoard."""
    model.eval()
    n = min(n_images, len(val_ds))
    indices = random.sample(range(len(val_ds)), n)

    for i, idx in enumerate(indices):
        item = val_ds[idx]
        z = item["z"].unsqueeze(0).to(device)
        decoded = model._decode(z)
        if decoded.shape[1] != 1:
            decoded = decoded[:, :1]
        # Per-image stretch for visualisation
        img = decoded[0]                            # (1, H, W)
        img = img - img.min()
        img = img / (img.max() - img.min() + 1e-8)
        writer.add_image(f"val/decoded/{i}", img, global_step=step)


# =============================================================================
# Argument parser
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train multi-task surprise + GMM predictor from VAE latents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── dataset ───────────────────────────────────────────────────────────────
    p.add_argument("--ds_root", type=str, required=True,
                   help="Path to DS_ROOT built by build_surprise_pred_dataset.py.")
    p.add_argument("--vae_model_name", type=str, required=True,
                   help="Sub-folder inside DS_ROOT containing latents + annotations.")
    p.add_argument("--vae_config", type=str, required=True,
                   help="VAE config JSON (to rebuild decoder architecture).")
    p.add_argument("--vae_weights", type=str, required=True,
                   help="Pretrained VAE weights (.pt).")

    # ── model ─────────────────────────────────────────────────────────────────
    p.add_argument("--dino_name", type=str, default="dinov2_vits14")
    p.add_argument("--hidden_dim", type=int, default=256,
                   help="Hidden dimension for trunk MLP.")

    # ── targets ───────────────────────────────────────────────────────────────
    p.add_argument("--target_mode", type=str, default="minmax",
                   choices=["raw", "minmax"],
                   help="Target format: minmax (direct) or raw (z-scored).")

    # ── splits ────────────────────────────────────────────────────────────────
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_frac", type=float, default=0.15)
    p.add_argument("--test_frac", type=float, default=0.10)

    # ── training ──────────────────────────────────────────────────────────────
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--amp", action="store_true", default=False,
                   help="Use automatic mixed precision.")
    p.add_argument("--loss_weights", type=float, nargs=2, default=[1.0, 1.0],
                   metavar=("W_SURPRISE", "W_GMM"),
                   help="Weights for loss = w1*L_surprise + w2*L_gmm.")

    # ── early stopping ────────────────────────────────────────────────────────
    p.add_argument("--early_metric", type=str, default="val_loss",
                   choices=["val_loss", "val_spearman_surprise", "val_spearman_gmm"])
    p.add_argument("--early_mode", type=str, default=None,
                   help="min or max.  Auto-selected from metric if not given.")
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--min_delta", type=float, default=1e-4)

    # ── checkpointing ─────────────────────────────────────────────────────────
    p.add_argument("--out_dir", type=str, default="./artifacts/runs/main/surprise_predictor",
                   help="Output directory for checkpoints + TB logs.")
    p.add_argument("--run_name", type=str, default="",
                   help="Run sub-folder name.  Empty = auto-generated.")
    p.add_argument("--save_every", type=int, default=0,
                   help="Save checkpoint every N epochs (0 = only best + final).")
    p.add_argument("--resume", type=str, default="",
                   help="Path to a checkpoint to resume training from.")

    return p.parse_args()


# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    args = parse_args()

    # ── seed ──────────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── device ────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print(f"Device: {device}")

    # ── early-stop mode auto ──────────────────────────────────────────────────
    if args.early_mode is None:
        args.early_mode = "min" if "loss" in args.early_metric else "max"

    # ── output dir ────────────────────────────────────────────────────────────
    run_name = args.run_name or (
        f"{args.vae_model_name}_{args.target_mode}_h{args.hidden_dim}_s{args.seed}"
    )
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = out_dir / "tb"
    print(f"Output dir: {out_dir}")

    # ── dataset ───────────────────────────────────────────────────────────────
    # Discover all valid stems from annotations first (full dataset, no z-score yet)
    tmp_ds = SurprisePredDataset(
        ds_root=args.ds_root,
        vae_model_name=args.vae_model_name,
        target_mode=args.target_mode,
    )
    all_stems = [e["stem"] for e in tmp_ds.entries]
    print(f"Total valid entries: {len(all_stems)}")

    train_stems, val_stems, test_stems = split_stems(
        all_stems, args.seed, args.val_frac, args.test_frac
    )
    print(f"Splits: train={len(train_stems)} val={len(val_stems)} test={len(test_stems)}")

    # Compute z-score stats on train split (needed only if raw mode)
    z_stats: Optional[Dict[str, float]] = None
    if args.target_mode == "raw":
        train_tmp = SurprisePredDataset(
            ds_root=args.ds_root,
            vae_model_name=args.vae_model_name,
            target_mode="raw",
            stem_list=train_stems,
        )
        z_stats = compute_z_score_stats(train_tmp.entries)
        print(f"z-score stats (train): {z_stats}")
        del train_tmp

    del tmp_ds

    # Build final datasets
    train_ds = SurprisePredDataset(
        args.ds_root, args.vae_model_name, args.target_mode,
        stem_list=train_stems, z_score_stats=z_stats,
    )
    val_ds = SurprisePredDataset(
        args.ds_root, args.vae_model_name, args.target_mode,
        stem_list=val_stems, z_score_stats=z_stats,
    )
    test_ds = SurprisePredDataset(
        args.ds_root, args.vae_model_name, args.target_mode,
        stem_list=test_stems, z_score_stats=z_stats,
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_surprise,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_surprise,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=torch.cuda.is_available(),
        collate_fn=_collate_surprise,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    print("Building model ...")
    model = SurprisePredictor(
        vae_config_path=args.vae_config,
        vae_weights_path=args.vae_weights,
        dino_name=args.dino_name,
        hidden_dim=args.hidden_dim,
        device=device,
    ).to(device)

    # Parameter summary
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Parameters: trainable={trainable/1e6:.2f}M  total={total/1e6:.2f}M")

    # ── optimizer ─────────────────────────────────────────────────────────────
    # Only optimise parameters that require grad (decoder + trunk + heads)
    opt_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=args.weight_decay)

    scaler = torch.amp.GradScaler("cuda") if args.amp and device.startswith("cuda") else None
    loss_w = tuple(args.loss_weights)

    # ── resume ────────────────────────────────────────────────────────────────
    start_epoch = 1
    best_value: Optional[float] = None
    best_epoch: Optional[int]   = None
    bad_epochs = 0
    history: List[Dict[str, Any]] = []

    if args.resume:
        print(f"Resuming from {args.resume} ...")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)
        if "optimizer_state" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_value  = ckpt.get("best_value")
        best_epoch  = ckpt.get("best_epoch")
        bad_epochs  = ckpt.get("bad_epochs", 0)
        print(f"  → Resuming at epoch {start_epoch}")

    # ── tensorboard ───────────────────────────────────────────────────────────
    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("run/config", json.dumps(vars(args), indent=2))

    # ── checkpoint helper ─────────────────────────────────────────────────────
    def _save_ckpt(path: Path, epoch: int, extra: Optional[Dict] = None) -> None:
        payload = {
            "epoch":          epoch,
            "model_state":    model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "args":           vars(args),
            "best_value":     best_value,
            "best_epoch":     best_epoch,
            "bad_epochs":     bad_epochs,
        }
        if z_stats is not None:
            payload["z_score_stats"] = z_stats
        if extra:
            payload.update(extra)
        torch.save(payload, path)

    # ── training loop ─────────────────────────────────────────────────────────
    best_ckpt_path = out_dir / "best_model.pt"

    pbar = tqdm(range(start_epoch, args.epochs + 1), desc="Training")
    for epoch in pbar:
        # --- train ---
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, loss_w, scaler, args.amp,
        )
        # --- validate ---
        val_stats = evaluate(model, val_loader, device, loss_w, args.amp)

        # --- pick early-stop metric ---
        metric_map = {
            "val_loss":             val_stats["loss_total"],
            "val_spearman_surprise": val_stats["spearman_surprise"],
            "val_spearman_gmm":      val_stats["spearman_gmm"],
        }
        current_value = metric_map[args.early_metric]

        # --- logging ---
        log_scalar_dict(writer, "train", train_stats, epoch)
        log_scalar_dict(writer, "val", val_stats, epoch)
        writer.add_scalar("early_stop/current_value", current_value, epoch)

        # --- decoded images (every 5 epochs) ---
        if epoch == 1 or epoch % 5 == 0:
            log_decoded_images(writer, model, val_ds, device, epoch)

        # --- history row ---
        row: Dict[str, Any] = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_stats.items()})
        row.update({f"val_{k}": v for k, v in val_stats.items()})
        history.append(row)

        # --- early stopping ---
        if _is_improvement(current_value, best_value, args.early_mode, args.min_delta):
            best_value = current_value
            best_epoch = epoch
            bad_epochs = 0
            _save_ckpt(best_ckpt_path, epoch)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}. "
                      f"Best epoch={best_epoch} best_value={best_value:.6f}")
                break

        # --- periodic checkpoint ---
        if args.save_every > 0 and epoch % args.save_every == 0:
            _save_ckpt(out_dir / f"ckpt_epoch_{epoch}.pt", epoch)

        pbar.set_postfix({
            "tr_loss":  f"{train_stats['loss_total']:.4f}",
            "val_loss": f"{val_stats['loss_total']:.4f}",
            "sp_surp":  f"{val_stats['spearman_surprise']:.3f}",
            "sp_gmm":   f"{val_stats['spearman_gmm']:.3f}",
            "best_ep":  best_epoch,
            "bad":      bad_epochs,
        })

    # ── final checkpoint ──────────────────────────────────────────────────────
    _save_ckpt(out_dir / "final_model.pt", epoch)

    # ── load best + test ──────────────────────────────────────────────────────
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        print(f"Loaded best checkpoint: epoch={ckpt['epoch']} "
              f"best_value={ckpt.get('best_value', '?')}")

    test_stats = evaluate(model, test_loader, device, loss_w, args.amp)
    log_scalar_dict(writer, "test", test_stats, best_epoch or epoch)
    print("TEST RESULTS:")
    for k, v in test_stats.items():
        print(f"  {k}: {v:.6f}")

    # ── persist ───────────────────────────────────────────────────────────────
    history_path = out_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    writer.close()

    print(f"\nSaved history → {history_path}")
    print(f"TensorBoard:    tensorboard --logdir {tb_dir}")


if __name__ == "__main__":
    main()

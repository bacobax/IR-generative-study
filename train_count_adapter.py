#!/usr/bin/env python3
"""
Train a residual prototype adapter for count-conditioned DINO features.

Model:
    z_hat(c) = z_base + W phi(c)

Where:
    - z_base is a learnable base embedding, initialized from the global mean feature
    - phi(c) is a small MLP on the scalar count
    - W projects the low-rank residual to feature dimension

Training targets:
    z_c = mean feature embedding of all images with count c

Loss:
    L = lambda_proto * MSE(z_hat(c), z_c)
      + lambda_cos   * (1 - cosine(z_hat(c), z_c))
      + lambda_smooth * second_difference_smoothness

Features:
    - tqdm progress bars
    - TensorBoard logging
    - full checkpointing + full resume
    - easy inference from checkpoint

Expected inputs:
    data_dir/
        annotations.json
        *.npy   # uint16 1-channel images

The script can:
    1) extract DINO features
    2) compute count prototypes
    3) train the residual adapter
    4) save checkpoints
    5) resume training
    6) load a trained checkpoint and predict embeddings for counts

Example:
    python train_count_adapter.py \
        --data-dir /path/to/data \
        --output-dir runs/count_adapter \
        --epochs 400 \
        --feature-mode cls

Resume:
    python train_count_adapter.py \
        --data-dir /path/to/data \
        --output-dir runs/count_adapter \
        --resume runs/count_adapter/checkpoints/latest.pt

Inference:
    python train_count_adapter.py \
        --data-dir /path/to/data \
        --output-dir runs/count_adapter \
        --inference-only \
        --checkpoint runs/count_adapter/checkpoints/best.pt \
        --counts 1 2 3 4 5
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

from transformers import AutoImageProcessor, AutoModel


# =========================
# Utilities
# =========================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)

def minmax01(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    mn = float(arr.min())
    mx = float(arr.max())
    if mx <= mn:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr - mn) / (mx - mn)

def load_npy_image(path: str | Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"Expected HxW or HxWx1 array, got {arr.shape} for {path}")
    return arr

def cosine_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = F.normalize(pred, dim=-1)
    target = F.normalize(target, dim=-1)
    return 1.0 - (pred * target).sum(dim=-1).mean()

def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def compute_cache_key(data_dir: str, model_name: str, feature_mode: str, file_list: List[str]) -> str:
    h = hashlib.sha256()
    h.update(os.path.abspath(data_dir).encode())
    h.update(model_name.encode())
    h.update(feature_mode.encode())
    for f in sorted(file_list):
        h.update(f.encode())
    return h.hexdigest()[:16]

@torch.no_grad()
def evaluate_counts(
    model: nn.Module,
    counts: List[int],
    prototypes: Dict[int, np.ndarray],
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    ks = torch.tensor(counts, dtype=torch.float32, device=device)
    target = torch.tensor(
        np.stack([prototypes[c] for c in counts], axis=0),
        dtype=torch.float32, device=device,
    )
    pred = model(ks)

    per_l2 = torch.norm(pred - target, dim=-1)
    pred_n = F.normalize(pred, dim=-1)
    tgt_n = F.normalize(target, dim=-1)
    per_cos = (pred_n * tgt_n).sum(dim=-1)

    mse = F.mse_loss(pred, target).item()
    cos_l = (1.0 - per_cos.mean()).item()

    smooth = 0.0
    if len(counts) >= 3:
        sc = sorted(counts)
        ks_s = torch.tensor(sc, dtype=torch.float32, device=device)
        z_s = model(ks_s)
        smooth = ((z_s[2:] - 2 * z_s[1:-1] + z_s[:-2]) ** 2).mean().item()

    l2_per = {int(c): float(per_l2[i]) for i, c in enumerate(counts)}
    cos_per = {int(c): float(per_cos[i]) for i, c in enumerate(counts)}

    return {
        "l2": float(per_l2.mean()),
        "cosine": float(per_cos.mean()),
        "mse": mse,
        "cosine_loss": cos_l,
        "smoothness": smooth,
        "l2_per_count": l2_per,
        "cosine_per_count": cos_per,
    }

def compute_baseline_metrics(
    val_counts: List[int],
    train_counts: List[int],
    prototypes: Dict[int, np.ndarray],
    base_feature: np.ndarray,
) -> Dict[str, float]:
    global_l2s, global_coss = [], []
    interp_l2s, interp_coss = [], []
    sorted_train = sorted(train_counts)

    for c in val_counts:
        z_true = prototypes[c]
        norm_true = np.linalg.norm(z_true) + 1e-8

        g_l2 = float(np.linalg.norm(base_feature - z_true))
        g_cos = float(np.dot(base_feature, z_true) / (np.linalg.norm(base_feature) * norm_true))
        global_l2s.append(g_l2)
        global_coss.append(g_cos)

        lower = [t for t in sorted_train if t < c]
        upper = [t for t in sorted_train if t > c]
        if lower and upper:
            z_interp = 0.5 * (prototypes[lower[-1]] + prototypes[upper[0]])
        elif lower:
            z_interp = prototypes[lower[-1]]
        elif upper:
            z_interp = prototypes[upper[0]]
        else:
            z_interp = base_feature

        i_l2 = float(np.linalg.norm(z_interp - z_true))
        i_cos = float(np.dot(z_interp, z_true) / (np.linalg.norm(z_interp) * norm_true))
        interp_l2s.append(i_l2)
        interp_coss.append(i_cos)

    return {
        "global_mean_l2": float(np.mean(global_l2s)),
        "global_mean_cos": float(np.mean(global_coss)),
        "interp_l2": float(np.mean(interp_l2s)),
        "interp_cos": float(np.mean(interp_coss)),
    }

def make_pca_figure(
    model: nn.Module,
    counts_sorted: List[int],
    prototypes: Dict[int, np.ndarray],
    train_counts: List[int],
    val_counts: List[int],
    device: torch.device,
) -> plt.Figure:
    model.eval()
    with torch.no_grad():
        ks = torch.tensor(counts_sorted, dtype=torch.float32, device=device)
        preds = model(ks).cpu().numpy()
    trues = np.stack([prototypes[c] for c in counts_sorted], axis=0)
    all_vecs = np.concatenate([trues, preds], axis=0)
    pca = PCA(n_components=2)
    proj = pca.fit_transform(all_vecs)
    n = len(counts_sorted)
    true_proj, pred_proj = proj[:n], proj[n:]

    fig, ax = plt.subplots(figsize=(8, 6))
    train_set = set(train_counts)
    for i, c in enumerate(counts_sorted):
        color = "tab:blue" if c in train_set else "tab:red"
        ax.scatter(*true_proj[i], marker="o", c=color, s=60, zorder=3)
        ax.scatter(*pred_proj[i], marker="x", c=color, s=60, zorder=3)
        ax.annotate(str(c), true_proj[i], fontsize=7, ha="center", va="bottom")
    ax.scatter([], [], marker="o", c="tab:blue", label="true (train)")
    ax.scatter([], [], marker="o", c="tab:red", label="true (val)")
    ax.scatter([], [], marker="x", c="gray", label="predicted")
    ax.legend(fontsize=8)
    ax.set_title("PCA: True vs Predicted Prototypes")
    fig.tight_layout()
    return fig

def make_distance_heatmap(
    model: nn.Module,
    counts_sorted: List[int],
    prototypes: Dict[int, np.ndarray],
    device: torch.device,
) -> plt.Figure:
    model.eval()
    with torch.no_grad():
        ks = torch.tensor(counts_sorted, dtype=torch.float32, device=device)
        pred_mat = model(ks).cpu().numpy()
    true_mat = np.stack([prototypes[c] for c in counts_sorted], axis=0)

    pred_dist = cosine_distances(pred_mat)
    true_dist = cosine_distances(true_mat)
    labels = [str(c) for c in counts_sorted]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title in [(axes[0], true_dist, "True"), (axes[1], pred_dist, "Predicted")]:
        im = ax.imshow(mat, cmap="viridis")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)
        ax.set_yticklabels(labels, fontsize=6)
        ax.set_title(f"{title} Cosine Distance")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    return fig

def load_annotations(data_dir: Path) -> pd.DataFrame:
    annot_path = data_dir / "annotations.json"
    if not annot_path.exists():
        raise FileNotFoundError(f"annotations.json not found in {data_dir}")

    with open(annot_path, "r") as f:
        coco = json.load(f)

    images_df = pd.DataFrame(coco["images"])
    categories_df = pd.DataFrame(coco.get("categories", []))
    person_cat_ids = set(categories_df.loc[categories_df["name"] == "person", "id"].tolist())

    annotations_df = pd.DataFrame(coco.get("annotations", []))
    if len(annotations_df) == 0:
        raise ValueError("No annotations found in annotations.json")

    if "category_id" in annotations_df.columns and len(person_cat_ids) > 0:
        annotations_df = annotations_df[annotations_df["category_id"].isin(person_cat_ids)].copy()

    count_per_image = annotations_df.groupby("image_id").size().rename("person_count")

    df = images_df.merge(
        count_per_image,
        how="left",
        left_on="id",
        right_index=True,
    )

    df["person_count"] = df["person_count"].fillna(0).astype(int)
    df["path"] = df["file_name"].apply(lambda x: str(data_dir / x))
    df = df[df["path"].apply(lambda p: Path(p).exists())].reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No valid image files found after matching annotations to .npy files")

    return df


# =========================
# Feature extraction
# =========================

@torch.no_grad()
def extract_dino_features(
    df: pd.DataFrame,
    model_name: str,
    feature_mode: str,
    device: torch.device,
    batch_size: int = 32,
) -> Tuple[np.ndarray, pd.DataFrame]:
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()

    feats_all: List[torch.Tensor] = []
    meta_rows = []

    n = len(df)
    for start in tqdm(range(0, n, batch_size), desc="Extracting DINO features"):
        batch_df = df.iloc[start : start + batch_size]

        images = []
        for _, row in batch_df.iterrows():
            arr = load_npy_image(row["path"])
            arr01 = minmax01(arr)
            rgb = np.repeat(arr01[..., None], 3, axis=-1).astype(np.float32)
            images.append(rgb)

        proc = processor(images=images, return_tensors="pt", do_rescale=False)
        pixel_values = proc["pixel_values"].to(device)

        outputs = model(pixel_values=pixel_values)
        hidden = outputs.last_hidden_state  # [B, 1+N, D]

        cls = hidden[:, 0]
        patches = hidden[:, 1:]
        mean_patch = patches.mean(dim=1)

        if feature_mode == "cls":
            feat = cls
        elif feature_mode == "mean_patch":
            feat = mean_patch
        elif feature_mode == "concat":
            feat = torch.cat([cls, mean_patch], dim=-1)
        else:
            raise ValueError(f"Unknown feature_mode={feature_mode}")

        feats_all.append(feat.cpu())

        for _, row in batch_df.iterrows():
            meta_rows.append({
                "image_id": row["id"],
                "file_name": row["file_name"],
                "person_count": int(row["person_count"]),
                "path": row["path"],
            })

    features = torch.cat(feats_all, dim=0).numpy()
    feat_df = pd.DataFrame(meta_rows)
    feat_df["feat_idx"] = np.arange(len(feat_df))
    return features, feat_df


# =========================
# Model
# =========================

class ResidualCountAdapter(nn.Module):
    def __init__(
        self,
        d_feat: int,
        rank: int,
        hidden: int,
        base_feature: np.ndarray,
        count_min: int,
        count_max: int,
        learn_base: bool = True,
    ):
        super().__init__()
        self.d_feat = d_feat
        self.rank = rank
        self.count_min = float(count_min)
        self.count_max = float(max(count_max, count_min + 1))

        base_tensor = torch.tensor(base_feature, dtype=torch.float32)
        if learn_base:
            self.base = nn.Parameter(base_tensor.clone())
        else:
            self.register_buffer("base", base_tensor.clone())

        self.mlp = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, rank),
        )
        self.W = nn.Parameter(torch.randn(rank, d_feat) * 0.02)

    def normalize_count(self, c: torch.Tensor) -> torch.Tensor:
        c = c.float()
        return ((c - self.count_min) / (self.count_max - self.count_min)).unsqueeze(-1)

    def residual(self, c: torch.Tensor) -> torch.Tensor:
        a = self.mlp(self.normalize_count(c))  # [B, rank]
        return a @ self.W  # [B, d_feat]

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.base.unsqueeze(0) + self.residual(c)


# =========================
# Training config
# =========================

@dataclass
class TrainConfig:
    data_dir: str
    output_dir: str
    model_name: str = "facebook/dinov2-base"
    feature_mode: str = "cls"  # cls | mean_patch | concat
    device: str = "auto"
    seed: int = 42

    rank: int = 128
    hidden: int = 128
    learn_base: bool = True

    epochs: int = 400
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    lambda_proto: float = 1.0
    lambda_cos: float = 0.5
    lambda_smooth: float = 0.05

    min_count_samples: int = 1
    save_every: int = 25

    batch_counts: int = 0  # 0 => full prototype set every epoch

    resume: Optional[str] = None
    checkpoint: Optional[str] = None
    inference_only: bool = False
    counts: Optional[List[int]] = None

    base_warmup_epochs: int = 0

    cache: bool = False
    cache_dir: str = "data/cache/dino_cache"

    patience: int = 0

    loco: bool = False

    val_counts: Optional[List[int]] = None
    figure_every: int = 20


# =========================
# Trainer
# =========================

class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        set_seed(cfg.seed)

        self.output_dir = Path(cfg.output_dir)
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.logs_dir = self.output_dir / "tb"
        self.artifacts_dir = self.output_dir / "artifacts"

        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.writer = SummaryWriter(str(self.logs_dir))

        self.epoch = 0
        self.best_loss = float("inf")

        self._prepare_data()
        self._build_model()
        self._build_optimizer()

    def _prepare_data(self) -> None:
        df = load_annotations(Path(self.cfg.data_dir))

        cache_hit = False
        if self.cfg.cache:
            file_list = df["file_name"].tolist()
            cache_key = compute_cache_key(
                self.cfg.data_dir, self.cfg.model_name, self.cfg.feature_mode, file_list,
            )
            cache_root = Path(self.cfg.cache_dir)
            cache_root.mkdir(parents=True, exist_ok=True)
            cached_features = cache_root / f"{cache_key}_features.npz"
            cached_meta = cache_root / f"{cache_key}_meta.csv"
            if cached_features.exists() and cached_meta.exists():
                saved = np.load(cached_features)
                self.features = saved["features"]
                self.feat_df = pd.read_csv(cached_meta)
                cache_hit = True
                print(f"Loaded cached DINO features from {cached_features}")

        if not cache_hit:
            self.features, self.feat_df = extract_dino_features(
                df=df,
                model_name=self.cfg.model_name,
                feature_mode=self.cfg.feature_mode,
                device=self.device,
                batch_size=32,
            )
            if self.cfg.cache:
                np.savez_compressed(cached_features, features=self.features)
                self.feat_df.to_csv(cached_meta, index=False)

        count_sizes = self.feat_df["person_count"].value_counts().sort_index()
        valid_counts = count_sizes[count_sizes >= self.cfg.min_count_samples].index.tolist()
        self.valid_counts = sorted(valid_counts)

        if len(self.valid_counts) < 3:
            raise ValueError("Need at least 3 valid counts for smoothness regularization")

        self.count_sizes = {int(k): int(v) for k, v in count_sizes.items() if k in self.valid_counts}

        self.prototypes: Dict[int, np.ndarray] = {}
        for c in self.valid_counts:
            idx = self.feat_df.loc[self.feat_df["person_count"] == c, "feat_idx"].to_numpy()
            self.prototypes[int(c)] = self.features[idx].mean(axis=0).astype(np.float32)

        self.counts_sorted = sorted(self.prototypes.keys())
        self.proto_matrix = np.stack([self.prototypes[c] for c in self.counts_sorted], axis=0)
        self.base_feature = self.features.mean(axis=0).astype(np.float32)
        self.d_feat = int(self.proto_matrix.shape[1])

        save_json(
            {
                "valid_counts": self.valid_counts,
                "count_sizes": self.count_sizes,
                "feature_mode": self.cfg.feature_mode,
                "d_feat": self.d_feat,
            },
            self.artifacts_dir / "data_summary.json",
        )

        if self.cfg.val_counts:
            self.val_counts = sorted([c for c in self.cfg.val_counts if c in self.counts_sorted])
            self.train_counts = [c for c in self.counts_sorted if c not in self.val_counts]
        else:
            self.val_counts = []
            self.train_counts = list(self.counts_sorted)

    def _build_model(self) -> None:
        self.model = ResidualCountAdapter(
            d_feat=self.d_feat,
            rank=self.cfg.rank,
            hidden=self.cfg.hidden,
            base_feature=self.base_feature,
            count_min=min(self.counts_sorted),
            count_max=max(self.counts_sorted),
            learn_base=self.cfg.learn_base,
        ).to(self.device)
        if self.cfg.learn_base and self.cfg.base_warmup_epochs > 0:
            self._set_base_trainable(False)

    def _build_optimizer(self) -> None:
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )

    def _smoothness_loss(self, active_counts: List[int]) -> torch.Tensor:
        if len(active_counts) < 3:
            return torch.tensor(0.0, device=self.device)
        ks = torch.tensor(sorted(active_counts), dtype=torch.float32, device=self.device)
        z = self.model(ks)
        return ((z[2:] - 2 * z[1:-1] + z[:-2]) ** 2).mean()

    def _get_epoch_counts(self) -> List[int]:
        src = self.train_counts
        if self.cfg.batch_counts and self.cfg.batch_counts > 0:
            n = min(self.cfg.batch_counts, len(src))
            return sorted(random.sample(src, n))
        return src

    def _compute_loss(self, counts: List[int]) -> Tuple[torch.Tensor, Dict[str, float]]:
        ks = torch.tensor(counts, dtype=torch.float32, device=self.device)
        target = torch.tensor(
            np.stack([self.prototypes[c] for c in counts], axis=0),
            dtype=torch.float32,
            device=self.device,
        )

        pred = self.model(ks)

        l_proto = F.mse_loss(pred, target)
        l_cos = cosine_loss(pred, target)
        l_smooth = self._smoothness_loss(counts)

        loss = (
            self.cfg.lambda_proto * l_proto
            + self.cfg.lambda_cos * l_cos
            + self.cfg.lambda_smooth * l_smooth
        )

        stats = {
            "loss": float(loss.item()),
            "proto": float(l_proto.item()),
            "cos": float(l_cos.item()),
            "smooth": float(l_smooth.item()),
        }
        return loss, stats

    def save_checkpoint(self, name: str) -> None:
        ckpt = {
            "epoch": self.epoch,
            "best_loss": self.best_loss,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "config": asdict(self.cfg),
            "counts_sorted": self.counts_sorted,
            "count_sizes": self.count_sizes,
            "base_feature_init": self.base_feature,
            "prototypes": self.prototypes,
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.random.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
            "base_requires_grad": (
                bool(self.model.base.requires_grad)
                if hasattr(self.model, "base") and isinstance(self.model.base, torch.nn.Parameter)
                else False
            ),
        }
        path = self.ckpt_dir / name
        torch.save(ckpt, path)

    def load_checkpoint(self, path: str | Path) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epoch = int(ckpt["epoch"])
        self.best_loss = float(ckpt.get("best_loss", float("inf")))

        if self.cfg.learn_base and "base_requires_grad" in ckpt:
            self._set_base_trainable(bool(ckpt["base_requires_grad"]))

        if "rng_state" in ckpt:
            rng = ckpt["rng_state"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch_rng = rng["torch"]
            if not isinstance(torch_rng, torch.ByteTensor):
                torch_rng = torch_rng.cpu().byte()
            torch.random.set_rng_state(torch_rng.cpu())
            if torch.cuda.is_available() and rng.get("torch_cuda") is not None:
                torch.cuda.set_rng_state_all(rng["torch_cuda"])

    def train(self) -> None:
        if self.cfg.resume:
            self.load_checkpoint(self.cfg.resume)

        has_val = len(self.val_counts) > 0
        if has_val:
            baseline_metrics = compute_baseline_metrics(
                self.val_counts, self.train_counts, self.prototypes, self.base_feature,
            )
            for k, v in baseline_metrics.items():
                self.writer.add_scalar(f"baseline/{k}", v, 0)

        no_improve = 0
        pbar = tqdm(range(self.epoch + 1, self.cfg.epochs + 1), desc="Training")
        for epoch in pbar:
            self.epoch = epoch
            self._update_base_warmup(epoch)

            self.model.train()

            counts_epoch = self._get_epoch_counts()
            loss, stats = self._compute_loss(counts_epoch)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optimizer.step()

            self.writer.add_scalar("train/loss", stats["loss"], epoch)
            self.writer.add_scalar("train/loss_proto", stats["proto"], epoch)
            self.writer.add_scalar("train/loss_cos", stats["cos"], epoch)
            self.writer.add_scalar("train/loss_smooth", stats["smooth"], epoch)
            self.writer.add_scalar("train/grad_norm", float(grad_norm), epoch)

            train_eval = evaluate_counts(
                self.model, self.train_counts, self.prototypes, self.device,
            )
            self.writer.add_scalar("train_eval/l2", train_eval["l2"], epoch)
            self.writer.add_scalar("train_eval/cosine", train_eval["cosine"], epoch)
            self.writer.add_scalar("train_eval/mse", train_eval["mse"], epoch)
            self.writer.add_scalar("train_eval/cosine_loss", train_eval["cosine_loss"], epoch)
            self.writer.add_scalar("train_eval/smoothness", train_eval["smoothness"], epoch)
            for c, v in train_eval["l2_per_count"].items():
                self.writer.add_scalar(f"train_per_count/l2_count_{c}", v, epoch)
            for c, v in train_eval["cosine_per_count"].items():
                self.writer.add_scalar(f"train_per_count/cos_count_{c}", v, epoch)

            val_score = None
            if has_val:
                val_eval = evaluate_counts(
                    self.model, self.val_counts, self.prototypes, self.device,
                )
                self.writer.add_scalar("val/l2", val_eval["l2"], epoch)
                self.writer.add_scalar("val/cosine", val_eval["cosine"], epoch)
                self.writer.add_scalar("val/mse", val_eval["mse"], epoch)
                self.writer.add_scalar("val/cosine_loss", val_eval["cosine_loss"], epoch)
                self.writer.add_scalar("val/smoothness", val_eval["smoothness"], epoch)
                for c, v in val_eval["l2_per_count"].items():
                    self.writer.add_scalar(f"val_per_count/l2_count_{c}", v, epoch)
                for c, v in val_eval["cosine_per_count"].items():
                    self.writer.add_scalar(f"val_per_count/cos_count_{c}", v, epoch)

                self.writer.add_scalar("gap/l2", val_eval["l2"] - train_eval["l2"], epoch)
                self.writer.add_scalar("gap/cosine", train_eval["cosine"] - val_eval["cosine"], epoch)
                self.writer.add_scalar("gap/mse", val_eval["mse"] - train_eval["mse"], epoch)
                self.writer.add_scalar("gap/cosine_loss", val_eval["cosine_loss"] - train_eval["cosine_loss"], epoch)

                if baseline_metrics["interp_l2"] > 0:
                    self.writer.add_scalar("gain_vs_baseline/interp_l2", baseline_metrics["interp_l2"] - val_eval["l2"], epoch)
                    self.writer.add_scalar("gain_vs_baseline/interp_cos", val_eval["cosine"] - baseline_metrics["interp_cos"], epoch)

                val_score = val_eval["mse"] + 0.5 * val_eval["cosine_loss"]

            if self.cfg.figure_every > 0 and epoch % self.cfg.figure_every == 0:
                fig_pca = make_pca_figure(
                    self.model, self.counts_sorted, self.prototypes,
                    self.train_counts, self.val_counts, self.device,
                )
                self.writer.add_figure("figures/prototype_pca", fig_pca, epoch)
                plt.close(fig_pca)

                fig_heat = make_distance_heatmap(
                    self.model, self.counts_sorted, self.prototypes, self.device,
                )
                self.writer.add_figure("figures/predicted_distance_heatmap", fig_heat, epoch)
                plt.close(fig_heat)

            criterion = val_score if (has_val and val_score is not None) else stats["loss"]
            pbar.set_postfix(
                loss=f"{stats['loss']:.5f}",
                proto=f"{stats['proto']:.5f}",
                cos=f"{stats['cos']:.5f}",
                smooth=f"{stats['smooth']:.5f}",
                **({
                    "val_l2": f"{val_eval['l2']:.5f}",
                    "val_cos": f"{val_eval['cosine']:.5f}",
                } if has_val else {}),
            )

            if criterion < self.best_loss:
                self.best_loss = criterion
                self.save_checkpoint("best.pt")
                no_improve = 0
            else:
                no_improve += 1

            if self.cfg.patience > 0 and no_improve >= self.cfg.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {self.cfg.patience} epochs)")
                break

            if epoch % self.cfg.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch}.pt")
                self.save_checkpoint("latest.pt")

        self.save_checkpoint("final.pt")
        self.writer.close()

    @torch.no_grad()
    def predict_embeddings(self, counts: List[int]) -> Dict[int, np.ndarray]:
        self.model.eval()
        ks = torch.tensor(counts, dtype=torch.float32, device=self.device)
        pred = self.model(ks).cpu().numpy()
        return {int(c): pred[i] for i, c in enumerate(counts)}

    def load_for_inference(self, ckpt_path: str | Path) -> None:
        self.load_checkpoint(ckpt_path)
        self.model.eval()

    def _set_base_trainable(self, trainable: bool) -> None:
        if hasattr(self.model, "base") and isinstance(self.model.base, torch.nn.Parameter):
            self.model.base.requires_grad_(trainable)

    def _update_base_warmup(self, epoch: int) -> None:
        if not self.cfg.learn_base:
            return

        should_train_base = epoch > self.cfg.base_warmup_epochs
        self._set_base_trainable(should_train_base)

        self.writer.add_scalar("train/base_trainable", float(should_train_base), epoch)

    def run_loco(self) -> Dict[int, np.ndarray]:
        results = []
        predicted_prototypes: Dict[int, np.ndarray] = {}

        pbar = tqdm(self.counts_sorted, desc="LOCO splits")
        for c_val in pbar:
            train_counts = [c for c in self.counts_sorted if c != c_val]
            if len(train_counts) < 3:
                continue

            set_seed(self.cfg.seed)
            model = ResidualCountAdapter(
                d_feat=self.d_feat,
                rank=self.cfg.rank,
                hidden=self.cfg.hidden,
                base_feature=self.base_feature,
                count_min=min(train_counts),
                count_max=max(train_counts),
                learn_base=self.cfg.learn_base,
            ).to(self.device)

            optimizer = torch.optim.AdamW(
                model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay,
            )

            best_loss = float("inf")
            best_state = None
            no_improve = 0

            for epoch in range(1, self.cfg.epochs + 1):
                if self.cfg.learn_base and self.cfg.base_warmup_epochs > 0:
                    trainable = epoch > self.cfg.base_warmup_epochs
                    if hasattr(model, "base") and isinstance(model.base, nn.Parameter):
                        model.base.requires_grad_(trainable)

                model.train()
                ks = torch.tensor(train_counts, dtype=torch.float32, device=self.device)
                target = torch.tensor(
                    np.stack([self.prototypes[c] for c in train_counts], axis=0),
                    dtype=torch.float32, device=self.device,
                )
                pred = model(ks)
                l_proto = F.mse_loss(pred, target)
                l_cos = cosine_loss(pred, target)

                if len(train_counts) >= 3:
                    sorted_tc = sorted(train_counts)
                    ks_s = torch.tensor(sorted_tc, dtype=torch.float32, device=self.device)
                    z_s = model(ks_s)
                    l_smooth = ((z_s[2:] - 2 * z_s[1:-1] + z_s[:-2]) ** 2).mean()
                else:
                    l_smooth = torch.tensor(0.0, device=self.device)

                loss = (
                    self.cfg.lambda_proto * l_proto
                    + self.cfg.lambda_cos * l_cos
                    + self.cfg.lambda_smooth * l_smooth
                )

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
                optimizer.step()

                if loss.item() < best_loss:
                    best_loss = loss.item()
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
                    no_improve = 0
                else:
                    no_improve += 1

                if self.cfg.patience > 0 and no_improve >= self.cfg.patience:
                    break

            if best_state is not None:
                model.load_state_dict(best_state)

            model.eval()
            with torch.no_grad():
                c_tensor = torch.tensor([c_val], dtype=torch.float32, device=self.device)
                z_pred = model(c_tensor).cpu().numpy()[0]

            z_true = self.prototypes[c_val]
            l2_err = float(np.linalg.norm(z_pred - z_true))
            norm_true = np.linalg.norm(z_true) + 1e-8
            cos_sim = float(np.dot(z_pred, z_true) / (np.linalg.norm(z_pred) * norm_true))

            l2_mean = float(np.linalg.norm(self.base_feature - z_true))
            cos_mean = float(np.dot(self.base_feature, z_true) / (np.linalg.norm(self.base_feature) * norm_true))

            lower = [c for c in train_counts if c < c_val]
            upper = [c for c in train_counts if c > c_val]
            if lower and upper:
                z_interp = 0.5 * (self.prototypes[max(lower)] + self.prototypes[min(upper)])
            elif lower:
                z_interp = self.prototypes[max(lower)]
            elif upper:
                z_interp = self.prototypes[min(upper)]
            else:
                z_interp = self.base_feature
            l2_interp = float(np.linalg.norm(z_interp - z_true))
            cos_interp = float(np.dot(z_interp, z_true) / (np.linalg.norm(z_interp) * norm_true))

            self.writer.add_scalar("loco/model_l2", l2_err, c_val)
            self.writer.add_scalar("loco/model_cos", cos_sim, c_val)
            self.writer.add_scalar("loco/baseline_mean_l2", l2_mean, c_val)
            self.writer.add_scalar("loco/baseline_mean_cos", cos_mean, c_val)
            self.writer.add_scalar("loco/baseline_interp_l2", l2_interp, c_val)
            self.writer.add_scalar("loco/baseline_interp_cos", cos_interp, c_val)

            results.append({
                "count": c_val,
                "l2_error": l2_err,
                "cosine_similarity": cos_sim,
                "mean_l2": l2_mean,
                "mean_cos": cos_mean,
                "interp_l2": l2_interp,
                "interp_cos": cos_interp,
            })
            predicted_prototypes[c_val] = z_pred
            pbar.set_postfix(count=c_val, l2=f"{l2_err:.4f}", cos=f"{cos_sim:.4f}")

        results_df = pd.DataFrame(results)
        mean_model_l2 = results_df["l2_error"].mean()
        mean_model_cos = results_df["cosine_similarity"].mean()
        mean_mean_l2 = results_df["mean_l2"].mean()
        mean_mean_cos = results_df["mean_cos"].mean()
        mean_interp_l2 = results_df["interp_l2"].mean()
        mean_interp_cos = results_df["interp_cos"].mean()

        self.writer.add_scalar("loco/mean_model_l2", mean_model_l2, 0)
        self.writer.add_scalar("loco/mean_model_cos", mean_model_cos, 0)
        self.writer.add_scalar("loco/mean_baseline_mean_l2", mean_mean_l2, 0)
        self.writer.add_scalar("loco/mean_baseline_mean_cos", mean_mean_cos, 0)
        self.writer.add_scalar("loco/mean_baseline_interp_l2", mean_interp_l2, 0)
        self.writer.add_scalar("loco/mean_baseline_interp_cos", mean_interp_cos, 0)

        print("\n===== LOCO Results =====")
        print(results_df.to_string(index=False))
        print(f"\nMean Model   L2: {mean_model_l2:.6f}  Cos: {mean_model_cos:.6f}")
        print(f"Mean Global  L2: {mean_mean_l2:.6f}  Cos: {mean_mean_cos:.6f}")
        print(f"Mean Interp  L2: {mean_interp_l2:.6f}  Cos: {mean_interp_cos:.6f}")

        results_df.to_csv(self.artifacts_dir / "loco_results.csv", index=False)
        np.savez_compressed(
            self.artifacts_dir / "loco_predicted_prototypes.npz",
            counts=np.array(list(predicted_prototypes.keys()), dtype=np.int32),
            embeddings=np.stack(list(predicted_prototypes.values()), axis=0),
        )

        self.writer.close()
        return predicted_prototypes


# =========================
# Inference helper
# =========================

def run_inference(cfg: TrainConfig) -> None:
    if not cfg.checkpoint:
        raise ValueError("--checkpoint is required with --inference-only")

    trainer = Trainer(cfg)
    trainer.load_for_inference(cfg.checkpoint)

    counts = cfg.counts if cfg.counts is not None else trainer.counts_sorted
    preds = trainer.predict_embeddings(counts)

    out_path = Path(cfg.output_dir) / "artifacts" / "inference_embeddings.npz"
    np.savez_compressed(
        out_path,
        counts=np.array(counts, dtype=np.int32),
        embeddings=np.stack([preds[c] for c in counts], axis=0),
    )
    print(f"Saved predicted embeddings to: {out_path}")

    # Optional console summary
    print("\nPredicted embeddings:")
    for c in counts:
        print(f"count={c:2d}  norm={np.linalg.norm(preds[c]):.4f}")


# =========================
# CLI
# =========================

def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)

    parser.add_argument("--model-name", type=str, default="facebook/dinov2-base")
    parser.add_argument("--feature-mode", type=str, choices=["cls", "mean_patch", "concat"], default="cls")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--rank", type=int, default=128)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--learn-base", action="store_true")
    parser.add_argument("--base-warmup-epochs", type=int, default=0)

    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--lambda-proto", type=float, default=1.0)
    parser.add_argument("--lambda-cos", type=float, default=0.5)
    parser.add_argument("--lambda-smooth", type=float, default=0.05)

    parser.add_argument("--min-count-samples", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--batch-counts", type=int, default=0)

    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--inference-only", action="store_true")
    parser.add_argument("--counts", type=int, nargs="*", default=None)

    parser.add_argument("--cache", action="store_true", help="Cache DINO features for reuse")
    parser.add_argument("--cache-dir", type=str, default="data/cache/dino_cache")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (0=disabled)")
    parser.add_argument("--loco", action="store_true", help="Run Leave-One-Count-Out validation")
    parser.add_argument("--val-counts", type=int, nargs="*", default=None, help="Held-out counts for validation")
    parser.add_argument("--figure-every", type=int, default=20, help="Log TB figures every N epochs")

    args = parser.parse_args()

    return TrainConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        model_name=args.model_name,
        feature_mode=args.feature_mode,
        device=args.device,
        seed=args.seed,
        rank=args.rank,
        hidden=args.hidden,
        learn_base=args.learn_base,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        lambda_proto=args.lambda_proto,
        lambda_cos=args.lambda_cos,
        lambda_smooth=args.lambda_smooth,
        min_count_samples=args.min_count_samples,
        save_every=args.save_every,
        batch_counts=args.batch_counts,
        resume=args.resume,
        checkpoint=args.checkpoint,
        inference_only=args.inference_only,
        counts=args.counts,
        base_warmup_epochs=args.base_warmup_epochs,
        cache=args.cache,
        cache_dir=args.cache_dir,
        patience=args.patience,
        loco=args.loco,
        val_counts=args.val_counts,
        figure_every=args.figure_every,
    )


# =========================
# Main
# =========================

def main() -> None:
    cfg = parse_args()

    if cfg.inference_only:
        run_inference(cfg)
        return

    trainer = Trainer(cfg)

    if cfg.loco:
        trainer.run_loco()
        return

    trainer.train()

    # Convenience: save embeddings from best checkpoint after training
    best_ckpt = Path(cfg.output_dir) / "checkpoints" / "best.pt"
    if best_ckpt.exists():
        trainer.load_for_inference(best_ckpt)
        preds = trainer.predict_embeddings(trainer.counts_sorted)
        out_path = Path(cfg.output_dir) / "artifacts" / "best_predicted_embeddings.npz"
        np.savez_compressed(
            out_path,
            counts=np.array(trainer.counts_sorted, dtype=np.int32),
            embeddings=np.stack([preds[c] for c in trainer.counts_sorted], axis=0),
        )
        print(f"Saved best predicted embeddings to: {out_path}")


if __name__ == "__main__":
    main()
import argparse
from src.core.configs.config_loader import apply_yaml_defaults
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train masked cluster reconstruction model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file. CLI flags override config values.")

    parser.add_argument("--data_root", type=str, default="./v18")
    parser.add_argument("--split", type=str, default="train", help="Split folder used to build stem list")
    parser.add_argument("--dino_name", type=str, default="dinov2_vits14")
    parser.add_argument("--k_regions", type=int, default=5)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_frac", type=float, default=0.20)
    parser.add_argument("--test_frac", type=float, default=0.10)

    parser.add_argument("--use_proj", action="store_true", default=True)
    parser.add_argument("--no_use_proj", action="store_false", dest="use_proj")
    parser.add_argument("--emb_dim", type=int, default=32)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto",
                        help="Device to use for training/eval.")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--mask_prob", type=float, default=0.50)

    parser.add_argument("--early_metric", type=str, default="val_masked_acc",
                        choices=["val_loss", "val_masked_acc", "val_full_acc"])
    parser.add_argument("--early_mode", type=str, default="max", choices=["min", "max"])
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--save_best", action="store_true", default=True)
    parser.add_argument("--no_save_best", action="store_false", dest="save_best")

    parser.add_argument("--runs_root", type=str, default="./artifacts/runs/main/cluster_reconstruction",
                        help="Root directory for training runs (weights + tensorboard).")
    parser.add_argument("--run_name", type=str, default="",
                        help="Run folder name under runs_root. Empty = auto-generated.")
    parser.add_argument("--out_dir", type=str, default="",
                        help="Full output directory override. If set, runs_root/run_name is ignored.")

    # ---- Optional: compute typicality during evaluation (expensive) ----
    parser.add_argument("--eval_typicality", action="store_true", default=False,
                        help="If set, compute typicality score on val/test (slower).")
    parser.add_argument("--typicality_chunk", type=int, default=32,
                        help="Number of masked positions processed per forward pass (bigger = faster, more VRAM).")
    parser.add_argument("--typicality_max_batches", type=int, default=0,
                        help="If >0, limit typicality computation to first N batches (for quick checks).")

    preliminary, _ = parser.parse_known_args()
    apply_yaml_defaults(parser, preliminary.config)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return device_arg


class ClusterReconDataset(Dataset):
    def __init__(self, img_dir: Path, cluster_maps_dir: Path, stems: List[str], k_regions: int):
        self.img_dir = img_dir
        self.cluster_maps_dir = cluster_maps_dir
        self.stems = stems
        self.k_regions = k_regions

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        stem = self.stems[idx]
        cmap_path = self.cluster_maps_dir / f"{stem}_clusters.npy"

        lbl = np.load(cmap_path).astype(np.int64)
        # fix possible 1..K indexing
        if lbl.min() == 1 and lbl.max() == self.k_regions:
            lbl = lbl - 1
        lbl = np.clip(lbl, 0, self.k_regions - 1)

        return {
            "cluster_lbl": torch.from_numpy(lbl).long(),  # (H,W)
            "stem": stem,
        }


def collate(batch):
    lbl = torch.stack([b["cluster_lbl"] for b in batch], dim=0)  # (B,H,W)
    stems = [b["stem"] for b in batch]
    return {"cluster_lbl": lbl, "stem": stems}


def stems_from_annotations(ann_path: Path) -> List[str]:
    ann = json.loads(ann_path.read_text())
    stems = []
    for img in ann.get("images", []):
        stems.append(Path(img["file_name"]).stem)
    return stems


def build_stems(img_dir: Path, ann_path: Path, cluster_maps_dir: Path) -> List[str]:
    if ann_path.exists():
        stems = stems_from_annotations(ann_path)
    else:
        stems = [
            p.stem
            for p in sorted(img_dir.glob("*.npy"))
            if not p.name.endswith("_clusters.npy")
        ]

    filtered = []
    for stem in stems:
        img_ok = (img_dir / f"{stem}.npy").exists()
        cmap_ok = (cluster_maps_dir / f"{stem}_clusters.npy").exists()
        if img_ok and cmap_ok:
            filtered.append(stem)

    if not filtered:
        raise RuntimeError("No usable stems found after filtering")

    return filtered


def split_stems(stems: List[str], seed: int, val_frac: float, test_frac: float):
    rng = random.Random(seed)
    stems_all = stems.copy()
    rng.shuffle(stems_all)

    n = len(stems_all)
    n_test = int(round(test_frac * n))
    n_val = int(round(val_frac * n))

    test_stems = stems_all[:n_test]
    val_stems = stems_all[n_test:n_test + n_val]
    train_stems = stems_all[n_test + n_val:]

    if not train_stems:
        raise RuntimeError("Train split is empty. Adjust val_frac/test_frac.")

    return train_stems, val_stems, test_stems


class MaskedClusterModel(nn.Module):
    """
    Masked token model on cluster grids.

    Input tokens: (B,H,W) int64 in [0..K] where K is MASK_ID
    Outputs:
    - logits: (B,K,H,W) over real clusters 0..K-1
    - feats (optional): (B,hidden,H,W) backbone features used for semantic embedding
    """
    def __init__(self, k_regions: int, emb_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.k_regions = k_regions
        self.emb = nn.Embedding(num_embeddings=k_regions + 1, embedding_dim=emb_dim)

        self.backbone = nn.Sequential(
            nn.Conv2d(emb_dim, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(hidden, k_regions, 1)

    def forward(self, tokens_bhw: torch.Tensor, return_features: bool = False):
        x = self.emb(tokens_bhw)                          # (B,H,W,emb)
        x = x.permute(0, 3, 1, 2).contiguous()            # (B,emb,H,W)
        feats = self.backbone(x)                          # (B,hidden,H,W)
        logits = self.head(feats)                         # (B,K,H,W)
        if return_features:
            return logits, feats
        return logits


def masked_ce_loss(logits_bkhw: torch.Tensor, target_bhw: torch.Tensor, mask_bhw: torch.Tensor) -> torch.Tensor:
    b, k, h, w = logits_bkhw.shape
    logits = logits_bkhw.permute(0, 2, 3, 1).reshape(b * h * w, k)
    target = target_bhw.reshape(b * h * w)
    mask = mask_bhw.reshape(b * h * w)

    logits_m = logits[mask]
    target_m = target[mask]
    if logits_m.numel() == 0:
        return torch.zeros([], device=logits_bkhw.device)

    return F.cross_entropy(logits_m, target_m)


def is_improvement(current: float, best: float, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > best + min_delta
    return current < best - min_delta


@torch.no_grad()
def extract_semantic_embeddings(
    loader: DataLoader,
    model: MaskedClusterModel,
    device: str,
) -> Tuple[np.ndarray, List[str]]:
    """
    Extract one semantic embedding per sample from the backbone feature map.
    Layer choice:
    - use feats = backbone output (B,hidden,H,W)
    - global feature g = meanpool over (H,W) -> (B,hidden)
    - L2-normalize
    """
    model.eval()
    all_feats: List[torch.Tensor] = []
    all_stems: List[str] = []

    for batch in loader:
        tokens = batch["cluster_lbl"].to(device)  # (B,H,W)
        _, feats = model(tokens, return_features=True)
        g = feats.mean(dim=(2, 3))                # (B,hidden)
        g = F.normalize(g, dim=1)
        all_feats.append(g.cpu())
        all_stems.extend(batch["stem"])

    emb = torch.cat(all_feats, dim=0).numpy()
    return emb, all_stems


@torch.no_grad()
def compute_typicality_scores(
    loader: DataLoader,
    model: MaskedClusterModel,
    device: str,
    mask_id: int,
    typicality_chunk: int = 32,
    max_batches: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Typicality (pseudo-likelihood) for each sample y:
    typicality(y) = mean_{(i,j)} [ -log p_theta(y_ij | y_{\\ij}) ]

    Implementation details:
    - for each position (i,j) we mask only that token and query the model
    - p_theta(y_ij | y_{\\ij}) comes from softmax(logits at (i,j)) evaluated at the GT token y_ij
    - we accumulate -log-prob across all positions and average

    Efficient chunked version:
    - flatten positions P=H*W
    - process positions in chunks of size M (typicality_chunk)
    - replicate tokens B times per position in the chunk -> (B*M,H,W)
    - mask the designated position in each replica
    - forward once, gather log-prob at that masked position for the GT token
    - accumulate into per-sample totals
    """
    model.eval()
    all_scores: List[torch.Tensor] = []
    all_stems: List[str] = []

    batch_idx = 0
    for batch in tqdm(loader, desc="Computing typicality scores"):
        batch_idx += 1
        if max_batches > 0 and batch_idx > max_batches:
            break

        y = batch["cluster_lbl"].to(device)  # (B,H,W), values in [0..K-1]
        B, H, W = y.shape
        P = H * W

        # (P,2) list of (i,j)
        coords = torch.stack(torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        ), dim=-1).reshape(P, 2)  # (P,2)

        total_surprise = torch.zeros(B, device=device, dtype=torch.float32)

        # loop over chunks of positions
        for start in range(0, P, typicality_chunk):
            end = min(P, start + typicality_chunk)
            m = end - start
            chunk_coords = coords[start:end]  # (m,2)

            # replicate y for each position in chunk: (B*m,H,W)
            x = y.unsqueeze(1).expand(B, m, H, W).reshape(B * m, H, W).clone()

            # indices mapping replica r = b*m + t
            b_idx = torch.arange(B, device=device).repeat_interleave(m)         # (B*m,)
            t_idx = torch.arange(m, device=device).repeat(B)                    # (B*m,)
            ij = chunk_coords[t_idx]                                            # (B*m,2)
            ii = ij[:, 0]
            jj = ij[:, 1]

            # mask designated position
            x[torch.arange(B * m, device=device), ii, jj] = mask_id

            # forward
            logits = model(x)  # (B*m,K,H,W)
            # log-probs at masked locations -> (B*m,K)
            lp = F.log_softmax(logits[torch.arange(B * m, device=device), :, ii, jj], dim=1)

            # GT tokens for those locations -> (B*m,)
            gt = y[b_idx, ii, jj]

            # pick log p(gt) -> (B*m,)
            lp_gt = lp.gather(1, gt.unsqueeze(1)).squeeze(1)

            # accumulate -log p(gt) per original sample b
            # reshape to (B,m) then sum over m
            total_surprise += (-lp_gt).reshape(B, m).sum(dim=1)

        score = total_surprise / float(P)  # (B,)
        all_scores.append(score.detach().cpu())
        all_stems.extend(batch["stem"])

    scores = torch.cat(all_scores, dim=0).numpy()
    return scores, all_stems


def train_one_epoch(model, loader, optimizer, device, mask_prob, mask_id):
    model.train()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for batch in loader:
        y = batch["cluster_lbl"].to(device)

        mask = (torch.rand_like(y.float()) < mask_prob)
        x = y.clone()
        x[mask] = mask_id

        logits = model(x)
        loss = masked_ce_loss(logits, y, mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            pred = logits.argmax(dim=1)
            correct = ((pred == y) & mask).sum().float()
            denom = mask.sum().float().clamp_min(1.0)
            acc = (correct / denom).item()

        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

    total_loss /= max(n_batches, 1)
    total_acc /= max(n_batches, 1)
    return {"loss": total_loss, "masked_acc": total_acc}


@torch.no_grad()
def evaluate_model(loader, model, device, k_regions, mask_prob, mask_id):
    model.eval()

    total_loss = 0.0
    total_masked_correct = 0
    total_masked_count = 0
    total_full_correct = 0
    total_full_count = 0
    n_batches = 0

    true_counts = torch.zeros(k_regions, device=device)
    pred_counts = torch.zeros(k_regions, device=device)
    correct_counts = torch.zeros(k_regions, device=device)

    for batch in loader:
        y = batch["cluster_lbl"].to(device)

        mask = (torch.rand_like(y.float()) < mask_prob)
        x = y.clone()
        x[mask] = mask_id

        logits = model(x)
        loss = masked_ce_loss(logits, y, mask)
        pred = logits.argmax(dim=1)

        masked_correct = ((pred == y) & mask).sum().item()
        masked_count = mask.sum().item()

        total_loss += loss.item()
        total_masked_correct += masked_correct
        total_masked_count += masked_count

        total_full_correct += (pred == y).sum().item()
        total_full_count += y.numel()

        for k in range(k_regions):
            true_k = (y == k)
            pred_k = (pred == k)
            true_counts[k] += true_k.sum()
            pred_counts[k] += pred_k.sum()
            correct_counts[k] += (true_k & pred_k).sum()

        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    masked_acc = total_masked_correct / max(total_masked_count, 1)
    full_acc = total_full_correct / max(total_full_count, 1)

    recall = (correct_counts / true_counts.clamp_min(1)).cpu().numpy()
    precision = (correct_counts / pred_counts.clamp_min(1)).cpu().numpy()

    return {
        "loss": float(avg_loss),
        "masked_acc": float(masked_acc),
        "full_acc": float(full_acc),
        "recall_per_cluster": recall,
        "precision_per_cluster": precision,
    }


def log_eval_metrics(writer: SummaryWriter, prefix: str, stats: Dict, step: int) -> None:
    writer.add_scalar(f"{prefix}/loss", stats["loss"], step)
    writer.add_scalar(f"{prefix}/masked_acc", stats["masked_acc"], step)
    writer.add_scalar(f"{prefix}/full_acc", stats["full_acc"], step)

    recalls = stats["recall_per_cluster"]
    precisions = stats["precision_per_cluster"]
    for k, value in enumerate(recalls):
        writer.add_scalar(f"{prefix}/recall_cluster_{k}", float(value), step)
    for k, value in enumerate(precisions):
        writer.add_scalar(f"{prefix}/precision_cluster_{k}", float(value), step)


@torch.no_grad()
def log_random_val_reconstruction(
    writer: SummaryWriter,
    model: MaskedClusterModel,
    val_ds: Dataset,
    device: str,
    mask_prob: float,
    mask_id: int,
    k_regions: int,
    step: int,
) -> None:
    if len(val_ds) == 0:
        return

    model_was_training = model.training
    model.eval()

    idx = random.randrange(len(val_ds))
    item = val_ds[idx]
    stem = item.get("stem", f"idx_{idx}")

    # --- prepare input ---
    y = item["cluster_lbl"].to(device)  # (H,W) in [0..K-1]
    H, W = y.shape

    mask = (torch.rand_like(y.float()) < mask_prob)
    x = y.clone()
    x[mask] = mask_id  # MASK token

    logits = model(x.unsqueeze(0))      # (1,K,H,W)
    pred = logits.argmax(dim=1)[0]      # (H,W) in [0..K-1]

    # --- helper: integer label map -> RGB (3,H,W) using a fixed palette ---
    # We create a palette of size (K+1) so MASK has its own color (last entry).
    def label_to_rgb(lbl_hw: torch.Tensor, palette: torch.Tensor) -> torch.Tensor:
        # lbl_hw: (H,W) int64 in [0..K] where K==mask_id is allowed
        rgb_hwc = palette[lbl_hw.long().clamp(0, palette.shape[0] - 1)]  # (H,W,3)
        return rgb_hwc.permute(2, 0, 1).contiguous()  # (3,H,W)

    # Build a deterministic palette on CPU: K clusters + 1 for mask.
    # Use evenly spaced HSV-like colors (simple + deterministic, no matplotlib dependency).
    # Palette values in [0,1].
    def make_palette(n: int) -> torch.Tensor:
        # n colors -> (n,3)
        # simple deterministic "color wheel"
        t = torch.linspace(0, 1, steps=n)
        r = torch.clamp(torch.sin(2 * torch.pi * (t + 0.0)) * 0.5 + 0.5, 0, 1)
        g = torch.clamp(torch.sin(2 * torch.pi * (t + 1/3)) * 0.5 + 0.5, 0, 1)
        b = torch.clamp(torch.sin(2 * torch.pi * (t + 2/3)) * 0.5 + 0.5, 0, 1)
        pal = torch.stack([r, g, b], dim=1)
        return pal

    palette = make_palette(k_regions + 1)   # last color reserved for MASK
    # Force MASK to be a very visible fixed color (e.g. white-ish)
    palette[-1] = torch.tensor([1.0, 1.0, 1.0])

    palette = palette.to(device)

    # --- build visual maps ---
    gt_rgb = label_to_rgb(y, palette)         # (3,H,W)
    pred_rgb = label_to_rgb(pred, palette)    # (3,H,W)
    masked_rgb = label_to_rgb(x, palette)     # (3,H,W)

    # Error map: red where wrong, black where correct
    err = (pred != y).float()                 # (H,W)
    err_rgb = torch.stack([err, torch.zeros_like(err), torch.zeros_like(err)], dim=0)  # (3,H,W)

    # --- make a 2x2 grid: [GT | Pred] on top, [Masked | Error] bottom ---
    # TensorBoard expects (C,H,W) or (B,C,H,W). We'll build one (3, 2H, 2W).
    top = torch.cat([gt_rgb, pred_rgb], dim=2)        # (3,H,2W)
    bot = torch.cat([masked_rgb, err_rgb], dim=2)     # (3,H,2W)
    grid = torch.cat([top, bot], dim=1)               # (3,2H,2W)

    writer.add_image(f"val/recon_grid/random", grid, global_step=step)

    # Also log a scalar: masked accuracy for this sample (quick sanity)
    masked_count = mask.sum().item()
    if masked_count > 0:
        masked_acc = (((pred == y) & mask).sum().item()) / masked_count
        writer.add_scalar("val/recon_one_sample_masked_acc", masked_acc, global_step=step)

    if model_was_training:
        model.train()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = resolve_device(args.device)

    data_root = Path(args.data_root)
    img_dir = data_root / args.split
    ann_path = img_dir / "annotations.json"
    cluster_maps_dir = data_root / "dino_global_clusters" / f"maps_{args.dino_name}_k{args.k_regions}"
    centers_path = data_root / "dino_global_clusters" / f"centers_{args.dino_name}_k{args.k_regions}.pt"

    if not cluster_maps_dir.exists():
        raise FileNotFoundError(f"Missing: {cluster_maps_dir}")
    if not centers_path.exists():
        raise FileNotFoundError(f"Missing centers: {centers_path}")

    ckpt = torch.load(centers_path, map_location="cpu")
    centers = ckpt["centers"].float()
    if centers.shape[0] != args.k_regions:
        raise RuntimeError(f"centers K mismatch: {centers.shape}")
    dino_dim = int(centers.shape[1])

    emb_dim = args.emb_dim if args.use_proj else dino_dim
    mask_id = args.k_regions  # special token

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        run_name = args.run_name if args.run_name else f"k{args.k_regions}_emb{emb_dim}_seed{args.seed}"
        out_dir = Path(args.runs_root) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt_path = out_dir / "best_model.pt"
    history_path = out_dir / "history.json"
    tb_dir = out_dir / "tb"

    stems = build_stems(img_dir, ann_path, cluster_maps_dir)
    train_stems, val_stems, test_stems = split_stems(stems, args.seed, args.val_frac, args.test_frac)

    train_ds = ClusterReconDataset(img_dir, cluster_maps_dir, train_stems, args.k_regions)
    val_ds = ClusterReconDataset(img_dir, cluster_maps_dir, val_stems, args.k_regions)
    test_ds = ClusterReconDataset(img_dir, cluster_maps_dir, test_stems, args.k_regions)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate,
    )

    model = MaskedClusterModel(args.k_regions, emb_dim=emb_dim, hidden=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"device: {device}")
    print(f"loaded centers: {centers.shape} dino={ckpt.get('dino_name', '?')} k={ckpt.get('k', '?')}")
    print(f"stems total={len(stems)} train={len(train_stems)} val={len(val_stems)} test={len(test_stems)}")
    print(f"params: {n_params / 1e6:.2f}M")
    print(f"TensorBoard logdir: {tb_dir}")

    writer = SummaryWriter(log_dir=str(tb_dir))
    writer.add_text("run/config", json.dumps(vars(args), indent=2))

    history = []
    best_value = None
    best_epoch = None
    bad_epochs = 0

    pbar = tqdm(range(1, args.epochs + 1), desc="Training epochs")
    for epoch in pbar:
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            mask_prob=args.mask_prob,
            mask_id=mask_id,
        )
        val_stats = evaluate_model(
            loader=val_loader,
            model=model,
            device=device,
            k_regions=args.k_regions,
            mask_prob=args.mask_prob,
            mask_id=mask_id,
        )

        if args.early_metric == "val_loss":
            current_value = float(val_stats["loss"])
        elif args.early_metric == "val_masked_acc":
            current_value = float(val_stats["masked_acc"])
        else:
            current_value = float(val_stats["full_acc"])

        history_row = {
            "epoch": epoch,
            "train_loss": float(train_stats["loss"]),
            "train_masked_acc": float(train_stats["masked_acc"]),
            "val_loss": float(val_stats["loss"]),
            "val_masked_acc": float(val_stats["masked_acc"]),
            "val_full_acc": float(val_stats["full_acc"]),
            "val_recall_per_cluster": val_stats["recall_per_cluster"].tolist(),
            "val_precision_per_cluster": val_stats["precision_per_cluster"].tolist(),
        }
        history.append(history_row)

        writer.add_scalar("train/loss", train_stats["loss"], epoch)
        writer.add_scalar("train/masked_acc", train_stats["masked_acc"], epoch)
        log_eval_metrics(writer, "val", val_stats, epoch)
        log_random_val_reconstruction(
            writer=writer,
            model=model,
            val_ds=val_ds,
            device=device,
            mask_prob=args.mask_prob,
            mask_id=mask_id,
            k_regions=args.k_regions,
            step=epoch,
        )
        writer.add_scalar("early_stop/current_value", current_value, epoch)

        # Optional: typicality on val (expensive)
        if args.eval_typicality:
            val_typ, _ = compute_typicality_scores(
                loader=val_loader,
                model=model,
                device=device,
                mask_id=mask_id,
                typicality_chunk=args.typicality_chunk,
                max_batches=args.typicality_max_batches,
            )
            writer.add_scalar("val/typicality_mean", float(np.mean(val_typ)), epoch)
            writer.add_scalar("val/typicality_std", float(np.std(val_typ)), epoch)

        if is_improvement(current_value, best_value, args.early_mode, args.min_delta):
            best_value = current_value
            best_epoch = epoch
            bad_epochs = 0

            if args.save_best:
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state": model.state_dict(),
                        "best_metric": args.early_metric,
                        "best_value": best_value,
                        "k_regions": args.k_regions,
                        "mask_id": mask_id,
                        "mask_prob": args.mask_prob,
                        "emb_dim": emb_dim,
                        "hidden_dim": args.hidden_dim,
                    },
                    best_ckpt_path,
                )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(
                    f"Early stopping at epoch {epoch}. "
                    f"Best epoch={best_epoch} best_value={best_value:.6f}"
                )
                break

        pbar.set_postfix(
            {
                "train_loss": f"{train_stats['loss']:.4f}",
                "val_masked_acc": f"{val_stats['masked_acc']:.4f}",
                "val_full_acc": f"{val_stats['full_acc']:.4f}",
                "best_epoch": best_epoch,
                "bad_epochs": bad_epochs,
            }
        )

    if args.save_best and best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt["model_state"])
        print(
            "Loaded best checkpoint:",
            f"epoch={best_ckpt['epoch']}",
            f"best_value={best_ckpt['best_value']}",
        )

    test_stats = evaluate_model(
        loader=test_loader,
        model=model,
        device=device,
        k_regions=args.k_regions,
        mask_prob=args.mask_prob,
        mask_id=mask_id,
    )

    final_step = best_epoch if best_epoch is not None else len(history)
    log_eval_metrics(writer, "test", test_stats, int(final_step))

    # Optional: typicality on test (expensive)
    if args.eval_typicality:
        test_typ, _ = compute_typicality_scores(
            loader=test_loader,
            model=model,
            device=device,
            mask_id=mask_id,
            typicality_chunk=args.typicality_chunk,
            max_batches=args.typicality_max_batches,
        )
        writer.add_scalar("test/typicality_mean", float(np.mean(test_typ)), int(final_step))
        writer.add_scalar("test/typicality_std", float(np.std(test_typ)), int(final_step))

    print(
        "TEST -> "
        f"loss={test_stats['loss']:.4f} "
        f"masked_acc={test_stats['masked_acc']:.4f} "
        f"full_acc={test_stats['full_acc']:.4f}"
    )

    history_path.write_text(json.dumps(history, indent=2))
    writer.close()

    print(f"Saved history to: {history_path}")
    print(f"Run TensorBoard with: tensorboard --logdir {tb_dir}")
    print("NOTE: semantic embeddings + typicality extraction utilities are implemented:")
    print("  - extract_semantic_embeddings(loader, model, device)")
    print("  - compute_typicality_scores(loader, model, device, mask_id, typicality_chunk=..., max_batches=...)")


if __name__ == "__main__":
    main()
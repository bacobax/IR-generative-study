"""Per-checkpoint generation-quality evaluation for layout FM.

Standalone entrypoint, designed to be launched (optionally in parallel) for a
single saved checkpoint. It:

  1. builds the layout-FM model + VAE from a training preset and loads the
     checkpoint's EMA UNet weights;
  2. generates ``n_test`` images from real *test*-split layouts;
  3. generates ``n_unseen`` images from synthetic *unseen* layouts built by
     recombining real *train* boxes;
  4. scores KID / MMD / FID (DINOv2 features) between the test-layout
     generations and the real test images;
  5. runs the trained YOLO detector on the unseen-layout generations and scores
     bbox adherence (precision/recall/F1 @IoU, mean matched IoU, count-MAE);
  6. writes a per-checkpoint metrics JSON and appends a row to
     ``<out>/checkpoint_eval_metrics.csv``.

Reuses existing utilities: ``LayoutFMTrainer`` (model build),
``sample_layout_batch`` (generation), ``feature_extractors``/``generative_metrics``/
``mmd`` (metrics). No production training code is modified by this script.

Example
-------
    python -m scripts.eval_checkpoint_quality \
        --config configs/fm/train/presets/stay_layout_latent_v18_sd15ft_x8_256_minmax_3L_reg.yaml \
        --checkpoint .../UNET/unet_fm_best.pt --device cuda:0 --out /tmp/ckpt_eval
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from src.cli import train_flow_matching as T
from src.core.data import collate_layout_batch
from src.core.data.datasets import AnnotationLayoutDataset
from src.core.normalization import resize_and_normalize
from src.algorithms.training.layout_flow_matching_trainer import LayoutFMTrainer
from src.algorithms.inference.rare_layout_dataset_tools import sample_layout_batch
from src.evaluation.feature_extractors import build_feature_extractor, extract_features
from src.evaluation.generative_metrics import compute_fid, compute_kid
from src.evaluation.mmd import compute_rbf_mmd


# ── config build (reuse training CLI parser so presets resolve identically) ──
def build_cfg(preset: str):
    parser = T.build_parser()
    argv = ["--config", preset]
    args = parser.parse_args(argv)
    return T.merge_config_and_cli(
        T.FMTrainConfig, args.config, parser, args,
        flat_to_nested=T._FLAT_TO_NESTED, cli_argv=argv,
    )


def _epoch_from_checkpoint(path: str) -> Optional[int]:
    m = re.search(r"epoch_(\d+)", os.path.basename(path))
    if m:
        return int(m.group(1))
    if "best" in os.path.basename(path):
        return -1
    return None


# ── image domain helpers (identical preprocessing for real + generated) ──────
def _normalized_to_uint8(img: torch.Tensor) -> np.ndarray:
    """Map a normalized image tensor [C,H,W] to a HxW(or HxWx3) uint8 array."""
    x = img.detach().cpu().float()
    lo = -1.0 if float(x.min()) < 0.0 else 0.0
    x01 = ((x - lo) / (1.0 - lo)).clamp(0.0, 1.0)
    arr = (x01 * 255.0).round().byte().numpy()
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    elif arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _save_png(img: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_normalized_to_uint8(img)).save(path)


def _real_npy_to_png(npy_path: Path, out_path: Path, *, image_size: int, normalization_mode: str) -> None:
    arr = np.load(npy_path)
    if arr.ndim == 2:
        arr = arr[None, ...]
    t = torch.from_numpy(arr.copy()).float()
    norm = resize_and_normalize(t, image_size=image_size, normalization_mode=normalization_mode)
    _save_png(norm, out_path)


# ── unseen-layout synthesis: recombine real train boxes ──────────────────────
def synthesize_unseen_layouts(
    train_dataset: AnnotationLayoutDataset,
    *,
    k: int,
    image_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    """Build ``k`` synthetic samples from novel combinations of real train boxes.

    Pools every real (normalized) train box, samples an object count from the
    train per-image count distribution, then draws that many boxes from the pool.
    Boxes are emitted in ``image_size`` pixel coords so ``collate_layout_batch``
    re-derives the normalized form consistently.
    """
    rng = np.random.default_rng(seed)
    box_pool: List[np.ndarray] = []
    counts: List[int] = []
    person_label = train_dataset.category_ids[0] if train_dataset.category_ids else 0
    for i in range(len(train_dataset)):
        sample = train_dataset[i]
        boxes = sample["boxes_xyxy"]  # image_size coords
        n = int(sample["n_objects"])
        counts.append(n)
        if n > 0:
            box_pool.append(boxes.numpy().astype(np.float32))
    if not box_pool:
        raise RuntimeError("Train set has no boxes to recombine for unseen layouts.")
    all_boxes = np.concatenate(box_pool, axis=0)
    counts_arr = np.asarray([c for c in counts if c > 0], dtype=np.int64)

    samples: List[Dict[str, Any]] = []
    for j in range(k):
        n = int(rng.choice(counts_arr))
        idx = rng.integers(0, all_boxes.shape[0], size=n)
        boxes = torch.from_numpy(all_boxes[idx].copy()).float().clamp(0.0, float(image_size))
        labels = torch.full((n,), int(person_label), dtype=torch.long)
        samples.append({
            "pixel_values": torch.zeros(1, image_size, image_size, dtype=torch.float32),
            "boxes_xyxy": boxes,
            "labels": labels,
            "n_objects": n,
            "file_name": f"unseen_{j:05d}",
            "image_id": j,
            "label_names": ["person"] * n,
        })
    return samples


# ── generation ───────────────────────────────────────────────────────────────
def _generate(sampler, samples: Sequence[Dict[str, Any]], *, steps: int, seed: int,
              batch_size: int, out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for start in range(0, len(samples), batch_size):
        chunk = list(samples[start : start + batch_size])
        batch = collate_layout_batch(chunk)
        imgs = sample_layout_batch(sampler, batch, steps=steps, seed=seed + start)
        for i in range(imgs.shape[0]):
            p = out_dir / f"{chunk[i]['file_name']}.png"
            _save_png(imgs[i], p)
            paths.append(p)
    return paths


# ── YOLO adherence ───────────────────────────────────────────────────────────
def _iou_matrix(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    if pred.size == 0 or gt.size == 0:
        return np.zeros((pred.shape[0], gt.shape[0]), dtype=np.float64)
    px1, py1, px2, py2 = pred[:, 0:1], pred[:, 1:2], pred[:, 2:3], pred[:, 3:4]
    gx1, gy1, gx2, gy2 = gt[:, 0], gt[:, 1], gt[:, 2], gt[:, 3]
    ix1 = np.maximum(px1, gx1); iy1 = np.maximum(py1, gy1)
    ix2 = np.minimum(px2, gx2); iy2 = np.minimum(py2, gy2)
    iw = np.clip(ix2 - ix1, 0, None); ih = np.clip(iy2 - iy1, 0, None)
    inter = iw * ih
    pa = np.clip(px2 - px1, 0, None) * np.clip(py2 - py1, 0, None)
    ga = np.clip(gx2 - gx1, 0, None) * np.clip(gy2 - gy1, 0, None)
    union = pa + ga - inter
    return inter / np.clip(union, 1e-9, None)


def _match_one(pred: np.ndarray, gt: np.ndarray, *, iou_thr: float):
    """Greedy IoU matching → (tp, fp, fn, matched_ious)."""
    n_pred, n_gt = pred.shape[0], gt.shape[0]
    if n_gt == 0:
        return 0, n_pred, 0, []
    if n_pred == 0:
        return 0, 0, n_gt, []
    iou = _iou_matrix(pred, gt)
    pairs = sorted(
        ((iou[i, j], i, j) for i in range(n_pred) for j in range(n_gt)),
        key=lambda t: -t[0],
    )
    used_p, used_g, matched = set(), set(), []
    for v, i, j in pairs:
        if v < iou_thr:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(i); used_g.add(j); matched.append(float(v))
    tp = len(matched)
    return tp, n_pred - tp, n_gt - tp, matched


def yolo_adherence(
    image_paths: Sequence[Path],
    gt_boxes_per_image: Sequence[np.ndarray],
    *,
    weights: str,
    device: str,
    iou_thr: float,
    conf: float,
) -> Dict[str, float]:
    from ultralytics import YOLO

    model = YOLO(weights)
    tp = fp = fn = 0
    all_ious: List[float] = []
    count_abs_err: List[float] = []
    for path, gt in zip(image_paths, gt_boxes_per_image):
        res = model.predict(source=str(path), conf=conf, device=device, verbose=False)[0]
        pred = res.boxes.xyxy.detach().cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        t, f, n, ious = _match_one(pred, np.asarray(gt, dtype=np.float64), iou_thr=iou_thr)
        tp += t; fp += f; fn += n; all_ious.extend(ious)
        count_abs_err.append(abs(pred.shape[0] - int(gt.shape[0])))
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "adherence_precision": float(precision),
        "adherence_recall": float(recall),
        "adherence_f1": float(f1),
        "adherence_mean_iou": float(np.mean(all_ious)) if all_ious else 0.0,
        "adherence_count_mae": float(np.mean(count_abs_err)) if count_abs_err else 0.0,
        "adherence_tp": int(tp), "adherence_fp": int(fp), "adherence_fn": int(fn),
    }


# ── distribution metrics ─────────────────────────────────────────────────────
def distribution_metrics(
    gen_paths: Sequence[Path],
    real_paths: Sequence[Path],
    *,
    extractor_name: str,
    dinov2_model_name: str,
    device: str,
    feature_batch_size: int,
    cache_dir: Path,
) -> Dict[str, float]:
    cfg = {"dinov2": {"model_name": dinov2_model_name, "pooling": "mean"}}
    extractor = build_feature_extractor(extractor_name, cfg, device)
    cache_dir.mkdir(parents=True, exist_ok=True)
    real_feats = extract_features(
        [str(p) for p in real_paths], extractor,
        batch_size=feature_batch_size, cache_path=cache_dir / f"real_{extractor_name}.npz",
    )
    gen_feats = extract_features(
        [str(p) for p in gen_paths], extractor,
        batch_size=feature_batch_size, cache_path=cache_dir / f"gen_{extractor_name}.npz",
        force=True,
    )
    return {
        "FID": compute_fid(real_feats, gen_feats),
        "KID": compute_kid(real_feats, gen_feats),
        "MMD": compute_rbf_mmd(real_feats, gen_feats, bandwidths=[0.1, 1.0, 10.0]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Training preset yaml.")
    ap.add_argument("--checkpoint", required=True, help="UNet EMA weights .pt.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True, help="Output dir for this run's eval.")
    ap.add_argument("--n-test", type=int, default=None)
    ap.add_argument("--n-unseen", type=int, default=None)
    ap.add_argument("--steps", type=int, default=None)
    args = ap.parse_args()

    cfg = build_cfg(args.config)
    ce = cfg.checkpoint_eval
    device = args.device
    image_size = int(cfg.data.image_size)
    n_test = int(args.n_test if args.n_test is not None else ce.n_test_images)
    n_unseen = int(args.n_unseen if args.n_unseen is not None else ce.n_unseen_images)
    steps = int(args.steps if args.steps is not None else ce.sample_steps)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    epoch = _epoch_from_checkpoint(args.checkpoint)

    test_dir = ce.test_dir or f"data/raw/{cfg.data.dataset_id}/test"
    test_ann = ce.test_annotations or str(Path(test_dir) / "annotations.json")

    resolved = T.resolve_training_data(cfg.data)
    normalization_mode = resolved.normalization_mode

    # num_classes / categories must be set before building the model. Prefer the
    # run's saved layout metadata (guarantees the embedding size matches the
    # checkpoint); fall back to the train split's category count.
    if not cfg.layout_conditioning.num_classes:
        meta_path = Path(args.checkpoint).resolve().parent.parent / "layout_conditioning.json"
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text())
            cfg.layout_conditioning.num_classes = int(meta["num_classes"])
            cfg.layout_conditioning.category_id_to_name = {
                int(k): v for k, v in meta.get("category_id_to_name", {}).items()
            }
        else:
            probe_ds = AnnotationLayoutDataset(
                root_dir=resolved.train_dir, annotations_path=resolved.train_annotations_path,
                image_size=image_size, normalization_mode=normalization_mode, include_label_names=True,
            )
            cfg.layout_conditioning.num_classes = probe_ds.num_categories
            cfg.layout_conditioning.category_id_to_name = dict(probe_ds.category_id_to_name)

    # 1) model + vae + sampler
    cfg.device = device
    from src.core.normalization import norm_to_display
    trainer = LayoutFMTrainer.from_config(cfg, from_norm_to_display=norm_to_display)
    trainer.load_unet_weights(args.checkpoint, strict=True)
    if trainer.vae is not None and getattr(cfg.model, "vae_weights", None):
        trainer.load_vae_weights(cfg.model.vae_weights, strict=False)
    trainer.unet.eval()
    sampler = trainer._make_sampler()
    print(f"[ckpt-eval] loaded {args.checkpoint} (epoch={epoch}) on {device}", flush=True)

    # 2) test-layout generations
    test_ds = AnnotationLayoutDataset(
        root_dir=test_dir, annotations_path=test_ann,
        image_size=image_size, normalization_mode=normalization_mode, include_label_names=True,
    )
    n_test = min(n_test, len(test_ds))
    test_samples = [test_ds[i] for i in range(n_test)]
    gen_test_dir = out_dir / f"gen_test_epoch_{epoch}"
    gen_test_paths = _generate(sampler, test_samples, steps=steps, seed=ce.seed,
                               batch_size=ce.gen_batch_size, out_dir=gen_test_dir)

    # 3) real test reference pngs (shared across checkpoints → cached on disk)
    ref_dir = out_dir / "real_test_png"
    if not ref_dir.is_dir() or len(list(ref_dir.glob("*.png"))) < n_test:
        for s in test_samples:
            _real_npy_to_png(Path(test_dir) / s["file_name"], ref_dir / f"{Path(s['file_name']).stem}.png",
                             image_size=image_size, normalization_mode=normalization_mode)
    real_paths = sorted(ref_dir.glob("*.png"))[:n_test]

    # 4) unseen-layout generations (recombined train boxes)
    train_ds = AnnotationLayoutDataset(
        root_dir=resolved.train_dir,
        annotations_path=resolved.train_annotations_path,
        image_size=image_size, normalization_mode=normalization_mode, include_label_names=True,
    )
    unseen_samples = synthesize_unseen_layouts(train_ds, k=n_unseen, image_size=image_size, seed=ce.seed)
    gen_unseen_dir = out_dir / f"gen_unseen_epoch_{epoch}"
    gen_unseen_paths = _generate(sampler, unseen_samples, steps=steps, seed=ce.seed + 7,
                                 batch_size=ce.gen_batch_size, out_dir=gen_unseen_dir)
    unseen_gt_boxes = [s["boxes_xyxy"].numpy() for s in unseen_samples]

    # 5) metrics
    metrics: Dict[str, Any] = {"epoch": epoch, "checkpoint": os.path.basename(args.checkpoint)}
    metrics.update(distribution_metrics(
        gen_test_paths, real_paths,
        extractor_name=ce.feature_extractor, dinov2_model_name=ce.dinov2_model_name,
        device=device, feature_batch_size=ce.feature_batch_size, cache_dir=out_dir / "features",
    ))
    metrics.update(yolo_adherence(
        gen_unseen_paths, unseen_gt_boxes,
        weights=ce.yolo_weights, device=device, iou_thr=ce.adherence_iou, conf=ce.yolo_conf,
    ))

    # 6) outputs
    with open(out_dir / f"metrics_epoch_{epoch}.json", "w") as f:
        json.dump(metrics, f, indent=2)
    csv_path = out_dir / "checkpoint_eval_metrics.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if write_header:
            w.writeheader()
        w.writerow(metrics)
    print(f"[ckpt-eval] epoch={epoch} "
          f"FID={metrics['FID']:.3f} KID={metrics['KID']:.5f} MMD={metrics['MMD']:.5f} "
          f"adherence_f1={metrics['adherence_f1']:.3f} mean_iou={metrics['adherence_mean_iou']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()

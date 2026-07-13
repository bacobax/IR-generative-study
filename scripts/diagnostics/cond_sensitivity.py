"""D1 — Conditioning-sensitivity probe for STAY LayoutFM.

Loads the EMA-best UNet checkpoint and measures the flow-matching MSE under three
conditioning regimes, holding the noise/target/coupling FIXED so the only thing that
varies is what the UNet sees:

  (a) true     — boxes/labels/object_mask aligned to the target (correct layout)
  (b) shuffled — same tensors rolled by 1 along batch (valid layout, wrong image)
  (c) empty    — object_mask all-False (no objects)

If loss(a) ~= loss(b) ~= loss(c) the UNet ignores the STAY conditioning (H1).
If loss(a) << loss(b),(c) the conditioning is functional.

Run for both val and train splits. Read-only; no production code changed.
"""
from __future__ import annotations

import argparse
import copy
import sys

import torch
from torch.utils.data import DataLoader

from src.cli import train_flow_matching as T
from src.core.data import collate_layout_batch
from src.core.data.datasets import AnnotationLayoutDataset
from src.core.data.latent_cache import build_latent_cache_dataset
from src.algorithms.training.layout_flow_matching_trainer import LayoutFMTrainer


def build_cfg(preset: str):
    parser = T.build_parser()
    argv = ["--config", preset]
    args = parser.parse_args(argv)
    return T.merge_config_and_cli(
        T.FMTrainConfig, args.config, parser, args,
        flat_to_nested=T._FLAT_TO_NESTED, cli_argv=argv,
    )


def build_datasets(cfg):
    rd = T.resolve_training_data(cfg.data)
    train_ds = AnnotationLayoutDataset(
        root_dir=rd.train_dir, annotations_path=rd.train_annotations_path,
        image_size=cfg.data.image_size, normalization_mode=rd.normalization_mode,
        include_label_names=True, subset_manifest=rd.train_subset_manifest,
    )
    eval_ds = AnnotationLayoutDataset(
        root_dir=rd.val_dir, annotations_path=rd.val_annotations_path,
        image_size=cfg.data.image_size, normalization_mode=rd.normalization_mode,
        include_label_names=True,
    )
    cfg.layout_conditioning.num_classes = train_ds.num_categories
    cfg.layout_conditioning.category_id_to_name = dict(train_ds.category_id_to_name)

    if cfg.latent_cache.enabled:
        dev = cfg.resolved_device()
        train_ds = build_latent_cache_dataset(
            base_dataset=train_ds, model_cfg=cfg.model, dataset_id=cfg.data.dataset_id,
            train_dir=rd.train_dir, val_dir=rd.val_dir, split="train",
            image_size=cfg.data.image_size, subset_manifest=rd.train_subset_manifest,
            normalization_mode=rd.normalization_mode, augment_config=cfg.augment,
            latent_cache_cfg=cfg.latent_cache, device=dev,
            strict_load=cfg.training.strict_load, batch_size=cfg.latent_cache.encode_batch_size,
        )
        eval_ds = build_latent_cache_dataset(
            base_dataset=eval_ds, model_cfg=cfg.model, dataset_id=cfg.data.dataset_id,
            train_dir=rd.train_dir, val_dir=rd.val_dir, split="val",
            image_size=cfg.data.image_size, subset_manifest=None,
            normalization_mode=rd.normalization_mode, augment_config=None,
            latent_cache_cfg=cfg.latent_cache, device=dev,
            strict_load=cfg.training.strict_load, batch_size=cfg.latent_cache.encode_batch_size,
        )
    return train_ds, eval_ds


def roll_cond(cond):
    """Roll every per-object tensor by 1 along batch dim → valid layout, wrong image."""
    out = {}
    for k, v in cond.items():
        out[k] = torch.roll(v, shifts=1, dims=0) if torch.is_tensor(v) else v
    return out


def empty_cond(cond):
    out = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in cond.items()}
    out["object_mask"] = torch.zeros_like(cond["object_mask"], dtype=cond["object_mask"].dtype)
    return out


@torch.no_grad()
def probe_split(trainer, loader, *, n_batches, n_seeds, device):
    t_scale = trainer.t_scale
    sums = {"true": 0.0, "shuffled": 0.0, "empty": 0.0}
    count = 0
    for bi, batch in enumerate(loader):
        if bi >= n_batches:
            break
        x_fm = trainer.fm_input_from_batch(batch).to(device)
        B = x_fm.shape[0]
        cond = trainer.prepare_conditioning_kwargs(batch, device=device)
        for s in range(n_seeds):
            torch.manual_seed(1000 + s)
            z0 = torch.randn_like(x_fm)
            t = torch.rand(B, device=device)
            te = t[:, None, None, None]
            # Fix coupling + target using TRUE conditioning, then reuse for all regimes.
            x_target, perm = trainer._match_flow_targets_with_permutation(z0, x_fm, cond)
            cond_al = trainer._permute_conditioning_kwargs(cond, perm, B)
            zt = (1.0 - te) * z0 + te * x_target
            v_target = x_target - z0
            regimes = {
                "true": cond_al,
                "shuffled": roll_cond(cond_al),
                "empty": empty_cond(cond_al),
            }
            for name, ck in regimes.items():
                out = trainer.unet(zt, t * t_scale, **ck)
                pred = out.sample
                if trainer.train_target == "x0":
                    pred = (pred - zt) / (1.0 - te).clamp(min=1e-5)
                sums[name] += float(torch.nn.functional.mse_loss(pred, v_target).item())
            count += 1
    return {k: v / max(count, 1) for k, v in sums.items()}, count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n_batches", type=int, default=6)
    ap.add_argument("--n_seeds", type=int, default=4)
    args = ap.parse_args()

    cfg = build_cfg(args.config)
    device = cfg.resolved_device()
    train_ds, eval_ds = build_datasets(cfg)

    trainer = LayoutFMTrainer.from_config(cfg, from_norm_to_display=None)
    trainer.load_unet_weights(args.ckpt, strict=True)
    trainer.unet.eval()
    print(f"[probe] loaded {args.ckpt}", flush=True)

    loaders = {
        "val": DataLoader(eval_ds, batch_size=cfg.data.batch_size, shuffle=False,
                          num_workers=2, collate_fn=collate_layout_batch),
        "train": DataLoader(train_ds, batch_size=cfg.data.batch_size, shuffle=False,
                            num_workers=2, collate_fn=collate_layout_batch),
    }

    print(f"\n{'split':6s} {'true':>10s} {'shuffled':>10s} {'empty':>10s}"
          f" {'shuf-true':>10s} {'empty-true':>10s}  n")
    for split, loader in loaders.items():
        res, n = probe_split(trainer, loader, n_batches=args.n_batches,
                             n_seeds=args.n_seeds, device=device)
        print(f"{split:6s} {res['true']:10.4f} {res['shuffled']:10.4f} {res['empty']:10.4f}"
              f" {res['shuffled']-res['true']:+10.4f} {res['empty']-res['true']:+10.4f}  {n}",
              flush=True)


if __name__ == "__main__":
    main()

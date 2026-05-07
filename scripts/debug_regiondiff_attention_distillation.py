#!/usr/bin/env python3
"""Debug teacher/student RegionDiff attention-map distillation on one layout batch."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.algorithms.training.flow_matching_trainer import FlowMatchingTrainer
from src.algorithms.training.regiondiff_attention_distillation import (
    RegionDiffAttentionRecorder,
    _infer_square_resolution,
    _match_attention_layers,
    _resize_flat_map,
    compute_region_attention_distillation_loss,
    load_regiondiff_attention_teacher,
)
from src.cli.train import _FLAT_TO_NESTED, build_parser as build_train_parser
from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.fm_config import FMTrainConfig
from src.core.data.datasets import AnnotationLayoutDataset
from src.core.data.layout_batching import collate_layout_batch
from src.core.data.training_data import apply_dataset_subset, resolve_training_data
from src.core.visualization.layout_debug import draw_bbox_overlays, save_image_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save RegionDiff attention KD heatmaps for a teacher/student batch.",
    )
    parser.add_argument("--config", required=True, help="FM RegionDiff config YAML.")
    parser.add_argument("--teacher_checkpoint", default=None)
    parser.add_argument("--student_checkpoint", default=None)
    parser.add_argument("--output_dir", default="artifacts/debug/regiondiff_attention_distillation")
    parser.add_argument("--max_batches", type=int, default=1)
    parser.add_argument("--max_images", type=int, default=2)
    return parser.parse_args()


def _load_fm_config(path: str) -> FMTrainConfig:
    train_parser = build_train_parser()
    train_args = train_parser.parse_args(["--config", path])
    return merge_config_and_cli(
        FMTrainConfig,
        path,
        train_parser,
        train_args,
        flat_to_nested=_FLAT_TO_NESTED,
        cli_argv=["--config", path],
    )


def _pick_latest_unet_checkpoint(run_dir: str) -> str:
    unet_dir = Path(run_dir) / "UNET"
    candidates = []
    for path in unet_dir.glob("unet_fm_epoch_*.pt"):
        if path.name.endswith("_ckpt.pt"):
            continue
        match = re.search(r"_epoch_(\d+)\.pt$", path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        best = unet_dir / "unet_fm_best.pt"
        if best.is_file():
            return str(best)
        raise FileNotFoundError(f"No student checkpoint found under {unet_dir}")
    candidates.sort(key=lambda item: item[0])
    return str(candidates[-1][1])


def _build_one_loader(cfg: FMTrainConfig) -> tuple[DataLoader, Dict[int, str], int]:
    resolved = resolve_training_data(cfg.data)
    if resolved.train_annotations_path is None:
        raise ValueError("Debug attention KD requires a layout dataset with annotations.")
    base_dataset = AnnotationLayoutDataset(
        root_dir=resolved.train_dir,
        annotations_path=resolved.train_annotations_path,
        image_size=cfg.data.image_size,
        normalization_mode=resolved.normalization_mode,
        include_label_names=True,
    )
    dataset = apply_dataset_subset(
        base_dataset,
        cfg.data.max_train_samples,
        cfg.data.subset_strategy,
    )
    return (
        DataLoader(
            dataset,
            batch_size=min(int(cfg.data.batch_size), 4),
            shuffle=False,
            num_workers=0,
            collate_fn=collate_layout_batch,
        ),
        dict(base_dataset.category_id_to_name),
        int(base_dataset.num_categories),
    )


def _heatmap_rgb(flat: torch.Tensor) -> np.ndarray:
    values = flat.detach().float().cpu()
    values = values - values.min()
    values = values / values.max().clamp(min=1e-8)
    arr = values.numpy()
    rgb = np.zeros((*arr.shape, 3), dtype=np.uint8)
    rgb[..., 0] = (arr * 255).astype(np.uint8)
    rgb[..., 1] = (np.sqrt(arr) * 180).astype(np.uint8)
    rgb[..., 2] = ((1.0 - arr) * 80).astype(np.uint8)
    return rgb


def _save_panel(path: Path, teacher_map: torch.Tensor, student_map: torch.Tensor, resolution: int) -> None:
    teacher_img = _heatmap_rgb(teacher_map.view(resolution, resolution))
    student_img = _heatmap_rgb(student_map.view(resolution, resolution))
    diff_img = _heatmap_rgb((teacher_map - student_map).abs().view(resolution, resolution))
    sep = np.full((resolution, max(1, resolution // 32), 3), 255, dtype=np.uint8)
    panel = np.concatenate([teacher_img, sep, student_img, sep, diff_img], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(panel).resize((panel.shape[1] * 4, panel.shape[0] * 4), Image.NEAREST).save(path)


def _selected_category(label: int, selected: list[str], category_id_to_name: Dict[int, str]) -> bool:
    if not selected:
        return True
    label_name = str(category_id_to_name.get(int(label), f"class {label}")).lower().replace("_", " ")
    normalized = {str(item).lower().replace("_", " ") for item in selected}
    return str(int(label)) in normalized or label_name in normalized


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_fm_config(args.config)
    if args.teacher_checkpoint is not None:
        cfg.distillation.teacher_checkpoint = args.teacher_checkpoint
    cfg.distillation.enabled = True
    if not cfg.distillation.teacher_checkpoint:
        raise ValueError("Set --teacher_checkpoint or distillation.teacher_checkpoint in the config.")

    loader, category_id_to_name, num_categories = _build_one_loader(cfg)
    cfg.layout_conditioning.category_id_to_name = category_id_to_name
    cfg.layout_conditioning.num_classes = num_categories

    trainer = FlowMatchingTrainer.from_config(cfg)
    if cfg.model.vae_weights is not None and trainer.vae is not None and os.path.isfile(cfg.model.vae_weights):
        trainer.load_vae_weights(cfg.model.vae_weights, strict=cfg.training.strict_load)
    student_checkpoint = args.student_checkpoint or _pick_latest_unet_checkpoint(cfg.output.model_dir)
    trainer.load_unet_weights(student_checkpoint, strict=cfg.training.strict_load)
    trainer.unet.eval()

    teacher = load_regiondiff_attention_teacher(
        cfg.distillation.teacher_checkpoint,
        device=trainer.device,
    )
    for batch_idx, batch in enumerate(loader):
        if batch_idx >= int(args.max_batches):
            break
        x, cond_kw = trainer._prepare_batch(batch)
        with torch.no_grad():
            x_fm = trainer.encode_fm_input(x)
            state = trainer._sample_flow_matching_state(x_fm, cond_kw)

            student_recorder = RegionDiffAttentionRecorder(
                trainer.unet,
                selected_layers=cfg.distillation.selected_region_layers,
                detach=True,
            )
            with student_recorder:
                trainer._forward_flow_matching_unet(state)

            teacher_recorder = RegionDiffAttentionRecorder(
                teacher.unet,
                selected_layers=cfg.distillation.selected_region_layers,
                detach=True,
            )
            with teacher_recorder:
                teacher.forward_attention(
                    noisy_latents=state["zt"],
                    fm_t=state["t"],
                    cond_kwargs=state["cond_kwargs"],
                    detach_teacher=True,
                )

        kd_loss, diagnostics = compute_region_attention_distillation_loss(
            teacher_attention_maps=teacher_recorder.records,
            student_attention_maps=student_recorder.records,
            boxes_xyxy_norm=state["cond_kwargs"]["boxes_xyxy_norm"],
            labels=state["cond_kwargs"]["labels"],
            object_mask=state["cond_kwargs"]["object_mask"],
            timesteps=state["t"],
            distillation_config=cfg.distillation,
            category_id_to_name=category_id_to_name,
        )
        diagnostics["attention_kd_loss"] = float(kd_loss.detach().cpu().item())
        (output_dir / f"batch_{batch_idx:03d}_diagnostics.json").write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        pairs, _ = _match_attention_layers(
            teacher_recorder.records,
            student_recorder.records,
            selected_layers=cfg.distillation.selected_region_layers,
        )
        selected_categories = list(cfg.distillation.selected_categories or [])
        max_images = min(int(args.max_images), int(state["cond_kwargs"]["labels"].shape[0]))
        for pair_idx, (teacher_key, teacher_record, student_key, student_record) in enumerate(pairs):
            teacher_map = teacher_record.attention
            student_map = student_record.attention
            teacher_resolution = _infer_square_resolution(int(teacher_map.shape[1]))
            student_resolution = _infer_square_resolution(int(student_map.shape[1]))
            if teacher_resolution is None or student_resolution is None:
                continue
            for image_idx in range(max_images):
                for object_idx in range(int(state["cond_kwargs"]["labels"].shape[1])):
                    if not bool(state["cond_kwargs"]["object_mask"][image_idx, object_idx]):
                        continue
                    label = int(state["cond_kwargs"]["labels"][image_idx, object_idx].item())
                    if not _selected_category(label, selected_categories, category_id_to_name):
                        continue
                    if teacher_map.shape[-1] <= object_idx or student_map.shape[-1] <= object_idx:
                        continue
                    teacher_obj = _resize_flat_map(
                        teacher_map[image_idx, :, object_idx],
                        source_resolution=teacher_resolution,
                        target_resolution=student_resolution,
                    ).to(device=student_map.device)
                    student_obj = student_map[image_idx, :, object_idx]
                    label_name = str(category_id_to_name.get(label, f"class_{label}")).replace(" ", "_")
                    out_name = (
                        f"batch{batch_idx:03d}_pair{pair_idx:02d}_img{image_idx:02d}_"
                        f"obj{object_idx:02d}_{label_name}.png"
                    )
                    _save_panel(
                        output_dir / "heatmaps" / out_name,
                        teacher_obj,
                        student_obj,
                        student_resolution,
                    )

        with torch.no_grad():
            decoded = trainer.vae.decode(x_fm) if trainer.vae is not None else x_fm
            display = trainer.from_norm_to_display(decoded).clamp(0.0, 1.0).detach().cpu()
        overlay = draw_bbox_overlays(
            display[:max_images],
            boxes_xyxy=batch["boxes_xyxy"][:max_images],
            labels=batch["labels"][:max_images],
            object_mask=batch["object_mask"][:max_images],
        )
        save_image_batch(display[:max_images], output_dir=str(output_dir / "samples"), prefix=f"batch_{batch_idx:03d}_decoded")
        save_image_batch(overlay, output_dir=str(output_dir / "samples"), prefix=f"batch_{batch_idx:03d}_boxes")


if __name__ == "__main__":
    main()

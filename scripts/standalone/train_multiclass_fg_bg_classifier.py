#!/usr/bin/env python3
"""Train a multiclass FLIR crop classifier with explicit background class."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.algorithms.training.foreground_background_utils import (
    append_jsonl,
    compute_multiclass_metrics,
    load_training_checkpoint,
    save_training_checkpoint,
    select_best_thresholds_per_class,
)
from src.core.configs.config_loader import apply_yaml_defaults
from src.core.data.foreground_background_dataset import (
    MultiClassCropDataset,
    build_balanced_sample_weights,
    collate_foreground_background_batch,
)
from src.core.paths import checkpoints_root
from src.core.training_utils import (
    autocast_context,
    build_grad_scaler,
    build_scheduler,
    grad_norm,
    move_optimizer_state_to_device,
    resolve_precision_settings,
)
from src.models.foreground_background_classifier import MultiClassForegroundBackgroundClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a multiclass FLIR crop classifier")
    parser.add_argument("--config", type=str, default=None, help="YAML preset. CLI flags override preset values.")
    parser.add_argument("--dataset_id", type=str, default="flir_private_proxy_alignment_v18")
    parser.add_argument("--dataset_root", type=str, default="", help="Optional dataset root override for tests or custom exports.")
    parser.add_argument("--input_size", type=int, default=128)
    parser.add_argument("--context_ratio", type=float, default=1.25)
    parser.add_argument("--negative_iou_threshold", type=float, default=0.01)
    parser.add_argument("--negative_max_retries", type=int, default=64)
    parser.add_argument("--context_preview_size", type=int, default=192)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_val_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--mixed_precision", type=str, default="auto", choices=["auto", "bf16", "fp16", "no"])
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="warmup_cosine", choices=["none", "warmup_cosine", "constant_with_warmup"])
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--scalar_every_steps", type=int, default=20)
    parser.add_argument("--image_every", type=int, default=1)
    parser.add_argument("--max_logged_images", type=int, default=8)
    parser.add_argument("--output_dir", type=str, default="")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--resume", type=str, default="")

    preliminary, _ = parser.parse_known_args()
    apply_yaml_defaults(parser, preliminary.config)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_arg)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is not available.")
    return device


def auto_run_name() -> str:
    return datetime.now().strftime("multiclass_fgbg_%Y%m%d_%H%M%S")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    run_name = args.run_name or auto_run_name()
    return checkpoints_root() / "multiclass_foreground_background_filter" / "runs" / run_name


def _display_from_normalized(batch: torch.Tensor) -> torch.Tensor:
    return ((batch.detach().cpu().to(torch.float32) + 1.0) * 0.5).clamp(0.0, 1.0)


def _make_grid(images: Sequence[torch.Tensor], *, nrow: int = 4) -> torch.Tensor:
    if not images:
        return torch.zeros(3, 1, 1, dtype=torch.float32)
    batch = torch.stack([img.to(torch.float32) for img in images], dim=0)
    if batch.shape[1] == 1:
        batch = batch.repeat(1, 3, 1, 1)
    elif batch.shape[1] > 3:
        batch = batch[:, :3]
    batch = batch.clamp(0.0, 1.0)
    nrow = max(1, min(int(nrow), batch.shape[0]))
    ncol = int(math.ceil(batch.shape[0] / nrow))
    _, channels, height, width = batch.shape
    grid = torch.zeros(channels, ncol * height, nrow * width, dtype=torch.float32)
    for idx, image in enumerate(batch):
        row = idx // nrow
        col = idx % nrow
        grid[:, row * height:(row + 1) * height, col * width:(col + 1) * width] = image
    return grid


def _log_image_panels(writer: SummaryWriter, prefix: str, image_panels: Dict[str, torch.Tensor], epoch: int) -> None:
    for tag, image in image_panels.items():
        writer.add_image(f"{prefix}/{tag}", image, global_step=epoch)


def _collect_class_previews(
    display_images: torch.Tensor,
    labels: torch.Tensor,
    metadata: Sequence[Dict[str, Any]],
    *,
    background_class_index: int,
    max_logged_images: int,
) -> Dict[str, List[torch.Tensor]]:
    foreground: List[torch.Tensor] = []
    background: List[torch.Tensor] = []
    for image, label, meta in zip(display_images, labels, metadata):
        if int(label.item()) == int(background_class_index):
            if len(background) < max_logged_images:
                background.append(image.detach().cpu())
            continue
        if len(foreground) < max_logged_images:
            foreground.append(image.detach().cpu())
    return {"foreground": foreground, "background": background}


def evaluate_model(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    dataset: MultiClassCropDataset,
    device: torch.device,
    criterion: nn.Module,
    precision_settings,
    max_logged_images: int,
) -> Dict[str, Any]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    positive_area_ratios: List[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    fg_preview: List[torch.Tensor] = []
    bg_preview: List[torch.Tensor] = []
    wrong_class_examples: List[Tuple[float, torch.Tensor]] = []
    below_threshold_examples: List[Tuple[float, torch.Tensor]] = []
    bg_confusions: List[Tuple[float, torch.Tensor]] = []
    preview_metadata: List[Dict[str, Any]] = []

    with torch.no_grad():
        for batch in dataloader:
            images = batch["pixel_values"].to(device)
            labels = batch["label"].to(device)
            metadata = batch["metadata"]
            with autocast_context(precision_settings):
                logits = model(images)
                loss = criterion(logits, labels)
            batch_size = images.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

            logits_cpu = logits.detach().cpu().to(torch.float32).numpy()
            labels_cpu = labels.detach().cpu().numpy()
            logits_list.append(logits_cpu)
            labels_list.append(labels_cpu)
            positive_area_ratios.append(
                np.asarray(
                    [
                        float(item["source_area_ratio"]) if not bool(item.get("is_background", False)) else float(item["crop_area_ratio"])
                        for item in metadata
                    ],
                    dtype=np.float32,
                )
            )

            display_images = _display_from_normalized(images)
            previews = _collect_class_previews(
                display_images,
                labels.detach().cpu(),
                metadata,
                background_class_index=dataset.background_class_index,
                max_logged_images=max_logged_images,
            )
            if len(fg_preview) < max_logged_images:
                fg_preview.extend(previews["foreground"][: max_logged_images - len(fg_preview)])
            if len(bg_preview) < max_logged_images:
                bg_preview.extend(previews["background"][: max_logged_images - len(bg_preview)])
            if len(preview_metadata) < max_logged_images:
                preview_metadata.extend(metadata[: max_logged_images - len(preview_metadata)])

        logits_all = np.concatenate(logits_list, axis=0) if logits_list else np.zeros((0, dataset.num_classes), dtype=np.float32)
        labels_all = np.concatenate(labels_list, axis=0) if labels_list else np.zeros(0, dtype=np.int64)
        area_ratios_all = np.concatenate(positive_area_ratios, axis=0) if positive_area_ratios else np.zeros(0, dtype=np.float32)
        threshold_payload = select_best_thresholds_per_class(
            logits=logits_all,
            labels=labels_all,
            foreground_class_indices=list(range(dataset.background_class_index)),
        )
        metrics = compute_multiclass_metrics(
            logits=logits_all,
            labels=labels_all,
            background_class_index=dataset.background_class_index,
            positive_area_ratios=area_ratios_all,
        )

        probs = torch.softmax(torch.from_numpy(logits_all), dim=1).numpy() if logits_all.size else np.zeros((0, dataset.num_classes), dtype=np.float32)
        pred = probs.argmax(axis=1) if probs.size else np.zeros(0, dtype=np.int64)
        for idx, (label, prediction) in enumerate(zip(labels_all.tolist(), pred.tolist())):
            if label == dataset.background_class_index:
                if prediction != dataset.background_class_index and len(bg_confusions) < max_logged_images:
                    bg_confusions.append((float(probs[idx, prediction]), display_images[idx % len(display_images)].detach().cpu()))
                continue
            if prediction != label and len(wrong_class_examples) < max_logged_images:
                wrong_class_examples.append((float(probs[idx, prediction]), display_images[idx % len(display_images)].detach().cpu()))
            elif prediction == label:
                class_threshold = float(threshold_payload["thresholds"].get(str(int(label)), 0.5))
                if float(probs[idx, label]) < class_threshold and len(below_threshold_examples) < max_logged_images:
                    below_threshold_examples.append((1.0 - float(probs[idx, label]), display_images[idx % len(display_images)].detach().cpu()))

    return {
        "loss": total_loss / max(1, total_count),
        "metrics": metrics,
        "logits": logits_all,
        "labels": labels_all,
        "thresholds": threshold_payload["thresholds"],
        "threshold_metrics": threshold_payload["metrics_by_class"],
        "image_panels": {
            "foreground_crops_grid": _make_grid(fg_preview, nrow=4),
            "background_crops_grid": _make_grid(bg_preview, nrow=4),
            "hard_wrong_class_examples": _make_grid([img for _, img in wrong_class_examples], nrow=4),
            "hard_below_threshold_examples": _make_grid([img for _, img in below_threshold_examples], nrow=4),
            "hard_background_confusions": _make_grid([img for _, img in bg_confusions], nrow=4),
            "source_context_grid": _make_grid(list(dataset.render_context_previews(preview_metadata)), nrow=4),
        },
    }


def run_train_epoch(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler,
    criterion: nn.Module,
    writer: SummaryWriter,
    epoch: int,
    global_step: int,
    device: torch.device,
    precision_settings,
    scalar_every_steps: int,
    max_logged_images: int,
    background_class_index: int,
) -> Tuple[Dict[str, float], int, Dict[str, torch.Tensor]]:
    model.train()
    running_loss = 0.0
    running_count = 0
    running_grad_norm = 0.0
    fg_preview: List[torch.Tensor] = []
    bg_preview: List[torch.Tensor] = []
    progress = tqdm(dataloader, desc=f"train epoch {epoch}", leave=False)
    for batch in progress:
        images = batch["pixel_values"].to(device)
        labels = batch["label"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(precision_settings):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            step_grad_norm = grad_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            step_grad_norm = grad_norm(model.parameters())
            optimizer.step()
        if scheduler is not None:
            scheduler.step()

        batch_size = images.shape[0]
        running_loss += float(loss.item()) * batch_size
        running_count += batch_size
        running_grad_norm += float(step_grad_norm)

        display_images = _display_from_normalized(images)
        previews = _collect_class_previews(
            display_images,
            labels.detach().cpu(),
            batch["metadata"],
            background_class_index=background_class_index,
            max_logged_images=max_logged_images,
        )
        if len(fg_preview) < max_logged_images:
            fg_preview.extend(previews["foreground"][: max_logged_images - len(fg_preview)])
        if len(bg_preview) < max_logged_images:
            bg_preview.extend(previews["background"][: max_logged_images - len(bg_preview)])

        global_step += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")
        if scalar_every_steps > 0 and global_step % scalar_every_steps == 0:
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
            writer.add_scalar("train/grad_norm", float(step_grad_norm), global_step)

    return (
        {
            "loss": running_loss / max(1, running_count),
            "grad_norm": running_grad_norm / max(1, len(dataloader)),
            "lr": float(optimizer.param_groups[0]["lr"]),
        },
        global_step,
        {
            "foreground_crops_grid": _make_grid(fg_preview, nrow=4),
            "background_crops_grid": _make_grid(bg_preview, nrow=4),
        },
    )


def _save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = resolve_device(args.device)
    precision_settings = resolve_precision_settings(device, args.mixed_precision)
    output_dir = resolve_output_dir(args)
    checkpoints_dir = output_dir / "checkpoints"
    tensorboard_dir = output_dir / "tensorboard"
    metrics_dir = output_dir / "metrics"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = MultiClassCropDataset(
        split="train",
        dataset_id=args.dataset_id,
        dataset_root=args.dataset_root or None,
        input_size=args.input_size,
        context_ratio=args.context_ratio,
        negative_iou_threshold=args.negative_iou_threshold,
        negative_max_retries=args.negative_max_retries,
        seed=args.seed,
        max_samples=args.max_train_samples,
        context_preview_size=args.context_preview_size,
    )
    val_dataset = MultiClassCropDataset(
        split="val",
        dataset_id=args.dataset_id,
        dataset_root=args.dataset_root or None,
        input_size=args.input_size,
        context_ratio=args.context_ratio,
        negative_iou_threshold=args.negative_iou_threshold,
        negative_max_retries=args.negative_max_retries,
        seed=args.seed,
        max_samples=args.max_val_samples,
        context_preview_size=args.context_preview_size,
    )
    test_dataset = MultiClassCropDataset(
        split="test",
        dataset_id=args.dataset_id,
        dataset_root=args.dataset_root or None,
        input_size=args.input_size,
        context_ratio=args.context_ratio,
        negative_iou_threshold=args.negative_iou_threshold,
        negative_max_retries=args.negative_max_retries,
        seed=args.seed,
        max_samples=args.max_test_samples,
        context_preview_size=args.context_preview_size,
    )

    sampler = WeightedRandomSampler(
        weights=build_balanced_sample_weights(train_dataset),
        num_samples=len(train_dataset),
        replacement=True,
    )
    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_foreground_background_batch,
    }
    train_loader = DataLoader(train_dataset, sampler=sampler, shuffle=False, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = MultiClassForegroundBackgroundClassifier(num_classes=train_dataset.num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    total_steps = max(1, len(train_loader) * max(1, int(args.epochs)))
    scheduler = build_scheduler(
        optimizer,
        scheduler_name=args.scheduler,
        total_steps=total_steps,
        warmup_ratio=float(args.warmup_ratio),
        min_lr_ratio=float(args.min_lr_ratio),
    )
    scaler = build_grad_scaler(precision_settings)
    criterion = nn.CrossEntropyLoss()

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    config_payload = dict(vars(args))
    config_payload["resolved_device"] = str(device)
    config_payload["resolved_output_dir"] = str(output_dir.resolve())
    dataset_stats = {"train": train_dataset.stats(), "val": val_dataset.stats(), "test": test_dataset.stats()}
    writer.add_text("run/config", json.dumps(config_payload, indent=2, sort_keys=True), global_step=0)
    writer.add_text("run/dataset_stats", json.dumps(dataset_stats, indent=2, sort_keys=True), global_step=0)

    start_epoch = 1
    global_step = 0
    best_val_macro_f1 = float("-inf")
    best_val_metrics: Dict[str, Any] = {}
    best_test_metrics: Dict[str, Any] = {}
    best_thresholds: Dict[str, float] = {}
    best_threshold_metrics: Dict[str, Dict[str, float]] = {}
    epochs_without_improvement = 0

    if args.resume:
        payload = load_training_checkpoint(
            args.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        move_optimizer_state_to_device(optimizer, device)
        start_epoch = int(payload.get("epoch", 0)) + 1
        global_step = int(payload.get("global_step", 0))
        best_val_macro_f1 = float(payload.get("best_val_metric", best_val_macro_f1))
        best_val_metrics = dict(payload.get("best_val_metrics", {}))
        best_test_metrics = dict(payload.get("best_test_metrics", {}))
        best_thresholds = dict(payload.get("per_class_thresholds", {}))
        best_threshold_metrics = dict(payload.get("per_class_threshold_selection_metrics", {}))

    for epoch in range(start_epoch, int(args.epochs) + 1):
        train_metrics, global_step, train_panels = run_train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            criterion=criterion,
            writer=writer,
            epoch=epoch,
            global_step=global_step,
            device=device,
            precision_settings=precision_settings,
            scalar_every_steps=int(args.scalar_every_steps),
            max_logged_images=int(args.max_logged_images),
            background_class_index=train_dataset.background_class_index,
        )
        val_eval = evaluate_model(
            model=model,
            dataloader=val_loader,
            dataset=val_dataset,
            device=device,
            criterion=criterion,
            precision_settings=precision_settings,
            max_logged_images=int(args.max_logged_images),
        )
        test_eval = evaluate_model(
            model=model,
            dataloader=test_loader,
            dataset=test_dataset,
            device=device,
            criterion=criterion,
            precision_settings=precision_settings,
            max_logged_images=int(args.max_logged_images),
        )

        writer.add_scalar("val/loss", float(val_eval["loss"]), epoch)
        writer.add_scalar("val/accuracy", float(val_eval["metrics"]["accuracy"]), epoch)
        writer.add_scalar("val/macro_f1", float(val_eval["metrics"]["macro_f1"]), epoch)
        writer.add_scalar("val/macro_recall", float(val_eval["metrics"]["macro_recall"]), epoch)
        writer.add_scalar("val/foreground_exact_match_rate", float(val_eval["metrics"]["foreground_exact_match_rate"]), epoch)
        writer.add_scalar("val/background_recall", float(val_eval["metrics"]["background_recall"]), epoch)
        writer.add_scalar("val/foreground_recall_tiny", float(val_eval["metrics"].get("foreground_recall_tiny", 0.0)), epoch)
        writer.add_scalar("val/foreground_recall_small", float(val_eval["metrics"].get("foreground_recall_small", 0.0)), epoch)
        writer.add_scalar("val/foreground_recall_medium_large", float(val_eval["metrics"].get("foreground_recall_medium_large", 0.0)), epoch)
        writer.add_scalar("test/accuracy", float(test_eval["metrics"]["accuracy"]), epoch)
        writer.add_scalar("test/macro_f1", float(test_eval["metrics"]["macro_f1"]), epoch)
        writer.add_scalar("test/macro_recall", float(test_eval["metrics"]["macro_recall"]), epoch)
        writer.add_scalar("test/foreground_exact_match_rate", float(test_eval["metrics"]["foreground_exact_match_rate"]), epoch)
        writer.add_scalar("test/background_recall", float(test_eval["metrics"]["background_recall"]), epoch)

        for class_idx, threshold in sorted(val_eval["thresholds"].items(), key=lambda item: int(item[0])):
            writer.add_scalar(f"val_thresholds/class_{class_idx}", float(threshold), epoch)

        probs = torch.softmax(torch.from_numpy(val_eval["logits"]), dim=1).numpy() if val_eval["logits"].size else np.zeros((0, train_dataset.num_classes), dtype=np.float32)
        if probs.size:
            writer.add_histogram("val/max_probability", probs.max(axis=1), epoch)
            target_probs = probs[np.arange(probs.shape[0]), val_eval["labels"]]
            writer.add_histogram("val/target_class_probability", target_probs, epoch)

        if int(args.image_every) > 0 and epoch % int(args.image_every) == 0:
            _log_image_panels(writer, "train", train_panels, epoch)
            _log_image_panels(writer, "val", val_eval["image_panels"], epoch)

        epoch_row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": {"loss": float(val_eval["loss"]), **{k: float(v) for k, v in val_eval["metrics"].items()}},
            "test": {"loss": float(test_eval["loss"]), **{k: float(v) for k, v in test_eval["metrics"].items()}},
            "per_class_thresholds": {str(k): float(v) for k, v in val_eval["thresholds"].items()},
        }
        append_jsonl(metrics_dir / "per_epoch.jsonl", [epoch_row])

        improved = float(val_eval["metrics"]["macro_f1"]) > (best_val_macro_f1 + float(args.min_delta))
        if improved:
            best_val_macro_f1 = float(val_eval["metrics"]["macro_f1"])
            best_val_metrics = {"loss": float(val_eval["loss"]), **{k: float(v) for k, v in val_eval["metrics"].items()}}
            best_test_metrics = {"loss": float(test_eval["loss"]), **{k: float(v) for k, v in test_eval["metrics"].items()}}
            best_thresholds = {str(k): float(v) for k, v in val_eval["thresholds"].items()}
            best_threshold_metrics = {
                str(k): {mk: float(mv) for mk, mv in metrics.items()}
                for k, metrics in val_eval["threshold_metrics"].items()
            }
            save_training_checkpoint(
                checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config=config_payload,
                best_val_metric=best_val_macro_f1,
                best_threshold=0.0,
                best_val_metrics=best_val_metrics,
                best_test_metrics=best_test_metrics,
                extra_payload={
                    "classifier_mode": "multiclass",
                    "background_class_index": int(train_dataset.background_class_index),
                    "num_foreground_classes": int(train_dataset.background_class_index),
                    "category_id_to_name": {str(k): str(v) for k, v in train_dataset.category_id_to_name.items()},
                    "model_index_to_category_id": {str(k): int(v) for k, v in train_dataset.model_index_to_category_id.items()},
                    "category_id_to_model_index": {str(k): int(v) for k, v in train_dataset.category_id_to_model_index.items()},
                    "per_class_thresholds": best_thresholds,
                    "per_class_threshold_selection_metrics": best_threshold_metrics,
                },
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        save_training_checkpoint(
            checkpoints_dir / "latest.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            global_step=global_step,
            config=config_payload,
            best_val_metric=best_val_macro_f1,
            best_threshold=0.0,
            best_val_metrics=best_val_metrics,
            best_test_metrics=best_test_metrics,
            extra_payload={
                "classifier_mode": "multiclass",
                "background_class_index": int(train_dataset.background_class_index),
                "num_foreground_classes": int(train_dataset.background_class_index),
                "category_id_to_name": {str(k): str(v) for k, v in train_dataset.category_id_to_name.items()},
                "model_index_to_category_id": {str(k): int(v) for k, v in train_dataset.model_index_to_category_id.items()},
                "category_id_to_model_index": {str(k): int(v) for k, v in train_dataset.category_id_to_model_index.items()},
                "per_class_thresholds": best_thresholds,
                "per_class_threshold_selection_metrics": best_threshold_metrics,
            },
        )
        if epochs_without_improvement >= int(args.patience):
            break

    best_checkpoint_path = checkpoints_dir / "best.pt"
    if best_checkpoint_path.is_file():
        payload = load_training_checkpoint(best_checkpoint_path, model=model, map_location=device)
        best_val_macro_f1 = float(payload.get("best_val_metric", best_val_macro_f1))
        best_thresholds = dict(payload.get("per_class_thresholds", best_thresholds))
        best_threshold_metrics = dict(payload.get("per_class_threshold_selection_metrics", best_threshold_metrics))

    final_val = evaluate_model(
        model=model,
        dataloader=val_loader,
        dataset=val_dataset,
        device=device,
        criterion=criterion,
        precision_settings=precision_settings,
        max_logged_images=int(args.max_logged_images),
    )
    final_test = evaluate_model(
        model=model,
        dataloader=test_loader,
        dataset=test_dataset,
        device=device,
        criterion=criterion,
        precision_settings=precision_settings,
        max_logged_images=int(args.max_logged_images),
    )

    summary = {
        "classifier_mode": "multiclass",
        "config": config_payload,
        "output_dir": str(output_dir.resolve()),
        "best_checkpoint_path": str(best_checkpoint_path.resolve()),
        "latest_checkpoint_path": str((checkpoints_dir / "latest.pt").resolve()),
        "normalization_mode": train_dataset.normalization_mode,
        "input_size": int(args.input_size),
        "context_ratio": float(args.context_ratio),
        "negative_iou_threshold": float(args.negative_iou_threshold),
        "negative_max_retries": int(args.negative_max_retries),
        "best_val_macro_f1": float(best_val_macro_f1),
        "background_class_index": int(train_dataset.background_class_index),
        "num_foreground_classes": int(train_dataset.background_class_index),
        "category_id_to_name": {str(k): str(v) for k, v in train_dataset.category_id_to_name.items()},
        "model_index_to_category_id": {str(k): int(v) for k, v in train_dataset.model_index_to_category_id.items()},
        "category_id_to_model_index": {str(k): int(v) for k, v in train_dataset.category_id_to_model_index.items()},
        "per_class_thresholds": best_thresholds,
        "per_class_threshold_selection_metrics": best_threshold_metrics,
        "dataset_stats": dataset_stats,
        "best_val_metrics": {"loss": float(final_val["loss"]), **{k: float(v) for k, v in final_val["metrics"].items()}},
        "best_test_metrics": {"loss": float(final_test["loss"]), **{k: float(v) for k, v in final_test["metrics"].items()}},
    }
    _save_json(metrics_dir / "summary.json", summary)
    writer.flush()
    writer.close()
    print(f"Saved run to: {output_dir}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"TensorBoard: tensorboard --logdir {tensorboard_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Train a binary FLIR foreground/background crop classifier."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.algorithms.training.foreground_background_utils import (
    append_jsonl,
    compute_binary_metrics,
    load_training_checkpoint,
    save_training_checkpoint,
    select_best_threshold,
)
from src.core.configs.config_loader import apply_yaml_defaults
from src.core.data.foreground_background_dataset import (
    ForegroundBackgroundCropDataset,
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
from src.models.foreground_background_classifier import ForegroundBackgroundClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a FLIR foreground/background crop classifier")
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
    return datetime.now().strftime("fgbg_%Y%m%d_%H%M%S")


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    run_name = args.run_name or auto_run_name()
    return checkpoints_root() / "foreground_background_filter" / "runs" / run_name


def _display_from_normalized(batch: torch.Tensor) -> torch.Tensor:
    return ((batch.detach().cpu().to(torch.float32) + 1.0) * 0.5).clamp(0.0, 1.0)


def _make_grid(images: Sequence[torch.Tensor], *, nrow: int = 4) -> torch.Tensor:
    if not images:
        return torch.zeros(3, 1, 1, dtype=torch.float32)
    batch = torch.stack([img.to(torch.float32) for img in images], dim=0)
    if batch.ndim != 4:
        raise ValueError(f"Expected image batch with shape (B, C, H, W), got {tuple(batch.shape)}")
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


def _update_hard_examples(
    store: List[Tuple[float, torch.Tensor]],
    *,
    scores: np.ndarray,
    display_images: torch.Tensor,
    limit: int,
) -> None:
    for score, image in zip(scores.tolist(), display_images):
        store.append((float(score), image.detach().cpu()))
    store.sort(key=lambda item: item[0], reverse=True)
    del store[limit:]


def _extract_label_preview_images(
    display_images: torch.Tensor,
    labels: torch.Tensor,
    *,
    target_label: int,
    limit: int,
) -> List[torch.Tensor]:
    selected: List[torch.Tensor] = []
    for image, label in zip(display_images, labels):
        if int(round(float(label.item()))) != int(target_label):
            continue
        selected.append(image.detach().cpu())
        if len(selected) >= limit:
            break
    return selected


def _log_image_panels(
    writer: SummaryWriter,
    prefix: str,
    image_panels: Dict[str, torch.Tensor],
    epoch: int,
) -> None:
    for tag, image in image_panels.items():
        writer.add_image(f"{prefix}/{tag}", image, global_step=epoch)


def evaluate_model(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    dataset: ForegroundBackgroundCropDataset,
    device: torch.device,
    criterion: nn.Module,
    precision_settings,
    threshold: Optional[float],
    max_logged_images: int,
) -> Dict[str, Any]:
    model.eval()
    logits_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    positive_area_ratios: List[np.ndarray] = []
    pos_logits: List[np.ndarray] = []
    neg_logits: List[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    val_pos_preview: List[torch.Tensor] = []
    val_neg_preview: List[torch.Tensor] = []
    preview_metadata: List[Dict[str, Any]] = []
    hard_false_positives: List[Tuple[float, torch.Tensor]] = []
    hard_false_negatives: List[Tuple[float, torch.Tensor]] = []

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
            labels_cpu = labels.detach().cpu().to(torch.float32).numpy()
            logits_list.append(logits_cpu)
            labels_list.append(labels_cpu)
            positive_area_ratios.append(
                np.asarray(
                    [float(item["source_area_ratio"]) for item in metadata],
                    dtype=np.float32,
                )
            )

            pos_mask = labels_cpu >= 0.5
            neg_mask = ~pos_mask
            if pos_mask.any():
                pos_logits.append(logits_cpu[pos_mask])
            if neg_mask.any():
                neg_logits.append(logits_cpu[neg_mask])

            display_images = _display_from_normalized(images)
            if len(val_pos_preview) < max_logged_images:
                val_pos_preview.extend(
                    _extract_label_preview_images(
                        display_images,
                        labels.detach().cpu(),
                        target_label=1,
                        limit=max_logged_images - len(val_pos_preview),
                    )
                )
            if len(val_neg_preview) < max_logged_images:
                val_neg_preview.extend(
                    _extract_label_preview_images(
                        display_images,
                        labels.detach().cpu(),
                        target_label=0,
                        limit=max_logged_images - len(val_neg_preview),
                    )
                )
            if len(preview_metadata) < max_logged_images:
                remaining = max_logged_images - len(preview_metadata)
                preview_metadata.extend(metadata[:remaining])

            probs = 1.0 / (1.0 + np.exp(-logits_cpu))
            if threshold is not None:
                pred_pos = probs >= float(threshold)
                false_positive_scores = probs[(labels_cpu < 0.5) & pred_pos]
                false_positive_images = display_images[(labels.detach().cpu() < 0.5) & torch.from_numpy(pred_pos)]
                false_negative_scores = (1.0 - probs[(labels_cpu >= 0.5) & (~pred_pos)])
                false_negative_images = display_images[(labels.detach().cpu() >= 0.5) & (~torch.from_numpy(pred_pos))]
                _update_hard_examples(
                    hard_false_positives,
                    scores=np.asarray(false_positive_scores, dtype=np.float32),
                    display_images=false_positive_images,
                    limit=max_logged_images,
                )
                _update_hard_examples(
                    hard_false_negatives,
                    scores=np.asarray(false_negative_scores, dtype=np.float32),
                    display_images=false_negative_images,
                    limit=max_logged_images,
                )

    logits_all = np.concatenate(logits_list, axis=0) if logits_list else np.zeros(0, dtype=np.float32)
    labels_all = np.concatenate(labels_list, axis=0) if labels_list else np.zeros(0, dtype=np.float32)
    area_ratios_all = (
        np.concatenate(positive_area_ratios, axis=0)
        if positive_area_ratios
        else np.zeros(0, dtype=np.float32)
    )

    if threshold is None:
        threshold_payload = select_best_threshold(
            logits=logits_all,
            labels=labels_all,
            positive_area_ratios=area_ratios_all,
        )
        threshold = float(threshold_payload["threshold"])
        metrics = dict(threshold_payload["metrics"])
    else:
        metrics = compute_binary_metrics(
            logits=logits_all,
            labels=labels_all,
            threshold=float(threshold),
            positive_area_ratios=area_ratios_all,
        )

    image_panels = {
        "pos_crops_grid": _make_grid(val_pos_preview, nrow=4),
        "neg_crops_grid": _make_grid(val_neg_preview, nrow=4),
        "hard_false_positives": _make_grid([img for _, img in hard_false_positives], nrow=4),
        "hard_false_negatives": _make_grid([img for _, img in hard_false_negatives], nrow=4),
        "source_context_grid": _make_grid(
            [img for img in dataset.render_context_previews(preview_metadata)],
            nrow=4,
        ),
    }

    return {
        "loss": total_loss / max(1, total_count),
        "threshold": float(threshold),
        "metrics": metrics,
        "logits": logits_all,
        "labels": labels_all,
        "pos_logits": np.concatenate(pos_logits, axis=0) if pos_logits else np.zeros(0, dtype=np.float32),
        "neg_logits": np.concatenate(neg_logits, axis=0) if neg_logits else np.zeros(0, dtype=np.float32),
        "image_panels": image_panels,
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
) -> Tuple[Dict[str, float], int, Dict[str, torch.Tensor]]:
    model.train()
    running_loss = 0.0
    running_count = 0
    running_grad_norm = 0.0
    preview_pos: List[torch.Tensor] = []
    preview_neg: List[torch.Tensor] = []

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
        if len(preview_pos) < max_logged_images:
            preview_pos.extend(
                _extract_label_preview_images(
                    display_images,
                    labels.detach().cpu(),
                    target_label=1,
                    limit=max_logged_images - len(preview_pos),
                )
            )
        if len(preview_neg) < max_logged_images:
            preview_neg.extend(
                _extract_label_preview_images(
                    display_images,
                    labels.detach().cpu(),
                    target_label=0,
                    limit=max_logged_images - len(preview_neg),
                )
            )

        global_step += 1
        progress.set_postfix(loss=f"{loss.item():.4f}")

        if scalar_every_steps > 0 and global_step % scalar_every_steps == 0:
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/lr", float(optimizer.param_groups[0]["lr"]), global_step)
            writer.add_scalar("train/grad_norm", float(step_grad_norm), global_step)

    epoch_metrics = {
        "loss": running_loss / max(1, running_count),
        "grad_norm": running_grad_norm / max(1, len(dataloader)),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }
    image_panels = {
        "pos_crops_grid": _make_grid(preview_pos, nrow=4),
        "neg_crops_grid": _make_grid(preview_neg, nrow=4),
    }
    return epoch_metrics, global_step, image_panels


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

    train_dataset = ForegroundBackgroundCropDataset(
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
    val_dataset = ForegroundBackgroundCropDataset(
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
    test_dataset = ForegroundBackgroundCropDataset(
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

    loader_kwargs = {
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_foreground_background_batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_kwargs)

    model = ForegroundBackgroundClassifier().to(device)
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
    criterion = nn.BCEWithLogitsLoss()

    writer = SummaryWriter(log_dir=str(tensorboard_dir))
    config_payload = dict(vars(args))
    config_payload["resolved_device"] = str(device)
    config_payload["resolved_output_dir"] = str(output_dir.resolve())
    writer.add_text("run/config", json.dumps(config_payload, indent=2, sort_keys=True), global_step=0)
    dataset_stats = {
        "train": train_dataset.stats(),
        "val": val_dataset.stats(),
        "test": test_dataset.stats(),
    }
    writer.add_text("run/dataset_stats", json.dumps(dataset_stats, indent=2, sort_keys=True), global_step=0)

    start_epoch = 1
    global_step = 0
    best_val_f1 = float("-inf")
    best_threshold = 0.5
    best_val_metrics: Dict[str, Any] = {}
    best_test_metrics: Dict[str, Any] = {}
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
        best_val_f1 = float(payload.get("best_val_metric", best_val_f1))
        best_threshold = float(payload.get("best_threshold", best_threshold))
        best_val_metrics = dict(payload.get("best_val_metrics", {}))
        best_test_metrics = dict(payload.get("best_test_metrics", {}))

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
        )

        val_eval = evaluate_model(
            model=model,
            dataloader=val_loader,
            dataset=val_dataset,
            device=device,
            criterion=criterion,
            precision_settings=precision_settings,
            threshold=None,
            max_logged_images=int(args.max_logged_images),
        )
        test_eval = evaluate_model(
            model=model,
            dataloader=test_loader,
            dataset=test_dataset,
            device=device,
            criterion=criterion,
            precision_settings=precision_settings,
            threshold=val_eval["threshold"],
            max_logged_images=int(args.max_logged_images),
        )

        writer.add_scalar("val/loss", float(val_eval["loss"]), epoch)
        writer.add_scalar("val/accuracy", float(val_eval["metrics"]["accuracy"]), epoch)
        writer.add_scalar("val/balanced_accuracy", float(val_eval["metrics"]["balanced_accuracy"]), epoch)
        writer.add_scalar("val/precision", float(val_eval["metrics"]["precision"]), epoch)
        writer.add_scalar("val/recall", float(val_eval["metrics"]["recall"]), epoch)
        writer.add_scalar("val/f1", float(val_eval["metrics"]["f1"]), epoch)
        writer.add_scalar("val/threshold", float(val_eval["threshold"]), epoch)
        writer.add_scalar("val/recall_tiny", float(val_eval["metrics"].get("recall_tiny", 0.0)), epoch)
        writer.add_scalar("val/recall_small", float(val_eval["metrics"].get("recall_small", 0.0)), epoch)
        writer.add_scalar("val/recall_medium_large", float(val_eval["metrics"].get("recall_medium_large", 0.0)), epoch)

        writer.add_scalar("test/accuracy", float(test_eval["metrics"]["accuracy"]), epoch)
        writer.add_scalar("test/balanced_accuracy", float(test_eval["metrics"]["balanced_accuracy"]), epoch)
        writer.add_scalar("test/precision", float(test_eval["metrics"]["precision"]), epoch)
        writer.add_scalar("test/recall", float(test_eval["metrics"]["recall"]), epoch)
        writer.add_scalar("test/f1", float(test_eval["metrics"]["f1"]), epoch)
        writer.add_scalar("test/recall_tiny", float(test_eval["metrics"].get("recall_tiny", 0.0)), epoch)
        writer.add_scalar("test/recall_small", float(test_eval["metrics"].get("recall_small", 0.0)), epoch)
        writer.add_scalar("test/recall_medium_large", float(test_eval["metrics"].get("recall_medium_large", 0.0)), epoch)

        if val_eval["pos_logits"].size > 0:
            writer.add_histogram("val/logits_pos", val_eval["pos_logits"], epoch)
            writer.add_histogram("val/probs_pos", 1.0 / (1.0 + np.exp(-val_eval["pos_logits"])), epoch)
        if val_eval["neg_logits"].size > 0:
            writer.add_histogram("val/logits_neg", val_eval["neg_logits"], epoch)
            writer.add_histogram("val/probs_neg", 1.0 / (1.0 + np.exp(-val_eval["neg_logits"])), epoch)

        if int(args.image_every) > 0 and epoch % int(args.image_every) == 0:
            val_log_eval = evaluate_model(
                model=model,
                dataloader=val_loader,
                dataset=val_dataset,
                device=device,
                criterion=criterion,
                precision_settings=precision_settings,
                threshold=val_eval["threshold"],
                max_logged_images=int(args.max_logged_images),
            )
            _log_image_panels(writer, "train", train_panels, epoch)
            _log_image_panels(writer, "val", val_log_eval["image_panels"], epoch)

        epoch_row = {
            "epoch": epoch,
            "global_step": global_step,
            "train": train_metrics,
            "val": {
                "loss": float(val_eval["loss"]),
                **{k: float(v) for k, v in val_eval["metrics"].items() if isinstance(v, (int, float))},
            },
            "test": {
                "loss": float(test_eval["loss"]),
                **{k: float(v) for k, v in test_eval["metrics"].items() if isinstance(v, (int, float))},
            },
        }
        append_jsonl(metrics_dir / "per_epoch.jsonl", [epoch_row])

        improved = float(val_eval["metrics"]["f1"]) > (best_val_f1 + float(args.min_delta))
        if improved:
            best_val_f1 = float(val_eval["metrics"]["f1"])
            best_threshold = float(val_eval["threshold"])
            best_val_metrics = {
                "loss": float(val_eval["loss"]),
                **{k: float(v) for k, v in val_eval["metrics"].items() if isinstance(v, (int, float))},
            }
            best_test_metrics = {
                "loss": float(test_eval["loss"]),
                **{k: float(v) for k, v in test_eval["metrics"].items() if isinstance(v, (int, float))},
            }
            save_training_checkpoint(
                checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                global_step=global_step,
                config=config_payload,
                best_val_metric=best_val_f1,
                best_threshold=best_threshold,
                best_val_metrics=best_val_metrics,
                best_test_metrics=best_test_metrics,
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
            best_val_metric=best_val_f1,
            best_threshold=best_threshold,
            best_val_metrics=best_val_metrics,
            best_test_metrics=best_test_metrics,
        )

        if epochs_without_improvement >= int(args.patience):
            break

    best_checkpoint_path = checkpoints_dir / "best.pt"
    if best_checkpoint_path.is_file():
        payload = load_training_checkpoint(best_checkpoint_path, model=model, map_location=device)
        best_threshold = float(payload.get("best_threshold", best_threshold))
        best_val_f1 = float(payload.get("best_val_metric", best_val_f1))

    final_val = evaluate_model(
        model=model,
        dataloader=val_loader,
        dataset=val_dataset,
        device=device,
        criterion=criterion,
        precision_settings=precision_settings,
        threshold=best_threshold,
        max_logged_images=int(args.max_logged_images),
    )
    final_test = evaluate_model(
        model=model,
        dataloader=test_loader,
        dataset=test_dataset,
        device=device,
        criterion=criterion,
        precision_settings=precision_settings,
        threshold=best_threshold,
        max_logged_images=int(args.max_logged_images),
    )

    summary = {
        "config": config_payload,
        "output_dir": str(output_dir.resolve()),
        "best_checkpoint_path": str(best_checkpoint_path.resolve()),
        "latest_checkpoint_path": str((checkpoints_dir / "latest.pt").resolve()),
        "normalization_mode": train_dataset.normalization_mode,
        "input_size": int(args.input_size),
        "context_ratio": float(args.context_ratio),
        "negative_iou_threshold": float(args.negative_iou_threshold),
        "negative_max_retries": int(args.negative_max_retries),
        "chosen_threshold": float(best_threshold),
        "best_val_f1": float(best_val_f1),
        "dataset_stats": dataset_stats,
        "best_val_metrics": {
            "loss": float(final_val["loss"]),
            **{k: float(v) for k, v in final_val["metrics"].items() if isinstance(v, (int, float))},
        },
        "best_test_metrics": {
            "loss": float(final_test["loss"]),
            **{k: float(v) for k, v in final_test["metrics"].items() if isinstance(v, (int, float))},
        },
    }
    _save_json(metrics_dir / "summary.json", summary)
    writer.flush()
    writer.close()

    print(f"Saved run to: {output_dir}")
    print(f"Best checkpoint: {best_checkpoint_path}")
    print(f"TensorBoard: tensorboard --logdir {tensorboard_dir}")


if __name__ == "__main__":
    main()

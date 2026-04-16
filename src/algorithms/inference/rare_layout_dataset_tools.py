"""Helpers for rare-layout generation and foreground/background audits."""

from __future__ import annotations

import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm.auto import tqdm

from src.algorithms.inference.flow_matching_sampler import get_unet_sample_shape
from src.algorithms.inference.layout_flow_matching_sampler import LayoutFlowMatchingSampler
from src.core.data.annotations import build_category_id_to_name, index_annotations, load_coco_annotations
from src.core.data.dataset_targets import resolve_dataset_target
from src.core.data.datasets import AnnotationLayoutDataset
from src.core.data.layout_batching import collate_layout_batch
from src.core.normalization import norm_to_display as from_norm_to_display
from src.core.paths import checkpoints_root, generated_root
from src.models.fm_unet import load_unet_config
from src.models.foreground_background_classifier import (
    ForegroundBackgroundClassifier,
    MultiClassForegroundBackgroundClassifier,
)
from src.models.stay_layout_conditioned_unet import build_stay_layout_conditioned_unet
from src.models.vae import build_vae_from_config, freeze_vae, load_vae_config, load_vae_weights


@dataclass(frozen=True)
class LayoutRarityRecord:
    dataset_idx: int
    sample: Dict[str, Any]
    combos: List[str]
    combo_pairs: List[Tuple[str, str]]
    layout_rarity: float
    n_objects: int
    file_name: str

    def to_summary_dict(self) -> Dict[str, Any]:
        return {
            "dataset_idx": int(self.dataset_idx),
            "combos": list(self.combos),
            "combo_pairs": [{"category": category, "size_bin": size_bin} for category, size_bin in self.combo_pairs],
            "layout_rarity": float(self.layout_rarity),
            "n_objects": int(self.n_objects),
            "file_name": str(self.file_name),
            "image_id": self.sample.get("image_id"),
        }


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_json(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def auto_generation_run_name() -> str:
    return datetime.now().strftime("rare_layout_%Y%m%d_%H%M%S")


def auto_filter_run_name(base_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"filter_{base_name}_{timestamp}"


def resolve_generation_output_dir(output_dir: str | Path | None, run_name: str | None) -> Path:
    if output_dir:
        return Path(output_dir)
    return generated_root() / (run_name or auto_generation_run_name())


def maybe_latest_checkpoint(weights_dir: Path, prefix: str) -> Path | None:
    numbered: List[Tuple[int, Path]] = []
    for candidate in weights_dir.glob(f"{prefix}*.pt"):
        suffix = candidate.stem.replace(prefix, "", 1).strip("_")
        try:
            numbered.append((int(suffix), candidate))
        except ValueError:
            continue
    if not numbered:
        return None
    numbered.sort(key=lambda item: item[0])
    return numbered[-1][1]


def load_sampler_from_pipeline(
    pipeline_dir: str | Path,
    preset_path: str | Path,
    checkpoint_name: str,
    device: str | torch.device,
) -> tuple[dict, dict, torch.nn.Module, torch.nn.Module, LayoutFlowMatchingSampler]:
    pipeline_dir = Path(pipeline_dir)
    preset = load_yaml(preset_path)
    layout_meta = load_json(pipeline_dir / "layout_conditioning.json")
    unet_cfg = load_unet_config(str(pipeline_dir / "UNET" / "config.json"))
    vae_cfg = load_vae_config(str(pipeline_dir / "VAE" / "config.json"))

    category_id_to_name = {
        int(key): value for key, value in layout_meta.get("category_id_to_name", {}).items()
    }

    unet = build_stay_layout_conditioned_unet(
        unet_cfg,
        image_in_channels=int(layout_meta["image_in_channels"]),
        num_classes=int(layout_meta["num_classes"]),
        class_embed_dim=int(layout_meta["class_embed_dim"]),
        bbox_embed_dim=int(layout_meta["bbox_embed_dim"]),
        object_embed_dim=int(layout_meta["object_embed_dim"]),
        use_style_latent=bool(layout_meta["use_style_latent"]),
        style_latent_dim=int(layout_meta["style_latent_dim"]),
        style_seed=int(layout_meta["style_seed"]),
        mask_resolution=int(layout_meta["mask_resolution"]),
        mask_hidden_channels=int(layout_meta["mask_hidden_channels"]),
        mask_threshold=float(layout_meta["mask_threshold"]),
        edge_dilation=int(layout_meta["edge_dilation"]),
        injection_mode=str(layout_meta["injection_mode"]),
        use_masked_context=bool(layout_meta["use_masked_context"]),
        mask_overlap_loss_weight=float(layout_meta["mask_overlap_loss_weight"]),
        mask_sharpness_loss_weight=float(layout_meta["mask_sharpness_loss_weight"]),
        mask_activation_loss_weight=float(layout_meta["mask_activation_loss_weight"]),
        category_id_to_name=category_id_to_name,
        device=device,
    )

    unet_path = pipeline_dir / "UNET" / checkpoint_name
    if not unet_path.is_file():
        latest = maybe_latest_checkpoint(pipeline_dir / "UNET", "unet_fm_epoch")
        if latest is None:
            raise FileNotFoundError(
                f"Could not find checkpoint {checkpoint_name!r} or numbered UNET checkpoints in {pipeline_dir / 'UNET'}"
            )
        unet_path = latest

    unet_state = torch.load(unet_path, map_location=device)
    if isinstance(unet_state, dict) and "unet_state" in unet_state:
        unet_state = unet_state["unet_state"]
    unet.load_state_dict(unet_state, strict=True)
    unet.eval()

    vae = build_vae_from_config(vae_cfg, device=device)
    vae_weights_value = preset["model"].get("vae_weights")
    if vae_weights_value is not None:
        load_vae_weights(vae, str(Path(vae_weights_value)), map_location=device)
    freeze_vae(vae)

    sampler = LayoutFlowMatchingSampler.from_stable(
        unet,
        vae,
        device=device,
        t_scale=float(preset["training"]["t_scale"]),
        train_target=str(preset["training"]["train_target"]),
        from_norm_to_display=from_norm_to_display,
    )
    return preset, layout_meta, unet, vae, sampler


def build_layout_dataset(
    preset: Dict[str, Any],
    *,
    split: str,
    dataset_root: str | Path | None = None,
    dataset_id: Optional[str] = None,
) -> AnnotationLayoutDataset:
    active_dataset_id = str(dataset_id or preset["data"]["dataset_id"])
    target = resolve_dataset_target(active_dataset_id)
    root_dir = Path(dataset_root) if dataset_root is not None else target.split_dir(split)
    annotations_path = root_dir / "annotations.json"
    return AnnotationLayoutDataset(
        root_dir=str(root_dir),
        annotations_path=str(annotations_path),
        image_size=int(preset["data"]["image_size"]),
        normalization_mode=target.normalization_mode,
        include_label_names=True,
    )


def get_original_image_size(dataset: AnnotationLayoutDataset, sample: Dict[str, Any]) -> tuple[float, float]:
    image_meta = dataset.images_by_id.get(sample["image_id"], {})
    image_w = float(image_meta.get("width", sample["pixel_values"].shape[-1]))
    image_h = float(image_meta.get("height", sample["pixel_values"].shape[-2]))
    return max(1.0, image_w), max(1.0, image_h)


def compute_size_bin_thresholds(dataset: AnnotationLayoutDataset, *, show_progress: bool = False) -> np.ndarray:
    area_ratios: List[float] = []
    iterator = range(len(dataset))
    if show_progress:
        iterator = tqdm(iterator, desc="Scanning box sizes")
    for sample_idx in iterator:
        sample = dataset[sample_idx]
        image_w, image_h = get_original_image_size(dataset, sample)
        image_area = max(1.0, image_w * image_h)
        for box in sample["boxes_xyxy_original"].numpy().tolist():
            width = max(0.0, float(box[2] - box[0]))
            height = max(0.0, float(box[3] - box[1]))
            area_ratios.append((width * height) / image_area)
    if not area_ratios:
        raise RuntimeError("No boxes found in the dataset split, so size bins cannot be computed.")
    thresholds = np.quantile(np.asarray(area_ratios, dtype=np.float32), [1.0 / 3.0, 2.0 / 3.0])
    return thresholds.astype(np.float32)


def size_bin_name(area_ratio: float, thresholds: np.ndarray) -> str:
    if float(area_ratio) <= float(thresholds[0]):
        return "small"
    if float(area_ratio) <= float(thresholds[1]):
        return "med"
    return "big"


def sample_instance_combo_pairs(
    dataset: AnnotationLayoutDataset,
    sample: Dict[str, Any],
    thresholds: np.ndarray,
    category_id_to_name: Dict[int, str],
) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    image_w, image_h = get_original_image_size(dataset, sample)
    image_area = max(1.0, image_w * image_h)
    boxes = sample["boxes_xyxy_original"].numpy().tolist()
    labels = sample["labels"].numpy().tolist()
    for label, box in zip(labels, boxes):
        width = max(0.0, float(box[2] - box[0]))
        height = max(0.0, float(box[3] - box[1]))
        area_ratio = (width * height) / image_area
        category_name = category_id_to_name.get(int(label), str(int(label)))
        pairs.append((category_name, size_bin_name(area_ratio, thresholds)))
    return pairs


def combo_key(category_name: str, size_bin: str) -> str:
    return f"{category_name} | {size_bin}"


def resolve_generation_plot_keys(counter: Counter, top_k: int | None = None) -> List[str]:
    keys = list(counter.keys())
    if not keys:
        return []
    keys.sort(key=lambda key: (-counter[key], key))
    if top_k is not None:
        keys = keys[:top_k]
    return keys


def build_layout_rarity_records(
    dataset: AnnotationLayoutDataset,
    *,
    thresholds: np.ndarray,
    aggregation: str = "sum",
    show_progress: bool = False,
) -> tuple[List[LayoutRarityRecord], Counter]:
    category_id_to_name = dataset.category_id_to_name
    instance_counter: Counter = Counter()
    per_sample_pairs: List[Tuple[int, Dict[str, Any], List[Tuple[str, str]]]] = []

    iterator = range(len(dataset))
    if show_progress:
        iterator = tqdm(iterator, desc="Scoring layout rarity")
    for dataset_idx in iterator:
        sample = dataset[dataset_idx]
        if int(sample["n_objects"]) <= 0:
            continue
        pairs = sample_instance_combo_pairs(dataset, sample, thresholds, category_id_to_name)
        combos = [combo_key(category_name, size_bin) for category_name, size_bin in pairs]
        instance_counter.update(combos)
        per_sample_pairs.append((dataset_idx, sample, pairs))

    records: List[LayoutRarityRecord] = []
    for dataset_idx, sample, pairs in per_sample_pairs:
        combos = [combo_key(category_name, size_bin) for category_name, size_bin in pairs]
        rarity_values = [1.0 / float(instance_counter[combo]) for combo in combos]
        if aggregation == "mean":
            layout_rarity = float(np.mean(rarity_values))
        else:
            layout_rarity = float(np.sum(rarity_values))
        records.append(
            LayoutRarityRecord(
                dataset_idx=dataset_idx,
                sample=sample,
                combos=combos,
                combo_pairs=list(pairs),
                layout_rarity=layout_rarity,
                n_objects=int(sample["n_objects"]),
                file_name=str(sample["file_name"]),
            )
        )

    records.sort(key=lambda item: (item.layout_rarity, item.n_objects), reverse=True)
    return records, instance_counter


def select_layout_records(
    records: Sequence[LayoutRarityRecord],
    *,
    n_samples: int,
    min_objects: int,
    selection_mode: str = "rare_first",
    seed: int = 0,
) -> List[LayoutRarityRecord]:
    filtered = [record for record in records if int(record.n_objects) >= int(min_objects)]
    if selection_mode == "random":
        rng = random.Random(int(seed))
        rng.shuffle(filtered)
    return filtered[: max(0, int(n_samples))]


@torch.no_grad()
def sample_layout_batch(
    sampler: LayoutFlowMatchingSampler,
    batch: Dict[str, Any],
    *,
    steps: int,
    seed: int,
) -> torch.Tensor:
    sampler.unet.eval()
    batch_size = int(batch["pixel_values"].shape[0])
    shape = get_unet_sample_shape(sampler.unet, override=sampler._sample_shape)

    noise_generator = torch.Generator(device=sampler.device)
    noise_generator.manual_seed(int(seed))
    z = torch.randn(batch_size, *shape, generator=noise_generator, device=sampler.device)

    cond_kw = {
        "boxes_xyxy_norm": batch["boxes_xyxy_norm"].to(sampler.device),
        "labels": batch["labels"].to(sampler.device),
        "object_mask": batch["object_mask"].to(sampler.device),
    }
    if hasattr(sampler.unet, "sample_style_noise") and bool(getattr(sampler.unet, "use_style_latent", False)):
        style_generator = torch.Generator(device="cpu")
        style_generator.manual_seed(int(seed) + 1000)
        cond_kw["style_noise"] = sampler.unet.sample_style_noise(
            batch_size=batch_size,
            max_objects=int(batch["labels"].shape[1]),
            device="cpu",
            dtype=batch["boxes_xyxy_norm"].dtype,
            generator=style_generator,
        ).to(sampler.device)

    dt = 1.0 / float(steps)
    for step_idx in range(steps):
        t_val = step_idx / float(steps)
        t = torch.full((batch_size,), t_val, device=sampler.device)
        unet_sample = sampler.unet(z, t * sampler.t_scale, **cond_kw).sample
        if sampler.train_target == "x0":
            velocity = (unet_sample - z) / (1.0 - t[:, None, None, None]).clamp(min=1e-5)
        else:
            velocity = unet_sample
        z = z + velocity * dt

    return sampler.decode(z).detach().cpu()


def tensor_to_npy_array(image: torch.Tensor) -> np.ndarray:
    image = image.detach().cpu().to(torch.float32)
    return image.numpy().astype(np.float32, copy=False)


def _xyxy_to_coco_bbox(box_xyxy: Sequence[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _build_generated_categories(selected_records: Sequence[LayoutRarityRecord]) -> List[Dict[str, Any]]:
    categories_map: Dict[int, str] = {}
    for record in selected_records:
        sample = record.sample
        label_names = list(sample.get("label_names", []))
        labels = sample["labels"].tolist()
        for idx, label in enumerate(labels):
            if int(label) not in categories_map:
                name = label_names[idx] if idx < len(label_names) else str(int(label))
                categories_map[int(label)] = str(name)
    return [
        {"id": int(category_id), "name": categories_map[category_id]}
        for category_id in sorted(categories_map)
    ]


def initialize_generated_layout_export(
    *,
    output_dir: str | Path,
    selected_records: Sequence[LayoutRarityRecord],
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    images_dir = output_dir / "images"
    metadata_dir = output_dir / "metadata"
    images_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    return {
        "output_dir": output_dir,
        "images_dir": images_dir,
        "metadata_dir": metadata_dir,
        "categories": _build_generated_categories(selected_records),
        "coco_images": [],
        "coco_annotations": [],
        "provenance_rows": [],
        "annotation_id": 1,
        "image_id": 1,
    }


def append_generated_layout_batch(
    export_state: Dict[str, Any],
    *,
    records: Sequence[LayoutRarityRecord],
    generated_images: Sequence[torch.Tensor],
) -> None:
    images_dir = Path(export_state["images_dir"])
    coco_images: List[Dict[str, Any]] = export_state["coco_images"]
    coco_annotations: List[Dict[str, Any]] = export_state["coco_annotations"]
    provenance_rows: List[Dict[str, Any]] = export_state["provenance_rows"]
    annotation_id = int(export_state["annotation_id"])
    image_id = int(export_state["image_id"])

    for record, generated_image in zip(records, generated_images):
        file_name = f"sample_{image_id:06d}.npy"
        np.save(images_dir / file_name, tensor_to_npy_array(generated_image))
        _, image_h, image_w = generated_image.shape

        coco_images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(image_w),
                "height": int(image_h),
            }
        )

        labels = record.sample["labels"].tolist()
        boxes_xyxy = record.sample["boxes_xyxy"].tolist()
        for object_idx, (label, box_xyxy) in enumerate(zip(labels, boxes_xyxy)):
            coco_bbox = _xyxy_to_coco_bbox(box_xyxy)
            coco_annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": coco_bbox,
                    "area": float(coco_bbox[2] * coco_bbox[3]),
                    "iscrowd": 0,
                    "source_image_id": record.sample.get("image_id"),
                    "source_file_name": record.file_name,
                    "object_index": int(object_idx),
                }
            )
            annotation_id += 1

        provenance_rows.append(
            {
                "generated_image_id": int(image_id),
                "generated_file_name": file_name,
                "source_image_id": record.sample.get("image_id"),
                "source_file_name": record.file_name,
                "dataset_idx": int(record.dataset_idx),
                "n_objects": int(record.n_objects),
                "layout_rarity": float(record.layout_rarity),
                "combos": list(record.combos),
                "combo_pairs": [{"category": category, "size_bin": size_bin} for category, size_bin in record.combo_pairs],
            }
        )
        image_id += 1

    export_state["annotation_id"] = annotation_id
    export_state["image_id"] = image_id


def finalize_generated_layout_export(
    export_state: Dict[str, Any],
    *,
    split: str,
    dataset_id: str,
    selection_mode: str,
    rarity_aggregation: str,
    size_bin_thresholds: np.ndarray,
    pipeline_dir: str | Path,
    preset_path: str | Path,
    checkpoint_name: str,
    steps: int,
    seed: int,
    n_requested_samples: int,
) -> Dict[str, Any]:
    output_dir = Path(export_state["output_dir"])
    metadata_dir = Path(export_state["metadata_dir"])
    coco_images: List[Dict[str, Any]] = export_state["coco_images"]
    coco_annotations: List[Dict[str, Any]] = export_state["coco_annotations"]
    provenance_rows: List[Dict[str, Any]] = export_state["provenance_rows"]
    coco_payload = {
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": export_state["categories"],
    }
    save_json(output_dir / "annotations.json", coco_payload)
    write_jsonl(metadata_dir / "provenance.jsonl", provenance_rows)

    summary = {
        "dataset_id": str(dataset_id),
        "split": str(split),
        "selection_mode": str(selection_mode),
        "rarity_aggregation": str(rarity_aggregation),
        "n_requested_samples": int(n_requested_samples),
        "n_generated_samples": int(len(coco_images)),
        "n_annotations": int(len(coco_annotations)),
        "size_bin_thresholds": [float(v) for v in size_bin_thresholds.tolist()],
        "pipeline_dir": str(Path(pipeline_dir)),
        "preset_path": str(Path(preset_path)),
        "checkpoint_name": str(checkpoint_name),
        "steps": int(steps),
        "seed": int(seed),
        "images_dir": "images",
        "annotations_path": "annotations.json",
        "samples": provenance_rows,
    }
    save_json(metadata_dir / "summary.json", summary)
    return summary


def export_generated_layout_dataset(
    *,
    output_dir: str | Path,
    selected_records: Sequence[LayoutRarityRecord],
    generated_images: Sequence[torch.Tensor],
    split: str,
    dataset_id: str,
    selection_mode: str,
    rarity_aggregation: str,
    size_bin_thresholds: np.ndarray,
    pipeline_dir: str | Path,
    preset_path: str | Path,
    checkpoint_name: str,
    steps: int,
    seed: int,
) -> Dict[str, Any]:
    export_state = initialize_generated_layout_export(
        output_dir=output_dir,
        selected_records=selected_records,
    )
    append_generated_layout_batch(
        export_state,
        records=selected_records,
        generated_images=generated_images,
    )
    return finalize_generated_layout_export(
        export_state,
        split=split,
        dataset_id=dataset_id,
        selection_mode=selection_mode,
        rarity_aggregation=rarity_aggregation,
        size_bin_thresholds=size_bin_thresholds,
        pipeline_dir=pipeline_dir,
        preset_path=preset_path,
        checkpoint_name=checkpoint_name,
        steps=steps,
        seed=seed,
        n_requested_samples=len(selected_records),
    )


def auto_find_latest_filter_run(runs_root: str | Path) -> Path:
    runs_root = Path(runs_root)
    candidates = [path for path in runs_root.glob("*/metrics/summary.json") if path.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No filter run summaries found under {runs_root}")
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].parents[1]


def resolve_filter_checkpoint_and_summary(
    *,
    run_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[Optional[Path], Path, Optional[Path]]:
    if run_dir:
        run_path = Path(run_dir)
        return run_path, run_path / "checkpoints" / "best.pt", run_path / "metrics" / "summary.json"
    if checkpoint_path is None:
        raise ValueError("Either run_dir or checkpoint_path must be provided.")
    checkpoint = Path(checkpoint_path)
    maybe_run_dir = checkpoint.parent.parent
    summary = maybe_run_dir / "metrics" / "summary.json"
    return (maybe_run_dir if maybe_run_dir.is_dir() else None), checkpoint, (summary if summary.is_file() else None)


def load_filter_from_run_or_checkpoint(
    *,
    device: str | torch.device,
    run_dir: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    threshold_override: Optional[float] = None,
) -> tuple[torch.nn.Module, Dict[str, Any], Any, int, float, Optional[Path]]:
    resolved_run_dir, resolved_checkpoint_path, resolved_summary_path = resolve_filter_checkpoint_and_summary(
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
    )
    if not resolved_checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing classifier checkpoint: {resolved_checkpoint_path}")

    summary: Dict[str, Any] = {}
    if resolved_summary_path is not None and resolved_summary_path.is_file():
        summary = load_json(resolved_summary_path)

    checkpoint = torch.load(resolved_checkpoint_path, map_location=device)
    classifier_mode = str(summary.get("classifier_mode", checkpoint.get("classifier_mode", "binary")))
    if classifier_mode == "multiclass":
        num_foreground_classes = int(summary.get("num_foreground_classes", checkpoint.get("num_foreground_classes", 0)))
        model = MultiClassForegroundBackgroundClassifier(num_classes=num_foreground_classes + 1).to(device)
    else:
        model = ForegroundBackgroundClassifier().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    if classifier_mode == "multiclass":
        threshold = (
            threshold_override
            if threshold_override is not None
            else summary.get("per_class_thresholds", checkpoint.get("per_class_thresholds", {}))
        )
    else:
        threshold = float(
            threshold_override
            if threshold_override is not None
            else summary.get("chosen_threshold", checkpoint.get("best_threshold", 0.5))
        )
    input_size = int(summary.get("input_size", checkpoint.get("config", {}).get("input_size", 128)))
    context_ratio = float(summary.get("context_ratio", checkpoint.get("config", {}).get("context_ratio", 1.25)))
    return model, summary, threshold, input_size, context_ratio, resolved_run_dir


def expand_box_with_context(
    box_xyxy: Sequence[float],
    *,
    image_w: int,
    image_h: int,
    context_ratio: float,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    new_w = width * float(context_ratio)
    new_h = height * float(context_ratio)
    x1 = max(0.0, cx - 0.5 * new_w)
    y1 = max(0.0, cy - 0.5 * new_h)
    x2 = min(float(image_w), cx + 0.5 * new_w)
    y2 = min(float(image_h), cy + 0.5 * new_h)
    x1_i = int(math.floor(x1))
    y1_i = int(math.floor(y1))
    x2_i = int(math.ceil(max(x2, x1_i + 1)))
    y2_i = int(math.ceil(max(y2, y1_i + 1)))
    return x1_i, y1_i, x2_i, y2_i


def _load_generated_dataset(dataset_dir: str | Path) -> tuple[dict, dict, dict, dict, Path]:
    dataset_dir = Path(dataset_dir)
    annotations = load_coco_annotations(dataset_dir / "annotations.json")
    images_by_id, anns_by_image_id, _ = index_annotations(annotations)
    categories = build_category_id_to_name(annotations)
    summary_path = dataset_dir / "metadata" / "summary.json"
    summary = load_json(summary_path) if summary_path.is_file() else {}
    return annotations, images_by_id, anns_by_image_id, categories, dataset_dir / "images"


def _ensure_channel_first(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image[None, ...]
    if image.ndim == 3 and image.shape[0] in (1, 3):
        return image
    if image.ndim == 3 and image.shape[-1] in (1, 3):
        return np.transpose(image, (2, 0, 1))
    raise ValueError(f"Unsupported generated image shape: {tuple(image.shape)}")


def _prepare_crop_batch(crops: Sequence[torch.Tensor], *, input_size: int, device: torch.device) -> torch.Tensor:
    resized = [
        F.interpolate(
            crop.to(torch.float32).unsqueeze(0),
            size=(input_size, input_size),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        for crop in crops
    ]
    crop_batch = torch.stack(resized, dim=0)
    if crop_batch.shape[1] != 1:
        crop_batch = crop_batch.mean(dim=1, keepdim=True)
    return crop_batch.to(device)


def _stats_rows_from_counter(counter: Dict[str, Dict[str, int]], *, group_name: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in sorted(counter):
        entry = counter[key]
        positive = int(entry["positive"])
        negative = int(entry["negative"])
        total = max(1, positive + negative)
        rows.append(
            {
                group_name: key,
                "positive_count": positive,
                "negative_count": negative,
                "total_count": positive + negative,
                "positive_ratio": positive / total,
                "negative_ratio": negative / total,
            }
        )
    return rows


def summarize_instance_statistics(instance_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    overall_positive = sum(int(row["is_positive"]) for row in instance_rows)
    overall_negative = len(instance_rows) - overall_positive
    overall_total = max(1, len(instance_rows))

    by_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
    by_size: Dict[str, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})
    by_joint: Dict[str, Dict[str, int]] = defaultdict(lambda: {"positive": 0, "negative": 0})

    for row in instance_rows:
        category = str(row["category_name"])
        size_bin = str(row["size_bin"])
        joint = combo_key(category, size_bin)
        target = "positive" if bool(row["is_positive"]) else "negative"
        by_category[category][target] += 1
        by_size[size_bin][target] += 1
        by_joint[joint][target] += 1

    return {
        "overall": {
            "positive_count": overall_positive,
            "negative_count": overall_negative,
            "total_count": len(instance_rows),
            "positive_ratio": overall_positive / overall_total,
            "negative_ratio": overall_negative / overall_total,
        },
        "by_category": _stats_rows_from_counter(by_category, group_name="category"),
        "by_size_bin": _stats_rows_from_counter(by_size, group_name="size_bin"),
        "by_category_size_bin": _stats_rows_from_counter(by_joint, group_name="category_size_bin"),
    }


def build_instance_confusion_matrix(
    instance_rows: Sequence[Dict[str, Any]],
    *,
    checkpoint_summary: Dict[str, Any],
) -> Dict[str, Any]:
    category_id_to_name = {
        str(k): str(v) for k, v in checkpoint_summary.get("category_id_to_name", {}).items()
    }
    for row in instance_rows:
        category_id_to_name.setdefault(str(row["expected_category_id"]), str(row.get("expected_category_name", row["expected_category_id"])))
    expected_category_ids = sorted({str(row["expected_category_id"]) for row in instance_rows}, key=lambda value: int(value) if value.isdigit() else value)
    labels = [category_id_to_name.get(category_id, category_id) for category_id in expected_category_ids] + ["background"]
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int32)
    for row in instance_rows:
        expected_label = category_id_to_name.get(str(row["expected_category_id"]), str(row["expected_category_id"]))
        predicted_category_id = row.get("predicted_category_id")
        predicted_label = "background" if predicted_category_id is None else category_id_to_name.get(str(predicted_category_id), str(predicted_category_id))
        matrix[label_to_index[expected_label], label_to_index[predicted_label]] += 1
    row_totals = matrix.sum(axis=1, keepdims=True).astype(np.float32)
    normalized = np.divide(
        matrix.astype(np.float32),
        np.maximum(row_totals, 1.0),
        out=np.zeros_like(matrix, dtype=np.float32),
        where=row_totals > 0,
    )
    return {
        "labels": labels,
        "matrix": matrix.tolist(),
        "matrix_counts": matrix.tolist(),
        "matrix_normalized": normalized.tolist(),
        "normalization": "row",
    }


def _normalize_for_display(image_array: np.ndarray) -> np.ndarray:
    image = np.asarray(image_array, dtype=np.float32)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = image[..., None]
    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    elif image.shape[-1] > 3:
        image = image[..., :3]
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value > min_value:
        image = (image - min_value) / (max_value - min_value)
    else:
        image = np.zeros_like(image, dtype=np.float32)
    return image.clip(0.0, 1.0)


def _save_category_example(
    *,
    row: Dict[str, Any],
    image_path: Path,
    output_path: Path,
    title: str,
) -> Path:
    image_array = _ensure_channel_first(np.load(image_path))
    display_image = _normalize_for_display(image_array)
    crop_x1, crop_y1, crop_x2, crop_y2 = [int(v) for v in row["crop_box_xyxy"]]
    bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [float(v) for v in row["bbox_xyxy"]]
    crop_image = display_image[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop_image.size == 0:
        crop_image = display_image

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 4.2))
    axes[0].imshow(crop_image)
    axes[0].set_title("crop")
    axes[0].axis("off")

    axes[1].imshow(display_image)
    rect = Rectangle(
        (bbox_x1, bbox_y1),
        max(1.0, bbox_x2 - bbox_x1),
        max(1.0, bbox_y2 - bbox_y1),
        linewidth=2.0,
        edgecolor="#d1495b",
        facecolor="none",
    )
    axes[1].add_patch(rect)
    axes[1].set_title("full image")
    axes[1].axis("off")
    fig.suptitle(title, fontsize=11)
    return _save_figure(fig, output_path)


def build_category_examples(
    *,
    instance_rows: Sequence[Dict[str, Any]],
    dataset_dir: str | Path,
    checkpoint_summary: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images"
    category_id_to_name = {
        str(k): str(v) for k, v in checkpoint_summary.get("category_id_to_name", {}).items()
    }
    for row in instance_rows:
        category_id_to_name.setdefault(str(row["expected_category_id"]), str(row.get("expected_category_name", row["expected_category_id"])))
    category_ids = sorted({str(row["expected_category_id"]) for row in instance_rows}, key=lambda value: int(value) if value.isdigit() else value)
    examples: Dict[str, Any] = {}

    for category_id in category_ids:
        category_name = category_id_to_name.get(category_id, category_id)
        fn_candidates = [
            row for row in instance_rows
            if str(row["expected_category_id"]) == category_id and not bool(row["is_positive"])
        ]
        fn_candidates.sort(key=lambda row: float(row.get("expected_class_probability", row.get("probability", 0.0))), reverse=True)

        fp_candidates = [
            row for row in instance_rows
            if str(row.get("predicted_category_id")) == category_id and str(row["expected_category_id"]) != category_id
        ]
        fp_candidates.sort(key=lambda row: float(row.get("expected_class_probability", row.get("probability", 1.0))))

        category_entry: Dict[str, Any] = {
            "category_id": int(category_id) if category_id.isdigit() else category_id,
            "category_name": category_name,
            "false_negative": None,
            "false_positive": None,
        }

        if fn_candidates:
            row = fn_candidates[0]
            output_path = output_dir / f"false_negative_{category_name.replace(' ', '_')}.png"
            saved_path = _save_category_example(
                row=row,
                image_path=images_dir / str(row["generated_file_name"]),
                output_path=output_path,
                title=f"{category_name} false negative | p_gt={float(row.get('expected_class_probability', row.get('probability', 0.0))):.3f}",
            )
            category_entry["false_negative"] = {
                "generated_file_name": row["generated_file_name"],
                "annotation_id": row["annotation_id"],
                "probability": float(row.get("expected_class_probability", row.get("probability", 0.0))),
                "path": str(saved_path),
            }

        if fp_candidates:
            row = fp_candidates[0]
            output_path = output_dir / f"false_positive_{category_name.replace(' ', '_')}.png"
            saved_path = _save_category_example(
                row=row,
                image_path=images_dir / str(row["generated_file_name"]),
                output_path=output_path,
                title=(
                    f"{category_name} false positive | expected={row.get('expected_category_name')} "
                    f"| p_gt={float(row.get('expected_class_probability', row.get('probability', 0.0))):.3f}"
                ),
            )
            category_entry["false_positive"] = {
                "generated_file_name": row["generated_file_name"],
                "annotation_id": row["annotation_id"],
                "probability": float(row.get("expected_class_probability", row.get("probability", 0.0))),
                "path": str(saved_path),
            }

        examples[category_name] = category_entry
    return examples


def _resolve_sweep_values(sweep_text: str) -> List[float]:
    if sweep_text.strip():
        values = [float(part.strip()) for part in sweep_text.split(",") if part.strip()]
    else:
        values = [round(step / 10.0, 2) for step in range(0, 11)]
    values = sorted({min(1.0, max(0.0, value)) for value in values})
    return values


def _count_combo_keys(image_rows: Sequence[Dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for row in image_rows:
        counter.update(row.get("combo_keys", []))
    return counter


def _count_instance_combo_keys(instance_rows: Sequence[Dict[str, Any]]) -> Counter:
    counter: Counter = Counter()
    for row in instance_rows:
        counter.update([combo_key(str(row["category_name"]), str(row["size_bin"]))])
    return counter


def _build_combo_frequency_counter_from_annotations(
    *,
    annotations_path: Path,
    thresholds: np.ndarray,
) -> Counter:
    coco = load_coco_annotations(annotations_path)
    images_by_id, _, _ = index_annotations(coco)
    category_id_to_name = build_category_id_to_name(coco)
    counter: Counter = Counter()
    for ann in coco.get("annotations", []):
        image_info = images_by_id.get(ann["image_id"], {})
        image_w = float(image_info.get("width", 1.0))
        image_h = float(image_info.get("height", 1.0))
        bbox = ann.get("bbox", [0.0, 0.0, 0.0, 0.0])
        area_ratio = (float(bbox[2]) * float(bbox[3])) / max(1.0, image_w * image_h)
        category_name = category_id_to_name.get(int(ann.get("category_id", -1)), str(ann.get("category_id", -1)))
        counter[combo_key(category_name, size_bin_name(area_ratio, thresholds))] += 1
    return counter


def _build_reference_combo_frequency_counter(
    *,
    dataset_id: str,
    split: str,
    thresholds: np.ndarray,
    fallback_instance_rows: Sequence[Dict[str, Any]],
) -> tuple[Counter, Dict[str, Any]]:
    try:
        target = resolve_dataset_target(dataset_id)
        annotations_path = target.annotations_path(split)
    except Exception:
        annotations_path = None

    if annotations_path is not None and annotations_path.is_file():
        counter = _build_combo_frequency_counter_from_annotations(
            annotations_path=annotations_path,
            thresholds=thresholds,
        )
        return counter, {
            "dataset_id": str(dataset_id),
            "split": str(split),
            "source": "real_dataset_train_split",
            "annotations_path": str(annotations_path),
            "total_instances": int(sum(counter.values())),
        }

    fallback_counter = _count_instance_combo_keys(fallback_instance_rows)
    return fallback_counter, {
        "dataset_id": str(dataset_id),
        "split": str(split),
        "source": "fallback_generated_instances",
        "annotations_path": None,
        "total_instances": int(sum(fallback_counter.values())),
    }


def _plot_generated_distribution_threshold_effects(
    *,
    baseline_counter: Counter,
    kept_counter: Counter,
    removed_counter: Counter,
    title_prefix: str,
    path: Path,
    top_k: int = 40,
) -> Path:
    keys = resolve_generation_plot_keys(baseline_counter, top_k=top_k)
    baseline_values = np.asarray([baseline_counter.get(key, 0) for key in keys], dtype=np.float32)
    kept_values = np.asarray([kept_counter.get(key, 0) for key in keys], dtype=np.float32)
    removed_values = np.asarray([removed_counter.get(key, 0) for key in keys], dtype=np.float32)

    fig, axes = plt.subplots(2, 1, figsize=(max(14, 0.55 * len(keys)), 10.0), sharex=True)
    x = np.arange(len(keys))
    axes[0].bar(x, baseline_values, color="#3a78c2", alpha=0.9, label="generated instances before image removal")
    axes[0].set_title(f"{title_prefix}: before image removal")
    axes[0].set_ylabel("count")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend()

    axes[1].bar(x, kept_values, color="#2a9d55", alpha=0.85, label="kept images' instances")
    axes[1].bar(x, removed_values, bottom=kept_values, color="#d1495b", alpha=0.8, label="removed images' instances")
    axes[1].set_title(f"{title_prefix}: kept vs removed after image threshold")
    axes[1].set_ylabel("count")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend()
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(keys, rotation=75, ha="right")
    return _save_figure(fig, path)


def _compute_global_discard_metrics(
    *,
    instance_rows: Sequence[Dict[str, Any]],
    removed_image_ids: set[int],
) -> Dict[str, Any]:
    total_valid = 0
    total_invalid = 0
    removed_valid = 0
    removed_invalid = 0
    for row in instance_rows:
        is_positive = bool(row["is_positive"])
        if is_positive:
            total_valid += 1
        else:
            total_invalid += 1
        if int(row["generated_image_id"]) in removed_image_ids:
            if is_positive:
                removed_valid += 1
            else:
                removed_invalid += 1
    return {
        "discarded_valid_count": int(removed_valid),
        "discarded_invalid_count": int(removed_invalid),
        "total_valid_count": int(total_valid),
        "total_invalid_count": int(total_invalid),
        "discarded_valid_ratio": float(removed_valid / max(1, total_valid)) if total_valid > 0 else 0.0,
        "discarded_invalid_ratio": float(removed_invalid / max(1, total_invalid)) if total_invalid > 0 else 0.0,
    }


def _compute_discarded_category_ratios(
    *,
    instance_rows: Sequence[Dict[str, Any]],
    removed_image_ids: set[int],
) -> List[Dict[str, Any]]:
    totals_by_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"valid": 0, "invalid": 0})
    removed_by_category: Dict[str, Dict[str, int]] = defaultdict(lambda: {"valid": 0, "invalid": 0})

    for row in instance_rows:
        category = str(row["category_name"])
        outcome_key = "valid" if bool(row["is_positive"]) else "invalid"
        totals_by_category[category][outcome_key] += 1
        if int(row["generated_image_id"]) in removed_image_ids:
            removed_by_category[category][outcome_key] += 1

    categories = sorted(totals_by_category.keys())
    ratio_rows: List[Dict[str, Any]] = []
    for category in categories:
        total_valid = int(totals_by_category[category]["valid"])
        total_invalid = int(totals_by_category[category]["invalid"])
        removed_valid = int(removed_by_category[category]["valid"])
        removed_invalid = int(removed_by_category[category]["invalid"])
        ratio_rows.append(
            {
                "category": category,
                "discarded_valid_count": removed_valid,
                "discarded_invalid_count": removed_invalid,
                "discarded_valid_ratio": removed_valid / max(1, total_valid) if total_valid > 0 else 0.0,
                "discarded_invalid_ratio": removed_invalid / max(1, total_invalid) if total_invalid > 0 else 0.0,
                "total_valid_count": total_valid,
                "total_invalid_count": total_invalid,
            }
        )
    ratio_rows.sort(
        key=lambda row: (
            -max(float(row["discarded_invalid_ratio"]), float(row["discarded_valid_ratio"])),
            row["category"],
        )
    )
    return ratio_rows


def _compute_combo_rarity_score(
    *,
    instance_rows: Sequence[Dict[str, Any]],
    removed_image_ids: set[int],
    reference_counter: Counter,
    score_alpha: float,
    score_beta: float,
    rarity_frequency_epsilon: float,
) -> Dict[str, Any]:
    totals_by_combo: Dict[str, Dict[str, int]] = defaultdict(lambda: {"valid": 0, "invalid": 0})
    removed_by_combo: Dict[str, Dict[str, int]] = defaultdict(lambda: {"valid": 0, "invalid": 0})

    for row in instance_rows:
        combo = combo_key(str(row["category_name"]), str(row["size_bin"]))
        outcome_key = "valid" if bool(row["is_positive"]) else "invalid"
        totals_by_combo[combo][outcome_key] += 1
        if int(row["generated_image_id"]) in removed_image_ids:
            removed_by_combo[combo][outcome_key] += 1

    combo_keys = sorted(totals_by_combo.keys())
    raw_weights: Dict[str, float] = {}
    for combo in combo_keys:
        reference_frequency = float(reference_counter.get(combo, 0))
        raw_weights[combo] = 1.0 / math.sqrt(reference_frequency + float(rarity_frequency_epsilon))
    weight_norm = sum(raw_weights.values()) or 1.0

    combo_rows: List[Dict[str, Any]] = []
    score = 0.0
    for combo in combo_keys:
        total_valid = int(totals_by_combo[combo]["valid"])
        total_invalid = int(totals_by_combo[combo]["invalid"])
        removed_valid = int(removed_by_combo[combo]["valid"])
        removed_invalid = int(removed_by_combo[combo]["invalid"])
        r_val = removed_valid / max(1, total_valid) if total_valid > 0 else 0.0
        r_inv = removed_invalid / max(1, total_invalid) if total_invalid > 0 else 0.0
        weight = raw_weights[combo] / weight_norm
        score_term = weight * (float(score_alpha) * r_inv - float(score_beta) * r_val)
        score += score_term
        combo_rows.append(
            {
                "category_size_bin": combo,
                "reference_frequency": int(reference_counter.get(combo, 0)),
                "weight": float(weight),
                "discarded_valid_count": removed_valid,
                "discarded_invalid_count": removed_invalid,
                "total_valid_count": total_valid,
                "total_invalid_count": total_invalid,
                "discarded_valid_ratio": float(r_val),
                "discarded_invalid_ratio": float(r_inv),
                "score_term": float(score_term),
            }
        )
    combo_rows.sort(key=lambda row: (-abs(float(row["score_term"])), row["category_size_bin"]))
    return {
        "score": float(score),
        "combo_rows": combo_rows,
    }


def _plot_discarded_category_ratios(
    *,
    ratio_rows: Sequence[Dict[str, Any]],
    title_prefix: str,
    path: Path,
) -> Optional[Path]:
    if not ratio_rows:
        return None
    labels = [str(row["category"]) for row in ratio_rows]
    discarded_valid = np.asarray([float(row["discarded_valid_ratio"]) for row in ratio_rows], dtype=np.float32)
    discarded_invalid = np.asarray([float(row["discarded_invalid_ratio"]) for row in ratio_rows], dtype=np.float32)
    x = np.arange(len(labels))
    width = 0.42

    fig, ax = plt.subplots(1, 1, figsize=(max(12, 0.55 * len(labels) + 3), 5.8))
    ax.bar(x - width / 2, discarded_invalid, width=width, color="#d1495b", alpha=0.88, label="discarded invalid ratio")
    ax.bar(x + width / 2, discarded_valid, width=width, color="#2a9d55", alpha=0.88, label="discarded valid ratio")
    ax.set_title(f"{title_prefix}: discarded per-category instance ratios")
    ax.set_ylabel("ratio in [0, 1]")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=75, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _style_axes(ax)
    return _save_figure(fig, path)


def _plot_threshold_selection_score_curve(
    *,
    threshold_rows: Sequence[Dict[str, Any]],
    recommended_value: Optional[float],
    path: Path,
) -> Path:
    x_values = [float(row["min_valid_object_fraction"]) for row in threshold_rows]
    y_values = [float(row["rarity_weighted_score"]) for row in threshold_rows]
    feasible_mask = np.asarray([bool(row.get("is_feasible_threshold", True)) for row in threshold_rows], dtype=bool)
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
    ax.plot(x_values, y_values, marker="o", color="#5b8def", linewidth=2.0, alpha=0.65, label="rarity-weighted score")
    if feasible_mask.any():
        feasible_x = np.asarray(x_values, dtype=np.float32)[feasible_mask]
        feasible_y = np.asarray(y_values, dtype=np.float32)[feasible_mask]
        ax.scatter(feasible_x, feasible_y, color="#2a9d55", s=60, label="feasible threshold")
    if (~feasible_mask).any():
        infeasible_x = np.asarray(x_values, dtype=np.float32)[~feasible_mask]
        infeasible_y = np.asarray(y_values, dtype=np.float32)[~feasible_mask]
        ax.scatter(infeasible_x, infeasible_y, color="#d1495b", marker="x", s=70, label="infeasible threshold")
    if recommended_value is not None:
        ax.axvline(float(recommended_value), color="#2a9d55", linestyle="-.", linewidth=2, label=f"recommended={float(recommended_value):.2f}")
    ax.set_xlabel("MIN_VALID_OBJECT_FRACTION")
    ax.set_ylabel("score")
    ax.set_title("Rare-aware threshold selection score")
    ax.legend()
    _style_axes(ax)
    return _save_figure(fig, path)


def _plot_threshold_global_discard_ratios(
    *,
    threshold_rows: Sequence[Dict[str, Any]],
    recommended_value: Optional[float],
    path: Path,
) -> Path:
    x_values = [float(row["min_valid_object_fraction"]) for row in threshold_rows]
    invalid_ratios = [float(row["global_discarded_invalid_ratio"]) for row in threshold_rows]
    valid_ratios = [float(row["global_discarded_valid_ratio"]) for row in threshold_rows]
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 4.8))
    ax.plot(x_values, invalid_ratios, marker="o", color="#d1495b", linewidth=2.0, label="discarded invalid ratio")
    ax.plot(x_values, valid_ratios, marker="o", color="#2a9d55", linewidth=2.0, label="discarded valid ratio")
    if recommended_value is not None:
        ax.axvline(float(recommended_value), color="#3a78c2", linestyle="-.", linewidth=1.8, label=f"recommended={float(recommended_value):.2f}")
    ax.set_xlabel("MIN_VALID_OBJECT_FRACTION")
    ax.set_ylabel("ratio in [0, 1]")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Global discarded-instance ratios across thresholds")
    ax.legend()
    _style_axes(ax)
    return _save_figure(fig, path)


def _plot_threshold_tradeoff_scatter(
    *,
    threshold_rows: Sequence[Dict[str, Any]],
    path: Path,
) -> Path:
    invalid_ratios = np.asarray([float(row["global_discarded_invalid_ratio"]) for row in threshold_rows], dtype=np.float32)
    valid_ratios = np.asarray([float(row["global_discarded_valid_ratio"]) for row in threshold_rows], dtype=np.float32)
    scores = np.asarray([float(row["rarity_weighted_score"]) for row in threshold_rows], dtype=np.float32)
    thresholds = [float(row["min_valid_object_fraction"]) for row in threshold_rows]
    feasible_mask = np.asarray([bool(row.get("is_feasible_threshold", True)) for row in threshold_rows], dtype=bool)
    fig, ax = plt.subplots(1, 1, figsize=(6.8, 5.5))
    scatter = ax.scatter(
        invalid_ratios[feasible_mask],
        valid_ratios[feasible_mask],
        c=scores[feasible_mask],
        cmap="coolwarm",
        s=70,
        marker="o",
        edgecolors="black",
        linewidths=0.5,
        label="feasible",
    ) if feasible_mask.any() else None
    if (~feasible_mask).any():
        ax.scatter(
            invalid_ratios[~feasible_mask],
            valid_ratios[~feasible_mask],
            c=scores[~feasible_mask],
            cmap="coolwarm",
            s=85,
            marker="x",
            linewidths=1.8,
            label="infeasible",
        )
    for x, y, threshold in zip(invalid_ratios, valid_ratios, thresholds):
        ax.text(float(x) + 0.01, float(y) + 0.01, f"{threshold:.2f}", fontsize=8)
    ax.set_xlabel("global discarded invalid ratio")
    ax.set_ylabel("global discarded valid ratio")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Threshold trade-off: remove invalid, keep valid")
    _style_axes(ax)
    if scatter is None:
        scatter = ax.scatter([], [], c=[], cmap="coolwarm")
    ax.legend()
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="rarity-weighted score")
    return _save_figure(fig, path)


def _build_image_level_stats_for_threshold(
    image_rows: Sequence[Dict[str, Any]],
    *,
    min_valid_object_fraction: Optional[float],
) -> Dict[str, Any]:
    if min_valid_object_fraction is None:
        return {
            "total_image_count": len(image_rows),
            "mean_valid_fraction": float(np.mean([float(row["valid_fraction"]) for row in image_rows])) if image_rows else 0.0,
            "median_valid_fraction": float(np.median([float(row["valid_fraction"]) for row in image_rows])) if image_rows else 0.0,
            "selected_min_valid_object_fraction": None,
            "valid_image_count": None,
            "invalid_image_count": None,
            "valid_image_ratio": None,
            "invalid_image_ratio": None,
        }
    valid_images = sum(int(float(row["valid_fraction"]) >= float(min_valid_object_fraction)) for row in image_rows)
    total_images = len(image_rows)
    invalid_images = total_images - valid_images
    return {
        "total_image_count": total_images,
        "mean_valid_fraction": float(np.mean([float(row["valid_fraction"]) for row in image_rows])) if image_rows else 0.0,
        "median_valid_fraction": float(np.median([float(row["valid_fraction"]) for row in image_rows])) if image_rows else 0.0,
        "selected_min_valid_object_fraction": float(min_valid_object_fraction),
        "valid_image_count": int(valid_images),
        "invalid_image_count": int(invalid_images),
        "valid_image_ratio": float(valid_images / max(1, total_images)),
        "invalid_image_ratio": float(invalid_images / max(1, total_images)),
    }


def build_per_image_threshold_sweep_analysis(
    *,
    instance_rows: Sequence[Dict[str, Any]],
    image_rows: Sequence[Dict[str, Any]],
    sweep_text: str,
    output_dir: Path,
    reference_dataset_id: str,
    reference_split: str,
    size_thresholds: np.ndarray,
    max_discarded_valid_threshold: float,
    score_alpha: float,
    score_beta: float,
) -> Dict[str, Any]:
    sweep_values = _resolve_sweep_values(sweep_text)
    baseline_counter = _count_combo_keys(image_rows)
    threshold_rows: List[Dict[str, Any]] = []
    charts_dir = output_dir / "threshold_sweep"
    charts_dir.mkdir(parents=True, exist_ok=True)
    rarity_frequency_epsilon = 1.0
    reference_counter, reference_meta = _build_reference_combo_frequency_counter(
        dataset_id=reference_dataset_id,
        split=reference_split,
        thresholds=size_thresholds,
        fallback_instance_rows=instance_rows,
    )
    reference_rows = [
        {
            "category_size_bin": combo,
            "reference_frequency": int(count),
        }
        for combo, count in sorted(reference_counter.items(), key=lambda item: (-item[1], item[0]))
    ]
    reference_table_path = output_dir / "reference_category_size_frequencies.csv"
    write_csv(reference_table_path, reference_rows)

    for value in sweep_values:
        kept_rows = [row for row in image_rows if float(row["valid_fraction"]) >= float(value)]
        removed_rows = [row for row in image_rows if float(row["valid_fraction"]) < float(value)]
        removed_image_ids = {int(row["generated_image_id"]) for row in removed_rows}
        kept_counter = _count_combo_keys(kept_rows)
        removed_counter = _count_combo_keys(removed_rows)
        chart_path = _plot_generated_distribution_threshold_effects(
            baseline_counter=baseline_counter,
            kept_counter=kept_counter,
            removed_counter=removed_counter,
            title_prefix=f"MIN_VALID_OBJECT_FRACTION = {value:.2f}",
            path=charts_dir / f"threshold_{value:.2f}.png",
        )
        discarded_category_ratio_rows = _compute_discarded_category_ratios(
            instance_rows=instance_rows,
            removed_image_ids=removed_image_ids,
        )
        discarded_category_ratio_chart = _plot_discarded_category_ratios(
            ratio_rows=discarded_category_ratio_rows,
            title_prefix=f"MIN_VALID_OBJECT_FRACTION = {value:.2f}",
            path=charts_dir / f"threshold_{value:.2f}_discarded_category_ratios.png",
        )
        global_discard_metrics = _compute_global_discard_metrics(
            instance_rows=instance_rows,
            removed_image_ids=removed_image_ids,
        )
        rarity_score_payload = _compute_combo_rarity_score(
            instance_rows=instance_rows,
            removed_image_ids=removed_image_ids,
            reference_counter=reference_counter,
            score_alpha=score_alpha,
            score_beta=score_beta,
            rarity_frequency_epsilon=rarity_frequency_epsilon,
        )
        threshold_rows.append(
            {
                "min_valid_object_fraction": float(value),
                "valid_image_count": len(kept_rows),
                "invalid_image_count": len(removed_rows),
                "valid_image_ratio": len(kept_rows) / max(1, len(image_rows)),
                "kept_combo_counts": dict(kept_counter),
                "removed_combo_counts": dict(removed_counter),
                **global_discard_metrics,
                "global_discarded_valid_count": int(global_discard_metrics["discarded_valid_count"]),
                "global_discarded_invalid_count": int(global_discard_metrics["discarded_invalid_count"]),
                "global_discarded_valid_ratio": float(global_discard_metrics["discarded_valid_ratio"]),
                "global_discarded_invalid_ratio": float(global_discard_metrics["discarded_invalid_ratio"]),
                "rarity_weighted_score": float(rarity_score_payload["score"]),
                "combo_rarity_metrics": rarity_score_payload["combo_rows"],
                "max_combo_discarded_valid_ratio": float(
                    max(
                        (float(row["discarded_valid_ratio"]) for row in rarity_score_payload["combo_rows"]),
                        default=0.0,
                    )
                ),
                "is_feasible_threshold": bool(
                    all(
                        float(row["discarded_valid_ratio"]) <= float(max_discarded_valid_threshold)
                        for row in rarity_score_payload["combo_rows"]
                    )
                ),
                "discarded_category_ratios": discarded_category_ratio_rows,
                "chart_path": str(chart_path),
                "discarded_category_ratio_chart_path": str(discarded_category_ratio_chart) if discarded_category_ratio_chart is not None else None,
            }
        )

    threshold_rows.sort(key=lambda row: float(row["min_valid_object_fraction"]))
    feasible_rows = [row for row in threshold_rows if bool(row["is_feasible_threshold"])]
    recommended_row = (
        max(
            feasible_rows,
            key=lambda row: (
                float(row["rarity_weighted_score"]),
                -float(row["global_discarded_valid_ratio"]),
                float(row["global_discarded_invalid_ratio"]),
                -float(row["min_valid_object_fraction"]),
            ),
        )
        if feasible_rows
        else None
    )
    metrics_table_path = output_dir / "threshold_selection_metrics.csv"
    write_csv(
        metrics_table_path,
        [
            {
                "min_valid_object_fraction": float(row["min_valid_object_fraction"]),
                "rarity_weighted_score": float(row["rarity_weighted_score"]),
                "global_discarded_invalid_count": int(row["discarded_invalid_count"]),
                "global_discarded_invalid_ratio": float(row["global_discarded_invalid_ratio"]),
                "global_discarded_valid_count": int(row["discarded_valid_count"]),
                "global_discarded_valid_ratio": float(row["global_discarded_valid_ratio"]),
                "max_combo_discarded_valid_ratio": float(row["max_combo_discarded_valid_ratio"]),
                "is_feasible_threshold": bool(row["is_feasible_threshold"]),
                "valid_image_count": int(row["valid_image_count"]),
                "invalid_image_count": int(row["invalid_image_count"]),
                "valid_image_ratio": float(row["valid_image_ratio"]),
            }
            for row in threshold_rows
        ],
    )

    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5))
    x_values = [row["min_valid_object_fraction"] for row in threshold_rows]
    y_values = [row["valid_image_ratio"] for row in threshold_rows]
    ax.plot(x_values, y_values, marker="o", color="#3a78c2")
    if recommended_row is not None:
        ax.axvline(
            float(recommended_row["min_valid_object_fraction"]),
            color="#2a9d55",
            linestyle="-.",
            linewidth=2,
            label=f"recommended={float(recommended_row['min_valid_object_fraction']):.2f}",
        )
    ax.set_xlabel("MIN_VALID_OBJECT_FRACTION")
    ax.set_ylabel("valid image ratio")
    ax.set_title("Valid image ratio across image-threshold sweep")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    _style_axes(ax)
    sweep_curve_path = _save_figure(fig, output_dir / "valid_image_ratio_sweep.png")
    score_curve_path = _plot_threshold_selection_score_curve(
        threshold_rows=threshold_rows,
        recommended_value=None if recommended_row is None else float(recommended_row["min_valid_object_fraction"]),
        path=output_dir / "threshold_selection_score_curve.png",
    )
    global_discard_path = _plot_threshold_global_discard_ratios(
        threshold_rows=threshold_rows,
        recommended_value=None if recommended_row is None else float(recommended_row["min_valid_object_fraction"]),
        path=output_dir / "threshold_selection_global_discard_ratios.png",
    )
    tradeoff_path = _plot_threshold_tradeoff_scatter(
        threshold_rows=threshold_rows,
        path=output_dir / "threshold_selection_tradeoff_scatter.png",
    )

    return {
        "sweep_values": sweep_values,
        "thresholds": threshold_rows,
        "valid_image_ratio_sweep_chart": str(sweep_curve_path),
        "reference_frequency_analysis": {
            **reference_meta,
            "score_alpha": float(score_alpha),
            "score_beta": float(score_beta),
            "rarity_frequency_epsilon": float(rarity_frequency_epsilon),
            "reference_table_path": str(reference_table_path),
        },
        "threshold_selection": {
            "max_discarded_valid_threshold": float(max_discarded_valid_threshold),
            "n_feasible_thresholds": int(len(feasible_rows)),
            "recommended_min_valid_object_fraction": None if recommended_row is None else float(recommended_row["min_valid_object_fraction"]),
            "recommended_row": None if recommended_row is None else {
                "min_valid_object_fraction": float(recommended_row["min_valid_object_fraction"]),
                "rarity_weighted_score": float(recommended_row["rarity_weighted_score"]),
                "global_discarded_invalid_ratio": float(recommended_row["global_discarded_invalid_ratio"]),
                "global_discarded_valid_ratio": float(recommended_row["global_discarded_valid_ratio"]),
                "max_combo_discarded_valid_ratio": float(recommended_row["max_combo_discarded_valid_ratio"]),
                "valid_image_ratio": float(recommended_row["valid_image_ratio"]),
            },
            "metrics_table_path": str(metrics_table_path),
            "score_curve_chart_path": str(score_curve_path),
            "global_discard_ratio_chart_path": str(global_discard_path),
            "tradeoff_scatter_chart_path": str(tradeoff_path),
        },
    }


def audit_generated_layout_dataset(
    *,
    dataset_dir: str | Path,
    filter_model: ForegroundBackgroundClassifier,
    threshold: float,
    filter_input_size: int,
    context_ratio: float,
    device: str | torch.device,
    crop_batch_size: int = 64,
    show_progress: bool = False,
    classifier_summary: Optional[Dict[str, Any]] = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    annotations, images_by_id, anns_by_image_id, categories, images_dir = _load_generated_dataset(dataset_dir)
    metadata_summary_path = Path(dataset_dir) / "metadata" / "summary.json"
    metadata_summary = load_json(metadata_summary_path) if metadata_summary_path.is_file() else {}
    size_thresholds = np.asarray(metadata_summary.get("size_bin_thresholds", [0.002, 0.01]), dtype=np.float32)
    classifier_summary = dict(classifier_summary or {})
    classifier_mode = str(classifier_summary.get("classifier_mode", "binary"))
    per_class_thresholds = classifier_summary.get("per_class_thresholds", {}) if classifier_mode == "multiclass" else {}
    background_class_index = int(classifier_summary.get("background_class_index", -1)) if classifier_mode == "multiclass" else -1
    category_id_to_model_index = {
        int(k): int(v) for k, v in classifier_summary.get("category_id_to_model_index", {}).items()
    } if classifier_mode == "multiclass" else {}
    model_index_to_category_id = {
        int(k): int(v) for k, v in classifier_summary.get("model_index_to_category_id", {}).items()
    } if classifier_mode == "multiclass" else {}
    confusion_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    filter_model.eval()
    device = torch.device(device)
    instance_rows: List[Dict[str, Any]] = []
    image_rows: List[Dict[str, Any]] = []
    iterator: Iterable[tuple[Any, dict]] = images_by_id.items()
    if show_progress:
        iterator = tqdm(list(iterator), desc="Auditing generated layouts")

    for image_id, image_info in iterator:
        image_path = images_dir / str(image_info["file_name"])
        image_array = _ensure_channel_first(np.load(image_path))
        image_tensor = torch.from_numpy(image_array).to(torch.float32)
        _, image_h, image_w = image_tensor.shape
        anns = list(anns_by_image_id.get(image_id, []))
        crops: List[torch.Tensor] = []
        crop_payloads: List[Dict[str, Any]] = []

        for ann in anns:
            x, y, w, h = [float(v) for v in ann["bbox"]]
            box_xyxy = (x, y, x + w, y + h)
            area_ratio = (w * h) / max(1.0, float(image_w * image_h))
            size_bin = size_bin_name(area_ratio, size_thresholds)
            x1, y1, x2, y2 = expand_box_with_context(
                box_xyxy,
                image_w=image_w,
                image_h=image_h,
                context_ratio=context_ratio,
            )
            crops.append(image_tensor[:, y1:y2, x1:x2])
            crop_payloads.append(
                {
                    "annotation_id": ann["id"],
                    "generated_image_id": image_id,
                    "generated_file_name": image_info["file_name"],
                    "source_image_id": ann.get("source_image_id"),
                    "source_file_name": ann.get("source_file_name"),
                    "category_id": int(ann["category_id"]),
                    "category_name": categories.get(int(ann["category_id"]), str(int(ann["category_id"]))),
                    "bbox_xywh": [float(v) for v in ann["bbox"]],
                    "bbox_xyxy": [float(box_xyxy[0]), float(box_xyxy[1]), float(box_xyxy[2]), float(box_xyxy[3])],
                    "crop_box_xyxy": [int(x1), int(y1), int(x2), int(y2)],
                    "size_bin": size_bin,
                    "normalized_area_ratio": float(area_ratio),
                }
            )

        probs_chunks: List[np.ndarray] = []
        for start_idx in range(0, len(crops), max(1, int(crop_batch_size))):
            crop_batch = _prepare_crop_batch(
                crops[start_idx:start_idx + max(1, int(crop_batch_size))],
                input_size=int(filter_input_size),
                device=device,
            )
            logits = filter_model(crop_batch).detach().cpu()
            if classifier_mode == "multiclass":
                probs_chunks.append(torch.softmax(logits, dim=1).numpy())
            else:
                probs_chunks.append(torch.sigmoid(logits).numpy())
        probs = np.concatenate(probs_chunks, axis=0) if probs_chunks else (
            np.zeros((0, background_class_index + 1), dtype=np.float32) if classifier_mode == "multiclass" else np.asarray([], dtype=np.float32)
        )

        n_positive = 0
        for payload_idx, payload in enumerate(crop_payloads):
            if classifier_mode == "multiclass":
                prob_vector = probs[payload_idx]
                predicted_model_index = int(np.argmax(prob_vector))
                predicted_category_id = None if predicted_model_index == background_class_index else model_index_to_category_id.get(predicted_model_index)
                predicted_category_name = "background" if predicted_model_index == background_class_index else categories.get(
                    int(predicted_category_id),
                    str(predicted_category_id),
                )
                expected_category_id = int(payload["category_id"])
                expected_model_index = int(category_id_to_model_index.get(expected_category_id, expected_category_id))
                expected_class_probability = float(prob_vector[expected_model_index])
                expected_class_threshold = float(per_class_thresholds.get(str(expected_category_id), 0.5))
                is_background_prediction = predicted_model_index == background_class_index
                is_class_match = predicted_model_index == expected_model_index
                passes_expected_class_threshold = expected_class_probability >= expected_class_threshold
                is_positive = bool(is_class_match and passes_expected_class_threshold and not is_background_prediction)
                n_positive += int(is_positive)
                payload["predicted_category_id"] = predicted_category_id
                payload["predicted_category_name"] = predicted_category_name
                payload["predicted_model_index"] = predicted_model_index
                payload["predicted_probability"] = float(prob_vector[predicted_model_index])
                payload["expected_category_id"] = expected_category_id
                payload["expected_category_name"] = payload["category_name"]
                payload["expected_class_probability"] = expected_class_probability
                payload["expected_class_threshold"] = expected_class_threshold
                payload["is_background_prediction"] = bool(is_background_prediction)
                payload["is_class_match"] = bool(is_class_match)
                payload["passes_expected_class_threshold"] = bool(passes_expected_class_threshold)
                payload["threshold"] = expected_class_threshold
                payload["probability"] = expected_class_probability
                payload["is_positive"] = is_positive
                confusion_counts[str(expected_category_id)][str(predicted_category_id if predicted_category_id is not None else "background")] += 1
            else:
                prob = float(probs[payload_idx])
                is_positive = bool(prob >= float(threshold))
                n_positive += int(is_positive)
                expected_category_id = int(payload["category_id"])
                expected_category_name = payload["category_name"]
                predicted_category_id = expected_category_id if is_positive else None
                predicted_category_name = expected_category_name if is_positive else "background"
                payload["probability"] = prob
                payload["threshold"] = float(threshold)
                payload["is_positive"] = is_positive
                payload["predicted_category_id"] = predicted_category_id
                payload["predicted_category_name"] = predicted_category_name
                payload["predicted_model_index"] = None
                payload["predicted_probability"] = prob if is_positive else 1.0 - prob
                payload["expected_category_id"] = expected_category_id
                payload["expected_category_name"] = expected_category_name
                payload["expected_class_probability"] = prob
                payload["expected_class_threshold"] = float(threshold)
                payload["is_background_prediction"] = bool(not is_positive)
                payload["is_class_match"] = bool(is_positive)
                payload["passes_expected_class_threshold"] = bool(is_positive)
                confusion_counts[str(expected_category_id)][str(predicted_category_id if predicted_category_id is not None else "background")] += 1
            instance_rows.append(payload)

        valid_fraction = float(n_positive / max(1, len(crop_payloads))) if crop_payloads else 0.0
        image_rows.append(
            {
                "generated_image_id": image_id,
                "generated_file_name": image_info["file_name"],
                "source_image_id": crop_payloads[0]["source_image_id"] if crop_payloads else None,
                "source_file_name": crop_payloads[0]["source_file_name"] if crop_payloads else None,
                "n_instances": len(crop_payloads),
                "n_positive_instances": n_positive,
                "n_negative_instances": len(crop_payloads) - n_positive,
                "valid_fraction": float(valid_fraction),
                "combo_keys": [combo_key(payload["category_name"], payload["size_bin"]) for payload in crop_payloads],
            }
        )

    stats = summarize_instance_statistics(instance_rows)
    stats["image_level"] = {
        "total_image_count": len(image_rows),
        "mean_valid_fraction": float(np.mean([float(row["valid_fraction"]) for row in image_rows])) if image_rows else 0.0,
        "median_valid_fraction": float(np.median([float(row["valid_fraction"]) for row in image_rows])) if image_rows else 0.0,
        "threshold": float(threshold) if classifier_mode != "multiclass" else None,
    }
    stats["generation_metadata"] = {
        "dataset_dir": str(dataset_dir),
        "size_bin_thresholds": [float(v) for v in size_thresholds.tolist()],
        "dataset_id": str(metadata_summary.get("dataset_id", "flir_private_proxy_alignment_v18")),
        "source_split": str(metadata_summary.get("split", "train")),
    }
    if classifier_mode == "multiclass":
        background_predictions = sum(int(row.get("is_background_prediction", False)) for row in instance_rows)
        wrong_class_predictions = sum(
            int((not row.get("is_background_prediction", False)) and (not row.get("is_class_match", False)))
            for row in instance_rows
        )
        below_threshold_predictions = sum(
            int(row.get("is_class_match", False) and (not row.get("passes_expected_class_threshold", True)))
            for row in instance_rows
        )
        exact_match_pass = sum(int(row["is_positive"]) for row in instance_rows)
        total_instances = max(1, len(instance_rows))
        stats["multiclass"] = {
            "background_prediction_count": background_predictions,
            "background_prediction_rate": background_predictions / total_instances,
            "wrong_class_count": wrong_class_predictions,
            "wrong_class_rate": wrong_class_predictions / total_instances,
            "below_threshold_count": below_threshold_predictions,
            "below_threshold_rate": below_threshold_predictions / total_instances,
            "exact_match_threshold_pass_count": exact_match_pass,
            "exact_match_threshold_pass_rate": exact_match_pass / total_instances,
            "per_class_thresholds": {str(k): float(v) for k, v in per_class_thresholds.items()},
            "confusion_counts": {str(k): {str(pk): int(pv) for pk, pv in pred.items()} for k, pred in confusion_counts.items()},
        }
    else:
        stats["binary"] = {
            "confusion_counts": {str(k): {str(pk): int(pv) for pk, pv in pred.items()} for k, pred in confusion_counts.items()},
        }
    return instance_rows, image_rows, stats


def export_audit_results(
    *,
    output_dir: str | Path,
    summary: Dict[str, Any],
    instance_rows: Sequence[Dict[str, Any]],
    image_rows: Sequence[Dict[str, Any]],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "per_instance_manifest.jsonl", instance_rows)
    write_jsonl(output_dir / "per_image_manifest.jsonl", image_rows)
    write_csv(output_dir / "per_instance_manifest.csv", list(instance_rows))
    write_csv(output_dir / "per_image_manifest.csv", list(image_rows))
    checkpoint_summary = dict(summary.get("checkpoint_summary", {}))
    per_instance_analysis = {
        "stats": summary["stats"],
        "confusion_matrix": build_instance_confusion_matrix(
            instance_rows,
            checkpoint_summary=checkpoint_summary,
        ),
        "category_examples": build_category_examples(
            instance_rows=instance_rows,
            dataset_dir=summary["generated_dataset_dir"],
            checkpoint_summary=checkpoint_summary,
            output_dir=output_dir / "per_instance" / "examples",
        ),
    }
    per_image_analysis = build_per_image_threshold_sweep_analysis(
        instance_rows=instance_rows,
        image_rows=image_rows,
        sweep_text=str(summary.get("min_valid_object_fraction_sweep", "")),
        output_dir=output_dir / "per_image",
        reference_dataset_id=str(summary["stats"].get("generation_metadata", {}).get("dataset_id", "flir_private_proxy_alignment_v18")),
        reference_split="train",
        size_thresholds=np.asarray(
            summary["stats"].get("generation_metadata", {}).get("size_bin_thresholds", [0.002, 0.01]),
            dtype=np.float32,
        ),
        max_discarded_valid_threshold=float(summary.get("max_discarded_valid_threshold", 1.0)),
        score_alpha=float(summary.get("score_alpha", 1.0)),
        score_beta=float(summary.get("score_beta", 1.0)),
    )
    summary_with_charts = dict(summary)
    summary_with_charts["per_instance_analysis"] = per_instance_analysis
    summary_with_charts["per_image_analysis"] = per_image_analysis
    recommended_value = per_image_analysis["threshold_selection"]["recommended_min_valid_object_fraction"]
    summary_with_charts["recommended_min_valid_object_fraction"] = recommended_value
    summary_with_charts["stats"] = dict(summary_with_charts["stats"])
    summary_with_charts["stats"]["image_level"] = _build_image_level_stats_for_threshold(
        image_rows,
        min_valid_object_fraction=(
            None if recommended_value is None else float(recommended_value)
        ),
    )
    chart_paths = generate_audit_charts(
        output_dir=output_dir,
        summary=summary_with_charts,
        instance_rows=instance_rows,
        image_rows=image_rows,
    )
    summary_with_charts["table_paths"] = [
        str(Path(per_image_analysis["reference_frequency_analysis"]["reference_table_path"]).relative_to(output_dir)),
        str(Path(per_image_analysis["threshold_selection"]["metrics_table_path"]).relative_to(output_dir)),
        "per_instance_manifest.csv",
        "per_image_manifest.csv",
    ]
    per_image_chart_paths = [per_image_analysis["valid_image_ratio_sweep_chart"]] + [
        row["chart_path"] for row in per_image_analysis["thresholds"]
    ] + [
        row["discarded_category_ratio_chart_path"]
        for row in per_image_analysis["thresholds"]
        if row.get("discarded_category_ratio_chart_path")
    ] + [
        per_image_analysis["threshold_selection"]["score_curve_chart_path"],
        per_image_analysis["threshold_selection"]["global_discard_ratio_chart_path"],
        per_image_analysis["threshold_selection"]["tradeoff_scatter_chart_path"],
    ]
    summary_with_charts["chart_paths"] = [str(path.relative_to(output_dir)) for path in chart_paths] + [
        str(Path(path).relative_to(output_dir)) for path in per_image_chart_paths
    ]
    save_json(output_dir / "summary.json", summary_with_charts)


def resolve_filter_output_dir(
    *,
    dataset_dir: str | Path,
    output_dir: str | Path | None,
    filter_run_dir: str | Path | None,
    checkpoint_path: str | Path | None,
) -> Path:
    if output_dir:
        return Path(output_dir)
    if filter_run_dir:
        base_name = Path(filter_run_dir).name
    elif checkpoint_path:
        base_name = Path(checkpoint_path).stem
    else:
        base_name = "classifier"
    return Path(dataset_dir) / auto_filter_run_name(base_name)


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(axis="x", alpha=0.18)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig: plt.Figure, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_overall_metrics(charts_dir: Path, stats: Dict[str, Any]) -> List[Path]:
    paths: List[Path] = []
    overall = stats["overall"]
    image_level = stats["image_level"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(
        ["positive", "negative"],
        [overall["positive_count"], overall["negative_count"]],
        color=["#2a9d55", "#d1495b"],
    )
    axes[0].set_title("Instance outcomes")
    axes[0].set_ylabel("count")
    for idx, value in enumerate([overall["positive_ratio"], overall["negative_ratio"]]):
        axes[0].text(idx, [overall["positive_count"], overall["negative_count"]][idx], f"{value:.1%}", ha="center", va="bottom")

    if image_level.get("valid_image_count") is not None and image_level.get("invalid_image_count") is not None:
        axes[1].bar(
            ["valid", "invalid"],
            [image_level["valid_image_count"], image_level["invalid_image_count"]],
            color=["#3a78c2", "#a13d63"],
        )
        selected_value = image_level.get("selected_min_valid_object_fraction")
        title = "Image validity"
        if selected_value is not None:
            title += f" @ recommended={float(selected_value):.2f}"
        axes[1].set_title(title)
        axes[1].set_ylabel("count")
        axes[1].text(0, image_level["valid_image_count"], f"{image_level['valid_image_ratio']:.1%}", ha="center", va="bottom")
        axes[1].text(1, image_level["invalid_image_count"], f"{image_level['invalid_image_ratio']:.1%}", ha="center", va="bottom")
    else:
        axes[1].bar(
            ["mean", "median"],
            [image_level.get("mean_valid_fraction", 0.0), image_level.get("median_valid_fraction", 0.0)],
            color=["#3a78c2", "#72b7b2"],
        )
        axes[1].set_title("Valid-fraction summary")
        axes[1].set_ylabel("fraction")
        axes[1].set_ylim(0.0, 1.0)
    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    paths.append(_save_figure(fig, charts_dir / "overall_metrics.png"))
    return paths


def _plot_probability_and_valid_fraction(charts_dir: Path, summary: Dict[str, Any], instance_rows: Sequence[Dict[str, Any]], image_rows: Sequence[Dict[str, Any]]) -> List[Path]:
    paths: List[Path] = []
    threshold_value = summary["threshold"]
    recommended_min_valid_object_fraction = summary.get("recommended_min_valid_object_fraction")
    classifier_mode = str(summary.get("checkpoint_summary", {}).get("classifier_mode", "binary"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    probabilities = np.asarray([float(row["probability"]) for row in instance_rows], dtype=np.float32)
    axes[0].hist(probabilities, bins=40, color="#4c78a8", alpha=0.85)
    probability_title = "Instance probability distribution"
    if classifier_mode != "multiclass":
        threshold = float(threshold_value)
        axes[0].axvline(threshold, color="#d1495b", linestyle="--", linewidth=2, label=f"threshold={threshold:.3f}")
    else:
        probability_title = "Expected-class probability distribution"
        axes[0].axvline(0.5, color="#999999", linestyle=":", linewidth=1.5, label="0.5 reference")
    axes[0].set_title(probability_title)
    axes[0].set_xlabel("classifier probability")
    axes[0].set_ylabel("count")
    axes[0].legend()

    valid_fractions = np.asarray([float(row["valid_fraction"]) for row in image_rows], dtype=np.float32)
    axes[1].hist(valid_fractions, bins=30, color="#72b7b2", alpha=0.85)
    if recommended_min_valid_object_fraction is not None:
        axes[1].axvline(
            float(recommended_min_valid_object_fraction),
            color="#2a9d55",
            linestyle="--",
            linewidth=2,
            label=f"recommended={float(recommended_min_valid_object_fraction):.2f}",
        )
    axes[1].set_title("Per-image valid fraction")
    axes[1].set_xlabel("valid object fraction")
    axes[1].set_ylabel("count")
    axes[1].legend()

    for ax in axes:
        ax.grid(axis="y", alpha=0.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    paths.append(_save_figure(fig, charts_dir / "probability_and_valid_fraction.png"))
    return paths


def _plot_multiclass_thresholds(charts_dir: Path, multiclass_stats: Dict[str, Any], checkpoint_summary: Dict[str, Any]) -> Optional[Path]:
    thresholds = multiclass_stats.get("per_class_thresholds", {})
    category_id_to_name = {str(k): str(v) for k, v in checkpoint_summary.get("category_id_to_name", {}).items()}
    if not thresholds:
        return None
    rows = []
    for category_id, threshold in thresholds.items():
        rows.append((category_id_to_name.get(str(category_id), str(category_id)), float(threshold)))
    rows.sort(key=lambda item: item[1])
    labels = [row[0] for row in rows]
    values = [row[1] for row in rows]
    height = max(4.0, 0.35 * len(labels) + 1.5)
    fig, ax = plt.subplots(1, 1, figsize=(10, height))
    y = np.arange(len(labels))
    ax.barh(y, values, color="#7a5cff")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("saved threshold")
    ax.set_title("Per-class saved thresholds")
    ax.invert_yaxis()
    _style_axes(ax)
    return _save_figure(fig, charts_dir / "per_class_thresholds.png")


def _plot_matrix_heatmap(
    *,
    charts_dir: Path,
    matrix: np.ndarray,
    labels_x: Sequence[str],
    labels_y: Sequence[str],
    title: str,
    path_name: str,
    cmap: str,
) -> Optional[Path]:
    if matrix.size == 0 or not labels_x or not labels_y:
        return None
    fig, ax = plt.subplots(
        1,
        1,
        figsize=(max(7, 0.45 * len(labels_x) + 2), max(5, 0.4 * len(labels_y) + 2)),
    )
    im = ax.imshow(matrix, aspect="auto", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("expected class")
    ax.set_xticks(np.arange(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=75, ha="right")
    ax.set_yticks(np.arange(len(labels_y)))
    ax.set_yticklabels(labels_y)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save_figure(fig, charts_dir / path_name)


def _plot_multiclass_confusion(
    charts_dir: Path,
    multiclass_stats: Dict[str, Any],
    checkpoint_summary: Dict[str, Any],
) -> List[Path]:
    confusion = multiclass_stats.get("confusion_counts", {})
    if not confusion:
        return []
    category_id_to_name = {str(k): str(v) for k, v in checkpoint_summary.get("category_id_to_name", {}).items()}
    expected_ids = sorted(confusion.keys(), key=str)
    predicted_ids = sorted({pred for row in confusion.values() for pred in row.keys()}, key=str)
    expected_labels = [category_id_to_name.get(str(cid), str(cid)) for cid in expected_ids]
    predicted_labels = [
        "background" if str(cid) == "background" else category_id_to_name.get(str(cid), str(cid))
        for cid in predicted_ids
    ]
    count_matrix = np.zeros((len(expected_ids), len(predicted_ids)), dtype=np.float32)
    for exp_idx, exp_id in enumerate(expected_ids):
        row = confusion.get(exp_id, {})
        for pred_idx, pred_id in enumerate(predicted_ids):
            count_matrix[exp_idx, pred_idx] = float(row.get(pred_id, 0))
    row_totals = count_matrix.sum(axis=1, keepdims=True)
    normalized_matrix = np.divide(
        count_matrix,
        np.maximum(row_totals, 1.0),
        out=np.zeros_like(count_matrix, dtype=np.float32),
        where=row_totals > 0,
    )
    chart_paths: List[Path] = []
    counts_path = _plot_matrix_heatmap(
        charts_dir=charts_dir,
        matrix=count_matrix,
        labels_x=predicted_labels,
        labels_y=expected_labels,
        title="Confusion heatmap (counts)",
        path_name="multiclass_confusion_heatmap_counts.png",
        cmap="viridis",
    )
    if counts_path is not None:
        chart_paths.append(counts_path)
    normalized_path = _plot_matrix_heatmap(
        charts_dir=charts_dir,
        matrix=normalized_matrix,
        labels_x=predicted_labels,
        labels_y=expected_labels,
        title="Confusion heatmap (row-normalized)",
        path_name="multiclass_confusion_heatmap_normalized.png",
        cmap="magma",
    )
    if normalized_path is not None:
        chart_paths.append(normalized_path)
    return chart_paths


def _plot_general_confusion_matrix(charts_dir: Path, confusion_matrix: Dict[str, Any]) -> List[Path]:
    labels = list(confusion_matrix.get("labels", []))
    count_matrix = np.asarray(confusion_matrix.get("matrix_counts", confusion_matrix.get("matrix", [])), dtype=np.float32)
    normalized_matrix = np.asarray(confusion_matrix.get("matrix_normalized", []), dtype=np.float32)
    chart_paths: List[Path] = []
    counts_path = _plot_matrix_heatmap(
        charts_dir=charts_dir,
        matrix=count_matrix,
        labels_x=labels,
        labels_y=labels,
        title="Confusion matrix (counts)",
        path_name="confusion_matrix_counts.png",
        cmap="viridis",
    )
    if counts_path is not None:
        chart_paths.append(counts_path)
    normalized_path = _plot_matrix_heatmap(
        charts_dir=charts_dir,
        matrix=normalized_matrix,
        labels_x=labels,
        labels_y=labels,
        title="Confusion matrix (row-normalized)",
        path_name="confusion_matrix_normalized.png",
        cmap="magma",
    )
    if normalized_path is not None:
        chart_paths.append(normalized_path)
    return chart_paths


def _plot_group_stacked_bars(rows: Sequence[Dict[str, Any]], *, label_key: str, title: str, path: Path) -> Path:
    sorted_rows = sorted(rows, key=lambda row: (-int(row["total_count"]), str(row[label_key])))
    labels = [str(row[label_key]) for row in sorted_rows]
    positive = np.asarray([int(row["positive_count"]) for row in sorted_rows], dtype=np.int32)
    negative = np.asarray([int(row["negative_count"]) for row in sorted_rows], dtype=np.int32)

    height = max(4.0, 0.35 * len(labels) + 1.5)
    fig, ax = plt.subplots(1, 1, figsize=(12, height))
    y = np.arange(len(labels))
    ax.barh(y, positive, color="#2a9d55", label="positive")
    ax.barh(y, negative, left=positive, color="#d1495b", label="negative")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("instance count")
    ax.set_title(title)
    ax.legend(loc="lower right")
    _style_axes(ax)
    return _save_figure(fig, path)


def _plot_group_ratio_bars(rows: Sequence[Dict[str, Any]], *, label_key: str, title: str, path: Path) -> Path:
    sorted_rows = sorted(rows, key=lambda row: (float(row["positive_ratio"]), -int(row["total_count"]), str(row[label_key])))
    labels = [str(row[label_key]) for row in sorted_rows]
    ratios = np.asarray([float(row["positive_ratio"]) for row in sorted_rows], dtype=np.float32)
    totals = np.asarray([int(row["total_count"]) for row in sorted_rows], dtype=np.int32)

    height = max(4.0, 0.35 * len(labels) + 1.5)
    fig, ax = plt.subplots(1, 1, figsize=(12, height))
    y = np.arange(len(labels))
    colors = plt.cm.RdYlGn(ratios)
    ax.barh(y, ratios, color=colors)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("positive ratio")
    ax.set_title(title)
    for idx, (ratio, total) in enumerate(zip(ratios.tolist(), totals.tolist())):
        ax.text(min(ratio + 0.015, 0.99), idx, f"n={total}", va="center", fontsize=8)
    _style_axes(ax)
    return _save_figure(fig, path)


def _plot_joint_heatmaps(charts_dir: Path, rows: Sequence[Dict[str, Any]]) -> List[Path]:
    paths: List[Path] = []
    if not rows:
        return paths

    parsed = []
    for row in rows:
        label = str(row["category_size_bin"])
        if " | " in label:
            category, size_bin = label.split(" | ", 1)
        else:
            category, size_bin = label, "unknown"
        parsed.append((category, size_bin, row))

    category_order = sorted({category for category, _size_bin, _row in parsed})
    size_order = ["small", "med", "big"]
    ratio_matrix = np.full((len(category_order), len(size_order)), np.nan, dtype=np.float32)
    count_matrix = np.zeros((len(category_order), len(size_order)), dtype=np.float32)
    category_to_idx = {name: idx for idx, name in enumerate(category_order)}
    size_to_idx = {name: idx for idx, name in enumerate(size_order)}

    for category, size_bin, row in parsed:
        if size_bin not in size_to_idx:
            continue
        ratio_matrix[category_to_idx[category], size_to_idx[size_bin]] = float(row["positive_ratio"])
        count_matrix[category_to_idx[category], size_to_idx[size_bin]] = float(row["total_count"])

    fig, axes = plt.subplots(1, 2, figsize=(12, max(4.5, 0.42 * len(category_order) + 1.5)))
    heatmaps = [
        (ratio_matrix, "Positive ratio by category x size", "RdYlGn", axes[0]),
        (np.log1p(count_matrix), "log(1 + count) by category x size", "Blues", axes[1]),
    ]
    for matrix, title, cmap, ax in heatmaps:
        im = ax.imshow(matrix, aspect="auto", cmap=cmap)
        ax.set_title(title)
        ax.set_xticks(np.arange(len(size_order)))
        ax.set_xticklabels(size_order)
        ax.set_yticks(np.arange(len(category_order)))
        ax.set_yticklabels(category_order)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    paths.append(_save_figure(fig, charts_dir / "category_size_heatmaps.png"))
    return paths


def generate_audit_charts(
    *,
    output_dir: str | Path,
    summary: Dict[str, Any],
    instance_rows: Sequence[Dict[str, Any]],
    image_rows: Sequence[Dict[str, Any]],
) -> List[Path]:
    output_dir = Path(output_dir)
    charts_dir = output_dir / "per_instance" / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)

    stats = summary["stats"]
    chart_paths: List[Path] = []
    chart_paths.extend(_plot_overall_metrics(charts_dir, stats))
    chart_paths.extend(_plot_probability_and_valid_fraction(charts_dir, summary, instance_rows, image_rows))
    chart_paths.append(
        _plot_group_stacked_bars(
            stats["by_category"],
            label_key="category",
            title="Per-category positive vs negative counts",
            path=charts_dir / "by_category_counts.png",
        )
    )
    chart_paths.append(
        _plot_group_ratio_bars(
            stats["by_category"],
            label_key="category",
            title="Per-category positive ratio",
            path=charts_dir / "by_category_positive_ratio.png",
        )
    )
    chart_paths.append(
        _plot_group_stacked_bars(
            stats["by_size_bin"],
            label_key="size_bin",
            title="Per-size positive vs negative counts",
            path=charts_dir / "by_size_bin_counts.png",
        )
    )
    chart_paths.append(
        _plot_group_ratio_bars(
            stats["by_size_bin"],
            label_key="size_bin",
            title="Per-size positive ratio",
            path=charts_dir / "by_size_bin_positive_ratio.png",
        )
    )
    chart_paths.extend(_plot_joint_heatmaps(charts_dir, stats["by_category_size_bin"]))
    multiclass_stats = stats.get("multiclass")
    checkpoint_summary = summary.get("checkpoint_summary", {})
    confusion_charts = _plot_general_confusion_matrix(
        charts_dir,
        build_instance_confusion_matrix(instance_rows, checkpoint_summary=checkpoint_summary),
    )
    chart_paths.extend(confusion_charts)
    if multiclass_stats:
        threshold_chart = _plot_multiclass_thresholds(charts_dir, multiclass_stats, checkpoint_summary)
        if threshold_chart is not None:
            chart_paths.append(threshold_chart)
        chart_paths.extend(_plot_multiclass_confusion(charts_dir, multiclass_stats, checkpoint_summary))
    return chart_paths

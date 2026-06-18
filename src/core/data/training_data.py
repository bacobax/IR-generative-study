"""Shared helpers for config-driven non-layout training datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from torch.utils.data import DataLoader, Dataset, Subset

from src.core.data.annotation_dataset import AnnotationFMDataset
from src.core.data.dataset_targets import resolve_dataset_target
from src.core.data.datasets import SingleChannelImageDataset
from src.core.data.transforms import ScheduledAugment256
from src.core.normalization import RAW_UINT16_PERCENTILE


@dataclass(frozen=True)
class ResolvedTrainingData:
    """Resolved dataset paths used by config-driven training CLIs."""

    train_dir: str
    val_dir: str
    train_annotations_path: Optional[str]
    val_annotations_path: Optional[str]
    normalization_mode: str
    train_subset_manifest: Optional[str] = None
    train_manifest_path: Optional[str] = None
    val_manifest_path: Optional[str] = None


@dataclass(frozen=True)
class NonLayoutTrainingData:
    """Built datasets and loaders for non-layout training."""

    train_base_dataset: Dataset
    eval_base_dataset: Dataset
    train_dataset: Dataset
    eval_dataset: Dataset
    train_loader: DataLoader
    eval_loader: DataLoader
    use_annotation_ds: bool


def resolve_training_data(data_config: Any) -> ResolvedTrainingData:
    """Resolve dataset directories, split annotations, and normalization mode."""
    train_dir = data_config.train_dir
    val_dir = data_config.val_dir
    train_annotations_path = getattr(data_config, "annotations_path", None)
    val_annotations_path = getattr(data_config, "annotations_path", None)
    normalization_mode = RAW_UINT16_PERCENTILE
    train_subset_manifest = getattr(data_config, "subset_manifest", None)
    train_manifest_path = None
    val_manifest_path = None

    dataset_id = getattr(data_config, "dataset_id", None)
    if dataset_id is not None:
        target = resolve_dataset_target(dataset_id)
        train_dir = str(target.split_dir("train"))
        val_dir = str(target.split_dir("val"))
        normalization_mode = target.normalization_mode
        train_manifest = target.manifest_path("train")
        val_manifest = target.manifest_path("val")
        train_manifest_path = str(train_manifest) if train_manifest is not None else None
        val_manifest_path = str(val_manifest) if val_manifest is not None else None

        if train_annotations_path is None and target.has_coco_annotations():
            train_annotations_path = str(target.annotations_path("train"))
            val_annotations_path = str(target.annotations_path("val"))

    # Explicit config override wins over the dataset_id default. Needed when the
    # FM input must match a VAE finetuned under a different normalization.
    override = getattr(data_config, "normalization_mode", None)
    if override:
        normalization_mode = override

    return ResolvedTrainingData(
        train_dir=train_dir,
        val_dir=val_dir,
        train_annotations_path=train_annotations_path,
        val_annotations_path=val_annotations_path,
        normalization_mode=normalization_mode,
        train_subset_manifest=train_subset_manifest,
        train_manifest_path=train_manifest_path,
        val_manifest_path=val_manifest_path,
    )


def apply_dataset_subset(dataset: Dataset, max_samples: Optional[int], strategy: str) -> Dataset:
    """Apply a deterministic debug subset without disturbing sample order."""
    if max_samples is None or max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    if strategy != "first_n":
        raise ValueError(
            f"Unsupported subset_strategy={strategy!r}. Only 'first_n' is supported."
        )
    return Subset(dataset, list(range(int(max_samples))))


def build_non_layout_dataloaders(
    *,
    data_config: Any,
    augment_config: Any,
    curriculum_config: Any,
    total_epochs: int,
    resolved_data: ResolvedTrainingData,
) -> NonLayoutTrainingData:
    """Build repo-native non-layout datasets and dataloaders."""
    aug_kwargs = dict(
        total_epochs=total_epochs,
        warmup_frac=augment_config.warmup_frac,
        ramp_frac=augment_config.ramp_frac,
        p_crop_warmup=augment_config.p_crop_warmup,
        p_crop_max=augment_config.p_crop_max,
        p_crop_final=augment_config.p_crop_final,
        p_rot_warmup=augment_config.p_rot_warmup,
        p_rot_max=augment_config.p_rot_max,
        p_rot_final=augment_config.p_rot_final,
        p_hflip_warmup=getattr(augment_config, "p_hflip_warmup", 0.0),
        p_hflip_max=getattr(augment_config, "p_hflip_max", 0.0),
        p_hflip_final=getattr(augment_config, "p_hflip_final", 0.0),
        image_size=data_config.image_size,
        normalization_mode=resolved_data.normalization_mode,
    )
    train_transform = ScheduledAugment256(**aug_kwargs)
    eval_transform = ScheduledAugment256(**aug_kwargs)

    use_annotation_ds = (
        resolved_data.train_annotations_path is not None
        and bool(getattr(curriculum_config, "enabled", False))
    )

    if use_annotation_ds:
        train_base_dataset = AnnotationFMDataset(
            root_dir=resolved_data.train_dir,
            annotations_path=resolved_data.train_annotations_path,
            text_mode=False,
            curriculum=curriculum_config,
            transform=train_transform,
            resize_target=data_config.image_size,
            normalization_mode=resolved_data.normalization_mode,
            subset_manifest=resolved_data.train_subset_manifest,
        )
        eval_base_dataset = AnnotationFMDataset(
            root_dir=resolved_data.val_dir,
            annotations_path=resolved_data.val_annotations_path,
            text_mode=False,
            curriculum=None,
            transform=eval_transform,
            resize_target=data_config.image_size,
            normalization_mode=resolved_data.normalization_mode,
        )
    else:
        train_base_dataset = SingleChannelImageDataset(
            root_dir=resolved_data.train_dir,
            transform=train_transform,
            subset_manifest=resolved_data.train_subset_manifest,
            manifest_path=resolved_data.train_manifest_path,
        )
        eval_base_dataset = SingleChannelImageDataset(
            root_dir=resolved_data.val_dir,
            transform=eval_transform,
            manifest_path=resolved_data.val_manifest_path,
        )

    train_dataset = apply_dataset_subset(
        train_base_dataset,
        getattr(data_config, "max_train_samples", None),
        getattr(data_config, "subset_strategy", "first_n"),
    )
    eval_dataset = apply_dataset_subset(
        eval_base_dataset,
        getattr(data_config, "max_val_samples", None),
        getattr(data_config, "subset_strategy", "first_n"),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=True,
    )

    return NonLayoutTrainingData(
        train_base_dataset=train_base_dataset,
        eval_base_dataset=eval_base_dataset,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        train_loader=train_loader,
        eval_loader=eval_loader,
        use_annotation_ds=use_annotation_ds,
    )

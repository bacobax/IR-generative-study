"""Dataset loading, preprocessing, and data pipeline utilities."""

from src.core.data.annotation_dataset import AnnotationFMDataset
from src.core.data.adapters import (
    DatasetAdapter,
    DatasetBuildRequest,
    DatasetBundle,
    RepoDatasetAdapter,
    canonical_batch_from_mapping,
    canonical_sample_from_mapping,
    normalize_batch,
    normalize_sample,
)
from src.core.data.datasets import (
    AnnotationLayoutDataset,
    BBoxConditioningDataset,
    NPYImageDataset,
    NPYStemDataset,
    TextImageDataset,
)
from src.core.data.foreground_background_dataset import (
    ForegroundBackgroundCropDataset,
    MultiClassCropDataset,
    build_balanced_sample_weights,
    collate_foreground_background_batch,
)
from src.core.data.layout_batching import collate_layout_batch
from src.core.data.schema import (
    CanonicalBatch,
    CanonicalSample,
    LayoutFields,
    SampleKeys,
    TextFields,
    TokenizerFields,
)
from src.core.data.subset_manifest import (
    filter_files_by_subset_manifest,
    load_subset_manifest_file_names,
    resolve_subset_manifest_path,
)
from src.core.data.training_data import (
    NonLayoutTrainingData,
    ResolvedTrainingData,
    apply_dataset_subset,
    build_non_layout_dataloaders,
    resolve_training_data,
)

__all__ = [
    "AnnotationFMDataset",
    "AnnotationLayoutDataset",
    "BBoxConditioningDataset",
    "CanonicalBatch",
    "CanonicalSample",
    "DatasetAdapter",
    "DatasetBuildRequest",
    "DatasetBundle",
    "ForegroundBackgroundCropDataset",
    "LayoutFields",
    "MultiClassCropDataset",
    "NonLayoutTrainingData",
    "NPYImageDataset",
    "NPYStemDataset",
    "RepoDatasetAdapter",
    "ResolvedTrainingData",
    "SampleKeys",
    "TextFields",
    "TextImageDataset",
    "TokenizerFields",
    "apply_dataset_subset",
    "build_balanced_sample_weights",
    "build_non_layout_dataloaders",
    "canonical_batch_from_mapping",
    "canonical_sample_from_mapping",
    "collate_foreground_background_batch",
    "collate_layout_batch",
    "filter_files_by_subset_manifest",
    "load_subset_manifest_file_names",
    "normalize_batch",
    "normalize_sample",
    "resolve_training_data",
    "resolve_subset_manifest_path",
]

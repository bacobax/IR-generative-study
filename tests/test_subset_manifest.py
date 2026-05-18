from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.core.data.datasets import NPYImageDataset
from src.core.data.subset_manifest import (
    filter_files_by_subset_manifest,
    load_subset_manifest_file_names,
)


def _write_manifest(path: Path, names: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"samples": [{"path": name} for name in names]}),
        encoding="utf-8",
    )
    return path


def test_subset_manifest_preserves_order_and_deduplicates(tmp_path: Path) -> None:
    manifest = _write_manifest(
        tmp_path / "subsets" / "train_subset.json",
        ["c.npy", "a.npy", "c.npy", "b.npy"],
    )

    assert load_subset_manifest_file_names(manifest) == ["c.npy", "a.npy", "b.npy"]
    assert filter_files_by_subset_manifest(
        ["a.npy", "b.npy", "c.npy"],
        "train_subset.json",
        split_dir=tmp_path / "train",
    ) == ["c.npy", "a.npy", "b.npy"]


def test_subset_manifest_reports_missing_files(tmp_path: Path) -> None:
    manifest = _write_manifest(tmp_path / "subset.json", ["present.npy", "missing.npy"])

    with pytest.raises(FileNotFoundError, match="missing.npy"):
        filter_files_by_subset_manifest(["present.npy"], manifest, split_dir=tmp_path)


def test_npy_image_dataset_uses_subset_manifest_order(tmp_path: Path) -> None:
    for index, name in enumerate(["a.npy", "b.npy", "c.npy"]):
        np.save(tmp_path / name, np.full((4, 4), index, dtype=np.uint8))
    manifest = _write_manifest(tmp_path / "subsets" / "train_first2.json", ["c.npy", "a.npy"])

    dataset = NPYImageDataset(str(tmp_path), subset_manifest=str(manifest))

    assert dataset.files == ["c.npy", "a.npy"]
    assert len(dataset) == 2

import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.core.data.adapters import DatasetBuildRequest, RepoDatasetAdapter
from src.core.data.dataset_targets import (
    resolve_dataset_target,
    supported_dataset_ids,
    target_to_dataset_build_request,
)
from src.core.normalization import RAW_UINT16_PERCENTILE, UINT8_LINEAR
from src.core.paths import flir_root, v18_root
from src.core.registry import REGISTRIES


def _install_fake_collate_modules(monkeypatch: pytest.MonkeyPatch):
    def fake_layout_collate(batch):
        return {"layout": batch}

    def fake_sd_layout_collate(batch):
        return {"sd_layout": batch}

    layout_module = ModuleType("src.core.data.layout_batching")
    layout_module.collate_layout_batch = fake_layout_collate
    sd_layout_module = ModuleType("src.algorithms.stable_diffusion.layout_data")
    sd_layout_module.collate_sd_layout_batch = fake_sd_layout_collate
    monkeypatch.setitem(sys.modules, "src.core.data.layout_batching", layout_module)
    monkeypatch.setitem(
        sys.modules,
        "src.algorithms.stable_diffusion.layout_data",
        sd_layout_module,
    )
    return fake_layout_collate, fake_sd_layout_collate


def test_v18_target_keeps_legacy_paths_and_normalization() -> None:
    target = resolve_dataset_target("v18")

    assert target.dataset_id == "v18"
    assert target.root == v18_root()
    assert target.normalization_mode == RAW_UINT16_PERCENTILE
    assert target.split_dir("train") == v18_root() / "train"
    assert target.split_dir("val") == v18_root() / "val"
    assert target.annotations_path("train") == (
        v18_root() / "train" / "annotations.json"
    )
    assert target.annotations_path("val") == v18_root() / "val" / "annotations.json"


def test_flir_proxy_target_keeps_legacy_paths_and_normalization() -> None:
    target = resolve_dataset_target("flir_private_proxy_alignment_v18")

    assert target.dataset_id == "flir_private_proxy_alignment_v18"
    assert target.root == flir_root()
    assert target.normalization_mode == UINT8_LINEAR
    assert target.split_dir("train") == flir_root() / "train"
    assert target.split_dir("val") == flir_root() / "val"
    assert target.annotations_path("train") == (
        flir_root() / "train" / "annotations.json"
    )
    assert target.annotations_path("val") == flir_root() / "val" / "annotations.json"


def test_adapter_split_and_annotation_resolution_for_arbitrary_split(tmp_path: Path) -> None:
    adapter = RepoDatasetAdapter(
        dataset_id="unit",
        root=tmp_path / "dataset",
        normalization_mode=RAW_UINT16_PERCENTILE,
    )

    assert adapter.split_dir("train") == tmp_path / "dataset" / "train"
    assert adapter.split_dir("holdout") == tmp_path / "dataset" / "holdout"
    assert adapter.annotations_path("holdout") == (
        tmp_path / "dataset" / "holdout" / "annotations.json"
    )


def test_adapter_loads_category_metadata_when_annotations_exist(tmp_path: Path) -> None:
    split_dir = tmp_path / "dataset" / "train"
    split_dir.mkdir(parents=True)
    (split_dir / "annotations.json").write_text(
        json.dumps(
            {
                "images": [],
                "annotations": [],
                "categories": [
                    {"id": 0, "name": "person"},
                    {"id": 2, "name": "car"},
                ],
            }
        ),
        encoding="utf-8",
    )
    adapter = RepoDatasetAdapter(
        dataset_id="unit",
        root=tmp_path / "dataset",
        normalization_mode=RAW_UINT16_PERCENTILE,
    )

    assert adapter.category_metadata("train") == {0: "person", 2: "car"}
    assert adapter.category_metadata("val") == {}


def test_adapter_resolves_task_specific_collate_functions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout_collate, sd_layout_collate = _install_fake_collate_modules(monkeypatch)
    adapter = RepoDatasetAdapter(
        dataset_id="unit",
        root=tmp_path / "dataset",
        normalization_mode=RAW_UINT16_PERCENTILE,
    )

    assert adapter.collate_fn_for_task("layout") is layout_collate
    assert adapter.collate_fn_for_task("fm_layout") is layout_collate
    assert adapter.collate_fn_for_task("sd_layout") is sd_layout_collate
    assert adapter.collate_fn_for_task("unknown") is None
    assert adapter.collate_fn_for_task(None) is None


def test_registered_dataset_adapter_builds_metadata_without_dataset_rewrite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout_collate, _ = _install_fake_collate_modules(monkeypatch)
    adapter = REGISTRIES.dataset_adapter["v18"]
    bundle = adapter.build(
        DatasetBuildRequest(dataset_id="v18", split="val", options={"task": "layout"})
    )

    assert bundle.dataset is None
    assert bundle.collate_fn is layout_collate
    assert bundle.normalization_mode == RAW_UINT16_PERCENTILE
    assert bundle.metadata["dataset_id"] == "v18"
    assert bundle.metadata["split_dir"].endswith("data/raw/v18/val")
    assert bundle.metadata["annotations_path"].endswith(
        "data/raw/v18/val/annotations.json"
    )


def test_target_to_dataset_build_request_includes_adapter_resolved_fields() -> None:
    target = resolve_dataset_target("flir_private_proxy_alignment_v18")
    request = target_to_dataset_build_request(target, split="test")

    assert request.dataset_id == "flir_private_proxy_alignment_v18"
    assert request.root_dir == flir_root() / "test"
    assert request.annotations_path == flir_root() / "test" / "annotations.json"
    assert request.options["normalization_mode"] == UINT8_LINEAR
    assert request.metadata["split_dir"].endswith(
        "data/raw/flir_private_proxy_alignment_v18/test"
    )
    assert request.metadata["annotations_path"].endswith(
        "data/raw/flir_private_proxy_alignment_v18/test/annotations.json"
    )


def test_supported_dataset_ids_and_unknown_id_error_are_backward_compatible() -> None:
    assert set(supported_dataset_ids()) == {"v18", "flir_private_proxy_alignment_v18"}

    with pytest.raises(
        ValueError,
        match="Unknown dataset_id='missing'.*flir_private_proxy_alignment_v18.*v18",
    ):
        resolve_dataset_target("missing")

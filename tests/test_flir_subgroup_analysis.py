"""Tests for the subgroup analysis backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.analysis.flir_subgroup.analysis import (
    build_per_class_image_count_distribution,
    canonical_subgroup_label,
    compute_collateral_damage,
    compute_holdout_table,
    get_union_holdout_image_keys,
    parse_subgroup_label,
)
from src.analysis.flir_subgroup.app import create_app
from src.analysis.flir_subgroup.context import build_analysis_context, clear_analysis_context_cache
from src.analysis.flir_subgroup.datasets import DatasetConfig, resolve_dataset_config


def _write_npy(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.full((100, 100), value, dtype=np.uint8))


def _create_synthetic_flir_dataset(root: Path) -> Path:
    train_dir = root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    categories = [
        {"id": 1, "name": "car"},
        {"id": 2, "name": "person"},
        {"id": 3, "name": "light"},
    ]
    images = [
        {"id": "img1", "width": 100, "height": 100, "file_name": "img1.npy"},
        {"id": "img2", "width": 100, "height": 100, "file_name": "img2.npy"},
        {"id": "img3", "width": 100, "height": 100, "file_name": "img3.npy"},
        {"id": "img4", "width": 100, "height": 100, "file_name": "img4.npy"},
    ]
    for idx, image in enumerate(images, start=1):
        _write_npy(train_dir / str(image["file_name"]), value=30 * idx)

    annotations = []

    def add_box(image_id: str, category_id: int, x: int, y: int, w: int, h: int) -> None:
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
            }
        )

    add_box("img1", 1, 8, 35, 30, 30)
    add_box("img1", 1, 12, 40, 30, 30)
    add_box("img1", 1, 16, 45, 30, 30)
    add_box("img1", 2, 70, 20, 10, 10)

    add_box("img2", 1, 40, 35, 20, 20)
    add_box("img2", 1, 45, 40, 20, 20)
    add_box("img2", 3, 82, 20, 8, 8)

    add_box("img3", 2, 35, 30, 18, 18)
    add_box("img3", 2, 38, 34, 18, 18)
    add_box("img3", 3, 8, 20, 28, 28)

    add_box("img4", 1, 72, 35, 12, 12)
    add_box("img4", 2, 10, 42, 26, 26)
    add_box("img4", 2, 14, 46, 26, 26)

    (train_dir / "annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}),
        encoding="utf-8",
    )
    (train_dir / "captions.json").write_text(
        json.dumps(
            {
                "img1": "cars dominate the frame",
                "img2": "two medium cars and a light",
                "img3": "people and a large light",
                "img4": "people with a small car",
            }
        ),
        encoding="utf-8",
    )
    return root


def _create_synthetic_v18_dataset(root: Path) -> Path:
    train_dir = root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    categories = [{"id": 1, "name": "person"}]
    images = [
        {"id": "p1", "width": 100, "height": 100, "file_name": "p1.npy"},
        {"id": "p2", "width": 100, "height": 100, "file_name": "p2.npy"},
        {"id": "p3", "width": 100, "height": 100, "file_name": "p3.npy"},
    ]
    for idx, image in enumerate(images, start=1):
        _write_npy(train_dir / str(image["file_name"]), value=50 * idx)

    annotations = []

    def add_box(image_id: str, x: int, y: int, w: int, h: int) -> None:
        annotations.append(
            {
                "id": len(annotations) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0,
            }
        )

    add_box("p1", 5, 15, 10, 12)
    add_box("p1", 35, 20, 20, 24)
    add_box("p1", 66, 25, 28, 32)
    add_box("p2", 10, 35, 12, 14)
    add_box("p2", 40, 40, 22, 22)
    add_box("p3", 70, 18, 30, 30)
    add_box("p3", 72, 50, 18, 18)

    (train_dir / "annotations.json").write_text(
        json.dumps({"images": images, "annotations": annotations, "categories": categories}),
        encoding="utf-8",
    )
    return root


def _dataset_registry(tmp_path: Path) -> dict[str, DatasetConfig]:
    return {
        "flir_private_proxy_alignment_v18": DatasetConfig(
            dataset_id="flir_private_proxy_alignment_v18",
            label="Synthetic FLIR",
            description="Synthetic multi-class FLIR-like dataset.",
            root=_create_synthetic_flir_dataset(tmp_path / "flir"),
            is_default=True,
        ),
        "v18": DatasetConfig(
            dataset_id="v18",
            label="Synthetic v18",
            description="Synthetic single-class v18-like dataset.",
            root=_create_synthetic_v18_dataset(tmp_path / "v18"),
            is_default=False,
        ),
    }


def _manual_phase_tables() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    target = canonical_subgroup_label("car", "large")
    secondary = canonical_subgroup_label("person", "small")
    image_stats_df = pd.DataFrame(
        [
            {"image_key": "train::1", "split": "train", "image_id": "1", "total_object_count": 4, "image_density": 0.4},
            {"image_key": "train::2", "split": "train", "image_id": "2", "total_object_count": 4, "image_density": 0.3},
            {"image_key": "train::3", "split": "train", "image_id": "3", "total_object_count": 3, "image_density": 0.2},
        ]
    )
    image_subgroup_df = pd.DataFrame(
        [
            {"image_key": "train::1", "split": "train", "image_id": "1", "subgroup": target, "subgroup_count": 3, "total_object_count": 4, "dominance_ratio": 0.75},
            {"image_key": "train::1", "split": "train", "image_id": "1", "subgroup": secondary, "subgroup_count": 1, "total_object_count": 4, "dominance_ratio": 0.25},
            {"image_key": "train::2", "split": "train", "image_id": "2", "subgroup": target, "subgroup_count": 1, "total_object_count": 4, "dominance_ratio": 0.25},
            {"image_key": "train::2", "split": "train", "image_id": "2", "subgroup": secondary, "subgroup_count": 2, "total_object_count": 4, "dominance_ratio": 0.50},
            {"image_key": "train::3", "split": "train", "image_id": "3", "subgroup": secondary, "subgroup_count": 1, "total_object_count": 3, "dominance_ratio": 1 / 3},
        ]
    )
    instance_df = pd.DataFrame(
        [
            {"image_key": "train::1", "subgroup": target, "class_label": "car"},
            {"image_key": "train::1", "subgroup": target, "class_label": "car"},
            {"image_key": "train::1", "subgroup": target, "class_label": "car"},
            {"image_key": "train::1", "subgroup": secondary, "class_label": "person"},
            {"image_key": "train::2", "subgroup": target, "class_label": "car"},
            {"image_key": "train::2", "subgroup": secondary, "class_label": "person"},
            {"image_key": "train::2", "subgroup": secondary, "class_label": "person"},
            {"image_key": "train::2", "subgroup": canonical_subgroup_label("light", "small"), "class_label": "light"},
            {"image_key": "train::3", "subgroup": canonical_subgroup_label("light", "small"), "class_label": "light"},
            {"image_key": "train::3", "subgroup": secondary, "class_label": "person"},
            {"image_key": "train::3", "subgroup": canonical_subgroup_label("car", "small"), "class_label": "car"},
        ]
    )
    return image_stats_df, image_subgroup_df, instance_df


def test_subgroup_label_round_trip() -> None:
    subgroup = canonical_subgroup_label("car", "large", "center")
    assert subgroup == "class=car | size=large | pos=center"
    assert parse_subgroup_label(subgroup) == {
        "class_label": "car",
        "size_bin": "large",
        "position_bin": "center",
    }


def test_holdout_and_collateral_helpers() -> None:
    image_stats_df, image_subgroup_df, instance_df = _manual_phase_tables()
    target = canonical_subgroup_label("car", "large")

    holdout_df = compute_holdout_table(image_subgroup_df, image_stats_df, target, thresholds=[0.2, 0.5])
    assert holdout_df.to_dict(orient="records") == [
        {
            "subgroup": target,
            "tau": 0.2,
            "heldout_n_images": 2,
            "heldout_fraction": 2 / 3,
            "mean_target_count": 2.0,
            "median_target_count": 2.0,
            "mean_dominance": 0.5,
        },
        {
            "subgroup": target,
            "tau": 0.5,
            "heldout_n_images": 1,
            "heldout_fraction": 1 / 3,
            "mean_target_count": 3.0,
            "median_target_count": 3.0,
            "mean_dominance": 0.75,
        },
    ]

    damage_df, summary = compute_collateral_damage(instance_df, image_subgroup_df, target, tau=0.5)
    assert summary["heldout_n_images"] == 1
    assert round(summary["collateral_other_loss_frac"], 6) == round(1 / 7, 6)
    target_row = damage_df.loc[damage_df["subgroup"] == target].iloc[0]
    assert int(target_row["count_loss"]) == 3
    assert float(target_row["loss_fraction"]) == 0.75


def test_union_holdout_and_class_count_distribution() -> None:
    _, image_subgroup_df, instance_df = _manual_phase_tables()
    target = canonical_subgroup_label("car", "large")
    other = canonical_subgroup_label("person", "small")

    union_keys = get_union_holdout_image_keys(image_subgroup_df, [target, other], tau=0.5)
    assert union_keys == ["train::1", "train::2"]

    dist_df = build_per_class_image_count_distribution(instance_df, heldout_keys=["train::1"])
    car_rows = dist_df.loc[dist_df["class_label"] == "car"].set_index("instance_count")
    assert int(car_rows.loc[1, "n_images_before"]) == 2
    assert int(car_rows.loc[1, "n_images_after"]) == 2
    assert int(car_rows.loc[3, "n_images_before"]) == 1
    assert int(car_rows.loc[3, "n_images_after"]) == 0


def test_dataset_registry_and_contexts_support_both_datasets(tmp_path: Path) -> None:
    clear_analysis_context_cache()
    registry = _dataset_registry(tmp_path)

    resolved_flir = resolve_dataset_config("flir_private_proxy_alignment_v18", registry=registry)
    resolved_v18 = resolve_dataset_config("v18", registry=registry)
    assert resolved_flir.root.name == "flir"
    assert resolved_v18.root.name == "v18"

    flir_context = build_analysis_context(dataset_id="flir_private_proxy_alignment_v18", dataset_registry=registry)
    v18_context = build_analysis_context(dataset_id="v18", dataset_registry=registry)

    assert flir_context.dataset_summary["n_classes"] == 3
    assert v18_context.dataset_summary["n_classes"] == 1
    assert v18_context.dataset_summary["classes"] == ["person"]
    assert v18_context.get_phase_bundle("phase1").selectable_groups_df["subgroup_label"].str.contains("class=person").all()
    assert v18_context.get_phase_bundle("phase2").selectable_groups_df["subgroup_label"].str.contains("pos=").all()


def test_api_endpoints_dataset_selection_and_preview_route(tmp_path: Path) -> None:
    clear_analysis_context_cache()
    registry = _dataset_registry(tmp_path)
    client = TestClient(create_app(dataset_registry=registry))

    datasets_response = client.get("/api/flir-analysis/datasets")
    assert datasets_response.status_code == 200
    datasets_payload = datasets_response.json()
    assert datasets_payload["default_dataset_id"] == "flir_private_proxy_alignment_v18"
    assert [row["dataset_id"] for row in datasets_payload["datasets"]] == ["flir_private_proxy_alignment_v18", "v18"]

    options_flir = client.get("/api/flir-analysis/options", params={"dataset": "flir_private_proxy_alignment_v18"})
    assert options_flir.status_code == 200
    options_flir_payload = options_flir.json()
    assert options_flir_payload["active_dataset"]["n_images"] == 4
    assert options_flir_payload["active_dataset"]["n_classes"] == 3
    assert options_flir_payload["constants"]["position_bin_edges"] == [0.0, 1 / 3, 2 / 3, 1.0]
    assert options_flir_payload["bin_explanations"]["size"]["example"]["preview_url"].startswith("/api/flir-analysis/images/flir_private_proxy_alignment_v18/")

    options_v18 = client.get("/api/flir-analysis/options", params={"dataset": "v18"})
    assert options_v18.status_code == 200
    options_v18_payload = options_v18.json()
    assert options_v18_payload["active_dataset"]["classes"] == ["person"]
    assert all(group["class_label"] == "person" for group in options_v18_payload["phase1"]["groups"])
    assert options_v18_payload["bin_explanations"]["position"]["example"]["boxes"]

    missing_dataset_response = client.post(
        "/api/flir-analysis/holdout-curves",
        json={"phase": "phase1", "groups": [{"class_label": "car", "size_bin": "large"}]},
    )
    assert missing_dataset_response.status_code == 422

    holdout_response = client.post(
        "/api/flir-analysis/holdout-curves",
        json={
            "dataset": "flir_private_proxy_alignment_v18",
            "phase": "phase1",
            "groups": [{"class_label": "car", "size_bin": "large"}],
        },
    )
    assert holdout_response.status_code == 200
    holdout_group = holdout_response.json()["groups"][0]
    assert holdout_group["subgroup_label"] == "class=car | size=large"
    assert len(holdout_group["series"]) == 7

    v18_holdout_response = client.post(
        "/api/flir-analysis/holdout-curves",
        json={
            "dataset": "v18",
            "phase": "phase2",
            "groups": [{"class_label": "person", "size_bin": "large", "position_bin": "right"}],
        },
    )
    assert v18_holdout_response.status_code == 200
    assert v18_holdout_response.json()["dataset"] == "v18"

    partition_response = client.post(
        "/api/flir-analysis/partition-comparisons",
        json={
            "dataset": "flir_private_proxy_alignment_v18",
            "phase": "phase1",
            "tau": 0.5,
            "groups": [{"class_label": "car", "size_bin": "large"}],
            "include_zero_counts": False,
        },
    )
    assert partition_response.status_code == 200
    partition_payload = partition_response.json()
    assert partition_payload["heldout_n_images"] == 1
    assert {row["class_label"] for row in partition_payload["per_class_image_count_distribution"]} == {"car", "person", "light"}

    examples_response = client.post(
        "/api/flir-analysis/examples",
        json={
            "dataset": "v18",
            "phase": "phase1",
            "tau": 0.5,
            "groups": [{"class_label": "person", "size_bin": "large"}],
            "example_count": 2,
        },
    )
    assert examples_response.status_code == 200
    examples_payload = examples_response.json()
    retained = examples_payload["groups"][0]["retained_examples"]
    assert all(example["preview_url"].startswith("/api/flir-analysis/images/v18/") for example in retained)

    preview_url = examples_payload["groups"][0]["held_out_examples"][0]["preview_url"]
    image_response = client.get(preview_url)
    assert image_response.status_code == 200
    assert image_response.headers["content-type"] == "image/png"

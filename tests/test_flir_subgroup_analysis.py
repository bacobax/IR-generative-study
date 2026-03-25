"""Tests for the FLIR subgroup analysis backend."""

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


def test_context_builds_from_discovered_dataset(tmp_path: Path) -> None:
    clear_analysis_context_cache()
    data_root = _create_synthetic_flir_dataset(tmp_path / "flir")
    context = build_analysis_context(data_root=data_root)

    assert context.dataset_summary["n_images"] == 4
    assert context.dataset_summary["analysis_splits"] == ["train"]
    assert context.get_phase_bundle("phase1").selectable_groups_df["subgroup_label"].str.contains("class=car").any()
    assert context.get_phase_bundle("phase2").selectable_groups_df["subgroup_label"].str.contains("pos=").all()


def test_api_endpoints_and_preview_route(tmp_path: Path) -> None:
    clear_analysis_context_cache()
    data_root = _create_synthetic_flir_dataset(tmp_path / "flir")
    client = TestClient(create_app(data_root=data_root))

    options_response = client.get("/api/flir-analysis/options")
    assert options_response.status_code == 200
    options_payload = options_response.json()
    assert options_payload["dataset"]["n_images"] == 4
    assert options_payload["constants"]["position_bin_labels"] == ["left", "center", "right"]

    phase1_labels = {row["subgroup_label"] for row in options_payload["phase1"]["groups"]}
    assert "class=car | size=large" in phase1_labels

    holdout_response = client.post(
        "/api/flir-analysis/holdout-curves",
        json={
            "phase": "phase1",
            "groups": [{"class_label": "car", "size_bin": "large"}],
        },
    )
    assert holdout_response.status_code == 200
    holdout_group = holdout_response.json()["groups"][0]
    assert holdout_group["subgroup_label"] == "class=car | size=large"
    assert len(holdout_group["series"]) == 7

    invalid_response = client.post(
        "/api/flir-analysis/holdout-curves",
        json={"phase": "phase1", "groups": [{"class_label": "missing", "size_bin": "large"}]},
    )
    assert invalid_response.status_code == 422

    collateral_response = client.post(
        "/api/flir-analysis/collateral",
        json={
            "phase": "phase1",
            "tau": 0.5,
            "groups": [{"class_label": "car", "size_bin": "large"}],
        },
    )
    assert collateral_response.status_code == 200
    collateral_group = collateral_response.json()["groups"][0]
    assert collateral_group["summary"]["heldout_n_images"] == 1
    assert collateral_group["dominance_histogram"]

    partition_response = client.post(
        "/api/flir-analysis/partition-comparisons",
        json={
            "phase": "phase1",
            "tau": 0.5,
            "groups": [{"class_label": "car", "size_bin": "large"}],
            "include_zero_counts": False,
        },
    )
    assert partition_response.status_code == 200
    partition_payload = partition_response.json()
    assert partition_payload["heldout_n_images"] == 1

    class_image_rows = {
        (row["partition"], row["class_label"]): row["n_images"]
        for row in partition_payload["class_image_distribution"]
    }
    assert class_image_rows[("held_out", "car")] == 1
    assert class_image_rows[("held_out", "person")] == 1
    assert class_image_rows[("train", "car")] == 2
    assert class_image_rows[("train", "light")] == 2
    assert class_image_rows[("train", "person")] == 2

    car_distribution = [
        row for row in partition_payload["per_class_image_count_distribution"] if row["class_label"] == "car"
    ]
    bucket_three = next(row for row in car_distribution if row["instance_count"] == 3)
    assert bucket_three["n_images_before"] == 1
    assert bucket_three["n_images_after"] == 0

    examples_response = client.post(
        "/api/flir-analysis/examples",
        json={
            "phase": "phase1",
            "tau": 0.5,
            "example_count": 3,
            "groups": [{"class_label": "car", "size_bin": "large"}],
        },
    )
    assert examples_response.status_code == 200
    example_group = examples_response.json()["groups"][0]
    assert len(example_group["held_out_examples"]) == 1
    assert any(example["selection_source"] == "same_class_fallback" for example in example_group["retained_examples"])

    preview_url = example_group["held_out_examples"][0]["preview_url"]
    preview_response = client.get(preview_url)
    assert preview_response.status_code == 200
    assert preview_response.headers["content-type"] == "image/png"
    assert preview_response.content.startswith(b"\x89PNG")

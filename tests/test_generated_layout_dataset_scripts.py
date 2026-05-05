"""Tests for rare-layout dataset generation and classifier audit scripts."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from scripts.standalone.filter_generated_layout_dataset import main as filter_generated_main
from scripts.standalone.generate_rare_layout_dataset import main as generate_rare_layout_main
from src.algorithms.inference.rare_layout_dataset_tools import (
    LayoutRarityRecord,
    audit_generated_layout_dataset,
    build_instance_confusion_matrix,
    build_layout_dataset,
    build_layout_rarity_records,
    compute_size_bin_thresholds,
    export_generated_layout_dataset,
    summarize_instance_statistics,
)
from src.models.foreground_background_classifier import ForegroundBackgroundClassifier


def test_confusion_matrix_keeps_predicted_only_multiclass_labels() -> None:
    matrix = build_instance_confusion_matrix(
        [
            {
                "expected_category_id": 0,
                "expected_category_name": "person",
                "predicted_category_id": 3,
                "predicted_category_name": "other vehicle",
            }
        ],
        checkpoint_summary={
            "category_id_to_name": {
                "0": "person",
                "3": "other vehicle",
            }
        },
    )

    assert matrix["labels"] == ["person", "other vehicle", "background"]
    assert matrix["matrix_counts"][0][1] == 1


def _write_split(split_dir: Path, *, split_name: str) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    images = []
    annotations = []
    annotation_id = 1
    for image_index in range(3):
        image_id = f"{split_name}-{image_index}"
        file_name = f"{image_id}.npy"
        image = np.zeros((32, 32), dtype=np.uint8)
        if image_index != 1:
            image[4:12, 4:12] = 220
        image[18:28, 18:28] = 100 + 10 * image_index
        np.save(split_dir / file_name, image)
        images.append(
            {
                "id": image_id,
                "file_name": file_name,
                "width": 32,
                "height": 32,
            }
        )
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [4, 4, 8, 8],
                "area": 64,
                "iscrowd": 0,
            }
        )
        annotation_id += 1
        annotations.append(
            {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": 2,
                "bbox": [18, 18, 10, 10],
                "area": 100,
                "iscrowd": 0,
            }
        )
        annotation_id += 1

    payload = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
    }
    (split_dir / "annotations.json").write_text(json.dumps(payload), encoding="utf-8")


def _make_dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "flir_like"
    for split in ("train", "val", "test"):
        _write_split(root / split, split_name=split)
    return root


def _make_filter_run(tmp_path: Path, *, threshold: float = 0.5) -> Path:
    run_dir = tmp_path / "filter_run"
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics").mkdir(parents=True, exist_ok=True)

    model = ForegroundBackgroundClassifier()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.head.bias.fill_(10.0)

    torch.save({"model_state": model.state_dict(), "best_threshold": threshold}, run_dir / "checkpoints" / "best.pt")
    (run_dir / "metrics" / "summary.json").write_text(
        json.dumps(
            {
                "chosen_threshold": threshold,
                "input_size": 32,
                "context_ratio": 1.25,
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_rarity_thresholds_are_deterministic(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path)
    preset = {"data": {"dataset_id": "flir_private_proxy_alignment_v18", "image_size": 32}}
    dataset = build_layout_dataset(preset, split="train", dataset_root=root / "train")
    thresholds_a = compute_size_bin_thresholds(dataset)
    thresholds_b = compute_size_bin_thresholds(dataset)
    records_a, counter_a = build_layout_rarity_records(dataset, thresholds=thresholds_a)
    records_b, counter_b = build_layout_rarity_records(dataset, thresholds=thresholds_b)

    assert np.allclose(thresholds_a, thresholds_b)
    assert counter_a == counter_b
    assert [record.file_name for record in records_a] == [record.file_name for record in records_b]


def test_export_generated_dataset_writes_coco_and_provenance(tmp_path: Path) -> None:
    root = _make_dataset_root(tmp_path)
    preset = {"data": {"dataset_id": "flir_private_proxy_alignment_v18", "image_size": 32}}
    dataset = build_layout_dataset(preset, split="train", dataset_root=root / "train")
    thresholds = compute_size_bin_thresholds(dataset)
    records, _ = build_layout_rarity_records(dataset, thresholds=thresholds)
    selected = records[:2]
    generated_images = [sample.sample["pixel_values"].repeat(3, 1, 1) for sample in selected]
    output_dir = tmp_path / "generated_ds"

    summary = export_generated_layout_dataset(
        output_dir=output_dir,
        selected_records=selected,
        generated_images=generated_images,
        split="train",
        dataset_id="flir_private_proxy_alignment_v18",
        selection_mode="rare_first",
        rarity_aggregation="sum",
        size_bin_thresholds=thresholds,
        pipeline_dir="dummy_pipeline",
        preset_path="dummy_preset.yaml",
        checkpoint_name="dummy.pt",
        steps=10,
        seed=7,
    )

    assert (output_dir / "annotations.json").is_file()
    assert (output_dir / "metadata" / "summary.json").is_file()
    assert (output_dir / "metadata" / "provenance.jsonl").is_file()
    assert (output_dir / "images" / "sample_000001.npy").is_file()
    payload = json.loads((output_dir / "annotations.json").read_text(encoding="utf-8"))
    assert len(payload["images"]) == 2
    assert len(payload["annotations"]) == sum(record.n_objects for record in selected)
    assert len(summary["samples"]) == 2


def test_summarize_instance_statistics_counts_joint_groups() -> None:
    rows = [
        {"category_name": "person", "size_bin": "small", "is_positive": True},
        {"category_name": "person", "size_bin": "small", "is_positive": False},
        {"category_name": "car", "size_bin": "big", "is_positive": True},
    ]
    summary = summarize_instance_statistics(rows)
    assert summary["overall"]["positive_count"] == 2
    by_joint = {row["category_size_bin"]: row for row in summary["by_category_size_bin"]}
    assert by_joint["person | small"]["negative_count"] == 1
    assert by_joint["car | big"]["positive_count"] == 1


def test_generate_and_filter_scripts_smoke_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _make_dataset_root(tmp_path)
    output_dir = tmp_path / "generated_output"
    filter_run_dir = _make_filter_run(tmp_path)

    import scripts.standalone.generate_rare_layout_dataset as generation_script

    def _fake_load_sampler(_pipeline_dir, _preset_path, _checkpoint_name, _device):
        preset = {"data": {"dataset_id": "flir_private_proxy_alignment_v18", "image_size": 32}}
        return preset, {}, object(), object(), object()

    def _fake_sample_layout_batch(_sampler, batch, *, steps, seed):
        del steps, seed
        return batch["pixel_values"].repeat(1, 3, 1, 1)

    monkeypatch.setattr(generation_script, "load_sampler_from_pipeline", _fake_load_sampler)
    monkeypatch.setattr(generation_script, "sample_layout_batch", _fake_sample_layout_batch)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "generate_rare_layout_dataset.py",
            "--dataset_root",
            str(root / "train"),
            "--output_dir",
            str(output_dir),
            "--n_samples",
            "2",
            "--batch_size",
            "2",
            "--device",
            "cpu",
        ],
    )
    generate_rare_layout_main()

    assert (output_dir / "annotations.json").is_file()
    assert (output_dir / "images" / "sample_000001.npy").is_file()

    filter_output_dir = tmp_path / "audit_output"
    filter_config_path = tmp_path / "filter_config.yaml"
    filter_config_path.write_text(
        "\n".join(
            [
                f"generated_dataset_dir: {output_dir}",
                f"filter_run_dir: {filter_run_dir}",
                f"output_dir: {filter_output_dir}",
                "batch_size: 8",
                "max_discarded_valid_threshold: 1.0",
                "score_alpha: 1.0",
                "score_beta: 1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "filter_generated_layout_dataset.py",
            "--config",
            str(filter_config_path),
            "--device",
            "cpu",
        ],
    )
    filter_generated_main()

    assert (filter_output_dir / "summary.json").is_file()
    assert (filter_output_dir / "per_image_manifest.jsonl").is_file()
    assert (filter_output_dir / "per_instance_manifest.jsonl").is_file()
    assert (filter_output_dir / "per_instance" / "charts").is_dir()
    assert (filter_output_dir / "per_instance" / "charts" / "overall_metrics.png").is_file()
    assert (filter_output_dir / "per_instance" / "charts" / "category_size_heatmaps.png").is_file()
    assert (filter_output_dir / "per_instance" / "charts" / "confusion_matrix_counts.png").is_file()
    assert (filter_output_dir / "per_instance" / "charts" / "confusion_matrix_normalized.png").is_file()
    assert (filter_output_dir / "per_image" / "threshold_sweep").is_dir()
    assert (filter_output_dir / "per_image" / "valid_image_ratio_sweep.png").is_file()
    assert (filter_output_dir / "per_image" / "threshold_selection_score_curve.png").is_file()
    assert (filter_output_dir / "per_image" / "threshold_selection_global_discard_ratios.png").is_file()
    assert (filter_output_dir / "per_image" / "threshold_selection_tradeoff_scatter.png").is_file()
    assert (filter_output_dir / "per_image" / "threshold_selection_metrics.csv").is_file()
    assert (filter_output_dir / "per_image" / "reference_category_size_frequencies.csv").is_file()
    summary = json.loads((filter_output_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["stats"]["overall"]["total_count"] > 0
    assert summary["stats"]["image_level"]["valid_image_count"] >= 1
    assert "chart_paths" in summary
    assert "table_paths" in summary
    assert "per_instance_analysis" in summary
    assert "per_image_analysis" in summary
    assert summary["per_image_analysis"]["thresholds"]
    first_threshold = summary["per_image_analysis"]["thresholds"][0]
    assert first_threshold["discarded_category_ratios"]
    assert first_threshold["discarded_category_ratio_chart_path"] is not None
    assert (filter_output_dir / first_threshold["discarded_category_ratio_chart_path"]).is_file()
    assert "rarity_weighted_score" in first_threshold
    threshold_selection = summary["per_image_analysis"]["threshold_selection"]
    assert "recommended_min_valid_object_fraction" in threshold_selection
    assert threshold_selection["max_discarded_valid_threshold"] == 1.0
    assert "n_feasible_thresholds" in threshold_selection
    assert (filter_output_dir / threshold_selection["score_curve_chart_path"]).is_file()
    assert (filter_output_dir / threshold_selection["global_discard_ratio_chart_path"]).is_file()
    assert (filter_output_dir / threshold_selection["tradeoff_scatter_chart_path"]).is_file()
    assert "is_feasible_threshold" in first_threshold
    assert "max_combo_discarded_valid_ratio" in first_threshold
    reference_analysis = summary["per_image_analysis"]["reference_frequency_analysis"]
    assert reference_analysis["score_alpha"] == 1.0
    assert reference_analysis["score_beta"] == 1.0
    assert (filter_output_dir / reference_analysis["reference_table_path"]).is_file()
    confusion = summary["per_instance_analysis"]["confusion_matrix"]
    assert "matrix_counts" in confusion
    assert "matrix_normalized" in confusion
    assert confusion["normalization"] == "row"

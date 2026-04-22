"""Tests for YOLO staged training config validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.cli import train_yolo
from src.cli.train_yolo import (
    _build_train_stages,
    _ordered_experiment_entries,
    _reuse_existing_train_summary,
    _validate_config_yaml_keys,
    build_parser,
    run_eval,
    run_experiment_a_all,
)
from src.core.configs.yolo_experiment_config import YOLOExperimentConfig


def test_build_train_stages_creates_frozen_and_finetune_phases() -> None:
    cfg = YOLOExperimentConfig()
    cfg.output.experiment_name = "exp_balanced"
    cfg.training.epochs = 12
    cfg.training.freeze_backbone_epochs = 3
    cfg.training.freeze_backbone_layers = 10
    cfg.training.backbone_lr_multiplier = 0.1
    cfg.training.backbone_param_prefixes = ["model.0.", "model.1."]

    stages = _build_train_stages(cfg)

    assert [stage["stage_name"] for stage in stages] == ["backbone_frozen", "full_finetune"]
    assert stages[0]["epochs"] == 3
    assert stages[0]["freeze"] == 10
    assert stages[1]["epochs"] == 9
    assert stages[1]["freeze"] is None
    assert stages[0]["run_name"] == "exp_balanced_phase1_frozen"
    assert stages[1]["run_name"] == "exp_balanced"


def test_validate_config_yaml_keys_rejects_unknown_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_yolo_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "training:",
                "  epochs: 10",
                "  lr0: 0.01",
                "  typo_lr: 0.1",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="training.typo_lr"):
        _validate_config_yaml_keys(str(config_path))


def test_validate_config_yaml_keys_accepts_baseline_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "good_yolo_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "baseline:",
                "  mode: baseline_b",
                "  rarity_alpha: 1.0",
                "  rarity_eps: 1.0e-6",
                "  image_score_top_k: 3",
                "  normalize_weights: true",
                "  targeted_aug_probability: 0.5",
                "  translate_fraction: 0.05",
                "  scale_min: 0.9",
                "  scale_max: 1.1",
                "  crop_scale_min: 0.85",
                "  crop_scale_max: 1.0",
                "  crop_min_rare_box_area_retained: 0.5",
                "  crop_max_attempts: 10",
                "  allow_horizontal_flip: true",
                "  seed: 7",
            ]
        ),
        encoding="utf-8",
    )

    assert _validate_config_yaml_keys(str(config_path))["baseline"]["mode"] == "baseline_b"


def test_validate_config_yaml_keys_accepts_eval_dataset_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "good_eval_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "evaluation:",
                "  dataset_yaml: data/derived/yolo-test-ds/full_train.yaml",
                "  split: val",
            ]
        ),
        encoding="utf-8",
    )

    evaluation = _validate_config_yaml_keys(str(config_path))["evaluation"]
    assert evaluation["dataset_yaml"] == "data/derived/yolo-test-ds/full_train.yaml"
    assert evaluation["split"] == "val"


def test_validate_config_yaml_keys_accepts_experiment_b_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "good_exp_b_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_b:",
                "  mode: fm_aug",
                "  invalid_instance_ratio_threshold: 0.25",
                "  generated_dataset_dir: artifacts/generated/yolo/exp_b/generated_candidates",
                "  augmented_yolo_root: artifacts/generated/yolo/exp_b/augmented_yolo",
                "  disable_ultralytics_augmentations: true",
                "  filter:",
                "    checkpoint_dir: artifacts/checkpoints/filter/checkpoints",
                "    batch_size: 16",
                "  fm:",
                "    checkpoint_path: artifacts/checkpoints/fm/UNET/unet_fm_epoch_270.pt",
                "    preset_path: configs/fm/train/presets/stay_layout_latent_flir_sd15_512.yaml",
                "    steps: 10",
                "    batch_size: 2",
                "  sd:",
                "    base_model: runwayml/stable-diffusion-v1-5",
                "    sd_steps: 20",
                "    guidance: 7.5",
                "    precision: fp16",
                "    max_tries: 3",
                "    prompt_mode: constant",
            ]
        ),
        encoding="utf-8",
    )

    experiment_b = _validate_config_yaml_keys(str(config_path))["experiment_b"]
    assert experiment_b["mode"] == "fm_aug"
    assert experiment_b["invalid_instance_ratio_threshold"] == 0.25
    assert experiment_b["fm"]["checkpoint_path"].endswith("unet_fm_epoch_270.pt")


def test_validate_config_yaml_keys_rejects_unknown_experiment_b_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_exp_b_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "experiment_b:",
                "  mode: fm_aug",
                "  typo_threshold: 0.25",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="experiment_b.typo_threshold"):
        _validate_config_yaml_keys(str(config_path))


def test_validate_config_yaml_keys_rejects_unknown_baseline_entries(tmp_path: Path) -> None:
    config_path = tmp_path / "bad_baseline_config.yaml"
    config_path.write_text(
        "\n".join(
            [
                "baseline:",
                "  mode: baseline_a",
                "  typo_probability: 0.5",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="baseline.typo_probability"):
        _validate_config_yaml_keys(str(config_path))


def test_ordered_experiment_entries_follow_launcher_order() -> None:
    cfg = YOLOExperimentConfig()
    cfg.launcher.ordered_config_paths = [
        "configs/yolo/exp_a/flir/exp_balanced.yaml",
        "configs/yolo/exp_a/flir/exp_unbalanced.yaml",
        "configs/yolo/exp_a/flir/exp_full_train.yaml",
        "configs/yolo/exp_a/flir/exp_full_train_baseline_a.yaml",
        "configs/yolo/exp_a/flir/exp_full_train_baseline_b.yaml",
    ]
    cfg.launcher.ordered_labels = [
        "balanced",
        "unbalanced",
        "full_train",
        "full_train_baseline_a",
        "full_train_baseline_b",
    ]

    assert _ordered_experiment_entries(cfg) == [
        ("balanced", "configs/yolo/exp_a/flir/exp_balanced.yaml"),
        ("unbalanced", "configs/yolo/exp_a/flir/exp_unbalanced.yaml"),
        ("full_train", "configs/yolo/exp_a/flir/exp_full_train.yaml"),
        ("full_train_baseline_a", "configs/yolo/exp_a/flir/exp_full_train_baseline_a.yaml"),
        ("full_train_baseline_b", "configs/yolo/exp_a/flir/exp_full_train_baseline_b.yaml"),
    ]


def test_run_eval_prefers_configured_eval_dataset_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = YOLOExperimentConfig()
    cfg.output.runs_root = str(tmp_path / "runs")
    cfg.output.checkpoints_root = str(tmp_path / "checkpoints")
    cfg.output.analysis_root = str(tmp_path / "analysis")
    cfg.data.dataset_yaml = "train.yaml"
    cfg.data.test_dataset_yaml = "test.yaml"
    cfg.evaluation.dataset_yaml = "eval.yaml"
    cfg.evaluation.split = "val"

    seen_dataset_paths: list[str] = []

    def fake_require_dataset_yaml(path: str) -> str:
        seen_dataset_paths.append(path)
        return f"/resolved/{path}"

    class FakeModel:
        def __init__(self, weights: str, *, task: str) -> None:
            self.weights = weights
            self.task = task

        def val(self, **kwargs):
            return type(
                "FakeResults",
                (),
                {
                    "results_dict": {
                        "metrics/precision(B)": 0.1,
                        "metrics/recall(B)": 0.2,
                        "metrics/mAP50(B)": 0.3,
                        "metrics/mAP50-95(B)": 0.4,
                    },
                    "box": type("FakeBox", (), {"map75": 0.35})(),
                    "names": {},
                    "nt_per_class": [],
                },
            )()

    monkeypatch.setattr(train_yolo, "_require_ultralytics", lambda: FakeModel)
    monkeypatch.setattr(train_yolo, "_require_dataset_yaml", fake_require_dataset_yaml)
    monkeypatch.setattr(train_yolo, "_set_seed", lambda seed, deterministic: None)
    monkeypatch.setattr(train_yolo, "_save_filtered_confusion_matrix_plots", lambda results, *, analysis_dir: {})

    summary = run_eval(cfg, weights_path="best.pt")

    assert seen_dataset_paths == ["eval.yaml"]
    assert summary["dataset_yaml"] == "/resolved/eval.yaml"
    assert summary["split"] == "val"


def test_run_exp_a_all_reuses_train_but_reruns_eval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = YOLOExperimentConfig()
    cfg.output.analysis_root = str(tmp_path / "analysis")
    cfg.launcher.ordered_config_paths = ["configs/yolo/exp_a/flir/exp_full_train.yaml"]
    cfg.launcher.ordered_labels = ["full_train"]

    child_cfg = YOLOExperimentConfig()
    child_cfg.output.experiment_name = "exp_full_train"
    child_cfg.data.dataset_yaml = "data/derived/yolo-test-ds/full_train.yaml"

    eval_calls: list[str] = []

    monkeypatch.setattr(train_yolo, "_require_config_file", lambda path, *, label="config": path)
    monkeypatch.setattr(train_yolo, "_load_cli_config", lambda **kwargs: child_cfg)
    monkeypatch.setattr(
        train_yolo,
        "_reuse_existing_train_summary",
        lambda cfg: {
            "best_weights_path": "/tmp/best.pt",
            "tensorboard_log_dir": "/tmp/run",
            "baseline_artifact_paths": {},
        },
    )
    monkeypatch.setattr(train_yolo, "run_train", lambda cfg: pytest.fail("training should be reused"))

    def fake_run_eval(cfg: YOLOExperimentConfig, *, weights_path: str | None = None) -> dict[str, float]:
        eval_calls.append(weights_path or "")
        return {
            "dataset_yaml": "/tmp/eval.yaml",
            "split": "val",
            "map": 0.4,
            "map50": 0.5,
            "map75": 0.45,
            "precision": 0.6,
            "recall": 0.7,
        }

    monkeypatch.setattr(train_yolo, "run_eval", fake_run_eval)

    result = run_experiment_a_all(cfg, parser=build_parser())

    assert eval_calls == ["/tmp/best.pt"]
    assert Path(result["comparison_json"]).exists()


def test_reuse_existing_train_summary_accepts_checkpoint_without_summary(tmp_path: Path) -> None:
    cfg = YOLOExperimentConfig()
    cfg.output.experiment_name = "exp_full_train"
    cfg.output.runs_root = str(tmp_path / "runs")
    cfg.output.checkpoints_root = str(tmp_path / "checkpoints")
    cfg.output.analysis_root = str(tmp_path / "analysis")
    checkpoint_dir = cfg.output.checkpoint_dir()
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pt").write_bytes(b"fake weights")

    summary = _reuse_existing_train_summary(cfg)

    assert summary is not None
    assert summary["best_weights_path"] == str((checkpoint_dir / "best.pt").resolve())
    assert summary["reused_from_checkpoint_without_train_summary"] is True

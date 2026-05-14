from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.cli.train_yolo import (
    _ordered_experiment_entries,
    _validate_config_yaml_keys,
    build_parser,
)
from src.core.configs.config_loader import load_yaml
from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.yolo_experiment_config import YOLOExperimentConfig


REPO_ROOT = Path(__file__).resolve().parent.parent
SNAPSHOT_PATH = REPO_ROOT / "tests" / "fixtures" / "yolo_config_effective_snapshots.json"


def _snapshot_payloads() -> dict[str, dict]:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def _concrete_yolo_config_paths() -> list[Path]:
    roots = [
        REPO_ROOT / "configs" / "yolo" / "exp_a",
        REPO_ROOT / "configs" / "yolo" / "exp_b",
    ]
    return sorted(
        path
        for root in roots
        for path in root.rglob("*.yaml")
        if not path.name.startswith("_")
    )


def test_yolo_effective_configs_match_pre_extends_snapshots() -> None:
    snapshots = _snapshot_payloads()
    current_paths = {str(path.relative_to(REPO_ROOT)): path for path in _concrete_yolo_config_paths()}

    assert set(current_paths) == set(snapshots)
    for rel_path, expected in snapshots.items():
        assert load_yaml(current_paths[rel_path]) == expected, rel_path


@pytest.mark.parametrize("rel_path", sorted(_snapshot_payloads()))
def test_yolo_concrete_configs_validate_after_extends(rel_path: str) -> None:
    if "/synthetic_generation/" in rel_path:
        pytest.skip("Synthetic generation presets use a separate schema from train_yolo.")
    _validate_config_yaml_keys(str(REPO_ROOT / rel_path))


def test_yolo_launcher_ordered_config_paths_are_unchanged() -> None:
    snapshots = _snapshot_payloads()
    parser = build_parser()

    for rel_path, expected in snapshots.items():
        expected_launcher = expected.get("launcher", {})
        if not expected_launcher.get("ordered_config_paths"):
            continue
        args = parser.parse_args([])
        cfg = merge_config_and_cli(
            YOLOExperimentConfig,
            str(REPO_ROOT / rel_path),
            parser,
            args,
            flat_to_nested={},
        )
        assert load_yaml(REPO_ROOT / rel_path).get("launcher", {}) == expected_launcher
        assert _ordered_experiment_entries(cfg) == list(
            zip(
                expected_launcher.get("ordered_labels", []),
                expected_launcher.get("ordered_config_paths", []),
            )
        )

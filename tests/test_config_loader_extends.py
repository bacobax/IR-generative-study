from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.cli.train import build_parser
from src.core.configs.config_loader import (
    is_experimental_config_shape,
    load_experimental_config,
    load_yaml,
    merge_config_and_cli,
)
from src.core.configs.fm_config import FMTrainConfig


@dataclass
class NestedConfig:
    depth: int = 1
    label: str = "default"


@dataclass
class TinyConfig:
    count: int = 1
    nested: NestedConfig = field(default_factory=NestedConfig)


def _write_yaml(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_load_yaml_extends_deep_merges_parent_then_child(tmp_path: Path) -> None:
    parent = _write_yaml(
        tmp_path / "base.yaml",
        """
name: parent
nested:
  keep: 1
  replace: old
  inner:
    a: 10
list_value:
  - parent
""",
    )
    child = _write_yaml(
        tmp_path / "child.yaml",
        f"""
extends: {parent.name}
name: child
nested:
  replace: new
  add: 2
  inner:
    b: 20
list_value:
  - child
""",
    )

    assert load_yaml(child) == {
        "name": "child",
        "nested": {
            "keep": 1,
            "replace": "new",
            "add": 2,
            "inner": {
                "a": 10,
                "b": 20,
            },
        },
        "list_value": ["child"],
    }
    assert "extends" not in load_yaml(child)


def test_load_yaml_extends_resolves_parent_relative_to_child_file(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "base.yaml",
        """
root_value: parent
nested:
  value: 1
""",
    )
    child = _write_yaml(
        tmp_path / "configs" / "child.yaml",
        """
extends: ../base.yaml
nested:
  value: 2
""",
    )

    assert load_yaml(child) == {
        "root_value": "parent",
        "nested": {"value": 2},
    }


def test_load_yaml_extends_missing_parent_raises_clear_file_not_found(tmp_path: Path) -> None:
    child = _write_yaml(
        tmp_path / "child.yaml",
        """
extends: missing/base.yaml
value: 1
""",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        load_yaml(child)

    message = str(exc_info.value)
    assert "YAML extends parent not found" in message
    assert "child.yaml" in message
    assert "missing/base.yaml" in message


def test_load_yaml_extends_cycle_raises_clear_value_error(tmp_path: Path) -> None:
    first = _write_yaml(
        tmp_path / "first.yaml",
        """
extends: second.yaml
value: first
""",
    )
    _write_yaml(
        tmp_path / "second.yaml",
        """
extends: first.yaml
value: second
""",
    )

    with pytest.raises(ValueError) as exc_info:
        load_yaml(first)

    message = str(exc_info.value)
    assert "cycle" in message.lower()
    assert "first.yaml" in message
    assert "second.yaml" in message


def test_merge_config_and_cli_accepts_inherited_yaml_and_cli_still_wins(tmp_path: Path) -> None:
    _write_yaml(
        tmp_path / "base.yaml",
        """
count: 3
nested:
  depth: 4
  label: parent
""",
    )
    child = _write_yaml(
        tmp_path / "child.yaml",
        """
extends: base.yaml
nested:
  label: child
""",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--count", type=int, default=1)
    parser.add_argument("--label", default="default")
    args = parser.parse_args(["--config", str(child), "--count", "9"])

    cfg = merge_config_and_cli(
        TinyConfig,
        args.config,
        parser,
        args,
        flat_to_nested={"count": "count", "label": "nested.label"},
    )

    assert cfg.count == 9
    assert cfg.nested.depth == 4
    assert cfg.nested.label == "child"


def test_old_fm_yaml_still_parses_through_permissive_loader() -> None:
    config_path = "configs/fm/train/default.yaml"
    parser = build_parser()
    args = parser.parse_args(["--config", config_path])

    cfg = merge_config_and_cli(
        FMTrainConfig,
        config_path,
        parser,
        args,
        cli_argv=["--config", config_path],
    )

    assert cfg.training.train_target == "v"
    assert cfg.model.unet_config == "configs/models/fm/stable_unet_config.json"


def test_old_config_path_still_ignores_unknown_keys(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "legacy_extra.yaml",
        """
unknown_top_level: ok_for_legacy
nested:
  depth: 5
  ignored: true
""",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args(["--config", str(config_path)])

    cfg = merge_config_and_cli(TinyConfig, config_path, parser, args)

    assert cfg.nested.depth == 5
    assert not hasattr(cfg, "unknown_top_level")
    assert not hasattr(cfg.nested, "ignored")


def test_experimental_config_yaml_parses_strict_shape() -> None:
    config_path = "configs/fm/train/presets/experimental/hf_diffusers_flir.yaml"

    raw = load_yaml(config_path)
    cfg = load_experimental_config(config_path)

    assert is_experimental_config_shape(raw)
    assert cfg.kind == "experimental_training_config"
    assert cfg.model.kind == "hf_diffusers"
    assert cfg.dataset.kind == "flir"
    assert cfg.artifact.manifest_name == "artifact_manifest.json"


def test_experimental_config_rejects_unknown_key_with_dotted_path(tmp_path: Path) -> None:
    config_path = _write_yaml(
        tmp_path / "bad_experimental.yaml",
        """
kind: experimental_training_config
schema_version: 1
model:
  kind: hf_diffusers
  family: stable_diffusion
  unexpected_model_key: nope
dataset:
  kind: flir
transform:
  kind: resize_normalize
task:
  kind: flow_matching
runtime:
  kind: local_runtime
artifact:
  kind: artifact_bundle
  output_dir: artifacts/test
""",
    )

    with pytest.raises(ValueError, match="model\\.unexpected_model_key"):
        load_experimental_config(config_path)

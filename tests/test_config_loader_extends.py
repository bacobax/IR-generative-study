from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from src.core.configs.config_loader import load_yaml, merge_config_and_cli


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

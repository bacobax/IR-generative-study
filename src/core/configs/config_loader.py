"""Lightweight YAML config loader with CLI-override merging.

Merge semantics
---------------
1. Start from dataclass defaults (built into ``FMTrainConfig`` / ``FMSampleConfig``).
2. Deep-merge values from YAML config file (overrides defaults).
3. Apply any CLI-provided arguments on top (overrides config file).

Only standard-library + PyYAML (``pip install pyyaml``) are required.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

import yaml

T = TypeVar("T")


# ═══════════════════════════════════════════════════════════════════════════
# YAML helpers
# ═══════════════════════════════════════════════════════════════════════════

def load_yaml(path: str | Path) -> Dict[str, Any]:
    """Load a YAML file and return its contents as a dict."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    return data if data is not None else {}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (mutates *base*)."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


# ═══════════════════════════════════════════════════════════════════════════
# Dataclass ↔ dict conversion
# ═══════════════════════════════════════════════════════════════════════════

def dataclass_to_dict(obj) -> Dict[str, Any]:
    """Convert a (possibly nested) dataclass to a plain dict."""
    if not is_dataclass(obj):
        return obj
    result = {}
    for f in fields(obj):
        val = getattr(obj, f.name)
        if is_dataclass(val):
            result[f.name] = dataclass_to_dict(val)
        else:
            result[f.name] = val
    return result


def dict_to_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    """Instantiate a (possibly nested) dataclass from a plain dict.

    Keys in *data* that don't correspond to dataclass fields are silently
    ignored so that YAML files can contain extra metadata.
    """
    kwargs = {}
    field_map = {f.name: f for f in fields(cls)}
    for key, val in data.items():
        if key not in field_map:
            continue
        ftype = field_map[key].type
        # Resolve the actual type for nested dataclasses
        actual_type = _resolve_field_type(cls, key)
        if actual_type is not None and is_dataclass(actual_type) and isinstance(val, dict):
            kwargs[key] = dict_to_dataclass(actual_type, val)
        else:
            kwargs[key] = val
    return cls(**kwargs)


def _resolve_field_type(cls: Type, field_name: str):
    """Return the concrete type of a dataclass field, or None."""
    import sys
    import typing

    for f in fields(cls):
        if f.name == field_name:
            tp = f.type
            # Handle string annotations (from __future__ annotations)
            if isinstance(tp, str):
                # Build a namespace from the module where cls is defined
                mod = sys.modules.get(cls.__module__, None)
                ns: dict = {}
                if mod is not None:
                    ns.update(vars(mod))
                ns.update(vars(typing))
                ns[cls.__name__] = cls
                try:
                    tp = eval(tp, ns)
                except Exception:
                    return None
            # Unwrap Optional[X] → X
            origin = getattr(tp, "__origin__", None)
            args = getattr(tp, "__args__", ())
            if origin is getattr(typing, "Union", None) and type(None) in args:
                # Optional[X] = Union[X, None]
                non_none = [a for a in args if a is not type(None)]
                if len(non_none) == 1:
                    tp = non_none[0]
            # Check if it's a dataclass directly
            if is_dataclass(tp):
                return tp
            return None
    return None


# ═══════════════════════════════════════════════════════════════════════════
# CLI + config-file merge
# ═══════════════════════════════════════════════════════════════════════════

def _cli_overrides(parser: argparse.ArgumentParser,
                   args: argparse.Namespace) -> Dict[str, Any]:
    """Return only the CLI arguments that were explicitly provided by the user.

    Compares *args* against the parser defaults so config-file values are
    not accidentally clobbered by argparse defaults.
    """
    defaults = vars(parser.parse_args([]))       # parse with no args → all defaults
    provided = vars(args)
    overrides = {}
    for key, val in provided.items():
        if key == "config":
            continue
        if val != defaults.get(key):
            overrides[key] = val
    return overrides


def merge_config_and_cli(
    config_cls: Type[T],
    yaml_path: str | Path | None,
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    *,
    flat_to_nested: Dict[str, str] | None = None,
) -> T:
    """Build a config dataclass from (defaults + YAML + CLI overrides).

    Parameters
    ----------
    config_cls:
        The target dataclass type (e.g. ``FMTrainConfig``).
    yaml_path:
        Path to YAML config file, or ``None`` to skip.
    parser:
        The argparse parser (used to detect explicit CLI overrides).
    args:
        The parsed argparse namespace.
    flat_to_nested:
        Mapping from flat CLI arg names to dotted paths into the nested
        dataclass.  E.g. ``{"train_dir": "data.train_dir"}``.

    Returns
    -------
    An instance of *config_cls* with all three layers merged.
    """
    # 1. Start from dataclass defaults (as dict)
    base = dataclass_to_dict(config_cls())

    # 2. Merge YAML on top
    if yaml_path is not None:
        yaml_data = load_yaml(yaml_path)
        _deep_merge(base, yaml_data)

    # 3. Merge explicit CLI overrides on top
    cli_overrides = _cli_overrides(parser, args)
    if flat_to_nested and cli_overrides:
        nested_overrides: Dict[str, Any] = {}
        for flat_key, val in cli_overrides.items():
            dotted = flat_to_nested.get(flat_key)
            if dotted is not None:
                _set_dotted(nested_overrides, dotted, val)
            else:
                nested_overrides[flat_key] = val
        _deep_merge(base, nested_overrides)
    elif cli_overrides:
        _deep_merge(base, cli_overrides)

    # 4. Instantiate the dataclass
    return dict_to_dataclass(config_cls, base)


def _set_dotted(d: dict, dotted_key: str, value: Any) -> None:
    """Set a value in a nested dict using a dot-separated key path."""
    parts = dotted_key.split(".")
    current = d
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


# ═══════════════════════════════════════════════════════════════════════════
# Generic YAML → argparse adapter
# ═══════════════════════════════════════════════════════════════════════════

def apply_yaml_defaults(
    parser: argparse.ArgumentParser,
    config_path: str | Path | None,
) -> None:
    """Load a YAML file and apply its values as parser defaults.

    This allows **any** argparse-based script to support ``--config``
    without restructuring its argument handling.  Values provided on the
    command line still take precedence (they override the YAML-set
    defaults just as they would override the original defaults).

    Parameters
    ----------
    parser:
        The argparse parser whose defaults will be updated.
    config_path:
        Path to a YAML config file.  If *None* or empty, this is a no-op.

    Notes
    -----
    YAML keys must match the parser ``dest`` names (typically the
    long-option name with leading dashes stripped and internal dashes
    converted to underscores).  Unknown keys are silently ignored.
    """
    if not config_path:
        return
    data = load_yaml(config_path)
    if not data:
        return
    # Only set defaults for keys that are recognised by the parser.
    known_dests = {a.dest for a in parser._actions}
    valid = {k: v for k, v in data.items() if k in known_dests}
    if valid:
        parser.set_defaults(**valid)

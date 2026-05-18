"""Canonical CLI for adapting pretrained Stable Diffusion 1.5 models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

from src.core.configs.config_loader import load_yaml


STAGE1 = "stage1"
REGIONDIFF_STAGE2 = "regiondiff-stage2"
_STAGES = (STAGE1, REGIONDIFF_STAGE2)


def _config_stage(config_path: str | None) -> str | None:
    """Infer the adaptation stage from the config path or config shape."""
    if not config_path:
        return None

    path = Path(config_path)
    parts = set(path.parts)
    if "sd_layout" in parts:
        return REGIONDIFF_STAGE2
    if "sd" in parts:
        return STAGE1

    data = load_yaml(path)
    if isinstance(data, dict):
        if {"stage1", "region", "area_loss"}.intersection(data):
            return REGIONDIFF_STAGE2
        if "pretrained_model_name_or_path" in data or "baseline_mode" in data:
            return STAGE1
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt a pretrained Stable Diffusion 1.5 model.",
        add_help=False,
    )
    parser.add_argument(
        "--stage",
        choices=_STAGES,
        default=None,
        help="Adaptation stage. Inferred from configs/sd/** or configs/sd_layout/** when omitted.",
    )
    parser.add_argument("--config", type=str, default=None)
    return parser


def _strip_stage_args(argv: list[str]) -> list[str]:
    stripped: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item == "--stage":
            index += 2
            continue
        if item.startswith("--stage="):
            index += 1
            continue
        stripped.append(item)
        index += 1
    return stripped


def main(argv: Optional[list[str]] = None) -> None:
    import sys

    effective_argv = list(argv) if argv is not None else sys.argv[1:]
    stage_args, _ = build_parser().parse_known_args(effective_argv)
    stage = stage_args.stage or _config_stage(stage_args.config)
    if stage is None:
        raise SystemExit(
            "Could not infer adaptation stage. Pass --stage stage1 or "
            "--stage regiondiff-stage2."
        )

    delegated_argv = _strip_stage_args(effective_argv)
    if stage == STAGE1:
        from src.cli.adapt_stable_diffusion_stage1 import main as stage_main
    elif stage == REGIONDIFF_STAGE2:
        from src.cli.adapt_stable_diffusion_regiondiff_stage2 import main as stage_main
    else:
        raise SystemExit(f"Unknown adaptation stage: {stage!r}")

    stage_main(delegated_argv)


if __name__ == "__main__":
    main()

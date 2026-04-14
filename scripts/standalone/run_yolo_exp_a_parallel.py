#!/usr/bin/env python3
"""Thin wrapper for ordered YOLO Experiment A runs.

This script intentionally delegates all train/eval orchestration to
``src.cli.train_yolo`` so there is only one implementation of the
experiment flow.

Example:
    python scripts/standalone/run_yolo_exp_a_parallel.py --gpu 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from src.core.paths import repo_root


REPO_ROOT = repo_root()
DEFAULT_CONFIG = REPO_ROOT / "configs" / "yolo" / "exp_a" / "flir" / "run_exp_a_all.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Wrapper for ordered YOLO Experiment A runs")
    parser.add_argument("--gpu", type=str, required=True,
                        help="GPU identifier to pass through to train_yolo.py, e.g. 0 or cuda:0.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--python", type=str, default=sys.executable,
                        help="Python interpreter used to launch src.cli.train_yolo.")
    return parser


def _resolve_device_arg(gpu: str) -> str:
    return gpu if gpu.startswith("cuda:") or gpu == "cpu" else f"cuda:{gpu}"


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cmd = [
        args.python,
        "-m",
        "src.cli.train_yolo",
        "--action",
        "run_exp_a_all",
        "--config",
        args.config,
        "--device",
        _resolve_device_arg(args.gpu),
    ]
    subprocess.run(cmd, cwd=str(REPO_ROOT), check=True)


if __name__ == "__main__":
    main()

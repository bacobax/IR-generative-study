#!/usr/bin/env python3
"""Validate that a PyTorch checkpoint can be loaded and inspected."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

REPO = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a PyTorch checkpoint on CPU and scan all tensors for basic "
            "integrity issues such as unreadable serialization, empty tensors, "
            "and NaN/Inf values."
        )
    )
    parser.add_argument(
        "checkpoint",
        nargs="?",
        default=(
            "artifacts/checkpoints/flow_matching/serious_runs/"
            "stay_layout_latent_flir_sd15_512_hflip/UNET/unet_last_ckpt.pt"
        ),
        help="Checkpoint path, relative to the repo root or absolute.",
    )
    parser.add_argument(
        "--strict-finite",
        action="store_true",
        help="Fail if any floating-point tensor contains NaN or Inf.",
    )
    return parser.parse_args()


def resolve_checkpoint(path_arg: str) -> Path:
    path = Path(path_arg).expanduser()
    if not path.is_absolute():
        path = REPO / path
    return path


def iter_tensors(obj: Any, prefix: str = "checkpoint"):
    if torch.is_tensor(obj):
        yield prefix, obj
    elif isinstance(obj, Mapping):
        for key, value in obj.items():
            yield from iter_tensors(value, f"{prefix}.{key}")
    elif isinstance(obj, (list, tuple)):
        for idx, value in enumerate(obj):
            yield from iter_tensors(value, f"{prefix}[{idx}]")


def summarize_top_level(obj: Any) -> None:
    print(f"Loaded object type: {type(obj).__name__}")
    if isinstance(obj, Mapping):
        keys = list(obj.keys())
        print(f"Top-level keys ({len(keys)}): {keys[:25]}")
        if len(keys) > 25:
            print(f"  ... {len(keys) - 25} more")


def main() -> int:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)

    print(f"Checkpoint: {checkpoint}")
    if not checkpoint.is_file():
        print("FAIL: checkpoint file does not exist")
        return 1

    size_mb = checkpoint.stat().st_size / (1024 * 1024)
    print(f"File size: {size_mb:.1f} MiB")

    try:
        try:
            ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(checkpoint, map_location="cpu")
    except Exception as exc:
        print(f"FAIL: torch.load could not read checkpoint: {exc!r}")
        return 1

    summarize_top_level(ckpt)

    tensor_count = 0
    scalar_count = 0
    element_count = 0
    bytes_count = 0
    nonfinite = []
    empty = []

    for name, tensor in iter_tensors(ckpt):
        tensor_count += 1
        numel = tensor.numel()
        element_count += numel
        bytes_count += numel * tensor.element_size()
        if numel == 0:
            empty.append(name)
        if tensor.ndim == 0:
            scalar_count += 1
        if tensor.is_floating_point() or tensor.is_complex():
            finite = torch.isfinite(tensor)
            if not bool(finite.all().item()):
                bad = int((~finite).sum().item())
                nonfinite.append((name, bad, tuple(tensor.shape), str(tensor.dtype)))

    print(f"Tensors found: {tensor_count}")
    print(f"Scalar tensors: {scalar_count}")
    print(f"Tensor elements: {element_count:,}")
    print(f"Tensor payload: {bytes_count / (1024 * 1024):.1f} MiB")

    if empty:
        print(f"WARNING: {len(empty)} empty tensors found")
        for name in empty[:20]:
            print(f"  empty: {name}")
    if nonfinite:
        print(f"WARNING: {len(nonfinite)} tensors contain NaN or Inf")
        for name, bad, shape, dtype in nonfinite[:20]:
            print(f"  non-finite: {name} bad={bad} shape={shape} dtype={dtype}")

    if tensor_count == 0:
        print("FAIL: checkpoint loaded but no tensors were found")
        return 1
    if nonfinite and args.strict_finite:
        print("FAIL: non-finite tensor values found")
        return 1

    print("PASS: checkpoint is readable by torch.load and tensor metadata is inspectable")
    return 0


if __name__ == "__main__":
    sys.exit(main())

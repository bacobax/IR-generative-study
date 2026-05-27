#!/usr/bin/env python3
"""Explore BigEarthNet-S2 patch grids for 5x5 NIR mosaics.

The script intentionally uses only Pillow and NumPy so it can run in a light
environment without rasterio/GDAL. It is meant for dataset reconnaissance, not
for producing final georeferenced GeoTIFF products.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DEFAULT_ROOT = Path("data/raw/BigEarthNet-S2")
DEFAULT_OUTPUT = Path("artifacts/analysis/bigearthnet_s2_mosaics/exploration_summary.json")
DEFAULT_EXAMPLE_DIR = Path("artifacts/analysis/bigearthnet_s2_mosaics/example")


@dataclass(frozen=True)
class PatchRef:
    scene_name: str
    patch_name: str
    h: int
    v: int
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan a BigEarthNet-S2 directory and estimate how many contiguous "
            "5x5 B08 mosaics can be built from patch folders."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Path to BigEarthNet-S2. Default: {DEFAULT_ROOT}",
    )
    parser.add_argument(
        "--band",
        default="B08",
        help="Band suffix to inspect and mosaic. Default: B08.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=5,
        help="Patch grid side length for a mosaic. Default: 5.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"JSON summary output path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--metadata-parquet",
        type=Path,
        help=(
            "Optional BigEarthNet v2 metadata.parquet path. If omitted, the "
            "script auto-detects metadata.parquet under --root, then under "
            "--root/.. . When readable, complete windows are also classified "
            "by official split."
        ),
    )
    parser.add_argument(
        "--ignore-metadata",
        action="store_true",
        help="Do not auto-detect or read metadata.parquet.",
    )
    parser.add_argument(
        "--max-band-shape-checks",
        type=int,
        default=2000,
        help="Maximum number of band files to open for shape/tag sampling.",
    )
    parser.add_argument(
        "--max-transform-checks",
        type=int,
        default=100,
        help="Maximum complete windows to validate with GeoTIFF tiepoint tags.",
    )
    parser.add_argument(
        "--example-output-dir",
        type=Path,
        default=DEFAULT_EXAMPLE_DIR,
        help=(
            "Directory for one stitched uint16 .npy and contrast-stretched PNG "
            f"preview. Default: {DEFAULT_EXAMPLE_DIR}"
        ),
    )
    parser.add_argument(
        "--no-example",
        action="store_true",
        help="Skip writing the example mosaic artifact.",
    )
    return parser.parse_args()


def find_metadata_path(root: Path, requested: Path | None, ignore: bool) -> Path | None:
    if ignore:
        return None
    if requested is not None:
        return requested
    candidates = [
        root / "metadata.parquet",
        root.parent / "metadata.parquet",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_metadata_splits(metadata_path: Path | None) -> tuple[dict[str, str] | None, dict[str, Any]]:
    if metadata_path is None:
        return None, {"path": None, "status": "not_found"}
    if not metadata_path.exists():
        return None, {"path": str(metadata_path), "status": "missing"}

    try:
        import pandas as pd
    except ImportError as exc:
        return None, {
            "path": str(metadata_path),
            "status": "unreadable",
            "error": f"pandas with parquet support is required: {exc}",
        }

    try:
        dataframe = pd.read_parquet(
            metadata_path,
            columns=["patch_id", "split"],
        )
    except Exception as exc:  # pragma: no cover - depends on local parquet engine.
        return None, {
            "path": str(metadata_path),
            "status": "unreadable",
            "error": str(exc),
        }

    split_by_patch = dict(zip(dataframe["patch_id"], dataframe["split"], strict=True))
    split_counts = dataframe["split"].value_counts(dropna=False).to_dict()
    split_counts = {str(key): int(value) for key, value in split_counts.items()}
    return split_by_patch, {
        "path": str(metadata_path),
        "status": "loaded",
        "row_count": int(len(dataframe)),
        "split_counts": split_counts,
        "unique_splits": sorted(split_counts),
        "duplicate_patch_ids": int(dataframe["patch_id"].duplicated().sum()),
        "null_patch_ids": int(dataframe["patch_id"].isna().sum()),
        "null_splits": int(dataframe["split"].isna().sum()),
    }


def parse_patch_dir(scene_name: str, patch_dir: Path) -> tuple[int, int] | None:
    prefix = f"{scene_name}_"
    if not patch_dir.name.startswith(prefix):
        return None
    suffix = patch_dir.name[len(prefix) :]
    parts = suffix.split("_")
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return None


def band_path(patch: PatchRef, band: str) -> Path:
    return patch.path / f"{patch.patch_name}_{band}.tif"


def scan_scene(scene_dir: Path, band: str) -> tuple[dict[tuple[int, int], PatchRef], int]:
    patches: dict[tuple[int, int], PatchRef] = {}
    duplicate_count = 0
    for patch_dir in scene_dir.iterdir():
        if not patch_dir.is_dir():
            continue
        parsed = parse_patch_dir(scene_dir.name, patch_dir)
        if parsed is None:
            continue
        h, v = parsed
        key = (h, v)
        if key in patches:
            duplicate_count += 1
            continue
        patches[key] = PatchRef(
            scene_name=scene_dir.name,
            patch_name=patch_dir.name,
            h=h,
            v=v,
            path=patch_dir,
        )
    return patches, duplicate_count


def is_complete_window(
    patch_keys: set[tuple[int, int]],
    anchor: tuple[int, int],
    window_size: int,
) -> bool:
    h0, v0 = anchor
    return all(
        (h0 + dh, v0 + dv) in patch_keys
        for dv in range(window_size)
        for dh in range(window_size)
    )


def window_patch_names(
    patches: dict[tuple[int, int], PatchRef],
    anchor: tuple[int, int],
    window_size: int,
) -> list[str]:
    h0, v0 = anchor
    return [
        patches[(h0 + dh, v0 + dv)].patch_name
        for dv in range(window_size)
        for dh in range(window_size)
    ]


def classify_window_split(
    patch_names: list[str],
    split_by_patch: dict[str, str],
) -> tuple[str, str | None]:
    splits: set[str] = set()
    for patch_name in patch_names:
        split = split_by_patch.get(patch_name)
        if split is None:
            return "metadata_incomplete", None
        splits.add(str(split))
    if len(splits) == 1:
        return "split_compliant", next(iter(splits))
    return "mixed_split", None


def summarize_scene(
    scene_dir: Path,
    patches: dict[tuple[int, int], PatchRef],
    duplicate_count: int,
    band: str,
    window_size: int,
    split_by_patch: dict[str, str] | None = None,
) -> dict[str, Any]:
    if not patches:
        return {
            "scene": scene_dir.name,
            "patch_count": 0,
            "duplicate_grid_positions": duplicate_count,
        }

    patch_keys = set(patches)
    hs = [h for h, _ in patch_keys]
    vs = [v for _, v in patch_keys]
    min_h, max_h = min(hs), max(hs)
    min_v, max_v = min(vs), max(vs)
    grid_width = max_h - min_h + 1
    grid_height = max_v - min_v + 1
    grid_cells = grid_width * grid_height
    missing_band_count = 0
    first_missing_band_paths: list[str] = []
    for patch in patches.values():
        candidate = band_path(patch, band)
        if not candidate.exists():
            missing_band_count += 1
            if len(first_missing_band_paths) < 10:
                first_missing_band_paths.append(str(candidate))

    complete_windows = 0
    first_complete_anchor: list[int] | None = None
    stride_offset_counts = Counter()
    split_compliant_counts: Counter[str] = Counter()
    split_compliant_stride_counts: dict[str, Counter[str]] = {}
    metadata_incomplete_windows = 0
    mixed_split_windows = 0
    max_h_anchor = max_h - window_size + 1
    max_v_anchor = max_v - window_size + 1
    if max_h_anchor >= min_h and max_v_anchor >= min_v:
        for v in range(min_v, max_v_anchor + 1):
            for h in range(min_h, max_h_anchor + 1):
                if is_complete_window(patch_keys, (h, v), window_size):
                    complete_windows += 1
                    stride_offset_counts[(h % window_size, v % window_size)] += 1
                    if split_by_patch is not None:
                        patch_names = window_patch_names(patches, (h, v), window_size)
                        status, split = classify_window_split(patch_names, split_by_patch)
                        if status == "split_compliant" and split is not None:
                            split_compliant_counts[split] += 1
                            split_compliant_stride_counts.setdefault(split, Counter())[
                                f"{h % window_size},{v % window_size}"
                            ] += 1
                        elif status == "metadata_incomplete":
                            metadata_incomplete_windows += 1
                        elif status == "mixed_split":
                            mixed_split_windows += 1
                    if first_complete_anchor is None:
                        first_complete_anchor = [h, v]

    best_offset: tuple[int, int] | None = None
    best_offset_count = 0
    if stride_offset_counts:
        best_offset, best_offset_count = stride_offset_counts.most_common(1)[0]

    stride_offsets = {
        f"{h_mod},{v_mod}": stride_offset_counts[(h_mod, v_mod)]
        for v_mod in range(window_size)
        for h_mod in range(window_size)
    }

    result = {
        "scene": scene_dir.name,
        "patch_count": len(patches),
        "duplicate_grid_positions": duplicate_count,
        "h_range": [min_h, max_h],
        "v_range": [min_v, max_v],
        "grid_width": grid_width,
        "grid_height": grid_height,
        "bounding_grid_cells": grid_cells,
        "grid_density": len(patches) / grid_cells if grid_cells else None,
        f"missing_{band}_files": missing_band_count,
        f"first_missing_{band}_files": first_missing_band_paths,
        f"complete_{window_size}x{window_size}_windows": complete_windows,
        f"first_complete_{window_size}x{window_size}_anchor": first_complete_anchor,
        "stride_offset_counts": stride_offsets,
        "best_stride_offset": list(best_offset) if best_offset is not None else None,
        "best_stride_offset_count": best_offset_count,
    }
    if split_by_patch is not None:
        result.update(
            {
                f"split_compliant_{window_size}x{window_size}_windows": dict(
                    split_compliant_counts
                ),
                f"metadata_incomplete_{window_size}x{window_size}_windows": metadata_incomplete_windows,
                f"mixed_split_{window_size}x{window_size}_windows": mixed_split_windows,
                "split_compliant_stride_offset_counts": {
                    split: dict(counter)
                    for split, counter in split_compliant_stride_counts.items()
                },
            }
        )
    return result


def sample_band_shapes(
    all_patches: list[PatchRef],
    band: str,
    max_checks: int,
) -> dict[str, Any]:
    shape_counts: Counter[str] = Counter()
    pixel_scale_counts: Counter[str] = Counter()
    geo_key_counts: Counter[str] = Counter()
    failures: list[str] = []
    if not all_patches or max_checks <= 0:
        return {
            "checked": 0,
            "shape_counts": {},
            "pixel_scale_tag_counts": {},
            "geo_ascii_tag_counts": {},
            "failures": [],
        }

    checked = 0
    if max_checks >= len(all_patches):
        sampled_indices = range(len(all_patches))
    else:
        sampled_indices = np.linspace(
            0,
            len(all_patches) - 1,
            num=max_checks,
            dtype=int,
        )

    for index in sampled_indices:
        patch = all_patches[int(index)]
        path = band_path(patch, band)
        if not path.exists():
            continue
        checked += 1
        try:
            with Image.open(path) as image:
                shape_key = f"{image.size[0]}x{image.size[1]}|{image.mode}"
                shape_counts[shape_key] += 1
                pixel_scale_counts[str(image.tag_v2.get(33550))] += 1
                geo_key_counts[str(image.tag_v2.get(34737))] += 1
        except Exception as exc:  # pragma: no cover - only for corrupt local data.
            failures.append(f"{path}: {exc}")
            if len(failures) >= 10:
                break

    return {
        "checked": checked,
        "shape_counts": dict(shape_counts),
        "pixel_scale_tag_counts": dict(pixel_scale_counts),
        "geo_ascii_tag_counts": dict(geo_key_counts),
        "failures": failures,
    }


def validate_window_transforms(
    patches: dict[tuple[int, int], PatchRef],
    anchor: tuple[int, int],
    band: str,
    window_size: int,
) -> list[str]:
    errors: list[str] = []
    h0, v0 = anchor
    top_left_patch = patches[(h0, v0)]
    top_left_path = band_path(top_left_patch, band)

    with Image.open(top_left_path) as top_left_image:
        top_left_size = top_left_image.size
        scale = top_left_image.tag_v2.get(33550)
        tiepoint = top_left_image.tag_v2.get(33922)
        geo_ascii = top_left_image.tag_v2.get(34737)

    if scale is None or tiepoint is None or len(tiepoint) < 6:
        return [f"{top_left_path}: missing GeoTIFF scale or tiepoint tags"]

    pixel_width = float(scale[0])
    pixel_height = float(scale[1])
    x0 = float(tiepoint[3])
    y0 = float(tiepoint[4])
    patch_width, patch_height = top_left_size
    tolerance = 1e-6

    for dv in range(window_size):
        for dh in range(window_size):
            patch = patches[(h0 + dh, v0 + dv)]
            path = band_path(patch, band)
            with Image.open(path) as image:
                if image.size != top_left_size:
                    errors.append(f"{path}: size {image.size} != {top_left_size}")
                if image.tag_v2.get(33550) != scale:
                    errors.append(f"{path}: pixel scale changed")
                if image.tag_v2.get(34737) != geo_ascii:
                    errors.append(f"{path}: CRS tag changed")
                patch_tiepoint = image.tag_v2.get(33922)
                if patch_tiepoint is None or len(patch_tiepoint) < 6:
                    errors.append(f"{path}: missing tiepoint")
                    continue
                expected_x = x0 + dh * patch_width * pixel_width
                expected_y = y0 - dv * patch_height * pixel_height
                actual_x = float(patch_tiepoint[3])
                actual_y = float(patch_tiepoint[4])
                if abs(actual_x - expected_x) > tolerance:
                    errors.append(f"{path}: x {actual_x} != {expected_x}")
                if abs(actual_y - expected_y) > tolerance:
                    errors.append(f"{path}: y {actual_y} != {expected_y}")
    return errors


def sample_transform_checks(
    scene_patches: dict[str, dict[tuple[int, int], PatchRef]],
    scene_summaries: list[dict[str, Any]],
    band: str,
    window_size: int,
    max_checks: int,
) -> dict[str, Any]:
    checked = 0
    failed = 0
    failures: list[dict[str, Any]] = []
    for summary in scene_summaries:
        if checked >= max_checks:
            break
        anchor = summary.get(f"first_complete_{window_size}x{window_size}_anchor")
        if anchor is None:
            continue
        scene = summary["scene"]
        errors = validate_window_transforms(
            scene_patches[scene],
            (int(anchor[0]), int(anchor[1])),
            band,
            window_size,
        )
        checked += 1
        if errors:
            failed += 1
            failures.append({"scene": scene, "anchor": anchor, "errors": errors[:10]})
    return {
        "checked_windows": checked,
        "failed_windows": failed,
        "failures": failures[:10],
    }


def write_example_mosaic(
    scene_patches: dict[str, dict[tuple[int, int], PatchRef]],
    scene_summaries: list[dict[str, Any]],
    band: str,
    window_size: int,
    output_dir: Path,
) -> dict[str, Any] | None:
    selected_summary: dict[str, Any] | None = None
    selected_anchor: list[int] | None = None
    for summary in scene_summaries:
        anchor = summary.get(f"first_complete_{window_size}x{window_size}_anchor")
        if anchor is not None:
            selected_summary = summary
            selected_anchor = anchor
            break

    if selected_summary is None or selected_anchor is None:
        return None

    scene = selected_summary["scene"]
    h0, v0 = int(selected_anchor[0]), int(selected_anchor[1])
    patches = scene_patches[scene]
    arrays: list[list[np.ndarray]] = []
    dtype: np.dtype[Any] | None = None
    patch_shape: tuple[int, int] | None = None

    for dv in range(window_size):
        row: list[np.ndarray] = []
        for dh in range(window_size):
            patch = patches[(h0 + dh, v0 + dv)]
            with Image.open(band_path(patch, band)) as image:
                array = np.asarray(image)
            if dtype is None:
                dtype = array.dtype
            if patch_shape is None:
                patch_shape = array.shape
            if array.dtype != dtype:
                raise ValueError(f"Mixed dtypes in example window for {scene} {h0}_{v0}")
            if array.shape != patch_shape:
                raise ValueError(f"Mixed shapes in example window for {scene} {h0}_{v0}")
            row.append(array)
        arrays.append(row)

    mosaic = np.block(arrays)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{scene}_{h0:02d}_{v0:02d}_{band}_{window_size}x{window_size}"
    npy_path = output_dir / f"{stem}.npy"
    png_path = output_dir / f"{stem}_preview.png"
    np.save(npy_path, mosaic)

    lo, hi = np.percentile(mosaic, [2, 98])
    if hi <= lo:
        preview = np.zeros(mosaic.shape, dtype=np.uint8)
    else:
        preview = np.clip((mosaic.astype(np.float32) - lo) / (hi - lo), 0, 1)
        preview = (preview * 255).astype(np.uint8)
    Image.fromarray(preview).save(png_path)

    return {
        "scene": scene,
        "anchor": [h0, v0],
        "band": band,
        "window_size": window_size,
        "patch_shape": list(patch_shape or []),
        "mosaic_shape": list(mosaic.shape),
        "dtype": str(mosaic.dtype),
        "npy": str(npy_path),
        "preview_png": str(png_path),
        "preview_percentiles": [float(lo), float(hi)],
    }


def main() -> None:
    args = parse_args()
    root = args.root
    if not root.exists():
        raise FileNotFoundError(root)

    metadata_path = find_metadata_path(root, args.metadata_parquet, args.ignore_metadata)
    split_by_patch, metadata_report = load_metadata_splits(metadata_path)

    scene_dirs = sorted(path for path in root.iterdir() if path.is_dir())
    scene_patches: dict[str, dict[tuple[int, int], PatchRef]] = {}
    scene_summaries: list[dict[str, Any]] = []
    all_patches: list[PatchRef] = []

    for scene_dir in scene_dirs:
        patches, duplicate_count = scan_scene(scene_dir, args.band)
        scene_patches[scene_dir.name] = patches
        all_patches.extend(patches.values())
        scene_summaries.append(
            summarize_scene(
                scene_dir=scene_dir,
                patches=patches,
                duplicate_count=duplicate_count,
                band=args.band,
                window_size=args.window_size,
                split_by_patch=split_by_patch,
            )
        )

    window_key = f"complete_{args.window_size}x{args.window_size}_windows"
    total_windows = sum(int(summary.get(window_key, 0)) for summary in scene_summaries)
    missing_band_key = f"missing_{args.band}_files"
    total_missing_band_files = sum(
        int(summary.get(missing_band_key, 0)) for summary in scene_summaries
    )
    total_bounding_grid_cells = sum(
        int(summary.get("bounding_grid_cells", 0)) for summary in scene_summaries
    )
    stride_offset_totals: Counter[str] = Counter()
    split_compliant_totals: Counter[str] = Counter()
    split_compliant_stride_totals: dict[str, Counter[str]] = {}
    for summary in scene_summaries:
        stride_offset_totals.update(summary.get("stride_offset_counts", {}))
        split_counts = summary.get(
            f"split_compliant_{args.window_size}x{args.window_size}_windows",
            {},
        )
        split_compliant_totals.update(split_counts)
        for split, counts in summary.get("split_compliant_stride_offset_counts", {}).items():
            split_compliant_stride_totals.setdefault(split, Counter()).update(counts)

    zero_offset_key = "0,0"
    per_scene_best_stride_total = sum(
        int(summary.get("best_stride_offset_count", 0)) for summary in scene_summaries
    )
    split_per_scene_best_stride_totals: dict[str, int] = {}
    for split, stride_counts in split_compliant_stride_totals.items():
        split_per_scene_best_stride_totals[split] = 0
    for summary in scene_summaries:
        split_stride_counts = summary.get("split_compliant_stride_offset_counts", {})
        for split, stride_counts in split_stride_counts.items():
            split_per_scene_best_stride_totals[split] = (
                split_per_scene_best_stride_totals.get(split, 0)
                + max((int(value) for value in stride_counts.values()), default=0)
            )
    best_global_stride_key, best_global_stride_count = max(
        stride_offset_totals.items(),
        key=lambda item: item[1],
        default=(zero_offset_key, 0),
    )
    best_global_stride_offset = [
        int(part) for part in best_global_stride_key.split(",")
    ]

    report: dict[str, Any] = {
        "root": str(root),
        "band": args.band,
        "window_size": args.window_size,
        "scene_count": len(scene_dirs),
        "patch_count": len(all_patches),
        f"missing_{args.band}_files": total_missing_band_files,
        "total_bounding_grid_cells": total_bounding_grid_cells,
        "overall_grid_density": len(all_patches) / total_bounding_grid_cells
        if total_bounding_grid_cells
        else None,
        "expected_mosaic_pixels_if_120px_patch": [
            args.window_size * 120,
            args.window_size * 120,
        ],
        "metadata_sidecars": {
            "metadata.parquet": str(metadata_path) if metadata_path else None,
            "metadata_for_patches_with_snow_cloud_or_shadow.parquet": next(
                (
                    str(candidate)
                    for candidate in [
                        root / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
                        root.parent
                        / "metadata_for_patches_with_snow_cloud_or_shadow.parquet",
                    ]
                    if candidate.exists()
                ),
                None,
            ),
        },
        "metadata": metadata_report,
        f"total_complete_{args.window_size}x{args.window_size}_windows": total_windows,
        "stride_offset_totals": dict(
            sorted(stride_offset_totals.items(), key=lambda item: item[0])
        ),
        "stride_zero_zero_complete_windows": int(stride_offset_totals[zero_offset_key]),
        "best_global_stride_offset": {
            "offset": best_global_stride_offset,
            "complete_windows": int(best_global_stride_count),
        },
        "per_scene_best_stride_complete_windows": per_scene_best_stride_total,
        f"scenes_with_complete_{args.window_size}x{args.window_size}_windows": sum(
            1 for summary in scene_summaries if int(summary.get(window_key, 0)) > 0
        ),
        "band_shape_sample": sample_band_shapes(
            all_patches,
            args.band,
            args.max_band_shape_checks,
        ),
        "transform_check_sample": sample_transform_checks(
            scene_patches,
            scene_summaries,
            args.band,
            args.window_size,
            args.max_transform_checks,
        ),
        "scenes": scene_summaries,
    }

    if split_by_patch is not None:
        split_stride_offset_totals = {
            split: dict(sorted(stride_counts.items()))
            for split, stride_counts in split_compliant_stride_totals.items()
        }
        all_split_stride_totals = Counter()
        for stride_counts in split_compliant_stride_totals.values():
            all_split_stride_totals.update(stride_counts)
        best_total_stride_key, best_total_stride_count = max(
            all_split_stride_totals.items(),
            key=lambda item: item[1],
            default=(zero_offset_key, 0),
        )
        split_stride_zero_zero = {
            split: int(stride_counts.get(zero_offset_key, 0))
            for split, stride_counts in split_compliant_stride_totals.items()
        }
        split_best_global_stride = {}
        for split, stride_counts in split_compliant_stride_totals.items():
            best_key, best_count = max(
                stride_counts.items(),
                key=lambda item: item[1],
                default=(zero_offset_key, 0),
            )
            split_best_global_stride[split] = {
                "offset": [int(part) for part in best_key.split(",")],
                "complete_windows": int(best_count),
            }
        report["split_compliant_windows"] = {
            "sliding": {
                split: int(count)
                for split, count in sorted(split_compliant_totals.items())
            },
            "stride_offset_totals": split_stride_offset_totals,
            "stride_zero_zero": dict(sorted(split_stride_zero_zero.items())),
            "best_total_stride_offset": {
                "offset": [int(part) for part in best_total_stride_key.split(",")],
                "complete_windows": int(best_total_stride_count),
            },
            "best_global_stride_offset": split_best_global_stride,
            "per_scene_best_stride_offset": dict(
                sorted(split_per_scene_best_stride_totals.items())
            ),
            "metadata_incomplete_sliding": sum(
                int(
                    summary.get(
                        f"metadata_incomplete_{args.window_size}x{args.window_size}_windows",
                        0,
                    )
                )
                for summary in scene_summaries
            ),
            "mixed_split_sliding": sum(
                int(summary.get(f"mixed_split_{args.window_size}x{args.window_size}_windows", 0))
                for summary in scene_summaries
            ),
        }

    if not args.no_example:
        report["example_mosaic"] = write_example_mosaic(
            scene_patches,
            scene_summaries,
            args.band,
            args.window_size,
            args.example_output_dir,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, sort_keys=True)
        file.write("\n")

    print(f"Wrote {args.output}")
    print(f"Scenes: {report['scene_count']}")
    print(f"Patches: {report['patch_count']}")
    print(f"Complete {args.window_size}x{args.window_size} windows: {total_windows}")
    print(
        "Stride-5 offset 0,0 windows: "
        f"{report['stride_zero_zero_complete_windows']}"
    )
    print(
        "Per-scene best stride-offset windows: "
        f"{report['per_scene_best_stride_complete_windows']}"
    )
    if split_by_patch is not None:
        print(
            "Split-compliant sliding windows: "
            f"{report['split_compliant_windows']['sliding']}"
        )
        print(
            "Split-compliant stride-5 offset 0,0 windows: "
            f"{report['split_compliant_windows']['stride_zero_zero']}"
        )
    if report.get("example_mosaic"):
        print(f"Example preview: {report['example_mosaic']['preview_png']}")


if __name__ == "__main__":
    main()

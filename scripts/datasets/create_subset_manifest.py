#!/usr/bin/env python3
"""Create lightweight nested subset manifests from a dataset manifest."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


Sample = dict[str, Any]

FIXED_AREA_BINS = (
    ("tiny", 0.0, 0.0025),
    ("small", 0.0025, 0.0100),
    ("medium", 0.0100, 0.0500),
    ("large", 0.0500, math.inf),
)
STRATIFICATION_FIELDS = [
    "dominant_class_group",
    "object_count_bin",
    "bbox_area_bin",
    "bbox_y_bin",
]


@dataclass(frozen=True)
class AreaBinSpec:
    mode: str
    labels: list[str]
    thresholds: list[float]


@dataclass(frozen=True)
class ImageSummary:
    sample: Sample
    sample_id: str
    num_instances: int
    class_counts: dict[str, int]
    dominant_class_group: str
    median_bbox_relative_area: float | None
    median_bbox_center_y: float | None
    median_bbox_aspect_ratio: float | None
    object_count_bin: str
    bbox_area_bin: str
    bbox_y_bin: str
    stratum: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create lightweight subset manifests without copying images or "
            "annotations."
        ),
        epilog=(
            "Examples:\n"
            "  Build one subset from a full manifest:\n"
            "    python scripts/datasets/create_subset_manifest.py \\\n"
            "      --source-manifest data/train_full_manifest.json \\\n"
            "      --method bbox_stratified \\\n"
            "      --num-samples 5000 \\\n"
            "      --output data/subsets/train_5000.json\n"
            "\n"
            "  Build a smaller nested subset from an existing larger subset:\n"
            "    python scripts/datasets/create_subset_manifest.py \\\n"
            "      --source-manifest data/train_full_manifest.json \\\n"
            "      --input-manifest data/subsets/train_10000.json \\\n"
            "      --method bbox_stratified \\\n"
            "      --num-samples 5000 \\\n"
            "      --output data/subsets/train_5000.json\n"
            "\n"
            "  Build nested subset sizes in one command:\n"
            "    python scripts/datasets/create_subset_manifest.py \\\n"
            "      --source-manifest data/train_full_manifest.json \\\n"
            "      --method bbox_stratified \\\n"
            "      --subset-sizes 5000 10000 \\\n"
            "      --output-dir data/subsets \\\n"
            "      --subset-name-prefix train\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        required=True,
        help=(
            "Path to the full source dataset manifest JSON. The script reads "
            "sample entries from this file and records this path in outputs."
        ),
    )
    parser.add_argument(
        "--input-manifest",
        type=Path,
        help=(
            "Optional existing subset manifest to sample from instead of the "
            "full source manifest. Use this to build a smaller nested subset."
        ),
    )
    parser.add_argument(
        "--method",
        choices=["bbox_stratified", "random", "tile_balanced"],
        required=True,
        help=(
            "Sampling method. bbox_stratified preserves image-level bbox strata; "
            "random performs deterministic uniform sampling without replacement; "
            "tile_balanced samples evenly across a tile/scene field."
        ),
    )

    single = parser.add_argument_group("single-output mode")
    single.add_argument(
        "--num-samples",
        type=int,
        help="Number of samples to select for one output manifest.",
    )
    single.add_argument(
        "--output",
        type=Path,
        help="Path where the single output subset manifest JSON will be written.",
    )
    single.add_argument(
        "--diagnostics-output",
        type=Path,
        help=(
            "Optional path for a standalone diagnostics JSON in single-output "
            "mode. Diagnostics are also embedded in the manifest."
        ),
    )

    multi = parser.add_argument_group("multi-output mode")
    multi.add_argument(
        "--subset-sizes",
        type=int,
        nargs="+",
        help=(
            "Strictly increasing subset sizes to build in one command, for "
            "example: --subset-sizes 5000 10000."
        ),
    )
    multi.add_argument(
        "--output-dir",
        type=Path,
        help="Directory where multi-output subset manifests will be written.",
    )
    multi.add_argument(
        "--subset-name-prefix",
        type=str,
        help=(
            "Filename prefix for multi-output manifests. A prefix of 'train' "
            "with size 5000 writes train_5000.json."
        ),
    )
    multi.add_argument(
        "--subset-name-suffix",
        type=str,
        help=(
            "Optional filename suffix for multi-output manifests. A prefix of "
            "'train', size 5000, and suffix 'tile_balanced' writes "
            "train_5000_tile_balanced.json."
        ),
    )
    multi.add_argument(
        "--diagnostics-dir",
        type=Path,
        help=(
            "Optional directory for one standalone diagnostics JSON per subset "
            "in multi-output mode."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for deterministic sampling. Default: 42.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing output manifest or diagnostics files.",
    )
    parser.add_argument(
        "--area-binning",
        choices=["quantile", "fixed"],
        default="quantile",
        help=(
            "BBox relative-area binning for bbox_stratified and diagnostics. "
            "'quantile' learns bins from candidate boxes; 'fixed' uses tiny, "
            "small, medium, large thresholds. Default: quantile."
        ),
    )
    parser.add_argument(
        "--num-area-bins",
        type=int,
        default=4,
        help="Number of quantile area bins when --area-binning quantile is used. Default: 4.",
    )
    parser.add_argument(
        "--y-position-bins",
        type=int,
        default=3,
        help=(
            "Number of median bbox center-y bins used in image-level strata. "
            "With 3 bins, labels are upper, middle, lower. Default: 3."
        ),
    )
    parser.add_argument(
        "--position-grid",
        type=int,
        default=3,
        help=(
            "Grid side length for bbox-center position diagnostics. A value of "
            "3 means a 3x3 grid. Default: 3."
        ),
    )
    parser.add_argument(
        "--min-stratum-size",
        type=int,
        default=1,
        help=(
            "Merge strata smaller than this count into __rare__. Default 1 "
            "preserves every stratum."
        ),
    )
    parser.add_argument(
        "--tile-field",
        type=str,
        default="scene",
        help="Sample field used as the tile key for --method tile_balanced. Default: scene.",
    )
    parser.add_argument(
        "--sample-id-field",
        type=str,
        default="id",
        help=(
            "Field to use as sample ID when normalizing manifests. Use sample_id "
            "for BigEarthNet JSONL manifests. Default: id."
        ),
    )
    parser.add_argument(
        "--sample-path-field",
        type=str,
        default="path",
        help=(
            "Field to use as sample path when normalizing manifests. Use image_path "
            "for BigEarthNet JSONL manifests. Default: path."
        ),
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    single_mode = args.num_samples is not None or args.output is not None
    multi_mode = (
        args.subset_sizes is not None
        or args.output_dir is not None
        or args.subset_name_prefix is not None
    )
    if single_mode == multi_mode:
        raise ValueError(
            "Choose exactly one mode: single-output (--num-samples and --output) "
            "or multi-output (--subset-sizes, --output-dir, --subset-name-prefix)."
        )
    if single_mode:
        if args.num_samples is None or args.output is None:
            raise ValueError("Single-output mode requires --num-samples and --output.")
        if args.diagnostics_dir is not None:
            raise ValueError("--diagnostics-dir is only valid in multi-output mode.")
    if multi_mode:
        if args.subset_sizes is None or args.output_dir is None or not args.subset_name_prefix:
            raise ValueError(
                "Multi-output mode requires --subset-sizes, --output-dir, "
                "and --subset-name-prefix."
            )
        if args.diagnostics_output is not None:
            raise ValueError("--diagnostics-output is only valid in single-output mode.")
        if sorted(args.subset_sizes) != list(args.subset_sizes) or len(set(args.subset_sizes)) != len(args.subset_sizes):
            raise ValueError("--subset-sizes must be strictly increasing.")
    if args.num_samples is not None and args.num_samples <= 0:
        raise ValueError("--num-samples must be > 0.")
    if args.subset_sizes is not None and any(size <= 0 for size in args.subset_sizes):
        raise ValueError("All --subset-sizes values must be > 0.")
    if args.num_area_bins <= 0:
        raise ValueError("--num-area-bins must be > 0.")
    if args.y_position_bins <= 0:
        raise ValueError("--y-position-bins must be > 0.")
    if args.position_grid <= 0:
        raise ValueError("--position-grid must be > 0.")
    if args.min_stratum_size <= 0:
        raise ValueError("--min-stratum-size must be > 0.")
    if args.method == "tile_balanced":
        if not args.tile_field:
            raise ValueError("--tile-field is required for --method tile_balanced.")
        if not args.sample_id_field:
            raise ValueError("--sample-id-field is required.")
        if not args.sample_path_field:
            raise ValueError("--sample-path-field is required.")


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")

    if path.suffix.lower() == ".jsonl":
        samples: list[Sample] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    sample = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Manifest JSONL line {line_number} is not valid JSON: {path}"
                    ) from exc
                if not isinstance(sample, dict):
                    raise ValueError(
                        f"Manifest JSONL line {line_number} must be an object: {path}"
                    )
                samples.append(sample)
        manifest = {"manifest_format": "jsonl", "samples": samples}
        validate_manifest(manifest, path)
        return manifest

    try:
        with path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Manifest is not valid JSON: {path}") from exc
    validate_manifest(manifest, path)
    return manifest


def validate_manifest(manifest: dict[str, Any], path: Path) -> None:
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be a JSON object: {path}")
    samples = manifest.get("samples")
    if not isinstance(samples, list):
        raise ValueError(f'Manifest must contain a "samples" list: {path}')
    if not all(isinstance(sample, dict) for sample in samples):
        raise ValueError(f'Every entry in "samples" must be an object: {path}')


def validate_samples_for_method(
    samples: Sequence[Sample],
    method: str,
    args: argparse.Namespace | None = None,
) -> None:
    if method in {"random", "tile_balanced"}:
        required = ["id", "path"]
    else:
        required = ["id", "path", "width", "height", "boxes"]
    for index, sample in enumerate(samples):
        for field in required:
            if field not in sample:
                raise ValueError(f"Sample at index {index} is missing required field {field!r}.")
        if method == "tile_balanced" and args is not None and args.tile_field not in sample:
            raise ValueError(
                f"Sample at index {index} is missing tile field {args.tile_field!r}."
            )
        if method == "bbox_stratified":
            boxes = sample["boxes"]
            if not isinstance(boxes, list):
                raise ValueError(f"Sample {sample['id']!r} has non-list boxes.")
            width = sample["width"]
            height = sample["height"]
            if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
                raise ValueError(f"Sample {sample['id']!r} width/height must be numeric.")
            if width <= 0 or height <= 0:
                raise ValueError(f"Sample {sample['id']!r} width/height must be positive.")
            for box_index, box in enumerate(boxes):
                if not isinstance(box, dict):
                    raise ValueError(f"Sample {sample['id']!r} box {box_index} must be an object.")
                if "bbox_xyxy" not in box:
                    raise ValueError(f"Sample {sample['id']!r} box {box_index} is missing bbox_xyxy.")
                bbox = box["bbox_xyxy"]
                if not isinstance(bbox, list) or len(bbox) != 4:
                    raise ValueError(
                        f"Sample {sample['id']!r} box {box_index} bbox_xyxy must be a 4-item list."
                    )
                try:
                    x1, y1, x2, y2 = [float(value) for value in bbox]
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Sample {sample['id']!r} box {box_index} bbox_xyxy values must be numeric."
                    ) from exc
                if not all(math.isfinite(value) for value in [x1, y1, x2, y2]):
                    raise ValueError(
                        f"Sample {sample['id']!r} box {box_index} bbox_xyxy values must be finite."
                    )
                if x2 <= x1 or y2 <= y1:
                    raise ValueError(
                        f"Sample {sample['id']!r} box {box_index} must have x2 > x1 and y2 > y1."
                    )
                if "class_id" not in box and "class_name" not in box:
                    raise ValueError(
                        f"Sample {sample['id']!r} box {box_index} needs class_id or class_name."
                    )


def normalize_sample_aliases(samples: Sequence[Sample], args: argparse.Namespace) -> list[Sample]:
    """Return copies with canonical id/path aliases while preserving original fields."""
    normalized: list[Sample] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ValueError(f"Sample at index {index} must be an object.")
        normalized_sample = dict(sample)
        if "id" not in normalized_sample:
            sample_id = normalized_sample.get(args.sample_id_field)
            if sample_id is None:
                raise ValueError(
                    f"Sample at index {index} is missing id and {args.sample_id_field!r}."
                )
            normalized_sample["id"] = str(sample_id)
        if "path" not in normalized_sample:
            sample_path = normalized_sample.get(args.sample_path_field)
            if sample_path is None:
                raise ValueError(
                    f"Sample {normalized_sample['id']!r} is missing path and "
                    f"{args.sample_path_field!r}."
                )
            normalized_sample["path"] = str(sample_path)
        normalized.append(normalized_sample)
    return normalized


def validate_unique_sample_ids(samples: Sequence[Sample], context: str) -> None:
    ids = [str(sample.get("id")) for sample in samples]
    duplicates = [sample_id for sample_id, count in Counter(ids).items() if count > 1]
    if duplicates:
        preview = ", ".join(sorted(duplicates)[:10])
        raise ValueError(f"Duplicate sample IDs in {context}: {preview}")


def class_label(box: dict[str, Any]) -> str:
    if box.get("class_name") is not None:
        return str(box["class_name"])
    return str(box["class_id"])


def median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def quantile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile for an empty sequence.")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def iter_valid_box_features(sample: Sample) -> Iterable[tuple[str, float, float, float]]:
    width = float(sample["width"])
    height = float(sample["height"])
    image_area = width * height
    for box in sample.get("boxes", []):
        x1, y1, x2, y2 = [float(value) for value in box["bbox_xyxy"]]
        bbox_width = max(0.0, x2 - x1)
        bbox_height = max(0.0, y2 - y1)
        if bbox_width <= 0.0 or bbox_height <= 0.0:
            continue
        relative_area = (bbox_width * bbox_height) / image_area
        center_y = ((y1 + y2) / 2.0) / height
        aspect_ratio = bbox_width / bbox_height
        yield class_label(box), relative_area, center_y, aspect_ratio


def compute_area_bins(samples: Sequence[Sample], args: argparse.Namespace) -> AreaBinSpec:
    if args.area_binning == "fixed":
        labels = [label for label, _, _ in FIXED_AREA_BINS]
        thresholds = [upper for _, _, upper in FIXED_AREA_BINS[:-1]]
        return AreaBinSpec(mode="fixed", labels=labels, thresholds=thresholds)

    values = sorted(
        relative_area
        for sample in samples
        for _, relative_area, _, _ in iter_valid_box_features(sample)
    )
    labels = [f"q{i + 1}" for i in range(args.num_area_bins)]
    if len(values) < 2 or args.num_area_bins == 1:
        return AreaBinSpec(mode="quantile", labels=labels, thresholds=[])
    thresholds = [
        quantile(values, i / args.num_area_bins)
        for i in range(1, args.num_area_bins)
    ]
    return AreaBinSpec(mode="quantile", labels=labels, thresholds=thresholds)


def assign_area_bin(relative_area: float | None, spec: AreaBinSpec) -> str:
    if relative_area is None:
        return "none"
    if spec.mode == "fixed":
        for label, lower, upper in FIXED_AREA_BINS:
            if lower <= relative_area < upper:
                return label
        return "large"
    index = 0
    while index < len(spec.thresholds) and relative_area >= spec.thresholds[index]:
        index += 1
    return spec.labels[min(index, len(spec.labels) - 1)]


def assign_y_bin(center_y: float | None, y_bins: int) -> str:
    if center_y is None:
        return "none"
    clipped = min(max(center_y, 0.0), 1.0)
    index = min(int(clipped * y_bins), y_bins - 1)
    if y_bins == 3:
        return ["upper", "middle", "lower"][index]
    return f"y{index + 1}"


def object_count_bin(num_instances: int) -> str:
    if num_instances == 0:
        return "0"
    if num_instances == 1:
        return "1"
    if num_instances <= 3:
        return "2-3"
    if num_instances <= 7:
        return "4-7"
    return "8+"


def dominant_class_group(class_counts: dict[str, int], num_instances: int) -> str:
    if num_instances == 0:
        return "empty"
    if not class_counts:
        return "unknown"
    top_class, top_count = sorted(class_counts.items(), key=lambda item: (-item[1], item[0]))[0]
    if top_count / num_instances >= 0.60:
        return top_class
    return "mixed"


def make_stratum_key(
    dominant_group: str,
    count_bin: str,
    area_bin: str,
    y_bin: str,
) -> str:
    return json.dumps([dominant_group, count_bin, area_bin, y_bin], separators=(",", ":"))


def compute_detection_image_summaries(
    samples: Sequence[Sample],
    args: argparse.Namespace,
    area_spec: AreaBinSpec | None = None,
) -> list[ImageSummary]:
    if area_spec is None:
        area_spec = compute_area_bins(samples, args)
    summaries = []
    for sample in samples:
        features = list(iter_valid_box_features(sample))
        num_instances = len(sample.get("boxes", []))
        class_counts = Counter(class_name for class_name, _, _, _ in features)
        relative_areas = [relative_area for _, relative_area, _, _ in features]
        center_ys = [center_y for _, _, center_y, _ in features]
        aspect_ratios = [aspect_ratio for _, _, _, aspect_ratio in features]
        median_area = median(relative_areas)
        median_center_y = median(center_ys)
        median_aspect_ratio = median(aspect_ratios)
        count_bin = object_count_bin(num_instances)
        area_bin = assign_area_bin(median_area, area_spec)
        y_bin = assign_y_bin(median_center_y, args.y_position_bins)
        dominant_group = dominant_class_group(dict(class_counts), num_instances)
        stratum = make_stratum_key(dominant_group, count_bin, area_bin, y_bin)
        summaries.append(
            ImageSummary(
                sample=sample,
                sample_id=str(sample["id"]),
                num_instances=num_instances,
                class_counts=dict(class_counts),
                dominant_class_group=dominant_group,
                median_bbox_relative_area=median_area,
                median_bbox_center_y=median_center_y,
                median_bbox_aspect_ratio=median_aspect_ratio,
                object_count_bin=count_bin,
                bbox_area_bin=area_bin,
                bbox_y_bin=y_bin,
                stratum=stratum,
            )
        )
    return summaries


def assign_detection_strata(
    summaries: Sequence[ImageSummary],
    min_stratum_size: int,
) -> dict[str, list[Sample]]:
    grouped: dict[str, list[ImageSummary]] = defaultdict(list)
    for summary in summaries:
        grouped[summary.stratum].append(summary)

    strata: dict[str, list[Sample]] = defaultdict(list)
    for stratum, members in grouped.items():
        target_stratum = "__rare__" if len(members) < min_stratum_size else stratum
        strata[target_stratum].extend(summary.sample for summary in members)

    return {
        stratum: sorted(samples, key=lambda sample: str(sample["id"]))
        for stratum, samples in strata.items()
    }


def proportional_allocation(strata_sizes: dict[str, int], target: int) -> dict[str, int]:
    total = sum(strata_sizes.values())
    if target <= 0:
        raise ValueError("Target sample count must be > 0.")
    if target > total:
        raise ValueError(f"Requested {target} samples but only {total} candidates are available.")

    allocations: dict[str, int] = {}
    remainders: dict[str, float] = {}
    for stratum, size in strata_sizes.items():
        exact = target * size / total
        allocation = min(math.floor(exact), size)
        allocations[stratum] = allocation
        remainders[stratum] = exact - allocation

    remaining = target - sum(allocations.values())
    while remaining > 0:
        eligible = [
            stratum
            for stratum, size in strata_sizes.items()
            if allocations[stratum] < size
        ]
        if not eligible:
            raise RuntimeError("Unable to allocate all requested samples across strata.")
        made_progress = False
        for stratum in sorted(eligible, key=lambda key: (-remainders[key], key)):
            if remaining == 0:
                break
            if allocations[stratum] >= strata_sizes[stratum]:
                continue
            allocations[stratum] += 1
            remaining -= 1
            made_progress = True
        if not made_progress:
            raise RuntimeError("Unable to allocate all requested samples across strata.")
    return allocations


def stratified_sample(
    strata: dict[str, list[Sample]],
    target: int,
    seed: int,
) -> list[Sample]:
    allocations = proportional_allocation(
        {stratum: len(samples) for stratum, samples in strata.items()},
        target,
    )
    rng = random.Random(seed)
    selected: list[Sample] = []
    for stratum in sorted(strata):
        candidates = sorted(strata[stratum], key=lambda sample: str(sample["id"]))
        count = allocations[stratum]
        if count == len(candidates):
            selected.extend(candidates)
        else:
            selected.extend(rng.sample(candidates, count))
    selected = sorted(selected, key=lambda sample: str(sample["id"]))
    validate_unique_sample_ids(selected, "selected samples")
    return selected


def random_sample(samples: Sequence[Sample], target: int, seed: int) -> list[Sample]:
    candidates = sorted(samples, key=lambda sample: str(sample["id"]))
    rng = random.Random(seed)
    selected = sorted(rng.sample(candidates, target), key=lambda sample: str(sample["id"]))
    validate_unique_sample_ids(selected, "selected samples")
    return selected


def group_samples_by_tile(
    samples: Sequence[Sample],
    tile_field: str,
) -> dict[str, list[Sample]]:
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for index, sample in enumerate(samples):
        tile = sample.get(tile_field)
        if tile is None:
            raise ValueError(
                f"Sample {sample.get('id', index)!r} is missing tile field {tile_field!r}."
            )
        grouped[str(tile)].append(sample)
    return {
        tile: sorted(tile_samples, key=lambda sample: str(sample["id"]))
        for tile, tile_samples in grouped.items()
    }


def tile_balanced_allocation(
    tile_sizes: dict[str, int],
    target: int,
) -> dict[str, int]:
    """Allocate an exact target as evenly as possible without oversampling."""
    if target <= 0:
        raise ValueError("Target sample count must be > 0.")
    total = sum(tile_sizes.values())
    if target > total:
        raise ValueError(f"Requested {target} samples but only {total} candidates are available.")
    if not tile_sizes:
        raise ValueError("Cannot allocate a tile-balanced subset from zero tiles.")

    tiles = sorted(tile_sizes)
    base = target // len(tiles)
    allocations = {tile: min(size, base) for tile, size in tile_sizes.items()}
    remaining = target - sum(allocations.values())

    while remaining > 0:
        eligible = [tile for tile in tiles if allocations[tile] < tile_sizes[tile]]
        if not eligible:
            raise RuntimeError("Unable to allocate all requested samples across tiles.")
        # Water-fill: raise the lowest selected tile counts first. Ties prefer
        # tiles with more remaining capacity, then tile name for determinism.
        eligible = sorted(
            eligible,
            key=lambda tile: (
                allocations[tile],
                -(tile_sizes[tile] - allocations[tile]),
                tile,
            ),
        )
        made_progress = False
        for tile in eligible:
            if remaining == 0:
                break
            if allocations[tile] >= tile_sizes[tile]:
                continue
            allocations[tile] += 1
            remaining -= 1
            made_progress = True
        if not made_progress:
            raise RuntimeError("Unable to allocate all requested samples across tiles.")

    return {tile: allocations[tile] for tile in tiles}


def tile_balanced_sample(
    samples: Sequence[Sample],
    target: int,
    seed: int,
    tile_field: str,
) -> list[Sample]:
    grouped = group_samples_by_tile(samples, tile_field)
    allocations = tile_balanced_allocation(
        {tile: len(tile_samples) for tile, tile_samples in grouped.items()},
        target,
    )
    rng = random.Random(seed)
    selected: list[Sample] = []
    for tile in sorted(grouped):
        candidates = grouped[tile]
        count = allocations[tile]
        if count == len(candidates):
            selected.extend(candidates)
        else:
            selected.extend(rng.sample(candidates, count))
    selected = sorted(
        selected,
        key=lambda sample: (str(sample.get(tile_field)), str(sample["id"])),
    )
    validate_unique_sample_ids(selected, "selected samples")
    return selected


def histogram_distribution(values: Iterable[str]) -> dict[str, float]:
    counts = Counter(values)
    total = sum(counts.values())
    if total == 0:
        return {}
    return {key: count / total for key, count in counts.items()}


def js_divergence(p_dist: dict[str, float], q_dist: dict[str, float]) -> float:
    keys = set(p_dist) | set(q_dist)
    if not keys:
        return 0.0
    midpoint = {key: 0.5 * (p_dist.get(key, 0.0) + q_dist.get(key, 0.0)) for key in keys}
    return 0.5 * kl_divergence(p_dist, midpoint, keys) + 0.5 * kl_divergence(q_dist, midpoint, keys)


def kl_divergence(
    p_dist: dict[str, float],
    q_dist: dict[str, float],
    keys: Iterable[str],
) -> float:
    total = 0.0
    for key in keys:
        p_value = p_dist.get(key, 0.0)
        q_value = q_dist.get(key, 0.0)
        if p_value > 0.0 and q_value > 0.0:
            total += p_value * math.log(p_value / q_value, 2)
    return total


def l1_distance(p_dist: dict[str, float], q_dist: dict[str, float]) -> float:
    keys = set(p_dist) | set(q_dist)
    return sum(abs(p_dist.get(key, 0.0) - q_dist.get(key, 0.0)) for key in keys)


def bbox_class_distribution(samples: Sequence[Sample]) -> dict[str, float]:
    labels = [
        class_label(box)
        for sample in samples
        for box in sample.get("boxes", [])
        if isinstance(box, dict) and ("class_name" in box or "class_id" in box)
    ]
    return histogram_distribution(labels)


def bbox_area_distribution(samples: Sequence[Sample], area_spec: AreaBinSpec) -> dict[str, float]:
    bins = [
        assign_area_bin(relative_area, area_spec)
        for sample in samples
        for _, relative_area, _, _ in iter_valid_box_features(sample)
    ]
    return histogram_distribution(bins)


def bbox_position_distribution(samples: Sequence[Sample], grid_size: int) -> dict[str, float]:
    cells = []
    for sample in samples:
        width = float(sample["width"])
        height = float(sample["height"])
        for box in sample.get("boxes", []):
            x1, y1, x2, y2 = [float(value) for value in box["bbox_xyxy"]]
            bbox_width = max(0.0, x2 - x1)
            bbox_height = max(0.0, y2 - y1)
            if bbox_width <= 0.0 or bbox_height <= 0.0:
                continue
            center_x = min(max(((x1 + x2) / 2.0) / width, 0.0), 1.0)
            center_y = min(max(((y1 + y2) / 2.0) / height, 0.0), 1.0)
            x_bin = min(int(center_x * grid_size), grid_size - 1)
            y_bin = min(int(center_y * grid_size), grid_size - 1)
            cells.append(f"{y_bin},{x_bin}")
    return histogram_distribution(cells)


def object_count_distribution(samples: Sequence[Sample]) -> dict[str, float]:
    return histogram_distribution(object_count_bin(len(sample.get("boxes", []))) for sample in samples)


def aspect_ratio_bin(aspect_ratio: float) -> str:
    if aspect_ratio < 0.5:
        return "tall"
    if aspect_ratio < 1.0:
        return "portrait"
    if aspect_ratio < 2.0:
        return "landscape"
    return "wide"


def aspect_ratio_distribution(samples: Sequence[Sample]) -> dict[str, float]:
    bins = [
        aspect_ratio_bin(aspect_ratio)
        for sample in samples
        for _, _, _, aspect_ratio in iter_valid_box_features(sample)
    ]
    return histogram_distribution(bins)


def sample_has_valid_bbox_metadata(sample: Sample) -> bool:
    if "width" not in sample or "height" not in sample or not isinstance(sample.get("boxes"), list):
        return False
    if not isinstance(sample["width"], (int, float)) or not isinstance(sample["height"], (int, float)):
        return False
    if sample["width"] <= 0 or sample["height"] <= 0:
        return False
    for box in sample["boxes"]:
        if not isinstance(box, dict):
            return False
        bbox = box.get("bbox_xyxy")
        if not isinstance(bbox, list) or len(bbox) != 4:
            return False
        if "class_id" not in box and "class_name" not in box:
            return False
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            return False
        if not all(math.isfinite(value) for value in [x1, y1, x2, y2]):
            return False
        if x2 <= x1 or y2 <= y1:
            return False
    return True


def has_bbox_metadata(samples: Sequence[Sample]) -> bool:
    return all(sample_has_valid_bbox_metadata(sample) for sample in samples)


def compute_basic_diagnostics(
    candidate_samples: Sequence[Sample],
    selected_samples: Sequence[Sample],
) -> dict[str, Any]:
    return {
        "num_candidates": len(candidate_samples),
        "num_selected_samples": len(selected_samples),
        "selected_fraction": len(selected_samples) / len(candidate_samples) if candidate_samples else 0.0,
    }


def compute_bbox_diagnostics(
    candidate_samples: Sequence[Sample],
    selected_samples: Sequence[Sample],
    args: argparse.Namespace,
) -> dict[str, Any]:
    diagnostics = compute_basic_diagnostics(candidate_samples, selected_samples)
    area_spec = compute_area_bins(candidate_samples, args)
    candidate_class = bbox_class_distribution(candidate_samples)
    selected_class = bbox_class_distribution(selected_samples)
    candidate_area = bbox_area_distribution(candidate_samples, area_spec)
    selected_area = bbox_area_distribution(selected_samples, area_spec)
    candidate_position = bbox_position_distribution(candidate_samples, args.position_grid)
    selected_position = bbox_position_distribution(selected_samples, args.position_grid)
    candidate_object_count = object_count_distribution(candidate_samples)
    selected_object_count = object_count_distribution(selected_samples)
    candidate_aspect = aspect_ratio_distribution(candidate_samples)
    selected_aspect = aspect_ratio_distribution(selected_samples)
    candidate_summaries = compute_detection_image_summaries(candidate_samples, args, area_spec)
    selected_summaries = compute_detection_image_summaries(selected_samples, args, area_spec)
    candidate_strata = assign_detection_strata(candidate_summaries, args.min_stratum_size)
    selected_strata = assign_detection_strata(selected_summaries, args.min_stratum_size)
    candidate_empty = sum(1 for sample in candidate_samples if len(sample.get("boxes", [])) == 0)
    selected_empty = sum(1 for sample in selected_samples if len(sample.get("boxes", [])) == 0)

    diagnostics.update(
        {
            "class_l1": l1_distance(candidate_class, selected_class),
            "bbox_area_jsd": js_divergence(candidate_area, selected_area),
            "bbox_position_jsd": js_divergence(candidate_position, selected_position),
            "object_count_jsd": js_divergence(candidate_object_count, selected_object_count),
            "aspect_ratio_jsd": js_divergence(candidate_aspect, selected_aspect),
            "candidate_empty_images": candidate_empty,
            "candidate_empty_fraction": candidate_empty / len(candidate_samples) if candidate_samples else 0.0,
            "selected_empty_images": selected_empty,
            "selected_empty_fraction": selected_empty / len(selected_samples) if selected_samples else 0.0,
            "num_candidate_strata": len(candidate_strata),
            "num_selected_strata": len(selected_strata),
            "area_binning": {
                "mode": area_spec.mode,
                "labels": area_spec.labels,
                "thresholds": area_spec.thresholds,
            },
            "distributions": {
                "candidate_class_frequency": candidate_class,
                "selected_class_frequency": selected_class,
                "candidate_bbox_area": candidate_area,
                "selected_bbox_area": selected_area,
                "candidate_bbox_position": candidate_position,
                "selected_bbox_position": selected_position,
                "candidate_object_count": candidate_object_count,
                "selected_object_count": selected_object_count,
                "candidate_aspect_ratio": candidate_aspect,
                "selected_aspect_ratio": selected_aspect,
            },
        }
    )
    return diagnostics


def label_distribution(samples: Sequence[Sample]) -> dict[str, float]:
    labels = [
        str(label)
        for sample in samples
        for label in sample.get("labels", [])
        if label is not None
    ]
    return histogram_distribution(labels)


def tile_count_stats(counts: Counter[str]) -> dict[str, Any]:
    values = sorted(counts.values())
    if not values:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": None,
        }
    return {
        "min": values[0],
        "max": values[-1],
        "mean": sum(values) / len(values),
        "median": median(values),
    }


def compute_tile_balanced_diagnostics(
    candidate_samples: Sequence[Sample],
    selected_samples: Sequence[Sample],
    args: argparse.Namespace,
) -> dict[str, Any]:
    diagnostics = compute_basic_diagnostics(candidate_samples, selected_samples)
    candidate_counts = Counter(str(sample[args.tile_field]) for sample in candidate_samples)
    selected_counts = Counter(str(sample[args.tile_field]) for sample in selected_samples)
    base_target_per_tile = len(selected_samples) // len(candidate_counts) if candidate_counts else 0
    allocation = {tile: selected_counts.get(tile, 0) for tile in sorted(candidate_counts)}
    underfull_tiles = [
        tile
        for tile, candidate_count in candidate_counts.items()
        if candidate_count < base_target_per_tile
    ]

    diagnostics.update(
        {
            "tile_field": args.tile_field,
            "num_candidate_tiles": len(candidate_counts),
            "num_selected_tiles": len(selected_counts),
            "base_target_per_tile": base_target_per_tile,
            "candidate_samples_per_tile": dict(sorted(candidate_counts.items())),
            "selected_samples_per_tile": dict(sorted(selected_counts.items())),
            "selected_samples_per_tile_stats": tile_count_stats(selected_counts),
            "tile_allocation": allocation,
            "underfull_tile_count": len(underfull_tiles),
            "underfull_tiles": sorted(underfull_tiles),
        }
    )
    if any("labels" in sample for sample in candidate_samples):
        candidate_labels = label_distribution(candidate_samples)
        selected_labels = label_distribution(selected_samples)
        diagnostics["label_l1"] = l1_distance(candidate_labels, selected_labels)
        diagnostics["label_jsd"] = js_divergence(candidate_labels, selected_labels)
        diagnostics["distributions"] = {
            "candidate_label_frequency": candidate_labels,
            "selected_label_frequency": selected_labels,
        }
    return diagnostics


def compute_diagnostics(
    candidate_samples: Sequence[Sample],
    selected_samples: Sequence[Sample],
    method: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    if method == "tile_balanced":
        return compute_tile_balanced_diagnostics(candidate_samples, selected_samples, args)
    if method == "bbox_stratified":
        return compute_bbox_diagnostics(candidate_samples, selected_samples, args)
    diagnostics = compute_basic_diagnostics(candidate_samples, selected_samples)
    if has_bbox_metadata(candidate_samples) and has_bbox_metadata(selected_samples):
        diagnostics.update(compute_bbox_diagnostics(candidate_samples, selected_samples, args))
    return diagnostics


def select_samples(
    candidate_samples: Sequence[Sample],
    args: argparse.Namespace,
    target: int,
) -> list[Sample]:
    if target > len(candidate_samples):
        raise ValueError(
            f"Requested {target} samples but candidate pool contains {len(candidate_samples)}."
        )
    if args.method == "random":
        return random_sample(candidate_samples, target, args.seed)
    if args.method == "tile_balanced":
        return tile_balanced_sample(candidate_samples, target, args.seed, args.tile_field)
    area_spec = compute_area_bins(candidate_samples, args)
    summaries = compute_detection_image_summaries(candidate_samples, args, area_spec)
    strata = assign_detection_strata(summaries, args.min_stratum_size)
    return stratified_sample(strata, target, args.seed)


def manifest_metadata(
    *,
    source_manifest: Path,
    input_manifest: Path | None,
    method: str,
    num_candidates: int,
    target: int,
    selected: Sequence[Sample],
    seed: int,
    args: argparse.Namespace,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "source_manifest": str(source_manifest),
        "input_manifest": None if input_manifest is None else str(input_manifest),
        "method": method,
        "num_candidates": num_candidates,
        "num_samples_requested": target,
        "num_samples_selected": len(selected),
        "seed": seed,
        "sampling_method": {
            "random": "uniform_random_without_replacement",
            "bbox_stratified": "representative_nested_bbox_stratified_image_level",
            "tile_balanced": "nested_tile_balanced_without_replacement",
        }[method],
        "diagnostics": diagnostics,
        "samples": list(selected),
    }
    if method == "tile_balanced":
        payload["tile_field"] = args.tile_field
        payload["tile_allocation"] = diagnostics.get("tile_allocation", {})
    if method == "bbox_stratified":
        payload["stratification"] = {
            "fields": STRATIFICATION_FIELDS,
            "area_binning": args.area_binning,
            "num_area_bins": args.num_area_bins,
            "y_position_bins": args.y_position_bins,
            "position_grid": f"{args.position_grid}x{args.position_grid}",
            "min_stratum_size": args.min_stratum_size,
        }
    return payload


def ensure_output_available(path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists. Pass --overwrite to replace it: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict[str, Any], overwrite: bool) -> None:
    ensure_output_available(path, overwrite)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=False)
        handle.write("\n")


def write_diagnostics(
    path: Path | None,
    diagnostics: dict[str, Any],
    overwrite: bool,
) -> None:
    if path is None:
        return
    write_json(path, diagnostics, overwrite)


def print_summary(
    *,
    method: str,
    source_manifest: Path,
    input_manifest: Path | None,
    candidates: int,
    requested: int,
    selected: int,
    seed: int,
    output: Path,
    diagnostics: dict[str, Any],
) -> None:
    print(f"Method: {method}")
    print(f"Source manifest: {source_manifest}")
    print(f"Input manifest: {input_manifest if input_manifest is not None else 'None'}")
    print(f"Candidate samples: {candidates}")
    print(f"Requested samples: {requested}")
    print(f"Selected samples: {selected}")
    print(f"Seed: {seed}")
    print(f"Output: {output}")
    print("Diagnostics:")
    for key in [
        "class_l1",
        "bbox_area_jsd",
        "bbox_position_jsd",
        "object_count_jsd",
        "aspect_ratio_jsd",
    ]:
        if key in diagnostics:
            print(f"  {key}: {diagnostics[key]:.6f}")
    if "class_l1" not in diagnostics:
        print(f"  num_candidates: {diagnostics['num_candidates']}")
        print(f"  num_selected_samples: {diagnostics['num_selected_samples']}")
    if method == "tile_balanced":
        stats = diagnostics["selected_samples_per_tile_stats"]
        print(f"  num_candidate_tiles: {diagnostics['num_candidate_tiles']}")
        print(f"  num_selected_tiles: {diagnostics['num_selected_tiles']}")
        print(
            "  selected_samples_per_tile: "
            f"min={stats['min']}, median={stats['median']}, "
            f"mean={stats['mean']:.3f}, max={stats['max']}"
        )
        if "label_l1" in diagnostics:
            print(f"  label_l1: {diagnostics['label_l1']:.6f}")
            print(f"  label_jsd: {diagnostics['label_jsd']:.6f}")


def build_one_subset(
    *,
    candidate_samples: Sequence[Sample],
    source_manifest: Path,
    input_manifest: Path | None,
    output: Path,
    target: int,
    args: argparse.Namespace,
    diagnostics_output: Path | None = None,
) -> list[Sample]:
    selected = select_samples(candidate_samples, args, target)
    diagnostics = compute_diagnostics(candidate_samples, selected, args.method, args)
    payload = manifest_metadata(
        source_manifest=source_manifest,
        input_manifest=input_manifest,
        method=args.method,
        num_candidates=len(candidate_samples),
        target=target,
        selected=selected,
        seed=args.seed,
        args=args,
        diagnostics=diagnostics,
    )
    write_json(output, payload, args.overwrite)
    write_diagnostics(diagnostics_output, diagnostics, args.overwrite)
    print_summary(
        method=args.method,
        source_manifest=source_manifest,
        input_manifest=input_manifest,
        candidates=len(candidate_samples),
        requested=target,
        selected=len(selected),
        seed=args.seed,
        output=output,
        diagnostics=diagnostics,
    )
    return selected


def validate_output_paths(args: argparse.Namespace) -> None:
    if args.output is not None:
        ensure_output_available(args.output, args.overwrite)
    if args.diagnostics_output is not None:
        ensure_output_available(args.diagnostics_output, args.overwrite)
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for size in args.subset_sizes:
            ensure_output_available(
                multi_output_path(args.output_dir, args.subset_name_prefix, size, args.subset_name_suffix),
                args.overwrite,
            )
    if args.diagnostics_dir is not None:
        args.diagnostics_dir.mkdir(parents=True, exist_ok=True)
        for size in args.subset_sizes:
            ensure_output_available(
                multi_diagnostics_path(
                    args.diagnostics_dir,
                    args.subset_name_prefix,
                    size,
                    args.subset_name_suffix,
                ),
                args.overwrite,
            )


def multi_output_path(
    output_dir: Path,
    prefix: str,
    size: int,
    suffix: str | None,
) -> Path:
    stem = f"{prefix}_{size}"
    if suffix:
        stem = f"{stem}_{suffix}"
    return output_dir / f"{stem}.json"


def multi_diagnostics_path(
    diagnostics_dir: Path,
    prefix: str,
    size: int,
    suffix: str | None,
) -> Path:
    stem = f"{prefix}_{size}"
    if suffix:
        stem = f"{stem}_{suffix}"
    return diagnostics_dir / f"{stem}_diagnostics.json"


def run_single_output(
    candidate_samples: Sequence[Sample],
    args: argparse.Namespace,
) -> None:
    assert args.num_samples is not None
    assert args.output is not None
    if args.num_samples > len(candidate_samples):
        raise ValueError(
            f"--num-samples ({args.num_samples}) exceeds candidate samples ({len(candidate_samples)})."
        )
    build_one_subset(
        candidate_samples=candidate_samples,
        source_manifest=args.source_manifest,
        input_manifest=args.input_manifest,
        output=args.output,
        target=args.num_samples,
        args=args,
        diagnostics_output=args.diagnostics_output,
    )


def run_multi_output(
    candidate_samples: Sequence[Sample],
    args: argparse.Namespace,
) -> None:
    assert args.subset_sizes is not None
    assert args.output_dir is not None
    assert args.subset_name_prefix is not None
    largest = args.subset_sizes[-1]
    if largest > len(candidate_samples):
        raise ValueError(
            f"Largest subset size ({largest}) exceeds candidate samples ({len(candidate_samples)})."
        )

    selected_by_size: dict[int, list[Sample]] = {}
    current_candidates = list(candidate_samples)
    current_input_manifest = args.input_manifest
    for size in sorted(args.subset_sizes, reverse=True):
        output = multi_output_path(
            args.output_dir,
            args.subset_name_prefix,
            size,
            args.subset_name_suffix,
        )
        diagnostics_output = (
            None
            if args.diagnostics_dir is None
            else multi_diagnostics_path(
                args.diagnostics_dir,
                args.subset_name_prefix,
                size,
                args.subset_name_suffix,
            )
        )
        selected = build_one_subset(
            candidate_samples=current_candidates,
            source_manifest=args.source_manifest,
            input_manifest=current_input_manifest,
            output=output,
            target=size,
            args=args,
            diagnostics_output=diagnostics_output,
        )
        selected_by_size[size] = selected
        current_candidates = selected
        current_input_manifest = output

    for small, large in zip(args.subset_sizes, args.subset_sizes[1:]):
        small_ids = {str(sample["id"]) for sample in selected_by_size[small]}
        large_ids = {str(sample["id"]) for sample in selected_by_size[large]}
        if not small_ids.issubset(large_ids):
            raise RuntimeError(f"Nested subset check failed: {small} is not a subset of {large}.")


def main() -> None:
    args = parse_args()
    source_manifest = load_manifest(args.source_manifest)
    input_manifest = load_manifest(args.input_manifest) if args.input_manifest is not None else None
    candidate_manifest = input_manifest if input_manifest is not None else source_manifest
    candidate_samples = normalize_sample_aliases(candidate_manifest["samples"], args)
    validate_samples_for_method(candidate_samples, args.method, args)
    validate_unique_sample_ids(candidate_samples, "candidate pool")
    validate_output_paths(args)

    if args.num_samples is not None:
        run_single_output(candidate_samples, args)
    else:
        run_multi_output(candidate_samples, args)


if __name__ == "__main__":
    main()

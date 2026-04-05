#!/usr/bin/env python3
"""Convert the reduced FLIR proxy subset into a v18-style dataset layout.

The output layout mirrors ``data/raw/v18``:

    <output_root>/
      train/
        *.npy
        annotations.json
        captions.json
      val/
        *.npy
        annotations.json
      test/
        *.npy
        annotations.json

By default the export is ``person``-only so it matches the current v18
annotation structure used in this repository:

- image files are flat ``.npy`` arrays saved directly in each split folder
- image ids are string stems matching the ``.npy`` filenames
- annotations use a single category ``{"id": 0, "name": "person"}``

Image arrays are saved exactly as loaded from the source FLIR thermal JPEGs:
no normalization, no resizing, no dtype conversion, and no value remapping.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image


SPLIT_TO_SOURCE_DIR = {
    "train": "images_thermal_train",
    "val": "images_thermal_val",
    "test": "video_thermal_test",
}

BASE_CAPTION = "overhead infrared surveillance image, circular field of view"


def repo_root() -> Path:
    """Resolve the repository root from this script location."""
    return Path(__file__).resolve().parents[2]


def default_derived_root() -> Path:
    return repo_root() / "data" / "derived" / "flir_private_proxy_alignment"


def default_flir_root() -> Path:
    return repo_root() / "data" / "raw" / "flir"


def default_output_root() -> Path:
    return repo_root() / "data" / "raw" / "flir_private_proxy_alignment_v18"


def caption_from_count(n_people: int) -> str:
    """Match the simple caption style used by the v18 dataset."""
    if n_people <= 0:
        return BASE_CAPTION
    if n_people == 1:
        return f"{BASE_CAPTION}, 1 person"
    return f"{BASE_CAPTION}, {n_people} people"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert data/derived/flir_private_proxy_alignment into a v18-style "
            "dataset with .npy images and v18-like COCO annotations."
        )
    )
    parser.add_argument(
        "--derived-root",
        type=Path,
        default=default_derived_root(),
        help="Directory containing reduced_flir_{train,val,test}_coco.json.",
    )
    parser.add_argument(
        "--flir-root",
        type=Path,
        default=default_flir_root(),
        help="Root directory of the raw FLIR dataset.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="Destination root for the v18-style dataset export.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=sorted(SPLIT_TO_SOURCE_DIR.keys()),
        help="Which splits to export.",
    )
    parser.add_argument(
        "--keep-all-categories",
        action="store_true",
        help=(
            "Keep all FLIR categories instead of exporting a person-only dataset. "
            "The default matches the current v18 schema."
        ),
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional cap for quick smoke tests.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing .npy files instead of skipping them.",
    )
    parser.add_argument(
        "--no-train-captions",
        action="store_true",
        help="Do not write train/captions.json.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def dedupe_images(images: Iterable[dict]) -> List[dict]:
    """Remove exact duplicate image entries while preserving order."""
    unique_by_id: Dict[object, dict] = {}
    ordered: List[dict] = []
    for image in images:
        image_id = image["id"]
        prev = unique_by_id.get(image_id)
        if prev is None:
            unique_by_id[image_id] = image
            ordered.append(image)
            continue
        if prev != image:
            raise ValueError(
                f"Conflicting duplicate image entry for id={image_id!r}: "
                f"{prev!r} != {image!r}"
            )
    return ordered


def build_category_remap(
    source_categories: List[dict],
    *,
    person_only: bool,
) -> Tuple[List[dict], Dict[int, int], int | None]:
    """Return output categories, source->dest id remap, and person category id."""
    source_name_by_id = {int(cat["id"]): str(cat["name"]) for cat in source_categories}
    person_source_id = next(
        (cat_id for cat_id, name in source_name_by_id.items() if name == "person"),
        None,
    )
    if person_only:
        if person_source_id is None:
            raise ValueError("Could not find a 'person' category in the source COCO.")
        return [{"id": 0, "name": "person"}], {person_source_id: 0}, 0

    ordered_source_ids = sorted(source_name_by_id)
    remap = {source_id: new_id for new_id, source_id in enumerate(ordered_source_ids)}
    output_categories = [
        {"id": remap[source_id], "name": source_name_by_id[source_id]}
        for source_id in ordered_source_ids
    ]
    person_output_id = remap.get(person_source_id) if person_source_id is not None else None
    return output_categories, remap, person_output_id


def stem_from_source_file_name(source_file_name: str) -> str:
    """Flatten a FLIR relative file path to the output v18-style image stem."""
    return Path(source_file_name).stem


def resolve_source_image_path(flir_root: Path, split: str, source_file_name: str) -> Path:
    if split not in SPLIT_TO_SOURCE_DIR:
        raise ValueError(
            f"Unknown FLIR source split {split!r}. Expected one of {sorted(SPLIT_TO_SOURCE_DIR)}."
        )
    source_dir = SPLIT_TO_SOURCE_DIR[split]
    return flir_root / source_dir / source_file_name


def load_image_array(path: Path) -> np.ndarray:
    """Load a FLIR thermal image without changing dtype or values."""
    with Image.open(path) as image:
        arr = np.asarray(image)
    return arr


def write_image_npy(
    source_path: Path,
    output_path: Path,
    *,
    overwrite: bool,
) -> np.ndarray:
    """Save a source FLIR image as .npy while preserving raw array content."""
    if output_path.exists() and not overwrite:
        return np.load(output_path)

    arr = load_image_array(source_path)
    np.save(output_path, arr, allow_pickle=False)
    return arr


def convert_split(
    *,
    split: str,
    derived_root: Path,
    flir_root: Path,
    output_root: Path,
    person_only: bool,
    max_images: int | None,
    overwrite: bool,
    write_train_captions: bool,
) -> None:
    input_coco_path = derived_root / f"reduced_flir_{split}_coco.json"
    if not input_coco_path.exists():
        raise FileNotFoundError(f"Missing derived COCO file: {input_coco_path}")

    source_coco = load_json(input_coco_path)
    unique_images = dedupe_images(source_coco.get("images", []))
    if max_images is not None:
        unique_images = unique_images[:max_images]

    selected_source_ids = {img["id"] for img in unique_images}
    source_annotations = [
        ann for ann in source_coco.get("annotations", [])
        if ann["image_id"] in selected_source_ids
    ]

    output_categories, category_remap, person_output_category_id = build_category_remap(
        source_coco.get("categories", []),
        person_only=person_only,
    )

    output_split_dir = output_root / split
    output_split_dir.mkdir(parents=True, exist_ok=True)

    output_images: List[dict] = []
    output_annotations: List[dict] = []
    source_id_to_output_id: Dict[object, str] = {}
    seen_output_filenames: set[str] = set()

    for image in unique_images:
        source_id = image["id"]
        output_id = stem_from_source_file_name(str(image["file_name"]))
        output_file_name = f"{output_id}.npy"
        if output_file_name in seen_output_filenames:
            raise ValueError(
                f"Duplicate output filename within split={split!r}: {output_file_name}"
            )
        seen_output_filenames.add(output_file_name)
        source_id_to_output_id[source_id] = output_id

        source_split = str(image.get("source_split", split))
        source_path = resolve_source_image_path(flir_root, source_split, str(image["file_name"]))
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source image: {source_path}")

        output_path = output_split_dir / output_file_name
        arr = write_image_npy(source_path, output_path, overwrite=overwrite)
        height, width = arr.shape[:2]

        output_images.append(
            {
                "id": output_id,
                "width": int(width),
                "height": int(height),
                "file_name": output_file_name,
            }
        )

    ann_id = 0
    for ann in source_annotations:
        source_category_id = int(ann["category_id"])
        if source_category_id not in category_remap:
            continue

        bbox = [float(x) for x in ann["bbox"]]
        area = float(ann.get("area", bbox[2] * bbox[3]))
        output_annotations.append(
            {
                "id": ann_id,
                "image_id": source_id_to_output_id[ann["image_id"]],
                "category_id": category_remap[source_category_id],
                "bbox": bbox,
                "area": area,
                "iscrowd": int(bool(ann.get("iscrowd", 0))),
            }
        )
        ann_id += 1

    output_coco = {
        "info": {
            "description": (
                "Reduced FLIR thermal proxy subset converted to a v18-style layout"
            ),
            "source_annotations": str(input_coco_path),
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "person_only": person_only,
        },
        "licenses": [],
        "categories": output_categories,
        "images": output_images,
        "annotations": output_annotations,
    }
    save_json(output_split_dir / "annotations.json", output_coco)

    if split == "train" and write_train_captions:
        counts_by_image_id = defaultdict(int)
        if person_output_category_id is not None:
            for ann in output_annotations:
                if ann["category_id"] == person_output_category_id:
                    counts_by_image_id[ann["image_id"]] += 1

        captions = {
            image["id"]: caption_from_count(counts_by_image_id.get(image["id"], 0))
            for image in output_images
        }
        save_json(output_split_dir / "captions.json", captions)

    image_count = len(output_images)
    ann_count = len(output_annotations)
    sample_dtype = None
    sample_shape = None
    if image_count:
        sample_path = output_split_dir / output_images[0]["file_name"]
        sample_arr = np.load(sample_path)
        sample_dtype = str(sample_arr.dtype)
        sample_shape = tuple(int(x) for x in sample_arr.shape)

    print(
        f"[{split}] images={image_count} annotations={ann_count} "
        f"categories={len(output_categories)} sample_dtype={sample_dtype} "
        f"sample_shape={sample_shape}"
    )


def main() -> None:
    args = parse_args()
    write_train_captions = not args.no_train_captions
    person_only = not args.keep_all_categories

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"Derived root: {args.derived_root}")
    print(f"FLIR root:    {args.flir_root}")
    print(f"Output root:  {args.output_root}")
    print(f"Person only:  {person_only}")
    print(f"Splits:       {args.splits}")
    if args.max_images_per_split is not None:
        print(f"Max images:   {args.max_images_per_split} per split")

    for split in args.splits:
        convert_split(
            split=split,
            derived_root=args.derived_root,
            flir_root=args.flir_root,
            output_root=args.output_root,
            person_only=person_only,
            max_images=args.max_images_per_split,
            overwrite=args.overwrite,
            write_train_captions=write_train_captions,
        )

    print("Done.")


if __name__ == "__main__":
    main()

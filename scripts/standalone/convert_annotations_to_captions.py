#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
from collections import defaultdict


BASE_CAPTION = "overhead infrared surveillance image, circular field of view"


def build_caption(count: int) -> str:
    if count <= 0:
        return BASE_CAPTION
    if count == 1:
        return f"{BASE_CAPTION}, 1 person"
    return f"{BASE_CAPTION}, {count} people"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert COCO-style annotations.json into a stem->caption JSON mapping."
    )
    parser.add_argument(
        "input_json",
        type=Path,
        help="Path to the input annotations.json file",
    )
    parser.add_argument(
        "output_json",
        type=Path,
        help="Path to the output JSON file",
    )
    args = parser.parse_args()

    with args.input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])

    # Count annotations per image_id
    counts_by_image_id = defaultdict(int)
    for ann in annotations:
        image_id = ann.get("image_id")
        if image_id is not None:
            counts_by_image_id[image_id] += 1

    result = {}

    for img in images:
        image_id = img.get("id")
        file_name = img.get("file_name", "")

        # stem of "abc.npy" -> "abc"
        stem = Path(file_name).stem

        count = counts_by_image_id.get(image_id, 0)
        result[stem] = build_caption(count)

    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Written {len(result)} captions to {args.output_json}")


if __name__ == "__main__":
    main()
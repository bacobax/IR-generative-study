#!/usr/bin/env python3
"""Download FLIR ADAS v2 and export the full thermal splits.

This script intentionally delegates the dataset conversion to
``build_flir_private_proxy_alignment_dataset.py --use-full-dataset``. It only
handles fetching/extracting the upstream zip and finding the extracted FLIR
root that contains the thermal split folders.

Example
-------
```bash
python scripts/standalone/download_and_build_flir_full_thermal_dataset.py \
  --output-root data/raw/flir_private_proxy_alignment_v18 \
  --overwrite
```
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Sequence
from urllib.parse import urlparse
from urllib.request import urlretrieve


DEFAULT_URL = "https://adas-dataset-v2.flirconservator.com/dataset/full/FLIR_ADAS_v2.zip"
THERMAL_SPLIT_DIRS = ("images_thermal_train", "images_thermal_val", "video_thermal_test")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_download_dir() -> Path:
    return repo_root() / "data" / "downloads" / "flir_adas_v2"


def default_extract_root() -> Path:
    return repo_root() / "data" / "raw" / "flir_adas_v2_download"


def default_output_root() -> Path:
    return repo_root() / "data" / "raw" / "flir_private_proxy_alignment_v18"


def default_zip_path(url: str, download_dir: Path) -> Path:
    parsed_name = Path(urlparse(url).path).name
    filename = parsed_name or "FLIR_ADAS_v2.zip"
    return download_dir / filename


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download FLIR ADAS v2, extract it, and export the full thermal "
            "dataset into the repository v18-style layout."
        )
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="FLIR ADAS v2 zip URL.")
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=default_download_dir(),
        help="Directory where the zip is cached.",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Existing or desired zip path. Defaults to --download-dir/<url filename>.",
    )
    parser.add_argument(
        "--extract-root",
        type=Path,
        default=default_extract_root(),
        help="Directory where the zip is extracted.",
    )
    parser.add_argument(
        "--flir-root",
        type=Path,
        default=None,
        help=(
            "Extracted FLIR root containing images_thermal_train, "
            "images_thermal_val, and video_thermal_test. If omitted, the script "
            "searches under --extract-root."
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root(),
        help="Destination for the v18-style full thermal dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="Thermal splits to export.",
    )
    parser.add_argument(
        "--max-images-per-split",
        type=int,
        default=None,
        help="Optional cap forwarded to the builder for smoke runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Forward --overwrite to the builder so existing .npy files are rewritten.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Download the zip again even if it already exists.",
    )
    parser.add_argument(
        "--force-extract",
        action="store_true",
        help="Extract the zip again even if an extracted FLIR root is already present.",
    )
    parser.add_argument(
        "--no-train-captions",
        action="store_true",
        help="Forward --no-train-captions to the builder.",
    )
    return parser.parse_args()


def download_zip(url: str, zip_path: Path, *, force: bool) -> Path:
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists() and not force:
        print(f"Using cached zip: {zip_path}")
        return zip_path

    print(f"Downloading {url}")
    print(f"Destination: {zip_path}")
    urlretrieve(url, zip_path)
    return zip_path


def safe_extract_zip(zip_path: Path, extract_root: Path) -> None:
    extract_root.mkdir(parents=True, exist_ok=True)
    root = extract_root.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            target = (extract_root / member.filename).resolve()
            if root != target and root not in target.parents:
                raise ValueError(f"Refusing to extract unsafe zip member: {member.filename}")
        archive.extractall(extract_root)


def is_flir_root(path: Path) -> bool:
    return all((path / split_dir / "coco.json").is_file() for split_dir in THERMAL_SPLIT_DIRS)


def find_flir_root(search_root: Path) -> Path:
    if is_flir_root(search_root):
        return search_root

    matches = [path for path in search_root.rglob("*") if path.is_dir() and is_flir_root(path)]
    if not matches:
        expected = ", ".join(THERMAL_SPLIT_DIRS)
        raise FileNotFoundError(
            f"Could not find an extracted FLIR root under {search_root}. "
            f"Expected a directory containing {expected}, each with coco.json."
        )
    matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def build_builder_command(
    *,
    flir_root: Path,
    output_root: Path,
    splits: Sequence[str],
    max_images_per_split: int | None,
    overwrite: bool,
    no_train_captions: bool,
) -> list[str]:
    builder_script = repo_root() / "scripts" / "standalone" / "build_flir_private_proxy_alignment_dataset.py"
    command = [
        sys.executable,
        str(builder_script),
        "--use-full-dataset",
        "--flir-root",
        str(flir_root),
        "--output-root",
        str(output_root),
        "--splits",
        *splits,
    ]
    if max_images_per_split is not None:
        command.extend(["--max-images-per-split", str(max_images_per_split)])
    if overwrite:
        command.append("--overwrite")
    if no_train_captions:
        command.append("--no-train-captions")
    return command


def main() -> None:
    args = parse_args()
    zip_path = args.zip_path or default_zip_path(args.url, args.download_dir)

    zip_path = download_zip(args.url, zip_path, force=args.force_download)

    if args.flir_root is not None:
        flir_root = args.flir_root
        if not is_flir_root(flir_root):
            raise FileNotFoundError(f"--flir-root does not look like a FLIR ADAS root: {flir_root}")
    else:
        existing_root = None if args.force_extract else _find_existing_flir_root(args.extract_root)
        if existing_root is None:
            print(f"Extracting {zip_path} into {args.extract_root}")
            safe_extract_zip(zip_path, args.extract_root)
            flir_root = find_flir_root(args.extract_root)
        else:
            flir_root = existing_root
            print(f"Using existing extracted FLIR root: {flir_root}")

    command = build_builder_command(
        flir_root=flir_root,
        output_root=args.output_root,
        splits=args.splits,
        max_images_per_split=args.max_images_per_split,
        overwrite=args.overwrite,
        no_train_captions=args.no_train_captions,
    )

    print("Running full thermal export:")
    print(" ".join(command))
    subprocess.run(command, check=True, cwd=repo_root())


def _find_existing_flir_root(search_root: Path) -> Path | None:
    try:
        return find_flir_root(search_root)
    except FileNotFoundError:
        return None


if __name__ == "__main__":
    main()

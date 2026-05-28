from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/datasets/create_subset_manifest.py")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_tile_balanced_jsonl_multi_output_is_exact_and_nested(tmp_path: Path) -> None:
    rows = []
    for tile, count in {"tile_a": 1, "tile_b": 3, "tile_c": 10, "tile_d": 10}.items():
        for index in range(count):
            sample_id = f"{tile}_{index:03d}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "image_path": f"images/train/{sample_id}.tif",
                    "scene": tile,
                    "split": "train",
                    "labels": ["forest"] if tile in {"tile_a", "tile_c"} else ["water"],
                }
            )
    source = tmp_path / "train.jsonl"
    out_dir = tmp_path / "subsets"
    write_jsonl(source, rows)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-manifest",
            str(source),
            "--method",
            "tile_balanced",
            "--subset-sizes",
            "6",
            "12",
            "--output-dir",
            str(out_dir),
            "--subset-name-prefix",
            "train",
            "--subset-name-suffix",
            "tile_balanced",
            "--diagnostics-dir",
            str(out_dir),
            "--tile-field",
            "scene",
            "--sample-id-field",
            "sample_id",
            "--sample-path-field",
            "image_path",
            "--seed",
            "7",
        ],
        cwd=Path.cwd(),
        check=True,
    )

    large = read_json(out_dir / "train_12_tile_balanced.json")
    small = read_json(out_dir / "train_6_tile_balanced.json")
    large_diag = read_json(out_dir / "train_12_tile_balanced_diagnostics.json")
    small_diag = read_json(out_dir / "train_6_tile_balanced_diagnostics.json")

    assert large["method"] == "tile_balanced"
    assert large["sampling_method"] == "nested_tile_balanced_without_replacement"
    assert len(large["samples"]) == 12
    assert len(small["samples"]) == 6
    assert large["tile_field"] == "scene"
    assert large["tile_allocation"] == large_diag["tile_allocation"]
    assert small["tile_allocation"] == small_diag["tile_allocation"]
    assert all("id" in sample and "path" in sample for sample in large["samples"])
    assert all(sample["path"].endswith(".tif") for sample in large["samples"])

    large_ids = {sample["id"] for sample in large["samples"]}
    small_ids = {sample["id"] for sample in small["samples"]}
    assert small_ids.issubset(large_ids)
    assert large_diag["num_candidate_tiles"] == 4
    assert large_diag["num_selected_tiles"] == 4
    assert large_diag["underfull_tile_count"] == 1
    assert large_diag["selected_samples_per_tile_stats"]["min"] == 1
    assert large_diag["selected_samples_per_tile_stats"]["max"] == 4
    assert small_diag["selected_samples_per_tile_stats"]["min"] == 1
    assert small_diag["selected_samples_per_tile_stats"]["max"] == 2
    assert "label_l1" in large_diag
    assert "candidate_label_frequency" in large_diag["distributions"]


def test_random_json_manifest_still_writes_flir_style_json(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    output = tmp_path / "subset.json"
    payload = {
        "samples": [
            {"id": f"sample_{index}", "path": f"sample_{index}.npy"}
            for index in range(5)
        ]
    }
    source.write_text(json.dumps(payload), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--source-manifest",
            str(source),
            "--method",
            "random",
            "--num-samples",
            "3",
            "--output",
            str(output),
            "--seed",
            "3",
        ],
        cwd=Path.cwd(),
        check=True,
    )

    subset = read_json(output)
    assert subset["method"] == "random"
    assert subset["sampling_method"] == "uniform_random_without_replacement"
    assert len(subset["samples"]) == 3
    assert all(sample["path"].endswith(".npy") for sample in subset["samples"])

"""Re-run the YOLO-adherence filter on the existing fm_balanced_aug candidates and
dump montages of kept/discarded GT-box crops (grouped by size bin) so the discards
can be eyeballed: does dropping these boxes actually make sense?

No FM generation — operates on the already-generated .npy images, so the discard set
matches the committed small_v4_fmbalaug results. Pin the GPU with CUDA_VISIBLE_DEVICES.

Usage:
  CUDA_VISIBLE_DEVICES=0 conda run -n diffusers-dev \
    python scripts/diagnostics/dump_fmbalaug_discarded_crops.py
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from src.algorithms.inference.yolo_adherence_filter import audit_generated_candidates_yolo

DATASET_DIR = Path(
    "artifacts/generated/yolo/exp_b/generated_candidates/small_v4_fmbalaug/fm_balanced_aug"
)
YOLO_WEIGHTS = Path(
    "artifacts/checkpoints/yolo/exp_v18_scratch_yolo11n/default_aug/best.pt"
)
CROPS_DIR = Path("artifacts/analysis/yolo/exp_v18_simple_yolo_tiny/small_v4_fmbalaug/discarded_crops")


def main() -> None:
    # The committed annotations.json was already filtered down to kept boxes by the
    # exp_b run, so we must audit the UNFILTERED set to see the discards. Temporarily
    # swap it in, then restore the filtered file in `finally` (non-destructive).
    annotations = DATASET_DIR / "annotations.json"
    unfiltered = DATASET_DIR / "annotations_unfiltered.json"
    if not unfiltered.is_file():
        raise FileNotFoundError(f"Missing {unfiltered}; cannot recover discarded boxes.")
    backup = Path(tempfile.mkstemp(suffix=".json", prefix="annot_filtered_")[1])
    shutil.copy2(annotations, backup)
    try:
        shutil.copy2(unfiltered, annotations)
        _instance_rows, _image_rows, stats = audit_generated_candidates_yolo(
            generated_dataset_dir=DATASET_DIR,
            yolo_weights=YOLO_WEIGHTS,
            device="cuda:0",
            iou_thr=0.5,
            conf=0.25,
            batch_size=16,
            crops_dir=CROPS_DIR,
            crop_context_frac=0.6,
            crop_tile=112,
            max_crops_per_bin=80,
        )
    finally:
        shutil.copy2(backup, annotations)
        backup.unlink(missing_ok=True)
    print(json.dumps(stats.get("crop_montages", {}), indent=2))


if __name__ == "__main__":
    main()

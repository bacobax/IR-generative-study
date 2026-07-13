"""Tests for the per-checkpoint quality evaluator (scripts/eval_checkpoint_quality).

Run with::

    python -m pytest tests/test_checkpoint_eval.py -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from src.core.data.datasets import AnnotationLayoutDataset  # noqa: E402
from src.core.normalization import UINT8_LINEAR  # noqa: E402
import scripts.eval_checkpoint_quality as E  # noqa: E402


def _make_train_data(tmpdir: str) -> tuple[str, str]:
    img_dir = os.path.join(tmpdir, "images")
    os.makedirs(img_dir, exist_ok=True)
    images, annotations = [], []
    ann_id = 1
    for i in range(6):
        fname = f"img_{i}.npy"
        np.save(os.path.join(img_dir, fname),
                np.random.randint(0, 255, size=(64, 64), dtype=np.uint8))
        images.append({"id": i, "file_name": fname, "width": 64, "height": 64})
        for _ in range((i % 3) + 1):  # 1..3 objects per image
            x, y = np.random.randint(0, 40), np.random.randint(0, 40)
            annotations.append({"id": ann_id, "image_id": i, "category_id": 0,
                                "bbox": [int(x), int(y), 20, 20], "area": 400, "iscrowd": 0})
            ann_id += 1
    payload = {"images": images, "annotations": annotations,
               "categories": [{"id": 0, "name": "person"}]}
    ann_path = os.path.join(tmpdir, "annotations.json")
    with open(ann_path, "w") as f:
        json.dump(payload, f)
    return img_dir, ann_path


# ── B-T1: unseen-layout synthesis ──
def test_synthesize_unseen_layouts_valid_and_reproducible():
    with tempfile.TemporaryDirectory() as tmp:
        img_dir, ann = _make_train_data(tmp)
        ds = AnnotationLayoutDataset(root_dir=img_dir, annotations_path=ann,
                                     image_size=64, normalization_mode=UINT8_LINEAR)
        a = E.synthesize_unseen_layouts(ds, k=10, image_size=64, seed=123)
        b = E.synthesize_unseen_layouts(ds, k=10, image_size=64, seed=123)
        assert len(a) == 10
        # reproducible by seed
        assert all(int(x["n_objects"]) == int(y["n_objects"]) for x, y in zip(a, b))
        assert all(torch.allclose(x["boxes_xyxy"], y["boxes_xyxy"]) for x, y in zip(a, b))
        for s in a:
            assert s["n_objects"] >= 1
            assert s["boxes_xyxy"].shape == (s["n_objects"], 4)
            assert s["labels"].shape == (s["n_objects"],)
            # boxes within image bounds
            assert float(s["boxes_xyxy"].min()) >= 0.0
            assert float(s["boxes_xyxy"].max()) <= 64.0
            # counts drawn from the train distribution (1..3)
            assert 1 <= int(s["n_objects"]) <= 3


# ── B-T2: metrics + adherence sanity ──
def test_iou_and_matching_perfect():
    gt = np.array([[10, 10, 50, 50], [60, 60, 90, 90]], dtype=float)
    tp, fp, fn, ious = E._match_one(gt.copy(), gt.copy(), iou_thr=0.5)
    assert (tp, fp, fn) == (2, 0, 0)
    assert pytest.approx(np.mean(ious), abs=1e-6) == 1.0


def test_matching_no_overlap():
    pred = np.array([[0, 0, 5, 5]], dtype=float)
    gt = np.array([[50, 50, 60, 60]], dtype=float)
    tp, fp, fn, ious = E._match_one(pred, gt, iou_thr=0.5)
    assert (tp, fp, fn) == (0, 1, 1)
    assert ious == []


def test_matching_empty_cases():
    gt = np.array([[10, 10, 20, 20]], dtype=float)
    assert E._match_one(np.zeros((0, 4)), gt, iou_thr=0.5) == (0, 0, 1, [])
    assert E._match_one(gt, np.zeros((0, 4)), iou_thr=0.5)[:3] == (0, 1, 0)


def test_distribution_metric_helpers_run():
    from src.evaluation.generative_metrics import compute_fid, compute_kid
    from src.evaluation.mmd import compute_rbf_mmd

    rng = np.random.default_rng(0)
    real = rng.normal(size=(20, 32)).astype(np.float32)
    gen = rng.normal(size=(20, 32)).astype(np.float32)
    assert np.isfinite(compute_fid(real, gen))
    assert np.isfinite(compute_kid(real, gen, subset_size=8, subsets=4))
    assert np.isfinite(compute_rbf_mmd(real, gen, bandwidths=[0.1, 1.0, 10.0]))


# ── normalized → uint8 image domain mapping ──
def test_normalized_to_uint8_ranges():
    neg = torch.full((1, 8, 8), -1.0)
    pos = torch.full((1, 8, 8), 1.0)
    assert int(E._normalized_to_uint8(neg).min()) == 0
    assert int(E._normalized_to_uint8(pos).max()) == 255


# ── B-T3: evaluator end-to-end smoke (skips if heavy artifacts absent) ──
@pytest.mark.skipif(
    not (os.path.isdir("data/raw/v18/test")
         and os.path.isfile("artifacts/checkpoints/flow_matching/serious_runs/"
                            "stay_layout_latent_v18_sd15ft_x8_256_minmax_3L/UNET/unet_fm_best.pt")),
    reason="requires v18 test split + a 3L checkpoint",
)
def test_evaluator_smoke(tmp_path):
    import subprocess
    cmd = [
        "python", "-m", "scripts.eval_checkpoint_quality",
        "--config", "configs/fm/train/presets/stay_layout_latent_v18_sd15ft_x8_256_minmax_3L.yaml",
        "--checkpoint", "artifacts/checkpoints/flow_matching/serious_runs/"
                        "stay_layout_latent_v18_sd15ft_x8_256_minmax_3L/UNET/unet_fm_best.pt",
        "--device", "cuda:0" if torch.cuda.is_available() else "cpu",
        "--out", str(tmp_path), "--n-test", "2", "--n-unseen", "2", "--steps", "2",
    ]
    subprocess.run(cmd, check=True, timeout=900)
    assert (tmp_path / "checkpoint_eval_metrics.csv").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))

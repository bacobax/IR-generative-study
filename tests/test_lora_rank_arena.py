from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from scripts.standalone import eval_lora_rank_arena as arena
from src.core.data.dataset_targets import DatasetTarget
from src.core.normalization import UINT8_LINEAR
from src.evaluation.generative_metrics import compute_fid, compute_kid
from src.evaluation.mmd import compute_rbf_mmd


def test_identical_features_have_near_zero_fid_and_mmd() -> None:
    rng = np.random.default_rng(123)
    features = rng.normal(size=(12, 5)).astype(np.float32)

    assert compute_fid(features, features) == pytest.approx(0.0, abs=1e-6)
    assert compute_rbf_mmd(features, features, bandwidths=[0.5, 1.0, 2.0]) == pytest.approx(0.0, abs=1e-12)


def test_kid_is_deterministic_with_seed() -> None:
    rng = np.random.default_rng(123)
    real = rng.normal(size=(20, 6)).astype(np.float32)
    generated = rng.normal(loc=0.2, size=(20, 6)).astype(np.float32)

    kid_a = compute_kid(real, generated, subsets=5, subset_size=8, seed=99)
    kid_b = compute_kid(real, generated, subsets=5, subset_size=8, seed=99)

    assert kid_a == pytest.approx(kid_b)


def test_ranking_uses_primary_secondary_tertiary_keys() -> None:
    rows = [
        {"label": "a", "rank": 8, "kid": 0.2, "fid": 1.0, "mmd": 0.1},
        {"label": "b", "rank": 16, "kid": 0.1, "fid": 4.0, "mmd": 0.1},
        {"label": "c", "rank": 32, "kid": 0.1, "fid": 2.0, "mmd": 0.3},
        {"label": "d", "rank": 64, "kid": 0.1, "fid": 2.0, "mmd": 0.2},
    ]

    ranked = arena.rank_metric_rows(
        rows,
        ranking_cfg={"primary": "kid", "secondary": "fid", "tertiary": "mmd"},
    )

    assert [row["label"] for row in ranked] == ["d", "c", "b", "a"]


def test_discover_reference_images_prefers_split_npy_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "dataset"
    split_dir = root / "val"
    images_dir = split_dir / "images"
    images_dir.mkdir(parents=True)
    split_dir.mkdir(exist_ok=True)
    np.save(split_dir / "b.npy", np.zeros((2, 2), dtype=np.uint8))
    np.save(split_dir / "a.npy", np.ones((2, 2), dtype=np.uint8))
    np.save(images_dir / "ignored.npy", np.ones((2, 2), dtype=np.uint8))

    target = DatasetTarget(
        dataset_id="fake",
        root=root,
        normalization_mode=UINT8_LINEAR,
    )
    monkeypatch.setattr(arena, "resolve_dataset_target", lambda dataset_id: target)

    paths, normalization_mode, resolved_split = arena.discover_reference_images(
        dataset_id="fake",
        reference_split="val",
        max_real_images=1,
    )

    assert paths == [split_dir / "a.npy"]
    assert normalization_mode == UINT8_LINEAR
    assert resolved_split == split_dir


def test_generate_samples_for_rank_with_fake_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Result:
        def __init__(self, count: int) -> None:
            self.images = [
                Image.fromarray(np.full((8, 8, 3), fill_value=idx, dtype=np.uint8))
                for idx in range(count)
            ]

    class _Pipe:
        def __call__(self, prompts, **kwargs):
            return _Result(len(prompts))

    monkeypatch.setattr(arena, "_build_pipeline", lambda *args, **kwargs: _Pipe())

    cfg = {
        "experiment": {"seed": 7},
        "generation": {
            "n_samples": 3,
            "batch_size": 2,
            "num_inference_steps": 1,
            "guidance_scale": 1.0,
            "prompt": "thermal image",
            "negative_prompt": "",
            "height": 8,
            "width": 8,
            "save_images": True,
            "resume_existing_images": True,
        },
    }
    rank_entry = {"rank": 8, "label": "lora_r8", "checkpoint_path": tmp_path / "lora"}

    paths = arena.generate_samples_for_rank(
        config=cfg,
        rank_entry=rank_entry,
        output_dir=tmp_path / "generated",
        device="cpu",
        dtype=torch.float32,
        force=False,
    )

    assert [path.name for path in paths] == [
        "sample_000000.png",
        "sample_000001.png",
        "sample_000002.png",
    ]
    assert (tmp_path / "generated" / "metadata.jsonl").is_file()


def test_normalize_lora_state_dict_keys_converts_diffusers_down_up_names() -> None:
    state = {
        "unet.block.to_q.lora.down.weight": torch.zeros(2, 3),
        "unet.block.to_q.lora.up.weight": torch.zeros(3, 2),
        "unet.block.proj_in.lora_A.weight": torch.zeros(2, 3, 1, 1),
    }

    normalized, changed = arena._normalize_lora_state_dict_keys(state)

    assert changed == 2
    assert "unet.block.to_q.lora_A.weight" in normalized
    assert "unet.block.to_q.lora_B.weight" in normalized
    assert "unet.block.proj_in.lora_A.weight" in normalized
    assert "unet.block.to_q.lora.down.weight" not in normalized


def test_expected_lora_targets_loaded_raises_for_missing_training_targets() -> None:
    class _LoRALinear(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = torch.nn.ModuleDict({"default_0": torch.nn.Linear(1, 1)})

    class _UNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj_in = _LoRALinear()

    class _Pipe:
        def __init__(self) -> None:
            self.unet = _UNet()

    rank_entry = {
        "label": "lora_r8",
        "stage1_manifest": {"lora_target_modules": ["proj_in", "to_q"]},
    }

    with pytest.raises(RuntimeError, match="to_q"):
        arena._assert_expected_lora_targets_loaded(_Pipe(), rank_entry)

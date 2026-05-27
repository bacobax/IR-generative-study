from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/select_best_checkpoint_and_compute_metrics.py").resolve()
SPEC = importlib.util.spec_from_file_location("checkpoint_selection", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
pipeline = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pipeline
SPEC.loader.exec_module(pipeline)


def _write_stage1_manifest(path: Path, *, baseline_mode: str = "sd_ir_lora") -> None:
    path.write_text(
        json.dumps(
            {
                "baseline_mode": baseline_mode,
                "pretrained_model_name_or_path": "runwayml/stable-diffusion-v1-5",
                "dataset_id": "flir_private_proxy_alignment_v18",
                "normalization_mode": "uint8_linear",
            }
        ),
        encoding="utf-8",
    )


def test_native_fm_checkpoint_discovery_identifiers_and_threshold(tmp_path: Path) -> None:
    unet_dir = tmp_path / "UNET"
    unet_dir.mkdir()
    (unet_dir / "unet_fm_best.pt").write_bytes(b"best")
    (unet_dir / "unet_fm_epoch_049.pt").write_bytes(b"old")
    (unet_dir / "unet_fm_epoch_050.pt").write_bytes(b"epoch50")
    (unet_dir / "unet_fm_epoch_100.pt").write_bytes(b"epoch100")
    (unet_dir / "unet_fm_epoch_100_ckpt.pt").write_bytes(b"resume")

    result = pipeline.discover_candidate_checkpoints(
        tmp_path,
        model_type="latent_flow_matching",
        checkpoint_min_epoch=50,
    )

    identifiers = [candidate.checkpoint_identifier for candidate in result.candidates]
    assert identifiers == ["best", "final", "epoch_050"]
    assert result.candidates[1].checkpoint_path.endswith("unet_fm_epoch_100.pt")
    assert any("epoch 49 < checkpoint_min_epoch 50" in item.reason for item in result.excluded)
    assert any("duplicate checkpoint path" in item.reason for item in result.excluded)


def test_native_sd_uncond_checkpoint_discovery(tmp_path: Path) -> None:
    unet_dir = tmp_path / "UNET"
    unet_dir.mkdir()
    (tmp_path / "SCHEDULER").mkdir()
    (unet_dir / "unet_sd_uncond_best.pt").write_bytes(b"best")
    (unet_dir / "unet_sd_uncond_epoch_075.pt").write_bytes(b"epoch75")

    result = pipeline.discover_candidate_checkpoints(
        tmp_path,
        model_type="sd_uncond",
        checkpoint_min_epoch=50,
    )

    assert [candidate.checkpoint_identifier for candidate in result.candidates] == [
        "best",
        "final",
    ]
    assert result.candidates[1].epoch == 75


def test_sd_lora_final_and_step_checkpoint_discovery(tmp_path: Path) -> None:
    _write_stage1_manifest(tmp_path / "stage1_manifest.json")
    (tmp_path / "pytorch_lora_weights.safetensors").write_bytes(b"final")
    old = tmp_path / "checkpoint-10"
    old.mkdir()
    (old / "pytorch_lora_weights.safetensors").write_bytes(b"old")
    step = tmp_path / "checkpoint-123"
    step.mkdir()
    (step / "pytorch_lora_weights.safetensors").write_bytes(b"step")
    missing = tmp_path / "checkpoint-456"
    missing.mkdir()

    result = pipeline.discover_candidate_checkpoints(
        tmp_path,
        model_type="sd_lora",
        checkpoint_min_step=100,
    )

    assert [candidate.checkpoint_identifier for candidate in result.candidates] == [
        "final",
        "step_000123",
    ]
    assert any("step 10 < checkpoint_min_step 100" in item.reason for item in result.excluded)
    assert any("missing Diffusers LoRA checkpoint weights" in item.reason for item in result.excluded)


def test_stage_seed_partitions_do_not_overlap() -> None:
    seeds = pipeline.make_stage_seeds(
        {
            "generation_seed": 1234,
            "stage1_num_images": 3,
            "stage2_extra_images": 4,
            "stage3_extra_images": 5,
            "stage1_seed_offset": 0,
            "stage2_seed_offset": 100000,
            "stage3_seed_offset": 200000,
        }
    )
    assert seeds["stage1"] == [1234, 1235, 1236]
    assert seeds["stage2"][0] == 101234
    assert seeds["stage3"][0] == 201234
    flattened = seeds["stage1"] + seeds["stage2"] + seeds["stage3"]
    assert len(flattened) == len(set(flattened))


def test_weighted_normalized_ranking_handles_identical_values() -> None:
    rows = [
        {"checkpoint_identifier": "a", "KID": 1.0, "FID": 5.0},
        {"checkpoint_identifier": "b", "KID": 1.0, "FID": 7.0},
        {"checkpoint_identifier": "c", "KID": 1.0, "FID": 6.0},
    ]

    ranked = pipeline.add_weighted_normalized_scores(rows, kid_weight=0.8, fid_weight=0.2)

    assert [row["checkpoint_identifier"] for row in ranked] == ["a", "c", "b"]
    assert all(row["normalized_KID"] == 0.0 for row in ranked)
    assert [row["rank"] for row in ranked] == [1, 2, 3]


def test_preview_writer_accepts_channel_first_arrays(tmp_path: Path) -> None:
    arr = (255 * pipeline.np.ones((1, 8, 8), dtype=pipeline.np.uint8))
    image_path = tmp_path / "sample_000000.npy"
    pipeline.np.save(image_path, arr)

    pipeline._save_preview_contact_sheet(
        [image_path],
        tmp_path / "preview_grid.png",
        normalization_mode=pipeline.UINT8_LINEAR,
        columns=1,
        tile_size=16,
    )

    assert (tmp_path / "preview_grid.png").is_file()

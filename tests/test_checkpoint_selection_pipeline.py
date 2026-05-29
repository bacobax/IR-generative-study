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


def _run_resolution(run_dir: Path, *, model_type: str) -> pipeline.RunResolution:
    return pipeline.RunResolution(
        run_identifier="run",
        run_dir=run_dir,
        model_type=model_type,
        sampler_name=None,
        sampling_config_path=None,
        preset={},
        generation_backend_used="test",
    )


def _cleanup(
    tmp_path: Path,
    run_dir: Path,
    *,
    model_type: str,
    stage2_ids: list[str],
) -> dict:
    discovery = pipeline.discover_candidate_checkpoints(
        run_dir,
        model_type=model_type,
        checkpoint_min_epoch=0,
        checkpoint_min_step=0,
    )
    return pipeline.cleanup_training_checkpoints(
        run=_run_resolution(run_dir, model_type=model_type),
        discovery=discovery,
        stage2_ranking=[{"checkpoint_identifier": checkpoint_id, "KID": idx} for idx, checkpoint_id in enumerate(stage2_ids)],
        run_output_dir=tmp_path / "selection_output",
    )


def test_cleanup_native_fm_preserves_stage2_top3_and_latest_epoch(tmp_path: Path) -> None:
    run_dir = tmp_path / "fm_run"
    unet_dir = run_dir / "UNET"
    unet_dir.mkdir(parents=True)
    (run_dir / "VAE").mkdir()
    for name in [
        "unet_fm_best.pt",
        "unet_fm_epoch_025.pt",
        "unet_fm_epoch_050.pt",
        "unet_fm_epoch_075.pt",
        "unet_fm_epoch_100.pt",
    ]:
        (unet_dir / name).write_bytes(name.encode())
    (run_dir / "VAE" / "config.json").write_text("{}", encoding="utf-8")

    result = _cleanup(
        tmp_path,
        run_dir,
        model_type="latent_flow_matching",
        stage2_ids=["best", "epoch_050", "epoch_075"],
    )

    assert (unet_dir / "unet_fm_best.pt").is_file()
    assert (unet_dir / "unet_fm_epoch_050.pt").is_file()
    assert (unet_dir / "unet_fm_epoch_075.pt").is_file()
    assert (unet_dir / "unet_fm_epoch_100.pt").is_file()
    assert not (unet_dir / "unet_fm_epoch_025.pt").exists()
    assert (run_dir / "VAE" / "config.json").is_file()
    assert (tmp_path / "selection_output" / "checkpoint_cleanup_plan.json").is_file()
    assert any(row["path"].endswith("unet_fm_epoch_025.pt") for row in result["deleted"])


def test_cleanup_native_sd_preserves_stage2_top3_and_latest_epoch(tmp_path: Path) -> None:
    run_dir = tmp_path / "sd_run"
    unet_dir = run_dir / "UNET"
    unet_dir.mkdir(parents=True)
    (run_dir / "SCHEDULER").mkdir()
    for name in [
        "unet_sd_uncond_best.pt",
        "unet_sd_uncond_epoch_010.pt",
        "unet_sd_uncond_epoch_030.pt",
        "unet_sd_uncond_epoch_050.pt",
        "unet_sd_uncond_epoch_090.pt",
    ]:
        (unet_dir / name).write_bytes(name.encode())

    _cleanup(
        tmp_path,
        run_dir,
        model_type="sd_uncond",
        stage2_ids=["epoch_030", "epoch_050", "best"],
    )

    assert (unet_dir / "unet_sd_uncond_best.pt").is_file()
    assert (unet_dir / "unet_sd_uncond_epoch_030.pt").is_file()
    assert (unet_dir / "unet_sd_uncond_epoch_050.pt").is_file()
    assert (unet_dir / "unet_sd_uncond_epoch_090.pt").is_file()
    assert not (unet_dir / "unet_sd_uncond_epoch_010.pt").exists()
    assert (run_dir / "SCHEDULER").is_dir()


def test_cleanup_native_preserves_sidecar_for_preserved_epoch(tmp_path: Path) -> None:
    run_dir = tmp_path / "fm_run"
    unet_dir = run_dir / "UNET"
    unet_dir.mkdir(parents=True)
    for name in [
        "unet_fm_epoch_025.pt",
        "unet_fm_epoch_025_ckpt.pt",
        "unet_fm_epoch_050.pt",
        "unet_fm_epoch_050_ckpt.pt",
        "unet_fm_epoch_100.pt",
    ]:
        (unet_dir / name).write_bytes(name.encode())

    _cleanup(
        tmp_path,
        run_dir,
        model_type="latent_flow_matching",
        stage2_ids=["epoch_050"],
    )

    assert (unet_dir / "unet_fm_epoch_050.pt").is_file()
    assert (unet_dir / "unet_fm_epoch_050_ckpt.pt").is_file()
    assert (unet_dir / "unet_fm_epoch_100.pt").is_file()
    assert not (unet_dir / "unet_fm_epoch_025.pt").exists()
    assert not (unet_dir / "unet_fm_epoch_025_ckpt.pt").exists()


def test_cleanup_lora_preserves_top3_latest_and_final_export(tmp_path: Path) -> None:
    run_dir = tmp_path / "lora_run"
    run_dir.mkdir()
    _write_stage1_manifest(run_dir / "stage1_manifest.json")
    (run_dir / "pytorch_lora_weights.safetensors").write_bytes(b"final")
    (run_dir / "logs").mkdir()
    for step in [10, 20, 30, 40, 50]:
        checkpoint_dir = run_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "pytorch_lora_weights.safetensors").write_bytes(str(step).encode())

    _cleanup(
        tmp_path,
        run_dir,
        model_type="sd_lora",
        stage2_ids=["step_000020", "step_000030", "step_000040"],
    )

    assert not (run_dir / "checkpoint-10").exists()
    for step in [20, 30, 40, 50]:
        assert (run_dir / f"checkpoint-{step}").is_dir()
    assert (run_dir / "pytorch_lora_weights.safetensors").is_file()
    assert (run_dir / "stage1_manifest.json").is_file()
    assert (run_dir / "logs").is_dir()


def test_cleanup_is_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "lora_run"
    run_dir.mkdir()
    _write_stage1_manifest(run_dir / "stage1_manifest.json")
    (run_dir / "pytorch_lora_weights.safetensors").write_bytes(b"final")
    for step in [10, 20, 30]:
        checkpoint_dir = run_dir / f"checkpoint-{step}"
        checkpoint_dir.mkdir()
        (checkpoint_dir / "pytorch_lora_weights.safetensors").write_bytes(str(step).encode())

    kwargs = {
        "model_type": "sd_lora",
        "stage2_ids": ["step_000020"],
    }
    first = _cleanup(tmp_path, run_dir, **kwargs)
    second = _cleanup(tmp_path, run_dir, **kwargs)

    assert any(row["path"].endswith("checkpoint-10") for row in first["deleted"])
    assert second["deleted"] == []
    assert (tmp_path / "selection_output" / "checkpoint_cleanup_result.json").is_file()


def test_analysis_preview_root_falls_back_to_output_root(tmp_path: Path) -> None:
    root = pipeline._analysis_preview_root(
        {"output_root": str(tmp_path / "selection"), "analysis_output_root": None},
        "run a",
    )

    assert root == tmp_path / "selection" / "run_a"


def test_cached_resume_survives_manually_deleted_selected_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "fm_run"
    unet_dir = run_dir / "UNET"
    unet_dir.mkdir(parents=True)
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps({"model_family": "flow_matching", "task": {"t_scale": 1000.0, "train_target": "v"}}),
        encoding="utf-8",
    )
    (unet_dir / "unet_fm_epoch_300.pt").write_bytes(b"latest")
    deleted_selected_path = unet_dir / "unet_fm_epoch_240.pt"

    output_root = tmp_path / "selection"
    run_output = output_root / "manual_cleanup_run"
    run_output.mkdir(parents=True)
    pipeline.save_json(
        run_output / "stage1_metrics.json",
        {
            "selected_top_k_checkpoints": ["epoch_240"],
            "ranking": [
                {
                    "checkpoint_identifier": "epoch_240",
                    "checkpoint_path": str(deleted_selected_path),
                    "KID": 0.1,
                    "FID": 1.0,
                    "selection_score": 0.0,
                    "rank": 1,
                }
            ],
        },
    )
    pipeline.save_json(
        run_output / "stage2_metrics.json",
        {
            "ranking": [
                {
                    "checkpoint_identifier": "epoch_240",
                    "checkpoint_path": str(deleted_selected_path),
                    "KID": 0.1,
                    "rank": 1,
                }
            ]
        },
    )
    pipeline.save_json(
        run_output / "final_metrics.json",
        {
            "selected_checkpoint_identifier": "epoch_240",
            "selected_checkpoint_path": str(deleted_selected_path),
            "total_generated_images": 3,
            "KID": 0.1,
            "FID": 1.0,
        },
    )
    images_dir = run_output / "epoch_240" / "stage3" / "generated_npy_images"
    images_dir.mkdir(parents=True)
    pipeline.np.save(images_dir / "sample_000000.npy", pipeline.np.zeros((1, 4, 4), dtype=pipeline.np.uint8))

    result = pipeline.run_one(
        {
            "run_identifier": "manual_cleanup_run",
            "run_dir": str(run_dir),
            "model_type": "latent_flow_matching",
            "sampling_config_path": None,
        },
        {
            "output_root": str(output_root),
            "analysis_output_root": None,
            "overwrite_existing_metrics": False,
            "save_analysis_previews": True,
            "analysis_preview_num_images": 1,
        },
    )

    assert result["final_selected_checkpoint"] == "epoch_240"
    assert result["selected_top_3_checkpoints"] == ["epoch_240"]
    assert (run_output / "checkpoint_selection_summary.json").is_file()
    assert (output_root / "manual_cleanup_run" / "epoch_240" / "stage3" / "preview_grid.png").is_file()


def test_cached_resume_cleanup_ignores_missing_cached_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "fm_run"
    unet_dir = run_dir / "UNET"
    unet_dir.mkdir(parents=True)
    (run_dir / "artifact_manifest.json").write_text(
        json.dumps({"model_family": "flow_matching", "task": {"t_scale": 1000.0, "train_target": "v"}}),
        encoding="utf-8",
    )
    (unet_dir / "unet_fm_epoch_300.pt").write_bytes(b"latest")
    deleted_selected_path = unet_dir / "unet_fm_epoch_240.pt"

    output_root = tmp_path / "selection"
    run_output = output_root / "manual_cleanup_run"
    run_output.mkdir(parents=True)
    payload_row = {
        "checkpoint_identifier": "epoch_240",
        "checkpoint_path": str(deleted_selected_path),
        "KID": 0.1,
        "FID": 1.0,
        "rank": 1,
    }
    pipeline.save_json(run_output / "stage1_metrics.json", {"selected_top_k_checkpoints": ["epoch_240"], "ranking": [payload_row]})
    pipeline.save_json(run_output / "stage2_metrics.json", {"ranking": [payload_row]})
    pipeline.save_json(
        run_output / "final_metrics.json",
        {
            "selected_checkpoint_identifier": "epoch_240",
            "selected_checkpoint_path": str(deleted_selected_path),
            "total_generated_images": 3,
            "KID": 0.1,
            "FID": 1.0,
        },
    )

    result = pipeline.run_one(
        {
            "run_identifier": "manual_cleanup_run",
            "run_dir": str(run_dir),
            "model_type": "latent_flow_matching",
            "sampling_config_path": None,
        },
        {
            "output_root": str(output_root),
            "analysis_output_root": None,
            "overwrite_existing_metrics": False,
            "save_analysis_previews": False,
        },
        cleanup_checkpoints=True,
    )

    assert result["final_selected_checkpoint"] == "epoch_240"
    assert (unet_dir / "unet_fm_epoch_300.pt").is_file()
    assert result["checkpoint_cleanup"]["deleted"] == []

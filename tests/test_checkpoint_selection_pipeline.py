from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


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


def test_pipeline_mode_defaults_to_legacy() -> None:
    assert pipeline.pipeline_mode({}) == "legacy_staged_kid_fid"
    assert pipeline.pipeline_mode({"pipeline_mode": "clean_fid_selection_publication"}) == "clean_fid_selection_publication"


def test_publication_seed_partitions_do_not_overlap() -> None:
    seeds = pipeline.make_publication_seeds(
        {
            "selection": {"selection_num_images": 3},
            "final": {
                "final_extra_images": 2,
                "reuse_selection_images_for_top1": True,
                "final_total_images": 5,
            },
            "generation": {
                "generation_seed": 1234,
                "selection_seed_offset": 0,
                "final_extra_seed_offset": 1000000,
            },
        }
    )

    assert seeds["selection"] == [1234, 1235, 1236]
    assert seeds["final_extra"] == [1001234, 1001235]
    assert len(seeds["selection"] + seeds["final_extra"]) == len(set(seeds["selection"] + seeds["final_extra"]))


def test_publication_seed_validation_rejects_final_total_mismatch() -> None:
    try:
        pipeline.make_publication_seeds(
            {
                "selection": {"selection_num_images": 3},
                "final": {
                    "final_extra_images": 2,
                    "reuse_selection_images_for_top1": True,
                    "final_total_images": 4,
                },
            }
        )
    except ValueError as exc:
        assert "final_total_images" in str(exc)
    else:
        raise AssertionError("Expected final_total_images validation to fail.")


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


def test_publication_ranking_uses_inception_fid_fallback_when_clean_fid_missing() -> None:
    ranked, effective = pipeline._rank_publication_selection_rows(
        [
            {"checkpoint_identifier": "a", "metric_values": {"inception_fid_fallback": 3.0}},
            {"checkpoint_identifier": "b", "metric_values": {"inception_fid_fallback": 1.0}},
        ],
        requested_metric="clean_fid",
        lower_is_better=True,
    )

    assert effective == "inception_fid_fallback"
    assert [row["checkpoint_identifier"] for row in ranked] == ["b", "a"]
    assert ranked[0]["requested_selection_metric"] == "clean_fid"
    assert ranked[0]["effective_selection_metric"] == "inception_fid_fallback"


def test_final_combined_manifest_contains_selection_then_final_extra(tmp_path: Path) -> None:
    run = pipeline.RunResolution(
        run_identifier="run",
        run_dir=tmp_path / "run",
        model_type="latent_flow_matching",
        sampler_name=None,
        sampling_config_path=None,
        preset={},
        generation_backend_used="native_flow_matching_sampler",
    )
    checkpoint = pipeline.CheckpointCandidate("epoch_001", str(tmp_path / "ckpt.pt"), "epoch", epoch=1)
    selection_paths = [tmp_path / "selection" / f"sample_{idx:06d}.npy" for idx in range(2)]
    final_paths = [tmp_path / "final_extra" / "sample_000000.npy"]

    manifest = pipeline._write_final_combined_manifest(
        manifest_path=tmp_path / "final_combined" / "image_manifest.json",
        selection_paths=selection_paths,
        final_extra_paths=final_paths,
        selected=checkpoint,
        run=run,
        seeds={"selection": [10, 11], "final_extra": [100]},
    )

    assert manifest["total_images"] == 3
    assert [row["phase"] for row in manifest["image_paths"]] == ["selection", "selection", "final_extra"]
    assert [row["seed"] for row in manifest["image_paths"]] == [10, 11, 100]
    assert (tmp_path / "final_combined" / "image_manifest.json").is_file()


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


def test_ensure_generated_stage_writes_analysis_preview_for_cached_stage(tmp_path: Path) -> None:
    run = pipeline.RunResolution(
        run_identifier="preview_run",
        run_dir=tmp_path / "run",
        model_type="latent_flow_matching",
        sampler_name=None,
        sampling_config_path=None,
        preset={"data": {"image_size": 512, "dataset_id": "flir_private_proxy_alignment_v18"}},
        generation_backend_used="native_flow_matching_sampler",
    )
    checkpoint = pipeline.CheckpointCandidate("best", str(tmp_path / "dummy.pt"), "best")
    stage_dir = tmp_path / "selection" / "preview_run" / "best" / "stage1"
    images_dir = stage_dir / "generated_npy_images"
    images_dir.mkdir(parents=True)
    arr = pipeline.np.arange(512 * 512, dtype=pipeline.np.uint8).reshape(512, 512)
    pipeline.np.save(images_dir / "sample_000000.npy", arr)

    pipeline.ensure_generated_stage(
        run=run,
        checkpoint=checkpoint,
        stage_dir=stage_dir,
        seeds=[123],
        config={
            "output_root": str(tmp_path / "selection"),
            "analysis_output_root": None,
            "save_analysis_previews": True,
            "analysis_preview_num_images": 1,
            "analysis_preview_tile_size": 512,
            "analysis_preview_columns": 1,
            "generated_min_std": 1.0,
        },
        device="cpu",
    )

    assert (
        tmp_path
        / "selection"
        / "preview_run"
        / "best"
        / "stage1"
        / "previews"
        / "sample_000000.png"
    ).is_file()
    assert (tmp_path / "selection" / "preview_run" / "best" / "stage1" / "preview_grid.png").is_file()


def test_ensure_generated_stage_writes_analysis_preview_after_generation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    run = pipeline.RunResolution(
        run_identifier="preview_run",
        run_dir=tmp_path / "run",
        model_type="latent_flow_matching",
        sampler_name=None,
        sampling_config_path=None,
        preset={"data": {"image_size": 512, "dataset_id": "flir_private_proxy_alignment_v18"}},
        generation_backend_used="native_flow_matching_sampler",
    )
    checkpoint = pipeline.CheckpointCandidate("best", str(tmp_path / "dummy.pt"), "best")
    stage_dir = tmp_path / "selection" / "preview_run" / "best" / "stage2"

    def fake_generate_native_samples(**kwargs):
        images_dir = kwargs["images_dir"]
        images_dir.mkdir(parents=True, exist_ok=True)
        arr = pipeline.np.arange(512 * 512, dtype=pipeline.np.uint8).reshape(512, 512)
        pipeline.save_npy_atomic(images_dir / "sample_000000.npy", arr)

    monkeypatch.setattr(pipeline, "generate_native_samples", fake_generate_native_samples)

    pipeline.ensure_generated_stage(
        run=run,
        checkpoint=checkpoint,
        stage_dir=stage_dir,
        seeds=[123],
        config={
            "output_root": str(tmp_path / "selection"),
            "analysis_output_root": None,
            "save_analysis_previews": True,
            "analysis_preview_num_images": 1,
            "analysis_preview_tile_size": 512,
            "analysis_preview_columns": 1,
            "generated_min_std": 1.0,
        },
        device="cpu",
    )

    assert (
        tmp_path
        / "selection"
        / "preview_run"
        / "best"
        / "stage2"
        / "previews"
        / "sample_000000.png"
    ).is_file()
    assert (tmp_path / "selection" / "preview_run" / "best" / "stage2" / "preview_grid.png").is_file()


def test_sd_stage1_generation_writes_only_npy_samples(tmp_path: Path, monkeypatch) -> None:
    class FakePipe:
        def __call__(self, *args, **kwargs):
            return SimpleNamespace(images=[object()])

    run = pipeline.RunResolution(
        run_identifier="sd_preview_run",
        run_dir=tmp_path / "run",
        model_type="sd_lora",
        sampler_name=None,
        sampling_config_path=None,
        preset={"data": {"image_size": 8}},
        generation_backend_used="sd_lora_pipeline",
    )
    checkpoint = pipeline.CheckpointCandidate("best", str(tmp_path / "dummy"), "best")
    images_dir = tmp_path / "generated_npy_images"
    images_dir.mkdir()

    monkeypatch.setattr(
        pipeline,
        "build_sd_stage1_pipeline",
        lambda *args, **kwargs: (FakePipe(), {"normalization_mode": pipeline.UINT8_LINEAR}),
    )
    monkeypatch.setattr(
        pipeline,
        "sd_output_to_npy",
        lambda *args, **kwargs: pipeline.np.ones((8, 8), dtype=pipeline.np.uint8),
    )

    pipeline.generate_sd_stage1_samples(
        run=run,
        checkpoint=checkpoint,
        images_dir=images_dir,
        seeds=[123],
        config={"num_inference_steps": 1, "generation_prompt": "test"},
        device="cpu",
    )

    assert (images_dir / "sample_000000.npy").is_file()
    assert not (images_dir / "sample_000000.png").exists()
    assert not list(images_dir.glob("*.png"))


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


def test_discover_reference_images_accepts_bigearthnet_validation_tiffs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.core.data.dataset_targets import (
        BigEarthNetS2B08DatasetAdapter,
        DEFAULT_DATASET_TARGETS,
        DatasetTarget,
    )
    from src.core.normalization import SENTINEL2_REFLECTANCE

    dataset_root = tmp_path / "bigearthnet"
    validation_dir = dataset_root / "images" / "validation"
    validation_dir.mkdir(parents=True)
    (validation_dir / "sample_b.tif").write_bytes(b"tif")
    (validation_dir / "sample_a.tiff").write_bytes(b"tiff")
    (validation_dir / "ignore.txt").write_text("ignore", encoding="utf-8")
    adapter = BigEarthNetS2B08DatasetAdapter(dataset_root)
    monkeypatch.setitem(
        DEFAULT_DATASET_TARGETS,
        "unit_bigearthnet",
        DatasetTarget(
            dataset_id="unit_bigearthnet",
            root=dataset_root,
            normalization_mode=SENTINEL2_REFLECTANCE,
            adapter=adapter,
        ),
    )

    paths, normalization_mode, reference_root = pipeline.discover_reference_images(
        {"dataset_id": "unit_bigearthnet", "real_reference_split": "val"},
        _run_resolution(tmp_path / "run", model_type="latent_flow_matching"),
    )

    assert [path.name for path in paths] == ["sample_a.tiff", "sample_b.tif"]
    assert normalization_mode == SENTINEL2_REFLECTANCE
    assert reference_root == validation_dir


def test_discover_reference_images_accepts_explicit_tiff_reference_path(tmp_path: Path) -> None:
    reference_dir = tmp_path / "references"
    reference_dir.mkdir()
    (reference_dir / "sample_b.tif").write_bytes(b"tif")
    (reference_dir / "sample_a.tiff").write_bytes(b"tiff")
    (reference_dir / "sample.npy").write_bytes(b"npy")

    paths, normalization_mode, reference_root = pipeline.discover_reference_images(
        {"real_reference_path": str(reference_dir), "real_reference_num_samples": 2},
        _run_resolution(tmp_path / "run", model_type="sd_uncond"),
    )

    assert [path.name for path in paths] == ["sample.npy", "sample_a.tiff"]
    assert normalization_mode == pipeline.UINT8_LINEAR
    assert reference_root == reference_dir


def test_discover_publication_reference_sources_train_val_test_combined(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.core.data.dataset_targets import DEFAULT_DATASET_TARGETS, DatasetTarget

    dataset_root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        split_dir.mkdir(parents=True)
        pipeline.np.save(split_dir / f"{split}_a.npy", pipeline.np.zeros((4, 4), dtype=pipeline.np.uint8))
        pipeline.np.save(split_dir / f"{split}_b.npy", pipeline.np.ones((4, 4), dtype=pipeline.np.uint8))
    monkeypatch.setitem(
        DEFAULT_DATASET_TARGETS,
        "unit_reference_sources",
        DatasetTarget(
            dataset_id="unit_reference_sources",
            root=dataset_root,
            normalization_mode=pipeline.UINT8_LINEAR,
        ),
    )

    sources = pipeline.discover_reference_sources(
        {
            "reference_data": {
                "dataset_id": "unit_reference_sources",
                "real_reference_splits": {"train": "train", "val": "val", "test": "test"},
                "real_reference_num_samples": {
                    "train": None,
                    "val": None,
                    "test": None,
                    "train_val_test": None,
                },
            }
        },
        _run_resolution(tmp_path / "run", model_type="latent_flow_matching"),
        ["train", "val", "test", "train_val_test"],
    )

    assert sources["train"]["num_real_images"] == 2
    assert sources["val"]["num_real_images"] == 2
    assert sources["test"]["num_real_images"] == 2
    assert sources["train_val_test"]["num_real_images"] == 6
    assert sources["train_val_test"]["splits"] == ["train", "val", "test"]
    assert [path.name for path in sources["train_val_test"]["paths"]] == [
        "train_a.npy",
        "train_b.npy",
        "val_a.npy",
        "val_b.npy",
        "test_a.npy",
        "test_b.npy",
    ]


def test_publication_preflight_reports_reference_counts_and_paths(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from src.core.data.dataset_targets import DEFAULT_DATASET_TARGETS, DatasetTarget

    dataset_root = tmp_path / "dataset"
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        split_dir.mkdir(parents=True)
        pipeline.np.save(split_dir / f"{split}.npy", pipeline.np.ones((8, 8), dtype=pipeline.np.uint8))
    monkeypatch.setitem(
        DEFAULT_DATASET_TARGETS,
        "unit_preflight_refs",
        DatasetTarget(
            dataset_id="unit_preflight_refs",
            root=dataset_root,
            normalization_mode=pipeline.UINT8_LINEAR,
        ),
    )

    run_dir = tmp_path / "run"
    unet_dir = run_dir / "UNET"
    unet_dir.mkdir(parents=True)
    (unet_dir / "config.json").write_text(
        json.dumps({"sample_size": 8, "in_channels": 4}),
        encoding="utf-8",
    )
    (unet_dir / "unet_fm_best.pt").write_bytes(b"best")
    (unet_dir / "unet_fm_epoch_050.pt").write_bytes(b"epoch")
    preset_path = tmp_path / "preset.yaml"
    preset_path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_id: unit_preflight_refs",
                "  image_size: 64",
                "training:",
                "  t_scale: 1000.0",
                "  train_target: v",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = pipeline.preflight_config(
        {
            "pipeline_mode": "clean_fid_selection_publication",
            "runs": [
                {
                    "run_identifier": "unit_run",
                    "run_dir": str(run_dir),
                    "model_type": "latent_flow_matching",
                    "sampling_config_path": str(preset_path),
                }
            ],
            "checkpoint_min_epoch": 50,
            "selection": {"selection_num_images": 2, "selection_reference_source": "val"},
            "final": {"final_extra_images": 1, "final_total_images": 3},
            "generation": {"device": "cpu"},
            "reference_data": {"dataset_id": "unit_preflight_refs"},
            "output": {"output_root": str(tmp_path / "out")},
        }
    )

    run_payload = payload["runs"][0]
    assert payload["pipeline_mode"] == "clean_fid_selection_publication"
    assert run_payload["selection_num_real_images"] == 1
    assert run_payload["final_reference_sources"]["train_val_test"]["num_real_images"] == 3
    assert run_payload["planned_num_generated_images_per_checkpoint"] == 2
    assert run_payload["planned_final_extra_images"] == 1
    assert run_payload["expected_output_paths"]["selection_metrics"].endswith("selection_metrics.json")


def test_validate_generation_dir_rejects_wrong_resolution_and_black_cache(tmp_path: Path) -> None:
    images_dir = tmp_path / "generated"
    images_dir.mkdir()
    pipeline.np.save(images_dir / "sample_000000.npy", pipeline.np.zeros((128, 128), dtype=pipeline.np.uint8))

    missing, complete = pipeline.validate_or_prepare_generation_dir(
        images_dir,
        n_images=1,
        overwrite=False,
        expected_hw=(512, 512),
        min_std=1e-6,
        normalization_mode=pipeline.UINT8_LINEAR,
    )

    assert missing == [0]
    assert complete is False
    assert not (images_dir / "sample_000000.npy").exists()

    pipeline.np.save(images_dir / "sample_000000.npy", pipeline.np.zeros((512, 512), dtype=pipeline.np.uint8))
    missing, complete = pipeline.validate_or_prepare_generation_dir(
        images_dir,
        n_images=1,
        overwrite=False,
        expected_hw=(512, 512),
        min_std=1e-6,
        normalization_mode=pipeline.UINT8_LINEAR,
    )

    assert missing == [0]
    assert complete is False
    assert not (images_dir / "sample_000000.npy").exists()

    pipeline.np.save(
        images_dir / "sample_000000.npy",
        pipeline.np.linspace(-1.0, 1.0, 512 * 512, dtype=pipeline.np.float32).reshape(512, 512),
    )
    missing, complete = pipeline.validate_or_prepare_generation_dir(
        images_dir,
        n_images=1,
        overwrite=False,
        expected_hw=(512, 512),
        min_std=1e-6,
        normalization_mode=pipeline.UINT8_LINEAR,
    )

    assert missing == [0]
    assert complete is False
    assert not (images_dir / "sample_000000.npy").exists()


def test_native_generation_saves_raw_domain_512_nonblack_arrays(tmp_path: Path, monkeypatch) -> None:
    from scripts.standalone import generate_checkpoint_quality_comparison as helpers

    class FakeSampler:
        device = "cpu"

        def sample(self, *, steps: int, batch_size: int):
            del steps, batch_size
            return pipeline.torch.zeros((1, 1, 64, 64))

        def sample_euler(self, *, steps: int, batch_size: int):
            del steps, batch_size
            return pipeline.torch.zeros((1, 1, 64, 64))

        def decode(self, latents):
            del latents
            return pipeline.torch.linspace(-1.0, 1.0, 512 * 512).reshape(1, 1, 512, 512)

    for model_family, model_type in [
        ("fm", "latent_flow_matching"),
        ("sd", "sd_uncond"),
    ]:
        fake_helpers = SimpleNamespace(
            resolve_run_dirs=lambda run_dir: SimpleNamespace(pipeline_dir=Path(run_dir)),
            detect_run_kind=lambda pipeline_dir, preset, model_family, active_family=model_family: SimpleNamespace(
                model_family=active_family,
                layout_conditioned=False,
                layout_variant="none",
            ),
            _build_fm_sampler=lambda **kwargs: FakeSampler(),
            _build_sd_sampler=lambda **kwargs: FakeSampler(),
            _normalization_mode_from_preset=lambda preset: pipeline.UINT8_LINEAR,
            tensor_to_output_array=helpers.tensor_to_output_array,
        )
        monkeypatch.setattr(pipeline, "_qcmp_helpers", lambda active_helpers=fake_helpers: active_helpers)
        images_dir = tmp_path / model_family / "generated"

        pipeline.generate_native_samples(
            run=_run_resolution(tmp_path / model_family / "run", model_type=model_type),
            checkpoint=pipeline.CheckpointCandidate("epoch_001", str(tmp_path / "dummy.pt"), "epoch", epoch=1),
            images_dir=images_dir,
            seeds=[123],
            config={"num_inference_steps": 1},
            device="cpu",
        )

        arr = pipeline.np.load(images_dir / "sample_000000.npy")
        assert arr.shape == (512, 512)
        assert arr.dtype == pipeline.np.uint8
        assert float(arr.std()) > 1.0


def test_sd_lora_generation_uses_preset_resolution(tmp_path: Path, monkeypatch) -> None:
    calls = []

    class FakePipe:
        def __call__(self, prompt, **kwargs):
            del prompt
            height = int(kwargs["height"])
            width = int(kwargs["width"])
            calls.append((height, width))
            image = pipeline.Image.fromarray(
                pipeline.np.full((height, width, 3), 128, dtype=pipeline.np.uint8)
            )
            return SimpleNamespace(images=[image])

    monkeypatch.setattr(
        pipeline,
        "build_sd_stage1_pipeline",
        lambda run, checkpoint, *, config, device: (
            FakePipe(),
            {"normalization_mode": pipeline.UINT8_LINEAR, "prompt_text": "thermal image"},
        ),
    )
    run = pipeline.RunResolution(
        run_identifier="lora",
        run_dir=tmp_path / "run",
        model_type="sd_lora",
        sampler_name=None,
        sampling_config_path=None,
        preset={"resolution": 512},
        generation_backend_used="diffusers_stable_diffusion_lora",
    )
    images_dir = tmp_path / "generated"

    pipeline.generate_sd_stage1_samples(
        run=run,
        checkpoint=pipeline.CheckpointCandidate("step_1", str(tmp_path / "checkpoint-1"), "step", step=1),
        images_dir=images_dir,
        seeds=[123],
        config={"num_inference_steps": 1},
        device="cpu",
    )

    arr = pipeline.np.load(images_dir / "sample_000000.npy")
    assert calls == [(512, 512)]
    assert arr.shape == (512, 512)

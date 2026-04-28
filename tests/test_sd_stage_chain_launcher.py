"""Tests for the SD Stage-1 -> RegionDiff chain launcher."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path("scripts/train/run_sd_stage1_then_regiondiff.py").resolve()
SPEC = importlib.util.spec_from_file_location("run_sd_stage1_then_regiondiff", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
chain = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = chain
SPEC.loader.exec_module(chain)


def _write_lora_stage1_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "pretrained_model_name_or_path: runwayml/stable-diffusion-v1-5",
                "dataset_id: flir_private_proxy_alignment_v18",
                "baseline_mode: sd_ir_lora",
                f"output_dir: {output_dir}",
                "num_train_epochs: 1",
                "max_train_steps: 1",
                "checkpointing_epochs: 1",
                "validation_prompt: null",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_unet_stage1_config(path: Path, output_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "pretrained_model_name_or_path: runwayml/stable-diffusion-v1-5",
                "dataset_id: flir_private_proxy_alignment_v18",
                "baseline_mode: sd_ir_unet",
                "unet_train_mode: full",
                f"output_dir: {output_dir}",
                "num_train_epochs: 1",
                "max_train_steps: 1",
                "checkpointing_epochs: 1",
                "validation_prompt: null",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_stage2_config(path: Path, output_dir: Path, stage1_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "data:",
                "  dataset_id: flir_private_proxy_alignment_v18",
                "  resolution: 64",
                "  batch_size: 1",
                "stage1:",
                "  pretrained_model_name_or_path: runwayml/stable-diffusion-v1-5",
                f"  stage1_dir: {stage1_dir}",
                "  stage1_checkpoint: null",
                "training:",
                "  num_train_epochs: 1",
                "  max_train_steps: 1",
                "  checkpointing_steps: 1",
                "validation:",
                "  num_validation_images: 1",
                "output:",
                f"  output_dir: {output_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _touch_lora_final(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage1_manifest.json").write_text("{}", encoding="utf-8")
    (output_dir / "pytorch_lora_weights.safetensors").write_bytes(b"lora")


def _touch_unet_final(output_dir: Path) -> None:
    unet_dir = output_dir / "unet"
    unet_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage1_manifest.json").write_text("{}", encoding="utf-8")
    (unet_dir / "diffusion_pytorch_model.safetensors").write_bytes(b"unet")


def _touch_stage2_final(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "regiondiff_unet.safetensors").write_bytes(b"regiondiff")
    (output_dir / "stage2_layout_manifest.json").write_text("{}", encoding="utf-8")


def _command_module(command: list[str]) -> str:
    return command[command.index("-m") + 1]


def _has_resume_latest(command: list[str]) -> bool:
    return command[-2:] == ["--resume_from_checkpoint", "latest"]


def test_fresh_run_calls_stage1_then_stage2_without_resume(tmp_path, monkeypatch) -> None:
    stage1_dir = tmp_path / "stage1_lora"
    stage2_dir = tmp_path / "stage2"
    stage1_config = tmp_path / "stage1.yaml"
    stage2_config = tmp_path / "stage2.yaml"
    _write_lora_stage1_config(stage1_config, stage1_dir)
    _write_stage2_config(stage2_config, stage2_dir, stage1_dir)
    calls: list[list[str]] = []

    def fake_run(command, cwd, check):
        calls.append(list(command))
        if _command_module(list(command)) == "src.cli.train_sd":
            _touch_lora_final(stage1_dir)
        else:
            _touch_stage2_final(stage2_dir)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(chain.subprocess, "run", fake_run)

    assert chain.main([
        "--project-root",
        str(tmp_path),
        "--stage1-config",
        str(stage1_config),
        "--stage2-config",
        str(stage2_config),
        "--mixed-precision",
        "fp16",
    ]) == 0

    assert [_command_module(call) for call in calls] == [
        "src.cli.train_sd",
        "src.cli.train_sd_layout",
    ]
    assert not any(_has_resume_latest(call) for call in calls)
    assert all(["--mixed_precision", "fp16"] == call[-2:] for call in calls)
    assert (stage2_dir / chain.CHAIN_MARKER_NAME).is_file()


def test_interrupted_stage1_rerun_resumes_latest_stage1_checkpoint(tmp_path, monkeypatch) -> None:
    stage1_dir = tmp_path / "stage1_lora"
    stage2_dir = tmp_path / "stage2"
    (stage1_dir / "checkpoint-7").mkdir(parents=True)
    stage1_config = tmp_path / "stage1.yaml"
    stage2_config = tmp_path / "stage2.yaml"
    _write_lora_stage1_config(stage1_config, stage1_dir)
    _write_stage2_config(stage2_config, stage2_dir, stage1_dir)
    calls: list[list[str]] = []

    def fake_run(command, cwd, check):
        calls.append(list(command))
        if _command_module(list(command)) == "src.cli.train_sd":
            _touch_lora_final(stage1_dir)
        else:
            _touch_stage2_final(stage2_dir)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(chain.subprocess, "run", fake_run)

    chain.main(["--project-root", str(tmp_path), "--stage1-config", str(stage1_config), "--stage2-config", str(stage2_config)])

    assert _command_module(calls[0]) == "src.cli.train_sd"
    assert _has_resume_latest(calls[0])
    assert not _has_resume_latest(calls[1])


def test_completed_stage1_missing_stage2_runs_only_stage2(tmp_path, monkeypatch) -> None:
    stage1_dir = tmp_path / "stage1_lora"
    stage2_dir = tmp_path / "stage2"
    _touch_lora_final(stage1_dir)
    stage1_config = tmp_path / "stage1.yaml"
    stage2_config = tmp_path / "stage2.yaml"
    _write_lora_stage1_config(stage1_config, stage1_dir)
    _write_stage2_config(stage2_config, stage2_dir, stage1_dir)
    calls: list[list[str]] = []

    def fake_run(command, cwd, check):
        calls.append(list(command))
        _touch_stage2_final(stage2_dir)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(chain.subprocess, "run", fake_run)

    chain.main(["--project-root", str(tmp_path), "--stage1-config", str(stage1_config), "--stage2-config", str(stage2_config)])

    assert [_command_module(call) for call in calls] == ["src.cli.train_sd_layout"]
    assert not _has_resume_latest(calls[0])


def test_interrupted_stage2_rerun_resumes_latest_stage2_checkpoint(tmp_path, monkeypatch) -> None:
    stage1_dir = tmp_path / "stage1_lora"
    stage2_dir = tmp_path / "stage2"
    _touch_lora_final(stage1_dir)
    (stage2_dir / "checkpoint-12").mkdir(parents=True)
    stage1_config = tmp_path / "stage1.yaml"
    stage2_config = tmp_path / "stage2.yaml"
    _write_lora_stage1_config(stage1_config, stage1_dir)
    _write_stage2_config(stage2_config, stage2_dir, stage1_dir)
    calls: list[list[str]] = []

    def fake_run(command, cwd, check):
        calls.append(list(command))
        _touch_stage2_final(stage2_dir)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(chain.subprocess, "run", fake_run)

    chain.main(["--project-root", str(tmp_path), "--stage1-config", str(stage1_config), "--stage2-config", str(stage2_config)])

    assert [_command_module(call) for call in calls] == ["src.cli.train_sd_layout"]
    assert _has_resume_latest(calls[0])


def test_completed_chain_skips_both_stages(tmp_path, monkeypatch) -> None:
    stage1_dir = tmp_path / "stage1_lora"
    stage2_dir = tmp_path / "stage2"
    _touch_lora_final(stage1_dir)
    _touch_stage2_final(stage2_dir)
    stage1_config = tmp_path / "stage1.yaml"
    stage2_config = tmp_path / "stage2.yaml"
    _write_lora_stage1_config(stage1_config, stage1_dir)
    _write_stage2_config(stage2_config, stage2_dir, stage1_dir)
    calls: list[list[str]] = []
    monkeypatch.setattr(chain.subprocess, "run", lambda command, cwd, check: calls.append(list(command)))

    chain.main(["--project-root", str(tmp_path), "--stage1-config", str(stage1_config), "--stage2-config", str(stage2_config)])

    assert calls == []


def test_full_unet_stage1_completion_uses_unet_export(tmp_path, monkeypatch) -> None:
    stage1_dir = tmp_path / "stage1_unet"
    stage2_dir = tmp_path / "stage2"
    _touch_unet_final(stage1_dir)
    stage1_config = tmp_path / "stage1.yaml"
    stage2_config = tmp_path / "stage2.yaml"
    _write_unet_stage1_config(stage1_config, stage1_dir)
    _write_stage2_config(stage2_config, stage2_dir, stage1_dir)
    calls: list[list[str]] = []

    def fake_run(command, cwd, check):
        calls.append(list(command))
        _touch_stage2_final(stage2_dir)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(chain.subprocess, "run", fake_run)

    chain.main(["--project-root", str(tmp_path), "--stage1-config", str(stage1_config), "--stage2-config", str(stage2_config)])

    assert [_command_module(call) for call in calls] == ["src.cli.train_sd_layout"]

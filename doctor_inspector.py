#!/usr/bin/env python3
"""Smoke doctor for training, generation, and analysis launch wiring.

The doctor intentionally keeps its outputs inside ``artifacts/doctor_inspector``
and its report under ``logs/``.  By default it removes the artifact sandbox at
the end of the run.

This is a wiring and smoke-config inspector, not a long training benchmark:
heavy model loading is avoided unless ``--full-model-runs`` is explicitly used.
"""

from __future__ import annotations

import argparse
import copy
import contextlib
import importlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

try:
    import yaml
except ModuleNotFoundError as exc:
    print(
        "doctor_inspector.py requires PyYAML in the Python interpreter that is "
        "running this script.\n"
        f"Current interpreter: {sys.executable}\n"
        "Install it for this interpreter with:\n"
        f"  {sys.executable} -m pip install pyyaml\n",
        file=sys.stderr,
    )
    raise SystemExit(2) from exc


REPO = Path(__file__).resolve().parent
ARTIFACT_ROOT = REPO / "artifacts" / "doctor_inspector"
LOG_ROOT = REPO / "logs"


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str = ""
    command: str | None = None
    seconds: float = 0.0


@dataclass
class DoctorContext:
    artifact_root: Path
    log_path: Path
    keep_artifacts: bool
    full_model_runs: bool
    results: list[CheckResult] = field(default_factory=list)

    @property
    def smoke_data(self) -> Path:
        return self.artifact_root / "data"

    @property
    def smoke_configs(self) -> Path:
        return self.artifact_root / "configs"

    @property
    def smoke_outputs(self) -> Path:
        return self.artifact_root / "outputs"


class Tee:
    """Write terminal output to stdout and a log file."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = self.path.open("w", encoding="utf-8")

    def write(self, text: str) -> None:
        sys.__stdout__.write(text)
        self.handle.write(text)
        self.handle.flush()

    def flush(self) -> None:
        sys.__stdout__.flush()
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


def _now() -> float:
    import time

    return time.perf_counter()


def _rel(path: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO))
    except Exception:
        return str(path)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data if isinstance(data, dict) else {}


def _write_yaml(path: Path, data: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return path


def _deep_set(data: dict[str, Any], dotted: str, value: Any) -> None:
    cur = data
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = cur.get(part)
        if not isinstance(child, dict):
            child = {}
            cur[part] = child
        cur = child
    cur[parts[-1]] = value


def _deep_update(data: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            _deep_update(data[key], value)
        else:
            data[key] = value
    return data


def _run_result(ctx: DoctorContext, name: str, fn: Callable[[], str | None], command: str | None = None) -> None:
    start = _now()
    try:
        detail = fn() or ""
        result = CheckResult(name=name, status="PASS", detail=detail, command=command, seconds=_now() - start)
    except Exception as exc:
        tb = traceback.format_exc(limit=12)
        result = CheckResult(
            name=name,
            status="FAIL",
            detail=f"{type(exc).__name__}: {exc}\n{tb}",
            command=command,
            seconds=_now() - start,
        )
    ctx.results.append(result)
    symbol = "PASS" if result.status == "PASS" else "FAIL"
    print(f"[{symbol}] {name} ({result.seconds:.2f}s)")
    if result.detail:
        for line in result.detail.strip().splitlines()[:16]:
            print(f"    {line}")
        if len(result.detail.strip().splitlines()) > 16:
            print("    ...")


def _skip_result(ctx: DoctorContext, name: str, detail: str) -> None:
    result = CheckResult(name=name, status="SKIP", detail=detail, seconds=0.0)
    ctx.results.append(result)
    print(f"[SKIP] {name}")
    for line in detail.strip().splitlines()[:16]:
        print(f"    {line}")


@contextlib.contextmanager
def patched_attr(module: Any, attr: str, value: Any):
    sentinel = object()
    old = getattr(module, attr, sentinel)
    setattr(module, attr, value)
    try:
        yield
    finally:
        if old is sentinel:
            delattr(module, attr)
        else:
            setattr(module, attr, old)


@contextlib.contextmanager
def patched_argv(argv: list[str]):
    old = sys.argv[:]
    sys.argv = argv[:]
    try:
        yield
    finally:
        sys.argv = old


@contextlib.contextmanager
def patched_env(updates: dict[str, str | None]):
    old = {key: os.environ.get(key) for key in updates}
    for key, value in updates.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def discover_configs() -> dict[str, list[Path]]:
    return {
        "fm_regiondiff": sorted((REPO / "configs/fm/train/presets").glob("regiondiff_*.yaml")),
        "fm_stay_layout": sorted((REPO / "configs/fm/train/presets").glob("stay_layout_latent_flir_*.yaml")),
        "fm_text_cfg": sorted((REPO / "configs/fm/train/presets").glob("text_cfg*.yaml")),
        "fm_uncond": sorted((REPO / "configs/fm/train/presets").glob("uncond*.yaml")),
        "sd": sorted((REPO / "configs/sd/train/presets").glob("*.yaml")),
        "sd_layout": sorted((REPO / "configs/sd_layout/train/presets").glob("*.yaml")),
        "sd_uncond": sorted((REPO / "configs/sd_uncond/train/presets").glob("*.yaml")),
        "vae": sorted((REPO / "configs/vae/train/presets").glob("*.yaml")),
        "generation": sorted((REPO / "configs/fm/generate/presets").glob("*.yaml"))
        + sorted((REPO / "configs/fm/sample/presets").glob("*.yaml")),
        "analysis": sorted((REPO / "configs/analysis/presets").glob("*.yaml")),
    }


def prepare_sandbox(ctx: DoctorContext) -> None:
    if ctx.artifact_root.exists():
        shutil.rmtree(ctx.artifact_root)
    ctx.smoke_data.mkdir(parents=True, exist_ok=True)
    ctx.smoke_configs.mkdir(parents=True, exist_ok=True)
    ctx.smoke_outputs.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(123)
    for split in ("train", "val", "generated_a", "generated_b"):
        split_dir = ctx.smoke_data / split
        split_dir.mkdir(parents=True, exist_ok=True)
        records = []
        images = []
        annotations = []
        for idx in range(4):
            fname = f"sample_{idx:05d}.npy"
            arr = rng.integers(0, 65535, size=(64, 64), dtype=np.uint16)
            np.save(split_dir / fname, arr)
            records.append({"file_name": fname, "text": f"thermal scene with person {idx}"})
            images.append({"id": idx + 1, "file_name": fname, "width": 64, "height": 64})
            annotations.append(
                {
                    "id": idx + 1,
                    "image_id": idx + 1,
                    "category_id": 1,
                    "bbox": [12.0, 10.0, 20.0, 24.0],
                    "area": 480.0,
                    "iscrowd": 0,
                }
            )
        with (split_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record) + "\n")
        with (split_dir / "captions.json").open("w", encoding="utf-8") as handle:
            json.dump({Path(r["file_name"]).stem: r["text"] for r in records}, handle, indent=2)
        with (split_dir / "annotations.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": [{"id": 1, "name": "person"}],
                },
                handle,
                indent=2,
            )

    # Generated-dir analysis expects subfolders containing npy files.
    gen_root = ctx.smoke_data / "generated"
    gen_root.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ctx.smoke_data / "generated_a", gen_root / "generator_a")
    shutil.copytree(ctx.smoke_data / "generated_b", gen_root / "generator_b")


def _base_fm_overrides(ctx: DoctorContext, name: str) -> dict[str, Any]:
    output = ctx.smoke_outputs / name
    return {
        "data": {
            "dataset_id": None,
            "train_dir": str(ctx.smoke_data / "train"),
            "val_dir": str(ctx.smoke_data / "val"),
            "annotations_path": str(ctx.smoke_data / "train" / "annotations.json"),
            "image_size": 32,
            "batch_size": 1,
            "num_workers": 0,
            "max_train_samples": 2,
            "max_val_samples": 2,
        },
        "training": {
            "epochs": 1,
            "save_every_n_epochs": 99,
            "eval_every": 0,
            "max_grad_norm": 1.0,
        },
        "sampling": {
            "sample_every": 0,
            "sample_every_steps": 0,
            "early_sanity_sample_epoch": 0,
            "sample_steps": 1,
            "sample_batch_size": 1,
        },
        "distillation": {
            "enabled": False,
            "teacher_checkpoint": None,
        },
        "optimizer": {"lr": 1e-5},
        "scheduler": {"name": "none", "warmup_ratio": 0.0},
        "ema": {"enabled": False, "start_step": 9999},
        "precision": {"mixed_precision": "no"},
        "output": {
            "model_dir": str(output / "checkpoints"),
            "log_dir": str(output / "logs"),
            "debug_dir": str(output / "debug"),
            "resume": None,
        },
        "device": "cpu",
    }


def make_smoke_config(ctx: DoctorContext, family: str, config_path: Path) -> Path:
    data = copy.deepcopy(_load_yaml(config_path))
    stem = config_path.stem
    out = ctx.smoke_configs / family / f"{stem}.smoke.yaml"

    if family in {"fm_regiondiff", "fm_stay_layout", "fm_uncond"}:
        _deep_update(data, _base_fm_overrides(ctx, f"{family}/{stem}"))
    elif family == "fm_text_cfg":
        _deep_update(
            data,
            {
                "data": {
                    "train_dir": str(ctx.smoke_data / "train"),
                    "val_dir": str(ctx.smoke_data / "val"),
                    "annotations_path": str(ctx.smoke_data / "train" / "annotations.json"),
                    "batch_size": 1,
                    "num_workers": 0,
                },
                "training": {
                    "epochs": 1,
                    "save_every_n_epochs": 99,
                    "eval_every": 0,
                },
                "sampling": {"sample_every": 0, "sample_steps": 1, "sample_batch_size": 1},
                "attention_vis": {"enabled": False},
                "output": {
                    "model_dir": str(ctx.smoke_outputs / family / stem / "checkpoints"),
                    "log_dir": str(ctx.smoke_outputs / family / stem / "logs"),
                    "resume": None,
                },
                "device": "cpu",
            },
        )
    elif family == "sd":
        _deep_update(
            data,
            {
                "dataset_id": None,
                "dataset_name": None,
                "dataset_config_name": None,
                "train_data_dir": str(ctx.smoke_data / "train"),
                "layout_annotations_path": str(ctx.smoke_data / "train" / "annotations.json"),
                "resolution": 32,
                "train_batch_size": 1,
                "num_train_epochs": 1,
                "max_train_steps": 1,
                "max_train_samples": 2,
                "dataloader_num_workers": 0,
                "validation_epochs": 99,
                "num_validation_images": 1,
                "validation_num_inference_steps": 1,
                "checkpointing_epochs": 99,
                "report_to": "tensorboard",
                "mixed_precision": "no",
                "output_dir": str(ctx.smoke_outputs / family / stem),
                "push_to_hub": False,
            },
        )
    elif family == "sd_layout":
        _deep_update(
            data,
            {
                "data": {
                    "dataset_id": "v18",
                    "train_data_dir": str(ctx.smoke_data / "train"),
                    "train_annotations": str(ctx.smoke_data / "train" / "annotations.json"),
                    "val_data_dir": str(ctx.smoke_data / "val"),
                    "val_annotations": str(ctx.smoke_data / "val" / "annotations.json"),
                    "resolution": 32,
                    "batch_size": 1,
                    "num_workers": 0,
                    "max_train_samples": 2,
                    "max_val_samples": 2,
                },
                "training": {
                    "num_train_epochs": 1,
                    "max_train_steps": 1,
                    "checkpointing_steps": 999,
                    "mixed_precision": "no",
                    "report_to": "tensorboard",
                },
                "validation": {
                    "validation_epochs": 99,
                    "num_validation_images": 1,
                    "validation_num_inference_steps": 1,
                },
                "output": {"output_dir": str(ctx.smoke_outputs / family / stem)},
            },
        )
    elif family == "sd_uncond":
        _deep_update(data, _base_fm_overrides(ctx, f"{family}/{stem}"))
        _deep_update(
            data,
            {
                "diffusion": {"num_train_timesteps": 4},
                "sampling": {"sample_every": 0, "sample_steps": 1, "sample_batch_size": 1},
            },
        )
    elif family == "vae":
        _deep_update(
            data,
            {
                "train_dir": str(ctx.smoke_data / "train"),
                "val_dir": str(ctx.smoke_data / "val"),
                "image_size": 32,
                "batch_size": 1,
                "num_workers": 0,
                "pin_memory": False,
                "epochs": 1,
                "patience": 1,
                "save_every_n_epochs": 99,
                "mixed_precision": "no",
                "model_dir": str(ctx.smoke_outputs / family / stem / "checkpoints"),
                "log_dir": str(ctx.smoke_outputs / family / stem / "logs"),
                "device": "cpu",
            },
        )
    elif family == "generation":
        _deep_update(
            data,
            {
                "metadata": str(ctx.smoke_data / "train" / "metadata.jsonl"),
                "max_samples": 1,
                "output_dir": str(ctx.smoke_outputs / family / stem),
                "device": "cpu",
                "fm_steps": 1,
                "fm_batch_size": 1,
                "sd_steps": 1,
                "precision": "fp32",
            },
        )
        if data.get("mode") == "sd15" and not data.get("stage1_dir") and not data.get("lora_dir"):
            data["lora_dir"] = str(ctx.smoke_outputs / "missing_lora_for_parse_only")
        if data.get("mode") == "fm" and not data.get("fm_pipeline_dir"):
            data["fm_pipeline_dir"] = str(ctx.smoke_outputs / "missing_fm_for_parse_only")
    elif family == "analysis":
        _deep_update(
            data,
            {
                "real_dir": str(ctx.smoke_data / "train"),
                "generated_dir": str(ctx.smoke_data / "generated"),
                "output_dir": str(ctx.smoke_outputs / family / stem),
                "max_samples": 2,
                "metrics_max_samples": 2,
                "metrics_pca_dim": 0,
                "skip_kl": True,
                "device": "cpu",
            },
        )
    else:
        raise ValueError(f"Unknown config family: {family}")

    return _write_yaml(out, data)


def smoke_fm_entrypoint(config_path: Path) -> str:
    cli = importlib.import_module("src.cli.train")

    def fake_run(cfg):
        assert cfg.data.batch_size == 1
        assert cfg.device == "cpu"
        Path(cfg.output.model_dir).mkdir(parents=True, exist_ok=True)
        return None

    with patched_attr(cli, "run_training", fake_run):
        cli.main(["--config", str(config_path)])
    return "parsed src.cli.train and invoked patched run_training"


def smoke_text_fm_entrypoint(config_path: Path) -> str:
    cli = importlib.import_module("src.cli.train_text_fm")

    def fake_run(cfg):
        assert cfg.data.batch_size == 1
        assert cfg.device == "cpu"
        Path(cfg.output.model_dir).mkdir(parents=True, exist_ok=True)
        return None

    with patched_attr(cli, "run_training", fake_run):
        cli.main(["--config", str(config_path)])
    return "parsed src.cli.train_text_fm and invoked patched run_training"


def smoke_sd_config(config_path: Path) -> str:
    parser_mod = importlib.import_module("src.algorithms.stable_diffusion.config")
    cfg = parser_mod.parse_args(["--config", str(config_path)])
    assert cfg.train_batch_size == 1
    assert cfg.max_train_steps == 1
    assert cfg.train_data_dir is not None
    return f"parsed SD config: baseline={cfg.baseline_mode}, output={_rel(cfg.output_dir)}"


def smoke_sd_layout_config(config_path: Path) -> str:
    parser_mod = importlib.import_module("src.core.configs.sd_layout_config")
    cfg = parser_mod.parse_args(["--config", str(config_path)])
    assert cfg.data.batch_size == 1
    assert cfg.training.max_train_steps == 1
    return f"parsed SD-layout config: mode={cfg.training.train_mode}, output={_rel(cfg.output.output_dir)}"


def smoke_sd_uncond_entrypoint(config_path: Path) -> str:
    cli = importlib.import_module("src.cli.train_sd_uncond")

    def fake_run(cfg):
        assert cfg.data.batch_size == 1
        assert cfg.device == "cpu"
        Path(cfg.output.model_dir).mkdir(parents=True, exist_ok=True)
        return None

    with patched_attr(cli, "run_training", fake_run):
        cli.main(["--config", str(config_path)])
    return "parsed src.cli.train_sd_uncond and invoked patched run_training"


def smoke_vae_config(config_path: Path) -> str:
    cli = importlib.import_module("src.cli.train_vae")
    with patched_argv(["train_vae.py", "--config", str(config_path)]):
        args = cli.parse_args()
    assert args.batch_size == 1
    assert args.device == "cpu"
    return f"parsed VAE config: model_dir={_rel(args.model_dir)}"


def smoke_generate_config(config_path: Path) -> str:
    cli = importlib.import_module("src.cli.generate")
    with patched_argv(["generate_datasets.py", "--config", str(config_path)]):
        args = cli.parse_args()
    assert args.max_samples == 1
    assert args.device == "cpu"
    if args.mode == "sd15":
        assert args.stage1_dir or args.lora_dir
    if args.mode == "fm":
        assert args.fm_pipeline_dir
    return f"parsed generate config: mode={args.mode}, output={_rel(args.output_dir)}"


def smoke_analysis_config(config_path: Path) -> str:
    try:
        mod = importlib.import_module("scripts.standalone.analyze_distribution_shift")
    except ModuleNotFoundError as exc:
        if exc.name and exc.name.startswith("scipy"):
            return (
                "SKIP optional analysis import: scipy.spatial is unavailable in this "
                "environment; py_compile still covers analysis syntax"
            )
        raise
    with patched_argv(["analyze_distribution_shift.py", "--config", str(config_path)]):
        args = mod.parse_args()
    assert args.real_dir
    assert args.generated_dir
    assert args.max_samples == 2
    return f"parsed analysis config: output={_rel(args.output_dir)}"


class FakeTokenizer:
    model_max_length = 8

    def __call__(self, text, *, max_length, padding, truncation, return_tensors):
        import torch

        return type("_TokenResult", (), {"input_ids": torch.zeros(1, max_length, dtype=torch.long)})()


def smoke_data_loaders(ctx: DoctorContext) -> str:
    import torch

    from src.algorithms.stable_diffusion.data import create_dataloader
    from src.algorithms.stable_diffusion.layout_data import StableDiffusionLayoutDataset, collate_sd_layout_batch
    from src.core.data import collate_layout_batch
    from src.core.data.annotation_dataset import AnnotationFMDataset
    from src.core.data.datasets import AnnotationLayoutDataset, NPYImageDataset
    from torch.utils.data import DataLoader

    train_dir = str(ctx.smoke_data / "train")
    ann_path = str(ctx.smoke_data / "train" / "annotations.json")

    npy = NPYImageDataset(train_dir)
    assert tuple(npy[0].shape) == (1, 64, 64)

    layout = AnnotationLayoutDataset(train_dir, ann_path, image_size=32)
    layout_batch = next(iter(DataLoader(layout, batch_size=1, collate_fn=collate_layout_batch)))
    assert layout_batch["pixel_values"].shape[-2:] == (32, 32)

    text = AnnotationFMDataset(train_dir, ann_path, text_mode=True, resize_target=32)
    text_sample = text[0]
    assert "text" in text_sample and isinstance(text_sample["pixel_values"], torch.Tensor)

    sd_loader, _ = create_dataloader(
        dataset_id=None,
        dataset_name=None,
        dataset_config_name=None,
        train_data_dir=train_dir,
        train_split="train",
        cache_dir=None,
        tokenizer=FakeTokenizer(),
        resolution=32,
        center_crop=False,
        random_flip=False,
        interpolation_mode="lanczos",
        image_column="image",
        caption_column="text",
        batch_size=1,
        num_workers=0,
        max_train_samples=1,
        seed=123,
        use_ir_preprocessing=True,
        prompt_text="thermal image",
    )
    sd_batch = next(iter(sd_loader))
    assert sd_batch["pixel_values"].shape[-2:] == (32, 32)

    sd_layout = StableDiffusionLayoutDataset(
        root_dir=train_dir,
        annotations_path=ann_path,
        tokenizer=FakeTokenizer(),
        resolution=32,
        normalization_mode="raw_uint16_percentile",
        prompt_mode="class_list",
        constant_prompt="thermal image",
        thermal_scene_suffix="in thermal scene.",
        use_captions_if_available=False,
        max_samples=1,
    )
    sd_layout_batch = collate_sd_layout_batch([sd_layout[0]])
    assert sd_layout_batch["input_ids"].shape == (1, FakeTokenizer.model_max_length)
    return "loaded NPY, FM layout, text-FM, SD, and SD-layout smoke batches"


def check_python_files(ctx: DoctorContext, files: Iterable[Path]) -> None:
    for path in files:
        _run_result(
            ctx,
            f"py_compile {_rel(path)}",
            lambda path=path: (py_compile.compile(str(path), doraise=True), "compiled")[1],
        )


def check_shell_scripts(ctx: DoctorContext, files: Iterable[Path]) -> None:
    for path in files:
        cmd = ["bash", "-n", str(path)]

        def run(path=path, cmd=cmd):
            proc = subprocess.run(cmd, cwd=REPO, text=True, capture_output=True, timeout=30)
            if proc.returncode != 0:
                raise RuntimeError((proc.stderr or proc.stdout).strip())
            return "bash -n clean"

        _run_result(ctx, f"bash -n {_rel(path)}", run, command=" ".join(cmd))


def run_config_matrix(ctx: DoctorContext) -> None:
    configs = discover_configs()
    for family, paths in configs.items():
        print(f"\n=== {family}: {len(paths)} config(s) ===")
        for original in paths:
            smoke_config = make_smoke_config(ctx, family, original)
            if family in {"fm_regiondiff", "fm_stay_layout", "fm_uncond"}:
                fn = lambda smoke_config=smoke_config: smoke_fm_entrypoint(smoke_config)
            elif family == "fm_text_cfg":
                fn = lambda smoke_config=smoke_config: smoke_text_fm_entrypoint(smoke_config)
            elif family == "sd":
                fn = lambda smoke_config=smoke_config: smoke_sd_config(smoke_config)
            elif family == "sd_layout":
                fn = lambda smoke_config=smoke_config: smoke_sd_layout_config(smoke_config)
            elif family == "sd_uncond":
                fn = lambda smoke_config=smoke_config: smoke_sd_uncond_entrypoint(smoke_config)
            elif family == "vae":
                fn = lambda smoke_config=smoke_config: smoke_vae_config(smoke_config)
            elif family == "generation":
                fn = lambda smoke_config=smoke_config: smoke_generate_config(smoke_config)
            elif family == "analysis":
                fn = lambda smoke_config=smoke_config: smoke_analysis_config(smoke_config)
            else:
                fn = lambda: "no checker"
            _run_result(ctx, f"{family} smoke {_rel(original)}", fn)


def run_launcher_checks(ctx: DoctorContext) -> None:
    print("\n=== launcher syntax ===")
    shell_files = sorted((REPO / "scripts").glob("*.sh"))
    shell_files += sorted((REPO / "scripts/train").glob("*.sh"))
    shell_files += sorted((REPO / "scripts/generate").glob("*.sh"))
    shell_files += sorted((REPO / "scripts/analyze").glob("*.sh"))
    check_shell_scripts(ctx, shell_files)

    print("\n=== active Python entrypoint syntax ===")
    py_files = [
        REPO / "train_sfm.py",
        REPO / "train_sd.py",
        REPO / "train_sd_uncond.py",
        REPO / "train_sd_layout.py",
        REPO / "train_vae.py",
        REPO / "generate_datasets.py",
        REPO / "scripts/standalone/analyze_distribution_shift.py",
        REPO / "src/cli/train.py",
        REPO / "src/cli/train_text_fm.py",
        REPO / "src/cli/train_sd.py",
        REPO / "src/cli/train_sd_layout.py",
        REPO / "src/cli/train_sd_uncond.py",
        REPO / "src/cli/train_vae.py",
        REPO / "src/cli/generate.py",
        REPO / "src/cli/sample.py",
        REPO / "src/cli/sample_text_fm.py",
    ]
    check_python_files(ctx, [path for path in py_files if path.exists()])


def _write_tiny_model_configs(ctx: DoctorContext) -> dict[str, Path]:
    """Write tiny real model configs used only by full doctor runs."""
    model_dir = ctx.artifact_root / "model_configs"
    model_dir.mkdir(parents=True, exist_ok=True)
    configs = {
        "latent_unet": {
            "sample_size": 16,
            "in_channels": 4,
            "out_channels": 4,
            "layers_per_block": 1,
            "block_out_channels": [16, 32],
            "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
            "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
            "norm_num_groups": 8,
        },
        "pixel_unet": {
            "sample_size": 32,
            "in_channels": 1,
            "out_channels": 1,
            "layers_per_block": 1,
            "block_out_channels": [16, 32],
            "down_block_types": ["DownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "UpBlock2D"],
            "norm_num_groups": 8,
        },
        "text_unet": {
            "sample_size": 16,
            "in_channels": 4,
            "out_channels": 4,
            "layers_per_block": 1,
            "block_out_channels": [16, 32],
            "down_block_types": ["CrossAttnDownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D"],
            "mid_block_type": "UNetMidBlock2DCrossAttn",
            "cross_attention_dim": 16,
            "attention_head_dim": 4,
            "norm_num_groups": 8,
        },
        "vae": {
            "spatial_dims": 2,
            "in_channels": 1,
            "out_channels": 1,
            "latent_channels": 4,
            "num_channels": [8, 16],
            "norm_num_groups": 8,
            "num_res_blocks": 1,
            "attention_levels": [False, False],
        },
    }
    paths: dict[str, Path] = {}
    for name, payload in configs.items():
        path = model_dir / f"{name}.json"
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        paths[name] = path
    return paths


def _full_base_overrides(
    ctx: DoctorContext,
    name: str,
    model_paths: dict[str, Path],
    *,
    layout: bool = False,
    regiondiff: bool = False,
    stay: bool = False,
) -> dict[str, Any]:
    updates = _base_fm_overrides(ctx, f"full/{name}")
    _deep_update(
        updates,
        {
            "data": {
                "image_size": 32,
                "batch_size": 1,
                "num_workers": 0,
                "max_train_samples": 1,
                "max_val_samples": 1,
            },
            "model": {
                "unet_config": str(model_paths["latent_unet"]),
                "vae_config": str(model_paths["vae"]),
                "vae_weights": None,
                "vae_pretrained_model_name_or_path": None,
                "vae_pretrained_subfolder": None,
                "pretrained_unet_path": None,
            },
            "training": {
                "epochs": 1,
                "save_every_n_epochs": 1,
                "eval_every": 0,
                "lr": 1.0e-4,
            },
            "sampling": {
                "sample_every": 0,
                "sample_every_steps": 0,
                "early_sanity_sample_epoch": 0,
                "sample_steps": 1,
                "sample_batch_size": 1,
                "fixed_validation_examples": 0,
                "save_debug_images": False,
            },
            "optimizer": {"name": "adamw", "lr": 1.0e-4, "weight_decay": 0.0},
            "scheduler": {"name": "none", "warmup_ratio": 0.0},
            "ema": {"enabled": False},
            "precision": {"mixed_precision": "no"},
            "device": "cpu",
        },
    )
    if not layout:
        updates["data"]["annotations_path"] = None
        updates["layout_conditioning"] = {"enabled": False}
        updates["trainer_name"] = "default_fm"
    elif regiondiff:
        _deep_update(
            updates,
            {
                "layout_conditioning": {
                    "enabled": True,
                    "variant": "regiondiff_v1",
                    "active_region_resolutions": [16, 8],
                    "layout_token_dim": 16,
                    "bbox_fourier_dim": 4,
                    "same_class_position_slots": 8,
                    "use_background_token": True,
                    "train_mode": "adapters_only",
                    "adapter_learning_rate": 1.0e-4,
                    "backbone_learning_rate": 1.0e-4,
                    "area_loss_enabled": False,
                },
                "trainer_name": "default_fm",
            },
        )
    elif stay:
        _deep_update(
            updates,
            {
                "model": {"unet_config": str(model_paths["latent_unet"])},
                "layout_conditioning": {
                    "enabled": True,
                    "variant": "stay_v2",
                    "class_embed_dim": 8,
                    "bbox_embed_dim": 8,
                    "object_embed_dim": 16,
                    "use_style_latent": True,
                    "style_latent_dim": 4,
                    "mask_resolution": 8,
                    "mask_hidden_channels": 8,
                    "edge_dilation": 0,
                    "log_internal_maps": False,
                },
                "trainer_name": "layout_fm",
            },
        )
    return updates


def _full_sd_uncond_overrides(
    ctx: DoctorContext,
    name: str,
    model_paths: dict[str, Path],
) -> dict[str, Any]:
    updates = _full_base_overrides(ctx, name, model_paths)
    updates["trainer_name"] = "sd_uncond"
    updates["sampler_name"] = "sd_uncond"
    return updates


def _write_full_config(ctx: DoctorContext, name: str, base: Path, updates: dict[str, Any]) -> Path:
    data = copy.deepcopy(_load_yaml(base))
    _deep_update(data, updates)
    return _write_yaml(ctx.smoke_configs / "full_model_runs" / f"{name}.yaml", data)


def _actual_fm_training(ctx: DoctorContext, name: str, config_path: Path) -> str:
    cli = importlib.import_module("src.cli.train")
    cli.main(["--config", str(config_path)])
    data = _load_yaml(config_path)
    model_dir = Path(data["output"]["model_dir"])
    unet_dir = model_dir / "UNET"
    weights = sorted(unet_dir.glob("unet_fm_epoch_*.pt"))
    if not weights:
        raise FileNotFoundError(f"No FM epoch weights produced in {unet_dir}")
    return f"actual {name} training produced {_rel(weights[-1])}"


def _actual_sd_uncond_training(ctx: DoctorContext, config_path: Path) -> str:
    cli = importlib.import_module("src.cli.train_sd_uncond")
    cli.main(["--config", str(config_path)])
    data = _load_yaml(config_path)
    model_dir = Path(data["output"]["model_dir"])
    weights = sorted((model_dir / "UNET").glob("unet_sd_uncond_epoch_*.pt"))
    if not weights:
        raise FileNotFoundError(f"No unconditional SD epoch weights produced in {model_dir / 'UNET'}")
    return f"actual unconditional SD training produced {_rel(weights[-1])}"


def _write_tiny_sd_pipeline(ctx: DoctorContext) -> Path:
    from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
    from transformers import CLIPTextConfig, CLIPTextModel, CLIPTokenizer

    pipeline_dir = ctx.smoke_outputs / "full" / "tiny_sd_pipeline"
    if (pipeline_dir / "model_index.json").is_file():
        return pipeline_dir

    tokenizer_src = ctx.artifact_root / "model_configs" / "tiny_clip_tokenizer"
    tokenizer_src.mkdir(parents=True, exist_ok=True)
    vocab = {
        "<|startoftext|>": 0,
        "<|endoftext|>": 1,
        "thermal</w>": 2,
        "image</w>": 3,
        "scene</w>": 4,
        "with</w>": 5,
        "person</w>": 6,
        "an</w>": 7,
        "of</w>": 8,
        "in</w>": 9,
        ".</w>": 10,
    }
    (tokenizer_src / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")
    (tokenizer_src / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
    tokenizer = CLIPTokenizer(str(tokenizer_src / "vocab.json"), str(tokenizer_src / "merges.txt"))
    tokenizer.model_max_length = 16
    tokenizer.init_kwargs["model_max_length"] = 16

    text_encoder = CLIPTextModel(
        CLIPTextConfig(
            vocab_size=len(vocab),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            max_position_embeddings=16,
            bos_token_id=0,
            eos_token_id=1,
            pad_token_id=1,
        )
    )
    vae = AutoencoderKL(
        sample_size=32,
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(8, 16),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=8,
    )
    unet = UNet2DConditionModel(
        sample_size=16,
        in_channels=4,
        out_channels=4,
        down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
        block_out_channels=(16, 32),
        layers_per_block=1,
        cross_attention_dim=16,
        attention_head_dim=4,
        norm_num_groups=8,
    )
    scheduler = DDPMScheduler(
        num_train_timesteps=4,
        beta_schedule="scaled_linear",
        beta_start=0.00085,
        beta_end=0.012,
        prediction_type="epsilon",
        clip_sample=False,
    )
    pipeline = StableDiffusionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=scheduler,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipeline.save_pretrained(pipeline_dir, safe_serialization=True)
    return pipeline_dir


def _actual_sd_stage1_training(ctx: DoctorContext, tiny_sd_dir: Path) -> Path:
    output_dir = ctx.smoke_outputs / "full" / "sd_stage1_actual"
    config = _write_yaml(
        ctx.smoke_configs / "full_model_runs" / "sd_stage1_actual.yaml",
        {
            "pretrained_model_name_or_path": str(tiny_sd_dir),
            "baseline_mode": "sd_ir_unet",
            "unet_train_mode": "full",
            "dataset_id": None,
            "dataset_name": None,
            "dataset_config_name": None,
            "train_data_dir": str(ctx.smoke_data / "train"),
            "resolution": 32,
            "train_batch_size": 1,
            "num_train_epochs": 1,
            "max_train_steps": 1,
            "max_train_samples": 1,
            "dataloader_num_workers": 0,
            "learning_rate": 1.0e-4,
            "lr_scheduler": "constant",
            "lr_warmup_steps": 0,
            "checkpointing_epochs": 99,
            "validation_prompt": None,
            "num_validation_images": 1,
            "validation_num_inference_steps": 1,
            "freeze_vae": True,
            "freeze_text_encoder": True,
            "mixed_precision": "no",
            "report_to": "tensorboard",
            "output_dir": str(output_dir),
            "logging_dir": "logs",
            "push_to_hub": False,
            "use_ir_preprocessing": True,
            "prediction_type": "epsilon",
        },
    )
    cli = importlib.import_module("src.cli.train_sd")
    accelerator_cls = cli.Accelerator

    def cpu_accelerator(*args, **kwargs):
        kwargs["cpu"] = True
        return accelerator_cls(*args, **kwargs)

    with patched_attr(cli, "get_least_used_cuda_gpu", lambda **_: (None, "doctor cpu smoke")):
        with patched_attr(cli, "Accelerator", cpu_accelerator):
            with patched_env({"CUDA_VISIBLE_DEVICES": ""}):
                with patched_argv(["train_sd.py", "--config", str(config)]):
                    cli.main()
    manifest = output_dir / "stage1_manifest.json"
    unet_config = output_dir / "unet" / "config.json"
    if not manifest.exists() or not unet_config.exists():
        raise FileNotFoundError(f"Stage-1 SD export missing under {output_dir}")
    return output_dir


def _actual_sd_layout_training(ctx: DoctorContext, tiny_sd_dir: Path, stage1_dir: Path) -> str:
    output_dir = ctx.smoke_outputs / "full" / "sd_layout_actual"
    config = _write_yaml(
        ctx.smoke_configs / "full_model_runs" / "sd_layout_actual.yaml",
        {
            "data": {
                "dataset_id": "v18",
                "train_data_dir": str(ctx.smoke_data / "train"),
                "train_annotations": str(ctx.smoke_data / "train" / "annotations.json"),
                "val_data_dir": str(ctx.smoke_data / "val"),
                "val_annotations": str(ctx.smoke_data / "val" / "annotations.json"),
                "resolution": 32,
                "batch_size": 1,
                "num_workers": 0,
                "max_train_samples": 1,
                "max_val_samples": 1,
            },
            "stage1": {
                "pretrained_model_name_or_path": str(tiny_sd_dir),
                "stage1_dir": str(stage1_dir),
                "stage1_checkpoint": None,
                "revision": None,
                "variant": None,
            },
            "prompt": {"prompt_mode": "constant", "constant_prompt": "thermal image"},
            "region": {
                "active_region_resolutions": [16, 8],
                "layout_token_dim": 16,
                "bbox_fourier_dim": 4,
                "same_class_position_slots": 8,
                "use_background_token": True,
            },
            "area_loss": {"enabled": False},
            "training": {
                "train_mode": "adapters_only",
                "adapter_learning_rate": 1.0e-4,
                "backbone_learning_rate": 1.0e-4,
                "num_train_epochs": 1,
                "max_train_steps": 1,
                "gradient_accumulation_steps": 1,
                "lr_scheduler": "constant",
                "lr_warmup_steps": 0,
                "checkpointing_steps": 999,
                "mixed_precision": "no",
                "report_to": "tensorboard",
                "vae_encode_dtype": "float32",
            },
            "validation": {
                "validation_epochs": 99,
                "num_validation_images": 1,
                "validation_num_inference_steps": 1,
                "guidance_scale": 1.0,
            },
            "output": {"output_dir": str(output_dir), "logging_dir": "logs"},
        },
    )
    cli = importlib.import_module("src.cli.train_sd_layout")
    accelerator_cls = cli.Accelerator

    def cpu_accelerator(*args, **kwargs):
        kwargs["cpu"] = True
        return accelerator_cls(*args, **kwargs)

    with patched_attr(cli, "get_least_used_cuda_gpu", lambda **_: (None, "doctor cpu smoke")):
        with patched_attr(cli, "Accelerator", cpu_accelerator):
            with patched_env({"CUDA_VISIBLE_DEVICES": ""}):
                with patched_argv(["train_sd_layout.py", "--config", str(config)]):
                    cli.main()
    manifest = output_dir / "stage2_layout_manifest.json"
    weights = output_dir / "regiondiff_unet.safetensors"
    if not manifest.exists() or not weights.exists():
        raise FileNotFoundError(f"Stage-2 SD-layout export missing under {output_dir}")
    return f"actual SD-layout stage-2 training exported {_rel(manifest)}"


def _actual_vae_training(ctx: DoctorContext, model_paths: dict[str, Path]) -> str:
    config = _write_yaml(
        ctx.smoke_configs / "full_model_runs" / "vae_actual.yaml",
        {
            "train_dir": str(ctx.smoke_data / "train"),
            "val_dir": str(ctx.smoke_data / "val"),
            "image_size": 32,
            "normalization_mode": "raw_uint16_percentile",
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "epochs": 1,
            "vae_json": str(model_paths["vae"]),
            "model_dir": str(ctx.smoke_outputs / "full" / "vae_actual"),
            "log_dir": str(ctx.smoke_outputs / "full" / "vae_actual" / "logs"),
            "patience": 1,
            "save_every_n_epochs": 1,
            "mixed_precision": "no",
            "device": "cpu",
            "scheduler": "none",
            "ema_decay": 0.0,
        },
    )
    cli = importlib.import_module("src.cli.train_vae")
    with patched_argv(["train_vae.py", "--config", str(config)]):
        cli.main()
    weights = Path(ctx.smoke_outputs / "full" / "vae_actual" / "VAE" / "vae_best.pt")
    if not weights.exists():
        raise FileNotFoundError(f"No VAE best weights produced at {weights}")
    return f"actual VAE training produced {_rel(weights)}"


def _write_random_fm_pipeline(ctx: DoctorContext, model_paths: dict[str, Path]) -> Path:
    from src.models.fm_unet import build_fm_unet_from_config, load_unet_config, save_unet_config
    from src.models.vae import build_vae_from_config, load_vae_config, save_vae_config, save_vae_weights
    import torch

    pipeline_dir = ctx.smoke_outputs / "full" / "fm_generation_pipeline"
    unet_dir = pipeline_dir / "UNET"
    vae_dir = pipeline_dir / "VAE"
    unet_dir.mkdir(parents=True, exist_ok=True)
    vae_dir.mkdir(parents=True, exist_ok=True)

    unet_cfg = load_unet_config(str(model_paths["latent_unet"]))
    vae_cfg = load_vae_config(str(model_paths["vae"]))
    unet = build_fm_unet_from_config(unet_cfg, device="cpu")
    vae = build_vae_from_config(vae_cfg, device="cpu")
    save_unet_config(unet_cfg, str(unet_dir / "config.json"))
    save_vae_config(vae_cfg, str(vae_dir / "config.json"))
    torch.save(unet.state_dict(), unet_dir / "unet_fm_epoch_1.pt")
    save_vae_weights(vae, str(vae_dir / "vae_best.pt"))
    return pipeline_dir


def _actual_fm_generation(ctx: DoctorContext, model_paths: dict[str, Path]) -> str:
    pipeline_dir = _write_random_fm_pipeline(ctx, model_paths)
    config = _write_yaml(
        ctx.smoke_configs / "full_model_runs" / "fm_generation_actual.yaml",
        {
            "mode": "fm",
            "metadata": str(ctx.smoke_data / "train" / "metadata.jsonl"),
            "max_samples": 1,
            "output_dir": str(ctx.smoke_outputs / "full" / "fm_generation"),
            "fm_pipeline_dir": str(pipeline_dir),
            "fm_steps": 1,
            "fm_batch_size": 1,
            "fm_t_scale": 1000.0,
            "device": "cpu",
        },
    )
    cli = importlib.import_module("src.cli.generate")
    with patched_argv(["generate_datasets.py", "--config", str(config)]):
        cli.main()
    sample = ctx.smoke_outputs / "full" / "fm_generation" / "sample_00000.npy"
    if not sample.exists():
        raise FileNotFoundError(f"No generated FM sample at {sample}")
    return f"actual FM generation produced {_rel(sample)}"


def _actual_analysis_smoke(ctx: DoctorContext) -> str:
    mod = importlib.import_module("scripts.standalone.analyze_distribution_shift")
    real, _ = mod.load_images(str(ctx.smoke_data / "train"), max_images=2)
    folders = mod.find_generated_folders(str(ctx.smoke_data / "generated"))
    if real.shape[0] != 2:
        raise RuntimeError(f"Expected 2 real images, got shape {real.shape}")
    if not folders:
        raise RuntimeError("Expected generated folders in smoke data")
    return f"actual analysis loaders read real_shape={tuple(real.shape)} and {len(folders)} generated folder(s)"


def run_full_model_runs(ctx: DoctorContext) -> None:
    """Optional tiny actual runs for users who want real model execution."""
    if not ctx.full_model_runs:
        print("\n=== full model runs ===")
        print("[SKIP] Heavy model/checkpoint loading disabled. Pass --full-model-runs to opt in.")
        return
    print("\n=== full model runs ===")
    model_paths = _write_tiny_model_configs(ctx)
    sd_state: dict[str, Path] = {}

    def tiny_sd_dir() -> Path:
        if "tiny_sd_dir" not in sd_state:
            sd_state["tiny_sd_dir"] = _write_tiny_sd_pipeline(ctx)
        return sd_state["tiny_sd_dir"]

    def stage1_dir() -> Path:
        if "stage1_dir" not in sd_state:
            sd_state["stage1_dir"] = _actual_sd_stage1_training(ctx, tiny_sd_dir())
        return sd_state["stage1_dir"]

    full_specs = [
        (
            "full actual FM uncond training",
            lambda: _actual_fm_training(
                ctx,
                "FM uncond",
                _write_full_config(
                    ctx,
                    "fm_uncond_actual",
                    REPO / "configs/fm/train/presets/uncond_latent_flir_sd15_512.yaml",
                    _full_base_overrides(ctx, "fm_uncond_actual", model_paths),
                ),
            ),
        ),
        (
            "full actual FM RegionDiff training",
            lambda: _actual_fm_training(
                ctx,
                "FM RegionDiff",
                _write_full_config(
                    ctx,
                    "fm_regiondiff_actual",
                    REPO / "configs/fm/train/presets/regiondiff_latent.yaml",
                    _full_base_overrides(ctx, "fm_regiondiff_actual", model_paths, layout=True, regiondiff=True),
                ),
            ),
        ),
        (
            "full actual FM STAY-layout training",
            lambda: _actual_fm_training(
                ctx,
                "FM STAY-layout",
                _write_full_config(
                    ctx,
                    "fm_stay_layout_actual",
                    REPO / "configs/fm/train/presets/stay_layout_latent_flir_sd15_512.yaml",
                    _full_base_overrides(ctx, "fm_stay_layout_actual", model_paths, layout=True, stay=True),
                ),
            ),
        ),
        (
            "full actual SD1.5 stage-1 training",
            lambda: f"actual SD stage-1 training exported {_rel(stage1_dir() / 'stage1_manifest.json')}",
        ),
        (
            "full actual SD-layout stage-2 training",
            lambda: _actual_sd_layout_training(ctx, tiny_sd_dir(), stage1_dir()),
        ),
        (
            "full actual unconditional SD training",
            lambda: _actual_sd_uncond_training(
                ctx,
                _write_full_config(
                    ctx,
                    "sd_uncond_actual",
                    REPO / "configs/sd_uncond/train/presets/uncond_latent_flir_sd15_512.yaml",
                    _full_sd_uncond_overrides(ctx, "sd_uncond_actual", model_paths),
                ),
            ),
        ),
        ("full actual VAE training", lambda: _actual_vae_training(ctx, model_paths)),
        ("full actual FM generation", lambda: _actual_fm_generation(ctx, model_paths)),
        ("full actual analysis loaders", lambda: _actual_analysis_smoke(ctx)),
    ]
    for name, fn in full_specs:
        _run_result(ctx, name, fn)


def print_summary(ctx: DoctorContext) -> int:
    total = len(ctx.results)
    failed = [r for r in ctx.results if r.status != "PASS"]
    skipped = [r for r in ctx.results if r.status == "SKIP"]
    failed = [r for r in failed if r.status != "SKIP"]
    passed = total - len(failed) - len(skipped)
    print("\n" + "=" * 80)
    print("Doctor Inspector Summary")
    print("=" * 80)
    print(f"Passed: {passed}")
    print(f"Skipped: {len(skipped)}")
    print(f"Failed: {len(failed)}")
    print(f"Total : {total}")
    print(f"Log   : {_rel(ctx.log_path)}")
    print(f"Artifacts sandbox: {_rel(ctx.artifact_root)}")
    if failed:
        print("\nFailures:")
        for result in failed:
            print(f"- {result.name}")
            first_line = result.detail.strip().splitlines()[0] if result.detail.strip() else ""
            if first_line:
                print(f"  {first_line}")
    if skipped:
        print("\nSkipped:")
        for result in skipped:
            print(f"- {result.name}")
    return 1 if failed else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repo smoke doctor checks.")
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep artifacts/doctor_inspector after the run.",
    )
    parser.add_argument(
        "--full-model-runs",
        action="store_true",
        help="Opt into heavyweight model-load/training/generation runs when supported.",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Override the report log path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = Path(args.log_file) if args.log_file else LOG_ROOT / f"doctor_inspector_{timestamp}.log"
    if not log_path.is_absolute():
        log_path = REPO / log_path

    tee = Tee(log_path)
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = tee
    sys.stderr = tee
    ctx = DoctorContext(
        artifact_root=ARTIFACT_ROOT,
        log_path=log_path,
        keep_artifacts=bool(args.keep_artifacts),
        full_model_runs=bool(args.full_model_runs),
    )
    exit_code = 1
    try:
        print("Doctor Inspector")
        print(f"Repo: {REPO}")
        print(f"Started: {datetime.now().isoformat(timespec='seconds')}")
        print(f"Full model runs: {ctx.full_model_runs}")
        prepare_sandbox(ctx)
        _run_result(ctx, "smoke data loaders", lambda: smoke_data_loaders(ctx))
        run_config_matrix(ctx)
        run_launcher_checks(ctx)
        run_full_model_runs(ctx)
        exit_code = print_summary(ctx)
        return exit_code
    finally:
        if not ctx.keep_artifacts and ctx.artifact_root.exists():
            shutil.rmtree(ctx.artifact_root)
            print(f"\nRemoved artifact sandbox: {_rel(ctx.artifact_root)}")
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        tee.close()


if __name__ == "__main__":
    raise SystemExit(main())

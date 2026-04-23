"""Tests for the smoked end-to-end pipeline orchestration."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_smoke_runner():
    path = Path("scripts/smoke/run_smoked_e2e_pipeline.py").resolve()
    spec = importlib.util.spec_from_file_location("run_smoked_e2e_pipeline", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_smoked_pipeline_stage_keys_are_unique() -> None:
    runner = _load_smoke_runner()
    stages = runner._stage_definitions("cpu")
    keys = [stage.key for stage in stages]
    assert len(keys) == len(set(keys))
    assert "fm_uncond" in keys
    assert "yolo_sd15_lora_aug" in keys


def test_smoked_pipeline_dry_run_writes_all_summary_formats(tmp_path: Path, monkeypatch) -> None:
    runner = _load_smoke_runner()
    monkeypatch.setattr(runner, "ROOT", Path.cwd())
    monkeypatch.setattr(runner, "SMOKE_ROOT", tmp_path / "smoked_e2e")
    monkeypatch.setattr(runner, "SUMMARY_DIR", tmp_path / "smoked_e2e" / "smoked_summary")
    monkeypatch.setattr(runner, "SUMMARY_JSON", tmp_path / "smoked_e2e" / "smoked_summary" / "smoked_e2e_summary.json")
    monkeypatch.setattr(runner, "SUMMARY_LOG", tmp_path / "smoked_e2e" / "smoked_summary" / "smoked_e2e.log")
    monkeypatch.setattr(runner, "SUMMARY_TXT", tmp_path / "smoked_e2e" / "smoked_summary" / "smoked_e2e_summary.txt")
    monkeypatch.setattr(runner, "SUMMARY_CSV", tmp_path / "smoked_e2e" / "smoked_summary" / "smoked_e2e_summary.csv")

    stages = runner._stage_definitions("cpu")[:2]
    rows = []
    completed = {}
    for stage in stages:
        row = runner.run_stage(stage, dry_run=True, completed=completed)
        rows.append(row)
        completed[stage.key] = row
    runner.write_summaries(rows, dry_run=True, device="cpu")

    assert runner.SUMMARY_JSON.is_file()
    assert runner.SUMMARY_LOG.is_file()
    assert runner.SUMMARY_TXT.is_file()
    assert runner.SUMMARY_CSV.is_file()
    payload = json.loads(runner.SUMMARY_JSON.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert [row["status"] for row in payload["stages"]] == ["dry_run", "dry_run"]

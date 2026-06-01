from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from src.analysis.flir_subgroup.app import create_app


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_run(run_dir: Path, *, checkpoint: str = "epoch_120", kid: float = 0.1) -> None:
    preview_grid = run_dir / checkpoint / "stage1" / "preview_grid.png"
    preview_grid.parent.mkdir(parents=True, exist_ok=True)
    preview_grid.write_bytes(b"png")
    (run_dir / "notes.txt").write_text("not an image", encoding="utf-8")

    stage1_row = {
        "rank": 1,
        "checkpoint_identifier": checkpoint,
        "checkpoint_path": f"/runs/{checkpoint}",
        "model_type": "latent_flow_matching",
        "sampler_name": "euler",
        "KID": kid,
        "FID": 10.0,
        "selection_score": 0.0,
        "num_generated_images": 20,
    }
    stage2_row = {
        "rank": 1,
        "checkpoint_identifier": checkpoint,
        "checkpoint_path": f"/runs/{checkpoint}",
        "model_type": "latent_flow_matching",
        "sampler_name": "euler",
        "KID": kid + 0.01,
        "total_generated_images": 70,
    }
    final_metrics = {
        "selected_checkpoint_identifier": checkpoint,
        "model_type": "latent_flow_matching",
        "sampler_name": "euler",
        "KID": kid + 0.02,
        "FID": 9.5,
        "MMD": 0.03,
        "total_generated_images": 120,
        "timestamp": "2026-05-31T00:00:00+00:00",
    }
    previews = {
        "stages": [
            {
                "checkpoint_identifier": checkpoint,
                "stage": "stage1",
                "num_preview_images": 1,
                "preview_grid": str(preview_grid),
                "preview_images": [str(preview_grid)],
            }
        ]
    }
    _write_json(run_dir / "stage1_metrics.json", {"ranking": [stage1_row], "selected_top_k_checkpoints": [checkpoint]})
    _write_json(run_dir / "stage2_metrics.json", {"ranking": [stage2_row], "selected_best_checkpoint": checkpoint})
    _write_json(run_dir / "final_metrics.json", final_metrics)
    _write_json(
        run_dir / "checkpoint_selection_summary.json",
        {
            "stage_1_full_ranking": [stage1_row],
            "stage_2_full_ranking": [stage2_row],
            "final_selected_checkpoint": checkpoint,
            "selected_top_3_checkpoints": [checkpoint],
            "final_metrics": final_metrics,
            "analysis_previews": previews,
        },
    )
    _write_json(run_dir / "preview_summary.json", previews)


def test_catalog_discovers_one_and_two_level_layouts(tmp_path: Path) -> None:
    root_run = tmp_path / "direct_run"
    nested_run = tmp_path / "flir_stage1_checkpoint_selection" / "nested_run"
    _write_run(root_run, checkpoint="epoch_120", kid=0.1)
    _write_run(nested_run, checkpoint="step_001000", kid=0.2)

    client = TestClient(create_app())
    response = client.post("/api/checkpoint-selection/catalog", json={"root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["root"] == str(tmp_path.resolve())
    assert payload["subroots"] == [None, "flir_stage1_checkpoint_selection"]
    assert {(row["subroot"], row["run"]) for row in payload["runs"]} == {
        (None, "direct_run"),
        ("flir_stage1_checkpoint_selection", "nested_run"),
    }
    direct = next(row for row in payload["runs"] if row["run"] == "direct_run")
    assert direct["status"] == "complete"
    assert direct["selected_checkpoint"] == "epoch_120"
    assert direct["metrics"]["KID"] == pytest.approx(0.12)
    assert direct["available_preview_stages"] == ["stage1"]


def test_run_detail_normalizes_rankings_metrics_and_previews(tmp_path: Path) -> None:
    _write_run(tmp_path / "subroot" / "run_a", checkpoint="step_002000", kid=0.4)
    client = TestClient(create_app())

    response = client.post(
        "/api/checkpoint-selection/run",
        json={"root": str(tmp_path), "subroot": "subroot", "run": "run_a"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["relative_path"] == "subroot/run_a"
    assert payload["stage1_ranking"][0]["checkpoint_identifier"] == "step_002000"
    assert payload["stage2_ranking"][0]["KID"] == 0.41000000000000003
    assert payload["final_metrics"]["FID"] == 9.5
    assert payload["previews"][0]["preview_grid"] == "subroot/run_a/step_002000/stage1/preview_grid.png"

    preview_response = client.get(
        "/api/checkpoint-selection/preview",
        params={"root": str(tmp_path), "relative_path": payload["previews"][0]["preview_grid"]},
    )
    assert preview_response.status_code == 200
    assert preview_response.content == b"png"


def test_malformed_json_warns_without_crashing(tmp_path: Path) -> None:
    run_dir = tmp_path / "broken_run"
    run_dir.mkdir()
    (run_dir / "stage1_metrics.json").write_text("{not json", encoding="utf-8")
    _write_json(run_dir / "final_metrics.json", {"selected_checkpoint_identifier": "final", "KID": 0.3})

    client = TestClient(create_app())
    response = client.post("/api/checkpoint-selection/catalog", json={"root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["runs"][0]["status"] == "complete"
    assert payload["warnings"]
    assert "stage1_metrics.json" in payload["warnings"][0]


def test_preview_rejects_traversal_and_non_images(tmp_path: Path) -> None:
    _write_run(tmp_path / "run_a")
    client = TestClient(create_app())

    traversal_response = client.get(
        "/api/checkpoint-selection/preview",
        params={"root": str(tmp_path), "relative_path": "../secret.png"},
    )
    assert traversal_response.status_code == 403

    non_image_response = client.get(
        "/api/checkpoint-selection/preview",
        params={"root": str(tmp_path), "relative_path": "run_a/notes.txt"},
    )
    assert non_image_response.status_code == 415

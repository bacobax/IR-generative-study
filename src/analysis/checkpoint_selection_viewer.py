"""FastAPI helpers for browsing checkpoint-selection analysis outputs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel


RUN_MARKER_FILES = (
    "checkpoint_selection_summary.json",
    "stage1_metrics.json",
    "stage2_metrics.json",
    "final_metrics.json",
)
PREVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
METRIC_KEYS = ("KID", "FID", "MMD", "Intra-LPIPS")


class CatalogRequest(BaseModel):
    """Request body for scanning a checkpoint-selection root."""

    root: str


class RunRequest(BaseModel):
    """Request body for loading one checkpoint-selection run."""

    root: str
    run: str
    subroot: str | None = None


def create_router() -> APIRouter:
    """Create the checkpoint-selection viewer router."""

    router = APIRouter(prefix="/api/checkpoint-selection", tags=["checkpoint-selection"])

    @router.post("/catalog")
    def catalog(request: CatalogRequest) -> dict[str, Any]:
        root = _resolve_root(request.root)
        runs, warnings = _discover_runs(root)
        subroots = sorted({row["subroot"] for row in runs if row["subroot"] is not None})
        if any(row["subroot"] is None for row in runs):
            subroots.insert(0, None)
        return {
            "root": str(root),
            "subroots": subroots,
            "runs": runs,
            "warnings": warnings,
        }

    @router.post("/run")
    def run_detail(request: RunRequest) -> dict[str, Any]:
        root = _resolve_root(request.root)
        run_dir = _resolve_run_dir(root, request.subroot, request.run)
        if not _is_run_dir(run_dir):
            raise HTTPException(status_code=404, detail=f"Analysis run not found: {request.run}")
        return _build_run_detail(root, run_dir, request.subroot)

    @router.get("/preview")
    def preview(
        root: str = Query(..., description="Checkpoint-selection root directory"),
        relative_path: str = Query(..., description="Preview image path relative to root"),
    ) -> FileResponse:
        resolved_root = _resolve_root(root)
        preview_path = _resolve_preview_path(resolved_root, relative_path)
        return FileResponse(preview_path)

    return router


def _resolve_root(root: str) -> Path:
    path = Path(root).expanduser().resolve()
    if not path.is_dir():
        raise HTTPException(status_code=404, detail=f"Checkpoint-selection root is not a directory: {root}")
    return path


def _resolve_run_dir(root: Path, subroot: str | None, run: str) -> Path:
    if not run or Path(run).is_absolute() or ".." in Path(run).parts:
        raise HTTPException(status_code=422, detail="run must be a relative folder name")
    if subroot:
        if Path(subroot).is_absolute() or ".." in Path(subroot).parts:
            raise HTTPException(status_code=422, detail="subroot must be a relative folder name")
        candidate = (root / subroot / run).resolve()
    else:
        candidate = (root / run).resolve()
    if not _is_under(candidate, root):
        raise HTTPException(status_code=403, detail="run path escapes the selected root")
    return candidate


def _resolve_preview_path(root: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    if rel.is_absolute() or ".." in rel.parts:
        raise HTTPException(status_code=403, detail="preview path must stay under the selected root")
    path = (root / rel).resolve()
    if not _is_under(path, root):
        raise HTTPException(status_code=403, detail="preview path escapes the selected root")
    if path.suffix.lower() not in PREVIEW_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Only preview image files can be served")
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Preview image not found: {relative_path}")
    return path


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _is_run_dir(path: Path) -> bool:
    return path.is_dir() and any((path / marker).is_file() for marker in RUN_MARKER_FILES)


def _discover_runs(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    runs: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen: set[Path] = set()

    for child in sorted(_iter_dirs(root), key=lambda path: path.name):
        if _is_run_dir(child):
            seen.add(child.resolve())
            runs.append(_build_catalog_row(root, child, subroot=None, warnings=warnings))
            continue
        for grandchild in sorted(_iter_dirs(child), key=lambda path: path.name):
            if not _is_run_dir(grandchild):
                continue
            resolved = grandchild.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            runs.append(_build_catalog_row(root, grandchild, subroot=child.name, warnings=warnings))

    runs.sort(key=lambda row: ((row["subroot"] or ""), row["run"]))
    return runs, warnings


def _iter_dirs(path: Path) -> Iterable[Path]:
    try:
        yield from (child for child in path.iterdir() if child.is_dir())
    except OSError:
        return


def _build_catalog_row(root: Path, run_dir: Path, *, subroot: str | None, warnings: list[str]) -> dict[str, Any]:
    payload = _load_run_payload(run_dir, warnings)
    detail = _normalize_run(root, run_dir, subroot=subroot, payload=payload)
    return {
        "subroot": subroot,
        "run": run_dir.name,
        "relative_path": _relative_posix(run_dir, root),
        "status": detail["status"],
        "selected_checkpoint": detail["selected_checkpoint"],
        "model_type": detail["model_type"],
        "sampler_name": detail["sampler_name"],
        "timestamp": detail["timestamp"],
        "metrics": detail["metrics"],
        "available_preview_stages": detail["available_preview_stages"],
    }


def _build_run_detail(root: Path, run_dir: Path, subroot: str | None) -> dict[str, Any]:
    warnings: list[str] = []
    payload = _load_run_payload(run_dir, warnings)
    detail = _normalize_run(root, run_dir, subroot=subroot, payload=payload)
    detail["warnings"] = warnings
    return detail


def _load_run_payload(run_dir: Path, warnings: list[str]) -> dict[str, Any]:
    return {
        "summary": _load_json(run_dir / "checkpoint_selection_summary.json", warnings),
        "stage1": _load_json(run_dir / "stage1_metrics.json", warnings),
        "stage2": _load_json(run_dir / "stage2_metrics.json", warnings),
        "final": _load_json(run_dir / "final_metrics.json", warnings),
        "preview": _load_json(run_dir / "preview_summary.json", warnings),
        "cleanup_plan": _load_json(run_dir / "checkpoint_cleanup_plan.json", warnings),
        "cleanup_result": _load_json(run_dir / "checkpoint_cleanup_result.json", warnings),
    }


def _load_json(path: Path, warnings: list[str]) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        warnings.append(f"{path}: {exc}")
        return None


def _normalize_run(
    root: Path,
    run_dir: Path,
    *,
    subroot: str | None,
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    summary = _as_mapping(payload.get("summary"))
    stage1_payload = _as_mapping(payload.get("stage1"))
    stage2_payload = _as_mapping(payload.get("stage2"))
    final_payload = _as_mapping(payload.get("final")) or _as_mapping(summary.get("final_metrics"))

    stage1_ranking = _ranking_rows(stage1_payload, summary.get("stage_1_full_ranking"))
    stage2_ranking = _ranking_rows(stage2_payload, summary.get("stage_2_full_ranking"))
    metrics = _metric_map(final_payload)
    selected_checkpoint = (
        summary.get("final_selected_checkpoint")
        or final_payload.get("selected_checkpoint_identifier")
        or stage2_payload.get("selected_best_checkpoint")
    )
    preview_source = _as_mapping(payload.get("preview")) or _as_mapping(summary.get("analysis_previews"))
    previews = _normalize_previews(root, preview_source)
    status = _status(stage1_ranking, stage2_ranking, final_payload, selected_checkpoint)

    return {
        "root": str(root),
        "subroot": subroot,
        "run": run_dir.name,
        "relative_path": _relative_posix(run_dir, root),
        "status": status,
        "selected_checkpoint": _to_json_safe(selected_checkpoint),
        "selected_top_checkpoints": _to_json_safe(
            summary.get("selected_top_3_checkpoints")
            or stage1_payload.get("selected_top_k_checkpoints")
            or []
        ),
        "model_type": _first_present(final_payload, stage1_ranking, stage2_ranking, "model_type"),
        "sampler_name": _first_present(final_payload, stage1_ranking, stage2_ranking, "sampler_name"),
        "generation_backend_used": _first_present(final_payload, stage1_ranking, stage2_ranking, "generation_backend_used"),
        "timestamp": _first_present(final_payload, [stage2_payload, stage1_payload, preview_source], [], "timestamp"),
        "metrics": metrics,
        "metric_directions": _to_json_safe(final_payload.get("metric_directions") or _default_metric_directions(metrics)),
        "stage1_ranking": _to_json_safe(stage1_ranking),
        "stage2_ranking": _to_json_safe(stage2_ranking),
        "final_metrics": _to_json_safe(final_payload),
        "summary": _to_json_safe(summary),
        "cleanup": _to_json_safe(payload.get("cleanup_result") or payload.get("cleanup_plan")),
        "previews": previews,
        "available_preview_stages": sorted({preview["stage"] for preview in previews if preview.get("stage")}),
    }


def _as_mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _ranking_rows(payload: Mapping[str, Any], fallback: Any) -> list[dict[str, Any]]:
    rows = payload.get("ranking") or payload.get("metrics") or fallback or []
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, Mapping)]


def _metric_map(payload: Mapping[str, Any]) -> dict[str, float | None]:
    return {key: _finite_float(payload.get(key)) for key in METRIC_KEYS if key in payload}


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _status(
    stage1_ranking: list[dict[str, Any]],
    stage2_ranking: list[dict[str, Any]],
    final_payload: Mapping[str, Any],
    selected_checkpoint: Any,
) -> str:
    if final_payload and selected_checkpoint:
        return "complete"
    if stage1_ranking or stage2_ranking or final_payload:
        return "partial"
    return "invalid"


def _first_present(
    primary: Mapping[str, Any],
    stage1_rows_or_payloads: list[Mapping[str, Any]] | Mapping[str, Any],
    stage2_rows: list[Mapping[str, Any]],
    key: str,
) -> Any:
    if primary.get(key) is not None:
        return _to_json_safe(primary.get(key))
    candidates: list[Mapping[str, Any]]
    if isinstance(stage1_rows_or_payloads, Mapping):
        candidates = [stage1_rows_or_payloads]
    else:
        candidates = list(stage1_rows_or_payloads)
    candidates.extend(stage2_rows)
    for row in candidates:
        if row.get(key) is not None:
            return _to_json_safe(row.get(key))
    return None


def _default_metric_directions(metrics: Mapping[str, Any]) -> dict[str, str]:
    directions = {key: "lower_is_better" for key in metrics}
    if "Intra-LPIPS" in directions:
        directions["Intra-LPIPS"] = "higher_is_better"
    return directions


def _normalize_previews(root: Path, preview_source: Mapping[str, Any]) -> list[dict[str, Any]]:
    stages = preview_source.get("stages", [])
    if not isinstance(stages, list):
        return []
    previews = []
    for stage in stages:
        if not isinstance(stage, Mapping):
            continue
        preview_grid = _relative_existing_image(root, stage.get("preview_grid"))
        preview_images = [
            relative
            for relative in (_relative_existing_image(root, image_path) for image_path in stage.get("preview_images", []))
            if relative is not None
        ]
        previews.append(
            {
                "checkpoint_identifier": _to_json_safe(stage.get("checkpoint_identifier")),
                "stage": _to_json_safe(stage.get("stage")),
                "num_preview_images": _to_json_safe(stage.get("num_preview_images")),
                "preview_grid": preview_grid,
                "preview_images": preview_images,
                "tile_size": _to_json_safe(stage.get("tile_size")),
                "columns": _to_json_safe(stage.get("columns")),
                "timestamp": _to_json_safe(stage.get("timestamp")),
            }
        )
    return previews


def _relative_existing_image(root: Path, value: Any) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value).expanduser()
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if not _is_under(resolved, root) or resolved.suffix.lower() not in PREVIEW_EXTENSIONS or not resolved.is_file():
        return None
    return _relative_posix(resolved, root)


def _relative_posix(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root).as_posix()


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _to_json_safe(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(child) for child in value]
    if isinstance(value, tuple):
        return [_to_json_safe(child) for child in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value

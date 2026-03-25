"""FastAPI routes for the FLIR subgroup analysis app."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Callable, Dict, List, Sequence
from urllib.parse import quote

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from src.analysis.flir_subgroup.analysis import (
    build_dominance_histogram,
    build_example_boxes,
    build_per_class_image_count_distribution,
    compare_partitions_from_holdout_keys,
    compute_collateral_damage,
    compute_holdout_table,
    get_union_holdout_image_keys,
    parse_subgroup_label,
    select_examples_for_group,
)
from src.analysis.flir_subgroup.constants import (
    ANALYSIS_SPLITS,
    DEFAULT_EXAMPLE_COUNT,
    DOMINANCE_THRESHOLDS,
    FEASIBILITY_RULES,
    FIXED_SIZE_BINS,
    POSITION_BIN_LABELS,
    POSITION_MODE,
    SIZE_BIN_LABELS,
    SIZE_BIN_METHOD,
)
from src.analysis.flir_subgroup.context import FlirSubgroupAnalysisContext, get_analysis_context
from src.analysis.flir_subgroup.data import render_preview_png_bytes
from src.analysis.flir_subgroup.schemas import (
    CollateralRequest,
    ExamplesRequest,
    GroupSpec,
    HoldoutCurvesRequest,
    PartitionComparisonsRequest,
    Phase,
)


def _records(df: pd.DataFrame) -> List[dict]:
    """Convert a dataframe to JSON-safe records."""

    if df.empty:
        return []
    safe_df = df.copy()
    safe_df = safe_df.where(pd.notnull(safe_df), None)
    return safe_df.to_dict(orient="records")


def _resolve_groups(
    context: FlirSubgroupAnalysisContext,
    phase: Phase,
    groups: Sequence[GroupSpec],
) -> List[dict]:
    """Validate structured group specs against cached selectable groups."""

    phase_bundle = context.get_phase_bundle(phase.value)
    selectable_by_label: Dict[str, dict] = {
        str(row["subgroup_label"]): row
        for row in _records(phase_bundle.selectable_groups_df)
    }

    resolved_groups: List[dict] = []
    for group in groups:
        subgroup_label = group.subgroup_label(phase)
        if phase == Phase.PHASE1 and group.position_bin is not None:
            raise HTTPException(status_code=422, detail="phase1 groups cannot include position_bin")
        if phase == Phase.PHASE2 and not group.position_bin:
            raise HTTPException(status_code=422, detail="phase2 groups require position_bin")

        try:
            metadata = selectable_by_label[subgroup_label]
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=f"Unknown subgroup for {phase.value}: {subgroup_label}") from exc

        resolved_groups.append(
            {
                "class_label": group.class_label,
                "size_bin": group.size_bin,
                "position_bin": group.position_bin,
                "subgroup_label": subgroup_label,
                "n_instances": int(metadata["n_instances"]),
                "n_images": int(metadata["n_images"]),
                "median_dominance": float(metadata["median_dominance"]),
            }
        )

    return resolved_groups


def _serialize_example_rows(
    context: FlirSubgroupAnalysisContext,
    example_df: pd.DataFrame,
    subgroup_label: str,
    instance_df: pd.DataFrame,
) -> List[dict]:
    """Serialize example rows for API responses."""

    if example_df.empty:
        return []

    image_lookup = context.image_table.set_index("image_key")
    class_label = parse_subgroup_label(subgroup_label)["class_label"]

    records: List[dict] = []
    for row in example_df.itertuples(index=False):
        if row.image_key not in image_lookup.index:
            continue

        image_meta = image_lookup.loc[row.image_key]
        image_instances = context.instance_table_raw.loc[context.instance_table_raw["image_key"] == row.image_key]
        class_count = int((image_instances["class_label"] == class_label).sum())

        records.append(
            {
                "image_key": str(row.image_key),
                "image_id": str(image_meta["image_id"]),
                "partition": str(row.partition),
                "selection_source": str(row.selection_source),
                "subgroup_label": subgroup_label,
                "dominance_ratio": float(getattr(row, "dominance_ratio", 0.0) or 0.0),
                "subgroup_count": int(getattr(row, "subgroup_count", 0) or 0),
                "class_count": class_count,
                "preview_url": f"/api/flir-analysis/images/{quote(str(row.image_key), safe='')}",
                "image_width": int(image_meta["image_width"]),
                "image_height": int(image_meta["image_height"]),
                "boxes": build_example_boxes(instance_df, str(row.image_key), subgroup_label),
            }
        )
    return records


def create_router(data_root: Path | None = None) -> APIRouter:
    """Create a router bound to a specific dataset root."""

    router = APIRouter(prefix="/api/flir-analysis", tags=["flir-analysis"])

    def get_context() -> FlirSubgroupAnalysisContext:
        root_arg = str(data_root.resolve()) if data_root is not None else None
        return get_analysis_context(root_arg)

    @router.get("/options")
    def get_options() -> dict:
        context = get_context()
        phase1_bundle = context.get_phase_bundle("phase1")
        phase2_bundle = context.get_phase_bundle("phase2")

        dataset_metadata = context.dataset_summary
        dataset_metadata["layout"] = _records(context.dataset_layout_df)
        dataset_metadata["root_metadata"] = _records(context.root_metadata_df)

        return {
            "dataset": dataset_metadata,
            "constants": {
                "analysis_splits": list(ANALYSIS_SPLITS),
                "size_bin_method": SIZE_BIN_METHOD,
                "size_bin_labels": list(SIZE_BIN_LABELS),
                "fixed_size_bins": FIXED_SIZE_BINS,
                "position_mode": POSITION_MODE,
                "position_bin_labels": list(POSITION_BIN_LABELS),
                "dominance_thresholds": list(DOMINANCE_THRESHOLDS),
                "feasibility_rules": FEASIBILITY_RULES,
            },
            "phase1": {
                "default_group": next(
                    row for row in _records(phase1_bundle.selectable_groups_df) if row["subgroup_label"] == phase1_bundle.default_subgroup_label
                ),
                "example_groups": phase1_bundle.example_subgroup_labels,
                "groups": _records(phase1_bundle.selectable_groups_df),
                "dominant_group_overview": _records(phase1_bundle.dominant_group_overview_df),
                "dominant_group_frequency": _records(phase1_bundle.dominant_group_frequency_df),
                "feasibility": _records(phase1_bundle.feasibility_df),
                "size_bin_spec": _records(phase1_bundle.size_bin_spec_df),
            },
            "phase2": {
                "default_group": next(
                    row for row in _records(phase2_bundle.selectable_groups_df) if row["subgroup_label"] == phase2_bundle.default_subgroup_label
                ),
                "example_groups": phase2_bundle.example_subgroup_labels,
                "groups": _records(phase2_bundle.selectable_groups_df),
                "dominant_group_overview": _records(phase2_bundle.dominant_group_overview_df),
                "dominant_group_frequency": _records(phase2_bundle.dominant_group_frequency_df),
                "size_bin_spec": _records(phase2_bundle.size_bin_spec_df),
            },
        }

    @router.post("/holdout-curves")
    def post_holdout_curves(request: HoldoutCurvesRequest) -> dict:
        context = get_context()
        phase_bundle = context.get_phase_bundle(request.phase.value)
        resolved_groups = _resolve_groups(context, request.phase, request.groups)
        thresholds = request.thresholds or list(DOMINANCE_THRESHOLDS)

        results = []
        for group in resolved_groups:
            holdout_df = compute_holdout_table(
                phase_bundle.image_subgroup_df,
                phase_bundle.image_stats_df,
                subgroup=group["subgroup_label"],
                thresholds=thresholds,
            )
            results.append({**group, "series": _records(holdout_df)})

        return {"phase": request.phase.value, "thresholds": thresholds, "groups": results}

    @router.post("/collateral")
    def post_collateral(request: CollateralRequest) -> dict:
        context = get_context()
        phase_bundle = context.get_phase_bundle(request.phase.value)
        resolved_groups = _resolve_groups(context, request.phase, request.groups)

        results = []
        for group in resolved_groups:
            damage_df, summary = compute_collateral_damage(
                phase_bundle.instance_df,
                phase_bundle.image_subgroup_df,
                subgroup=group["subgroup_label"],
                tau=request.tau,
            )
            dominance_hist_df = build_dominance_histogram(phase_bundle.image_subgroup_df, group["subgroup_label"])
            results.append(
                {
                    **group,
                    "summary": summary,
                    "damage_rows": _records(damage_df),
                    "dominance_histogram": _records(dominance_hist_df),
                }
            )

        return {"phase": request.phase.value, "tau": request.tau, "groups": results}

    @router.post("/partition-comparisons")
    def post_partition_comparisons(request: PartitionComparisonsRequest) -> dict:
        context = get_context()
        phase_bundle = context.get_phase_bundle(request.phase.value)
        resolved_groups = _resolve_groups(context, request.phase, request.groups)
        subgroup_labels = [group["subgroup_label"] for group in resolved_groups]
        heldout_keys = get_union_holdout_image_keys(phase_bundle.image_subgroup_df, subgroup_labels, request.tau)

        partition_tables = compare_partitions_from_holdout_keys(
            phase_bundle.image_stats_df,
            phase_bundle.instance_df,
            heldout_keys,
            selected_subgroups=subgroup_labels,
        )
        per_class_count_df = build_per_class_image_count_distribution(
            phase_bundle.instance_df,
            heldout_keys,
            include_zero_counts=request.include_zero_counts,
        )

        return {
            "phase": request.phase.value,
            "tau": request.tau,
            "groups": resolved_groups,
            "heldout_image_keys": heldout_keys,
            "heldout_n_images": len(heldout_keys),
            "numeric_summary": _records(partition_tables["numeric_summary_df"]),
            "class_distribution": _records(partition_tables["class_distribution_df"]),
            "class_image_distribution": _records(partition_tables["class_image_distribution_df"]),
            "subgroup_distribution": _records(partition_tables["subgroup_distribution_df"]),
            "cooccurring_class_distribution": _records(partition_tables["cooccurring_class_distribution_df"]),
            "per_class_image_count_distribution": _records(per_class_count_df),
        }

    @router.post("/examples")
    def post_examples(request: ExamplesRequest) -> dict:
        context = get_context()
        phase_bundle = context.get_phase_bundle(request.phase.value)
        resolved_groups = _resolve_groups(context, request.phase, request.groups)

        results = []
        for group in resolved_groups:
            example_tables = select_examples_for_group(
                phase_bundle.image_stats_df,
                phase_bundle.instance_df,
                phase_bundle.image_subgroup_df,
                subgroup=group["subgroup_label"],
                tau=request.tau,
                example_count=request.example_count or DEFAULT_EXAMPLE_COUNT,
            )
            results.append(
                {
                    **group,
                    "held_out_examples": _serialize_example_rows(
                        context,
                        example_tables["held_out_examples_df"],
                        group["subgroup_label"],
                        phase_bundle.instance_df,
                    ),
                    "retained_examples": _serialize_example_rows(
                        context,
                        example_tables["retained_examples_df"],
                        group["subgroup_label"],
                        phase_bundle.instance_df,
                    ),
                }
            )

        return {"phase": request.phase.value, "tau": request.tau, "example_count": request.example_count, "groups": results}

    @router.get("/images/{image_key:path}")
    def get_image_preview(image_key: str) -> StreamingResponse:
        context = get_context()
        image_table = context.image_table.set_index("image_key")
        if image_key not in image_table.index:
            raise HTTPException(status_code=404, detail=f"Unknown image_key: {image_key}")

        row = image_table.loc[image_key]
        file_path = row["file_path"]
        if not file_path:
            raise HTTPException(status_code=404, detail=f"No file path available for {image_key}")

        path = Path(str(file_path))
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"Image file not found for {image_key}")

        png_bytes = render_preview_png_bytes(path)
        return StreamingResponse(BytesIO(png_bytes), media_type="image/png")

    return router

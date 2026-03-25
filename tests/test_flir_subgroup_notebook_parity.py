"""Notebook parity checks for the refactored FLIR subgroup analysis code."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from src.analysis.flir_subgroup.analysis import compute_collateral_damage, compute_holdout_table
from src.analysis.flir_subgroup.context import build_analysis_context
from src.core.paths import flir_root


NOTEBOOK_PATH = Path("docs/notebooks/v18_scene_graph_score_analysis_flir.py")


def _load_notebook_namespace() -> dict:
    text = NOTEBOOK_PATH.read_text(encoding="utf-8")
    marker = "\nbenchmark_feasibility_df = build_feasibility_table("
    prefix = text.split(marker)[0]
    namespace: dict = {}
    os.environ.setdefault("MPLBACKEND", "Agg")
    exec(compile(prefix, str(NOTEBOOK_PATH), "exec"), namespace)
    return namespace


@pytest.mark.skipif(not NOTEBOOK_PATH.exists(), reason="Notebook mirror not found")
@pytest.mark.skipif(not flir_root().exists(), reason="FLIR dataset not available")
def test_notebook_parity_for_representative_groups() -> None:
    namespace = _load_notebook_namespace()
    context = build_analysis_context(data_root=flir_root())

    phase1_group = "class=car | size=large"
    phase2_group = "class=car | size=large | pos=center"

    phase1_bundle = context.get_phase_bundle("phase1")
    phase2_bundle = context.get_phase_bundle("phase2")

    ours_phase1_holdout = compute_holdout_table(
        phase1_bundle.image_subgroup_df,
        phase1_bundle.image_stats_df,
        phase1_group,
        thresholds=[0.5],
    )
    notebook_phase1_holdout = namespace["compute_holdout_table"](
        namespace["phase1_image_subgroup_df"],
        namespace["phase1_image_stats_df"],
        phase1_group,
        thresholds=[0.5],
    )
    pd.testing.assert_frame_equal(ours_phase1_holdout.reset_index(drop=True), notebook_phase1_holdout.reset_index(drop=True))

    ours_phase2_holdout = compute_holdout_table(
        phase2_bundle.image_subgroup_df,
        phase2_bundle.image_stats_df,
        phase2_group,
        thresholds=[0.5],
    )
    notebook_phase2_holdout = namespace["compute_holdout_table"](
        namespace["phase2_image_subgroup_df"],
        namespace["phase2_image_stats_df"],
        phase2_group,
        thresholds=[0.5],
    )
    pd.testing.assert_frame_equal(ours_phase2_holdout.reset_index(drop=True), notebook_phase2_holdout.reset_index(drop=True))

    ours_phase1_damage, ours_phase1_summary = compute_collateral_damage(
        phase1_bundle.instance_df,
        phase1_bundle.image_subgroup_df,
        phase1_group,
        tau=0.5,
    )
    notebook_phase1_damage, notebook_phase1_summary = namespace["compute_collateral_damage"](
        namespace["phase1_instance_table"],
        namespace["phase1_image_subgroup_df"],
        phase1_group,
        tau=0.5,
    )
    pd.testing.assert_frame_equal(ours_phase1_damage.reset_index(drop=True), notebook_phase1_damage.reset_index(drop=True))
    assert ours_phase1_summary == notebook_phase1_summary

    ours_phase2_damage, ours_phase2_summary = compute_collateral_damage(
        phase2_bundle.instance_df,
        phase2_bundle.image_subgroup_df,
        phase2_group,
        tau=0.5,
    )
    notebook_phase2_damage, notebook_phase2_summary = namespace["compute_collateral_damage"](
        namespace["phase2_instance_table"],
        namespace["phase2_image_subgroup_df"],
        phase2_group,
        tau=0.5,
    )
    pd.testing.assert_frame_equal(ours_phase2_damage.reset_index(drop=True), notebook_phase2_damage.reset_index(drop=True))
    assert ours_phase2_summary == notebook_phase2_summary

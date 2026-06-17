"""
Plot checkpoint-selection metrics from killarney_scratch results.

Generates two families of PNG charts via matplotlib:

1. Comparison charts (grouped bar)  — one per dataset × metric × reference source
   Saved under:  <out>/comparison/<dataset>/<ref>/<metric>.png

2. Trend charts (2×2 subplots)      — one per run, metrics vs checkpoint step
   Saved under:  <out>/trends/<dataset>/<run_name>.png

Usage
-----
    python scripts/plot_killarney_results.py
    python scripts/plot_killarney_results.py --root killarney_scratch --out killarney_scratch/_charts
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Display name  →  JSON key inside metric_values
METRICS: dict[str, str] = {
    "FID": "FID",
    "KID": "KID",
    "MMD": "MMD",
    "DINO_FD": "fd_dinov2",
}

REFERENCE_SOURCES: list[str] = ["test", "val", "train", "train_val_test"]
SUBSET_ORDER: list[str] = ["2k", "5k", "full"]
TRAINING_TYPES: list[str] = ["Diffusion", "Flow Matching", "SD1.5 LoRA", "SDXL LoRA"]

# Human-readable legend labels (internal key → display string)
TRAINING_TYPE_LABELS: dict[str, str] = {
    "Diffusion": "Diffusion from scratch",
    "Flow Matching": "Flow Matching OT from scratch",
    "SD1.5 LoRA": "SD1.5 LoRA",
    "SDXL LoRA": "SDXL LoRA",
}

# Colours consistent across all charts
TRAINING_TYPE_COLORS: dict[str, str] = {
    "Diffusion": "#4C72B0",
    "Flow Matching": "#DD8452",
    "SD1.5 LoRA": "#55A868",
    "SDXL LoRA": "#C44E52",
}

# Sentinel step for the "final" checkpoint (larger than any real step)
_FINAL_STEP_SENTINEL = 10_000_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _finite_float(value: Any) -> float | None:
    """Return value as float if it is finite, else None."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _load_json(path: Path) -> Any | None:
    """Load JSON from path, returning None on any error."""
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"  [warn] could not parse {path}: {exc}", file=sys.stderr)
        return None


def _dataset_display(dataset_dir_name: str) -> str:
    """Convert raw directory name into a short display label."""
    if "flir" in dataset_dir_name.lower():
        return "FLIR"
    if "bigearthnet" in dataset_dir_name.lower():
        return "BigEarthNet"
    return dataset_dir_name


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def classify_run(dataset_dir_name: str, run_dir_name: str) -> dict[str, str]:
    """
    Classify a run into {dataset, subset, training_type}.

    Subset dimension
    ----------------
    train_2000 / train_2040  →  2k
    train_5000 / train_5100  →  5k
    (neither)                →  full

    Training type (checked in order — first match wins)
    -------------
    sdxl + lora              →  SDXL LoRA
    lora (no sdxl)           →  SD1.5 LoRA
    _ot suffix/middle        →  Flow Matching
    else                     →  Diffusion
    """
    name = run_dir_name.lower()

    # Subset
    if re.search(r"train[_-](2000|2040)\b", name):
        subset = "2k"
    elif re.search(r"train[_-](5000|5100)\b", name):
        subset = "5k"
    else:
        subset = "full"

    # Training type
    if "sdxl" in name and "lora" in name:
        training_type = "SDXL LoRA"
    elif "lora" in name:
        training_type = "SD1.5 LoRA"
    elif re.search(r"_ot(_|$)", name):
        training_type = "Flow Matching"
    else:
        training_type = "Diffusion"

    return {
        "dataset": _dataset_display(dataset_dir_name),
        "subset": subset,
        "training_type": training_type,
    }


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_final_metrics(run_dir: Path) -> dict[str, dict[str, float | None]]:
    """
    Load headline (selected-checkpoint) metrics from final_metrics_summary.json.

    Returns  {ref_source: {metric_display_name: float | None}}
    Returns an empty dict if the file is absent or unusable.
    """
    data = _load_json(run_dir / "final_metrics_summary.json")
    if not isinstance(data, dict):
        return {}

    result: dict[str, dict[str, float | None]] = {}
    by_ref = data.get("metrics_by_reference_source")
    if not isinstance(by_ref, dict):
        return {}

    for ref, ref_block in by_ref.items():
        if not isinstance(ref_block, dict):
            continue
        mv = ref_block.get("metric_values")
        if not isinstance(mv, dict):
            continue
        result[ref] = {
            display: _finite_float(mv.get(json_key))
            for display, json_key in METRICS.items()
        }
    return result


def load_trend(run_dir: Path) -> list[dict[str, Any]]:
    """
    Load per-checkpoint trend data from selection_ranking.json.

    Returns a list of {step: int, ckpt_id: str, metrics: {display_name: float|None}},
    sorted by step ascending.  The pseudo-checkpoint "final" (step=null) is placed last.
    Returns an empty list if the file is absent or unusable.
    """
    data = _load_json(run_dir / "selection_ranking.json")
    if not isinstance(data, dict):
        return []

    ranking = data.get("ranking")
    if not isinstance(ranking, list):
        return []

    rows: list[dict[str, Any]] = []
    for entry in ranking:
        if not isinstance(entry, dict):
            continue
        raw_step = entry.get("step")
        sort_step = _FINAL_STEP_SENTINEL if raw_step is None else int(raw_step)
        mv = entry.get("metric_values") or {}
        rows.append({
            "sort_step": sort_step,
            "step": raw_step,
            "ckpt_id": entry.get("checkpoint_identifier", ""),
            "metrics": {
                display: _finite_float(mv.get(json_key))
                for display, json_key in METRICS.items()
            },
        })

    rows.sort(key=lambda r: r["sort_step"])
    return rows


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_runs(root: Path) -> list[dict[str, Any]]:
    """
    Walk root/<dataset_dir>/<run_dir>/ and build a list of run records.

    Each record:
        run_dir       Path
        run_name      str
        dataset_dir   str   (raw folder name)
        dataset       str   (display)
        subset        str
        training_type str
        final_metrics {ref: {metric: float|None}}  (may be empty)
        trend         [{sort_step, step, ckpt_id, metrics}]  (may be empty)
        status        "complete" | "trend_only" | "final_only" | "empty"
    """
    records: list[dict[str, Any]] = []
    skipped: list[str] = []

    for dataset_dir in sorted(root.iterdir()):
        if not dataset_dir.is_dir() or dataset_dir.name.startswith("_"):
            continue

        for run_dir in sorted(dataset_dir.iterdir()):
            if not run_dir.is_dir():
                continue

            classification = classify_run(dataset_dir.name, run_dir.name)
            final_metrics = load_final_metrics(run_dir)
            trend = load_trend(run_dir)

            has_final = bool(final_metrics)
            has_trend = bool(trend)

            if not has_final and not has_trend:
                skipped.append(run_dir.name)
                continue

            if has_final and has_trend:
                status = "complete"
            elif has_trend:
                status = "trend_only"
            else:
                status = "final_only"

            records.append({
                "run_dir": run_dir,
                "run_name": run_dir.name,
                "dataset_dir": dataset_dir.name,
                **classification,
                "final_metrics": final_metrics,
                "trend": trend,
                "status": status,
            })

    if skipped:
        print(f"  [info] skipped {len(skipped)} run(s) with no usable data: {skipped}",
              file=sys.stderr)

    return records


# ---------------------------------------------------------------------------
# Comparison charts
# ---------------------------------------------------------------------------

def plot_comparison(records: list[dict[str, Any]], out_dir: Path) -> int:
    """
    Emit grouped bar charts: dataset × reference_source × metric.

    Returns the number of PNG files written.
    """
    # Pre-group: data[dataset][ref][metric_display][subset][training_type] = float|None
    from collections import defaultdict
    data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: None))
    )))

    for rec in records:
        if not rec["final_metrics"]:
            continue
        ds = rec["dataset"]
        subset = rec["subset"]
        tt = rec["training_type"]
        for ref, metric_map in rec["final_metrics"].items():
            for metric_display, value in metric_map.items():
                data[ds][ref][metric_display][subset][tt] = value

    written = 0
    datasets = sorted(data.keys())

    for ds in datasets:
        for ref in REFERENCE_SOURCES:
            if ref not in data[ds]:
                continue
            for metric_display in METRICS:
                if metric_display not in data[ds][ref]:
                    continue

                metric_data = data[ds][ref][metric_display]

                # Determine which subsets / training types actually have data
                active_subsets = [s for s in SUBSET_ORDER if s in metric_data and
                                  any(metric_data[s].get(tt) is not None for tt in TRAINING_TYPES)]
                active_types = [tt for tt in TRAINING_TYPES if
                                any(metric_data.get(s, {}).get(tt) is not None for s in active_subsets)]

                if not active_subsets or not active_types:
                    continue

                n_groups = len(active_subsets)
                n_bars = len(active_types)
                bar_width = 0.7 / max(n_bars, 1)
                x = np.arange(n_groups)

                fig, ax = plt.subplots(figsize=(max(6, n_groups * n_bars * 0.9 + 2), 5))

                for i, tt in enumerate(active_types):
                    values = [metric_data.get(s, {}).get(tt) for s in active_subsets]
                    # Replace None with nan so bar is absent
                    heights = [v if v is not None else float("nan") for v in values]
                    offset = (i - (n_bars - 1) / 2) * bar_width
                    bars = ax.bar(
                        x + offset,
                        heights,
                        width=bar_width * 0.9,
                        label=TRAINING_TYPE_LABELS[tt],
                        color=TRAINING_TYPE_COLORS[tt],
                        alpha=0.85,
                        edgecolor="white",
                        linewidth=0.6,
                    )
                    # Value labels on bars
                    for bar, h in zip(bars, heights):
                        if not math.isnan(h):
                            ax.text(
                                bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + ax.get_ylim()[1] * 0.005,
                                f"{h:.3g}",
                                ha="center",
                                va="bottom",
                                fontsize=7,
                                rotation=45,
                            )

                ax.set_xticks(x)
                ax.set_xticklabels(active_subsets)
                ax.set_xlabel("Training subset size")
                ax.set_ylabel(metric_display)
                ax.set_title(f"{ds} — {metric_display} (ref: {ref})")
                ax.legend(title="Training type", bbox_to_anchor=(1.02, 1), loc="upper left",
                          fontsize=8)
                ax.grid(axis="y", linestyle="--", alpha=0.4)
                ax.set_axisbelow(True)

                # Re-do y-limit now that we know bar heights
                valid_heights = [h for h in [v for row in [metric_data.get(s, {}) for s in active_subsets]
                                              for v in row.values() if v is not None]]
                if valid_heights:
                    ymax = max(valid_heights)
                    ax.set_ylim(0, ymax * 1.18)

                plt.tight_layout()

                save_path = out_dir / "comparison" / ds / ref / f"{metric_display}.png"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                written += 1

    return written


# ---------------------------------------------------------------------------
# Trend charts
# ---------------------------------------------------------------------------

def plot_trends(records: list[dict[str, Any]], out_dir: Path) -> int:
    """
    Emit one 2×2 subplot PNG per run showing metric vs checkpoint step.

    Returns the number of PNG files written.
    """
    written = 0
    metric_names = list(METRICS.keys())  # ["FID", "KID", "MMD", "DINO_FD"]

    for rec in records:
        trend = rec["trend"]
        if not trend:
            continue

        ckpt_ids = [row["ckpt_id"] for row in trend]
        x = np.arange(len(ckpt_ids))

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        axes_flat = axes.flatten()

        tt_color = TRAINING_TYPE_COLORS.get(rec["training_type"], "#555555")

        for ax, metric_display in zip(axes_flat, metric_names):
            values = [row["metrics"].get(metric_display) for row in trend]
            valid_mask = [v is not None for v in values]
            y_plot = [v if v is not None else float("nan") for v in values]

            ax.plot(x, y_plot, marker="o", color=tt_color, linewidth=1.8,
                    markersize=5, markerfacecolor="white", markeredgewidth=1.5)

            # Mark selected checkpoint (lowest on trend if selection_metric == metric)
            # Just highlight the minimum for the dominant metric
            valid_vals = [(i, v) for i, v in enumerate(values) if v is not None]
            if valid_vals:
                best_i, best_v = min(valid_vals, key=lambda t: t[1])
                ax.scatter([best_i], [best_v], zorder=5, s=60, color="gold",
                           edgecolors=tt_color, linewidths=1.2)

            ax.set_xticks(x)
            ax.set_xticklabels(ckpt_ids, rotation=40, ha="right", fontsize=7)
            ax.set_ylabel(metric_display, fontsize=9)
            ax.set_title(metric_display, fontsize=10, fontweight="bold")
            ax.grid(linestyle="--", alpha=0.4)
            ax.set_axisbelow(True)

        # Group label in suptitle
        group_label = (f"{rec['dataset']} | subset={rec['subset']} | "
                       f"type={TRAINING_TYPE_LABELS[rec['training_type']]}")
        fig.suptitle(f"{rec['run_name']}\n{group_label}", fontsize=9, y=1.01)
        plt.tight_layout()

        save_path = out_dir / "trends" / rec["dataset"] / f"{rec['run_name']}.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written += 1

    return written


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate metric charts from killarney_scratch checkpoint-selection results."
    )
    parser.add_argument(
        "--root",
        default="killarney_scratch",
        help="Root directory containing the two dataset subdirectories (default: killarney_scratch).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory for charts (default: <root>/_charts).",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        sys.exit(f"Error: root directory does not exist: {root}")

    out_dir = Path(args.out).expanduser().resolve() if args.out else root / "_charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering runs under: {root}")
    records = discover_runs(root)
    print(f"Found {len(records)} runs  "
          f"({sum(1 for r in records if r['status'] == 'complete')} complete, "
          f"{sum(1 for r in records if r['status'] == 'trend_only')} trend-only, "
          f"{sum(1 for r in records if r['status'] == 'final_only')} final-only)")

    print("Generating comparison charts …")
    n_comp = plot_comparison(records, out_dir)
    print(f"  → wrote {n_comp} comparison PNGs")

    print("Generating trend charts …")
    n_trend = plot_trends(records, out_dir)
    print(f"  → wrote {n_trend} trend PNGs")

    print(f"\nDone. Charts saved under: {out_dir}")


if __name__ == "__main__":
    main()

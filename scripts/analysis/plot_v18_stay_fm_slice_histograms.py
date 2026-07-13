"""Plot v18 size-position slice histograms before and after STAY FM augmentation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd

from src.analysis.flir_subgroup.yolo_slice_stats import (
    POSITION_BIN_ORDER,
    SIZE_BIN_ORDER,
    assign_bins_from_thresholds,
    build_slice_counts_table,
    load_yolo_slice_dataset,
)


DEFAULT_SOURCE_YAML = "data/derived/yolo-test-ds_v18/full_train.yaml"
DEFAULT_AUGMENTED_YAML = (
    "artifacts/generated/yolo/exp_b/augmented_yolo/"
    "small_v4_fmaug_nofilter/fm_aug/full_train_synthetic_aug.yaml"
)
DEFAULT_OUTPUT_DIR = "artifacts/analysis/v18_stay_fm_slice_histograms"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Save 27-bucket size-position barplots for the skewed v18 train split "
            "and the STAY FM 1-to-1 no-filter augmented train split."
        )
    )
    parser.add_argument("--source-yaml", default=DEFAULT_SOURCE_YAML)
    parser.add_argument("--augmented-yaml", default=DEFAULT_AUGMENTED_YAML)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--source-output-stem", default="v18_full_train_skewed_27_slice_buckets")
    parser.add_argument("--augmented-output-stem", default="v18_stay_fm_1to1_nofilter_27_slice_buckets")
    parser.add_argument("--source-counts-stem", default="v18_full_train_slice_counts")
    parser.add_argument("--augmented-counts-stem", default="v18_stay_fm_1to1_nofilter_slice_counts")
    parser.add_argument("--source-title", default="v18 full_train: original skewed slice histogram")
    parser.add_argument(
        "--augmented-title",
        default="v18 full_train + STAY FM 1-to-1 no-filter: slice histogram",
    )
    return parser.parse_args()


def _slice_labels() -> list[str]:
    return [f"{size}\n{position}" for size in SIZE_BIN_ORDER for position in POSITION_BIN_ORDER]


def _counts_by_size_position(
    dataset_yaml: str | Path,
    *,
    q33: float,
    q67: float,
) -> pd.DataFrame:
    dataset = load_yolo_slice_dataset(dataset_yaml)
    rows = dataset.instance_df.copy()
    if not rows.empty:
        rows["size_bin"] = assign_bins_from_thresholds(rows["bbox_area_ratio"], q33, q67)
    class_labels = [dataset.names[idx] for idx in sorted(dataset.names)] or ["person"]
    counts = build_slice_counts_table(rows, class_labels)
    counts = (
        counts.groupby(["size_bin", "position_bin"], observed=False)["count"]
        .sum()
        .reindex(
            pd.MultiIndex.from_product(
                [SIZE_BIN_ORDER, POSITION_BIN_ORDER],
                names=["size_bin", "position_bin"],
            ),
            fill_value=0,
        )
        .rename("count")
        .reset_index()
    )
    counts["bucket"] = [f"{row.size_bin}_{row.position_bin}" for row in counts.itertuples(index=False)]
    return counts


def _augmented_counts_by_size_position(
    dataset_yaml: str | Path,
    *,
    q33: float,
    q67: float,
) -> pd.DataFrame:
    dataset = load_yolo_slice_dataset(dataset_yaml)
    rows = dataset.instance_df.copy()
    if not rows.empty:
        rows["size_bin"] = assign_bins_from_thresholds(rows["bbox_area_ratio"], q33, q67)
        rows["source"] = "real"
        rows.loc[rows["image_stem"].astype(str).str.startswith("synthetic_"), "source"] = "generated"
        rows.loc[rows["image_stem"].astype(str).str.startswith("real_"), "source"] = "real"

    dense_index = pd.MultiIndex.from_product(
        [SIZE_BIN_ORDER, POSITION_BIN_ORDER],
        names=["size_bin", "position_bin"],
    )
    if rows.empty:
        counts = pd.DataFrame(index=dense_index).reset_index()
        counts["real_count"] = 0
        counts["generated_count"] = 0
    else:
        counts = (
            rows.groupby(["size_bin", "position_bin", "source"], observed=False)
            .size()
            .rename("count")
            .reset_index()
            .pivot_table(
                index=["size_bin", "position_bin"],
                columns="source",
                values="count",
                aggfunc="sum",
                fill_value=0,
                observed=False,
            )
            .reindex(dense_index, fill_value=0)
            .rename(columns={"real": "real_count", "generated": "generated_count"})
            .reset_index()
        )
        for column in ("real_count", "generated_count"):
            if column not in counts:
                counts[column] = 0

    counts["real_count"] = counts["real_count"].astype(int)
    counts["generated_count"] = counts["generated_count"].astype(int)
    counts["count"] = counts["real_count"] + counts["generated_count"]
    counts["bucket"] = [f"{row.size_bin}_{row.position_bin}" for row in counts.itertuples(index=False)]
    return counts


def _plot_counts(counts: pd.DataFrame, *, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x = range(len(counts))
    bars = ax.bar(x, counts["count"].astype(int), color="#3b82f6", edgecolor="#1f2937", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel("Slice bucket (size x position)")
    ax.set_ylabel("Instance count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(_slice_labels(), rotation=70, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.bar_label(bars, labels=[str(int(v)) for v in counts["count"]], padding=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_stacked_counts(counts: pd.DataFrame, *, title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6))
    x = range(len(counts))
    real = counts["real_count"].astype(int)
    generated = counts["generated_count"].astype(int)
    real_bars = ax.bar(x, real, color="#2563eb", edgecolor="#1f2937", linewidth=0.5, label="Real")
    generated_bars = ax.bar(
        x,
        generated,
        bottom=real,
        color="#dc2626",
        edgecolor="#1f2937",
        linewidth=0.5,
        label="Generated",
    )
    ax.set_title(title)
    ax.set_xlabel("Slice bucket (size x position)")
    ax.set_ylabel("Instance count")
    ax.set_xticks(list(x))
    ax.set_xticklabels(_slice_labels(), rotation=70, ha="right", fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    ax.bar_label(real_bars, labels=[str(v) if v else "" for v in real], label_type="center", fontsize=7, color="white")
    ax.bar_label(
        generated_bars,
        labels=[str(v) if v else "" for v in generated],
        label_type="center",
        fontsize=7,
        color="white",
    )
    ax.bar_label(
        generated_bars,
        labels=[str(int(v)) for v in counts["count"]],
        padding=2,
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dataset = load_yolo_slice_dataset(args.source_yaml)
    q33 = source_dataset.thresholds.q33
    q67 = source_dataset.thresholds.q67

    source_counts = _counts_by_size_position(args.source_yaml, q33=q33, q67=q67)
    augmented_counts = _augmented_counts_by_size_position(args.augmented_yaml, q33=q33, q67=q67)

    source_counts.to_csv(output_dir / f"{args.source_counts_stem}.csv", index=False)
    augmented_counts.to_csv(output_dir / f"{args.augmented_counts_stem}.csv", index=False)

    _plot_counts(
        source_counts,
        title=args.source_title,
        output_path=output_dir / f"{args.source_output_stem}.png",
    )
    _plot_stacked_counts(
        augmented_counts,
        title=args.augmented_title,
        output_path=output_dir / f"{args.augmented_output_stem}.png",
    )

    print(f"Saved charts and count CSVs under: {output_dir.resolve()}")
    print(f"Frozen v18 size thresholds: q33={q33:.8f}, q67={q67:.8f}")
    print(f"Original instances: {int(source_counts['count'].sum())}")
    print(f"Augmented instances: {int(augmented_counts['count'].sum())}")


if __name__ == "__main__":
    main()

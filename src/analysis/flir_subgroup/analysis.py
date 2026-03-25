"""Notebook-equivalent subgroup analysis helpers for the FLIR app."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from src.analysis.flir_subgroup.constants import (
    DEFAULT_DOMINANCE_HISTOGRAM_BINS,
    DEFAULT_EXAMPLE_COUNT,
    DOMINANCE_THRESHOLDS,
    FEASIBILITY_RULES,
    FIXED_SIZE_BINS,
    MAX_EXAMPLE_SUBGROUPS,
    POSITION_MODE,
    SIZE_BIN_LABELS,
    SIZE_BIN_METHOD,
)


def canonical_subgroup_label(
    class_label: str,
    size_bin: str,
    position_bin: str | None = None,
) -> str:
    """Build the notebook-style canonical subgroup label."""

    components = [f"class={class_label}", f"size={size_bin}"]
    if position_bin is not None:
        components.append(f"pos={position_bin}")
    return " | ".join(components)


def parse_subgroup_label(subgroup: str) -> Dict[str, Optional[str]]:
    """Parse a canonical subgroup label into structured fields."""

    result: Dict[str, Optional[str]] = {"class_label": None, "size_bin": None, "position_bin": None}
    for part in subgroup.split("|"):
        cleaned = part.strip()
        key, value = cleaned.split("=", 1)
        if key == "class":
            result["class_label"] = value
        elif key == "size":
            result["size_bin"] = value
        elif key == "pos":
            result["position_bin"] = value
    return result


def assign_size_bins(
    instance_df: pd.DataFrame,
    *,
    method: str = SIZE_BIN_METHOD,
    labels: Sequence[str] = SIZE_BIN_LABELS,
    fixed_bins: Optional[Sequence[float]] = FIXED_SIZE_BINS,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Assign notebook-equivalent size bins."""

    df = instance_df.copy()
    values = df["bbox_area_norm"].astype(float).clip(lower=0.0)

    if method == "quantile":
        ranked = values.rank(method="first")
        df["size_bin"] = pd.qcut(ranked, q=len(labels), labels=list(labels))
        bin_spec = (
            df.groupby("size_bin", observed=True)["bbox_area_norm"]
            .agg(bin_min="min", bin_max="max", n_instances="size")
            .reset_index()
        )
        bin_spec["method"] = method
    elif method == "fixed":
        if fixed_bins is None:
            raise ValueError("Fixed size bins were requested but `fixed_bins` is None.")
        if len(fixed_bins) != len(labels) + 1:
            raise ValueError("Fixed size bins must have len(labels) + 1 edges.")
        df["size_bin"] = pd.cut(values, bins=fixed_bins, labels=list(labels), include_lowest=True)
        bin_spec = pd.DataFrame(
            {
                "size_bin": list(labels),
                "bin_min": list(fixed_bins[:-1]),
                "bin_max": list(fixed_bins[1:]),
                "method": method,
            }
        )
        counts = df["size_bin"].value_counts(dropna=False).rename_axis("size_bin").reset_index(name="n_instances")
        bin_spec = bin_spec.merge(counts, on="size_bin", how="left")
    else:
        raise ValueError(f"Unsupported size bin method: {method}")

    return df, bin_spec


def add_position_columns(instance_df: pd.DataFrame) -> pd.DataFrame:
    """Add notebook-equivalent position bins."""

    df = instance_df.copy()
    x = df["bbox_center_x_norm"].astype(float).clip(0.0, 1.0)
    y = df["bbox_center_y_norm"].astype(float).clip(0.0, 1.0)

    horizontal_labels = ["left", "center", "right"]
    vertical_labels = ["top", "middle", "bottom"]
    bins = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]

    df["position_bin_horizontal"] = pd.cut(x, bins=bins, labels=horizontal_labels, include_lowest=True)
    df["position_bin_vertical"] = pd.cut(y, bins=bins, labels=vertical_labels, include_lowest=True)
    df["position_bin_grid"] = df["position_bin_vertical"].astype(str) + "_" + df["position_bin_horizontal"].astype(str)

    center_distance = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
    df["center_distance_norm"] = center_distance / math.sqrt(0.5**2 + 0.5**2)
    return df


def build_subgroup_labels(
    instance_df: pd.DataFrame,
    *,
    include_position: bool = False,
    position_mode: str = POSITION_MODE,
) -> pd.DataFrame:
    """Create notebook-style subgroup labels."""

    df = instance_df.copy()
    components = ["class=" + df["class_label"].astype(str), "size=" + df["size_bin"].astype(str)]

    if include_position:
        if position_mode == "horizontal":
            position_values = df["position_bin_horizontal"].astype(str)
        elif position_mode == "grid_3x3":
            position_values = df["position_bin_grid"].astype(str)
        else:
            raise ValueError(f"Unsupported position mode: {position_mode}")
        components.append("pos=" + position_values)

    subgroup = components[0]
    for component in components[1:]:
        subgroup = subgroup + " | " + component
    df["subgroup"] = subgroup
    return df


def compute_image_level_subgroup_stats(
    image_df: pd.DataFrame,
    subgroup_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute notebook-equivalent image-level subgroup statistics."""

    image_base = image_df.copy()

    if subgroup_df.empty:
        empty_subgroups = pd.DataFrame(
            columns=[
                "image_key",
                "split",
                "image_id",
                "subgroup",
                "subgroup_count",
                "total_object_count",
                "dominance_ratio",
                "image_density",
                "is_dominant",
            ]
        )
        image_base["total_object_count"] = 0
        image_base["n_subgroups_present"] = 0
        image_base["total_bbox_area"] = 0.0
        image_base["image_density"] = 0.0
        image_base["dominant_subgroup"] = np.nan
        image_base["dominant_subgroup_count"] = 0
        image_base["dominant_ratio"] = 0.0
        image_base["dominant_tie_count"] = 0
        return image_base, empty_subgroups

    image_totals = (
        subgroup_df.groupby("image_key")
        .agg(total_object_count=("ann_id", "size"), n_subgroups_present=("subgroup", "nunique"), total_bbox_area=("bbox_area", "sum"))
        .reset_index()
    )
    image_totals["image_density"] = (
        image_totals["total_bbox_area"] / image_base.set_index("image_key").loc[image_totals["image_key"], "image_area"].to_numpy()
    )

    image_subgroup_df = (
        subgroup_df.groupby(["image_key", "split", "image_id", "subgroup"]).size().rename("subgroup_count").reset_index()
    )
    image_subgroup_df = image_subgroup_df.merge(image_totals, on="image_key", how="left")
    image_subgroup_df["dominance_ratio"] = image_subgroup_df["subgroup_count"] / image_subgroup_df["total_object_count"]

    max_count = image_subgroup_df.groupby("image_key")["subgroup_count"].transform("max")
    image_subgroup_df["is_at_max_count"] = image_subgroup_df["subgroup_count"] == max_count
    dominant_tie_count = image_subgroup_df.groupby("image_key")["is_at_max_count"].sum().rename("dominant_tie_count")

    dominant_rows = (
        image_subgroup_df.sort_values(
            ["image_key", "subgroup_count", "dominance_ratio", "subgroup"],
            ascending=[True, False, False, True],
        )
        .groupby("image_key")
        .head(1)
        .rename(
            columns={
                "subgroup": "dominant_subgroup",
                "subgroup_count": "dominant_subgroup_count",
                "dominance_ratio": "dominant_ratio",
            }
        )[["image_key", "dominant_subgroup", "dominant_subgroup_count", "dominant_ratio"]]
    )

    image_subgroup_df = image_subgroup_df.merge(dominant_rows[["image_key", "dominant_subgroup"]], on="image_key", how="left")
    image_subgroup_df["is_dominant"] = image_subgroup_df["subgroup"] == image_subgroup_df["dominant_subgroup"]

    image_stats_df = image_base.merge(image_totals, on="image_key", how="left")
    image_stats_df = image_stats_df.merge(dominant_rows, on="image_key", how="left")
    image_stats_df = image_stats_df.merge(dominant_tie_count.reset_index(), on="image_key", how="left")

    fill_zero_cols = [
        "total_object_count",
        "n_subgroups_present",
        "total_bbox_area",
        "image_density",
        "dominant_subgroup_count",
        "dominant_ratio",
        "dominant_tie_count",
    ]
    for column in fill_zero_cols:
        image_stats_df[column] = image_stats_df[column].fillna(0)

    return image_stats_df, image_subgroup_df


def build_analysis_tables(
    image_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    *,
    size_bin_method: str = SIZE_BIN_METHOD,
    size_bin_labels: Sequence[str] = SIZE_BIN_LABELS,
    fixed_size_bins: Optional[Sequence[float]] = FIXED_SIZE_BINS,
    include_position: bool = False,
    position_mode: str = POSITION_MODE,
) -> Dict[str, pd.DataFrame]:
    """Build notebook-equivalent phase analysis tables."""

    sized_df, size_bin_spec_df = assign_size_bins(instance_df, method=size_bin_method, labels=size_bin_labels, fixed_bins=fixed_size_bins)
    positioned_df = add_position_columns(sized_df)
    subgroup_df = build_subgroup_labels(positioned_df, include_position=include_position, position_mode=position_mode)
    image_stats_df, image_subgroup_df = compute_image_level_subgroup_stats(image_df, subgroup_df)
    return {
        "size_bin_spec_df": size_bin_spec_df,
        "instance_df": subgroup_df,
        "image_stats_df": image_stats_df,
        "image_subgroup_df": image_subgroup_df,
    }


def build_subgroup_frequency_table(instance_df: pd.DataFrame, image_subgroup_df: pd.DataFrame) -> pd.DataFrame:
    """Compute subgroup support by instance and image counts."""

    n_instances = instance_df.groupby("subgroup").size().rename("n_instances")
    n_images = image_subgroup_df.groupby("subgroup")["image_key"].nunique().rename("n_images")
    avg_present_count = image_subgroup_df.groupby("subgroup")["subgroup_count"].mean().rename("avg_instances_per_image_present")
    freq_df = pd.concat([n_instances, n_images, avg_present_count], axis=1).reset_index()
    return freq_df.sort_values(["n_instances", "n_images"], ascending=False).reset_index(drop=True)


def build_dominance_summary(image_subgroup_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize dominance ratios by subgroup."""

    summary = (
        image_subgroup_df.groupby("subgroup")["dominance_ratio"]
        .agg(
            mean_dominance="mean",
            median_dominance="median",
            q75_dominance=lambda series: float(series.quantile(0.75)),
            q90_dominance=lambda series: float(series.quantile(0.90)),
            n_images="size",
        )
        .reset_index()
    )
    return summary.sort_values(["median_dominance", "n_images"], ascending=False).reset_index(drop=True)


def choose_default_subgroup(freq_df: pd.DataFrame, preferred: Optional[str] = None, min_images: int = 20) -> str:
    """Choose the notebook-equivalent default subgroup."""

    if preferred is not None:
        if preferred not in set(freq_df["subgroup"]):
            raise ValueError(f"Requested subgroup not found: {preferred}")
        return preferred
    candidates = freq_df.loc[freq_df["n_images"] >= min_images]
    if candidates.empty:
        candidates = freq_df
    return str(candidates.sort_values(["n_images", "n_instances"], ascending=False).iloc[0]["subgroup"])


def compute_holdout_table(
    image_subgroup_df: pd.DataFrame,
    image_stats_df: pd.DataFrame,
    subgroup: str,
    thresholds: Sequence[float] = DOMINANCE_THRESHOLDS,
) -> pd.DataFrame:
    """Compute held-out size versus tau for one subgroup."""

    subgroup_rows = image_subgroup_df.loc[image_subgroup_df["subgroup"] == subgroup].copy()
    total_images = max(image_stats_df["image_key"].nunique(), 1)
    rows: List[dict] = []
    for tau in thresholds:
        heldout = subgroup_rows.loc[(subgroup_rows["subgroup_count"] >= 1) & (subgroup_rows["dominance_ratio"] >= tau)].copy()
        rows.append(
            {
                "subgroup": subgroup,
                "tau": float(tau),
                "heldout_n_images": int(heldout["image_key"].nunique()),
                "heldout_fraction": float(heldout["image_key"].nunique() / total_images),
                "mean_target_count": float(heldout["subgroup_count"].mean()) if not heldout.empty else 0.0,
                "median_target_count": float(heldout["subgroup_count"].median()) if not heldout.empty else 0.0,
                "mean_dominance": float(heldout["dominance_ratio"].mean()) if not heldout.empty else 0.0,
            }
        )
    return pd.DataFrame(rows)


def get_holdout_image_keys(image_subgroup_df: pd.DataFrame, subgroup: str, tau: float) -> List[str]:
    """Return held-out image keys for one subgroup at ``tau``."""

    heldout = image_subgroup_df.loc[
        (image_subgroup_df["subgroup"] == subgroup)
        & (image_subgroup_df["subgroup_count"] >= 1)
        & (image_subgroup_df["dominance_ratio"] >= tau)
    ]
    return sorted(heldout["image_key"].unique().tolist())


def get_union_holdout_image_keys(image_subgroup_df: pd.DataFrame, subgroups: Sequence[str], tau: float) -> List[str]:
    """Return the union hold-out set across selected subgroups."""

    heldout_keys: set[str] = set()
    for subgroup in subgroups:
        heldout_keys.update(get_holdout_image_keys(image_subgroup_df, subgroup, tau))
    return sorted(heldout_keys)


def compute_collateral_damage(
    instance_df: pd.DataFrame,
    image_subgroup_df: pd.DataFrame,
    subgroup: str,
    tau: float,
) -> Tuple[pd.DataFrame, dict]:
    """Compute notebook-equivalent collateral damage for one subgroup."""

    heldout_keys = set(get_holdout_image_keys(image_subgroup_df, subgroup, tau))
    retained_instance_df = instance_df.loc[~instance_df["image_key"].isin(heldout_keys)].copy()

    before = instance_df.groupby("subgroup").size().rename("count_before")
    after = retained_instance_df.groupby("subgroup").size().rename("count_after")
    damage_df = pd.concat([before, after], axis=1).fillna(0).reset_index()
    damage_df["count_before"] = damage_df["count_before"].astype(int)
    damage_df["count_after"] = damage_df["count_after"].astype(int)
    damage_df["count_loss"] = damage_df["count_before"] - damage_df["count_after"]
    damage_df["loss_fraction"] = damage_df["count_loss"] / damage_df["count_before"].replace(0, np.nan)
    damage_df["loss_fraction"] = damage_df["loss_fraction"].fillna(0.0)
    damage_df["is_target_subgroup"] = damage_df["subgroup"] == subgroup
    damage_df = damage_df.sort_values(["loss_fraction", "count_loss"], ascending=False).reset_index(drop=True)

    other_mask = ~damage_df["is_target_subgroup"]
    collateral_other_loss_frac = float(
        damage_df.loc[other_mask, "count_loss"].sum() / max(damage_df.loc[other_mask, "count_before"].sum(), 1)
    )
    summary = {
        "subgroup": subgroup,
        "tau": float(tau),
        "heldout_n_images": len(heldout_keys),
        "collateral_other_loss_frac": collateral_other_loss_frac,
    }
    return damage_df, summary


def compare_partitions_from_holdout_keys(
    image_stats_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    heldout_keys: Iterable[str],
    *,
    selected_subgroups: Sequence[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Build train-vs-held-out comparisons from an explicit hold-out set."""

    heldout_keys_set = set(heldout_keys)
    selected_subgroups = list(selected_subgroups or [])

    image_partition_df = image_stats_df.copy()
    image_partition_df["partition"] = np.where(image_partition_df["image_key"].isin(heldout_keys_set), "held_out", "train")

    instance_partition_df = instance_df.copy()
    instance_partition_df["partition"] = np.where(instance_partition_df["image_key"].isin(heldout_keys_set), "held_out", "train")

    numeric_summary_df = (
        image_partition_df.groupby("partition")
        .agg(
            n_images=("image_key", "nunique"),
            mean_total_object_count=("total_object_count", "mean"),
            median_total_object_count=("total_object_count", "median"),
            mean_density=("image_density", "mean"),
            median_density=("image_density", "median"),
        )
        .reset_index()
    )

    class_distribution_df = instance_partition_df.groupby(["partition", "class_label"]).size().rename("count").reset_index()
    class_distribution_df["fraction"] = class_distribution_df.groupby("partition")["count"].transform(lambda s: s / max(s.sum(), 1))

    subgroup_distribution_df = instance_partition_df.groupby(["partition", "subgroup"]).size().rename("count").reset_index()
    subgroup_distribution_df["fraction"] = subgroup_distribution_df.groupby("partition")["count"].transform(lambda s: s / max(s.sum(), 1))

    class_image_distribution_df = (
        instance_partition_df.groupby(["partition", "class_label"])["image_key"].nunique().rename("n_images").reset_index()
    )
    class_image_distribution_df["fraction"] = class_image_distribution_df.groupby("partition")["n_images"].transform(
        lambda s: s / max(s.sum(), 1)
    )

    if selected_subgroups:
        target_images = (
            instance_partition_df.loc[instance_partition_df["subgroup"].isin(selected_subgroups), ["image_key", "partition"]]
            .drop_duplicates()
        )
        cooccurring_df = instance_partition_df.merge(target_images, on=["image_key", "partition"], how="inner")
        cooccurring_df = cooccurring_df.groupby(["partition", "class_label"]).size().rename("count").reset_index()
        cooccurring_df["fraction"] = cooccurring_df.groupby("partition")["count"].transform(lambda s: s / max(s.sum(), 1))
    else:
        cooccurring_df = pd.DataFrame(columns=["partition", "class_label", "count", "fraction"])

    return {
        "image_partition_df": image_partition_df,
        "instance_partition_df": instance_partition_df,
        "numeric_summary_df": numeric_summary_df,
        "class_distribution_df": class_distribution_df,
        "class_image_distribution_df": class_image_distribution_df,
        "subgroup_distribution_df": subgroup_distribution_df,
        "cooccurring_class_distribution_df": cooccurring_df,
    }


def build_per_class_image_count_distribution(
    instance_df: pd.DataFrame,
    heldout_keys: Iterable[str],
    *,
    include_zero_counts: bool = False,
) -> pd.DataFrame:
    """Reproduce the notebook's per-class before/after image-count distribution."""

    heldout_keys_set = set(heldout_keys)

    image_class_counts = instance_df.groupby(["image_key", "class_label"]).size().rename("instance_count").reset_index()
    all_images = pd.DataFrame({"image_key": sorted(instance_df["image_key"].drop_duplicates().tolist())})
    all_classes = pd.DataFrame({"class_label": sorted(instance_df["class_label"].dropna().unique().tolist())})

    if include_zero_counts:
        full_grid = all_images.assign(_tmp=1).merge(all_classes.assign(_tmp=1), on="_tmp").drop(columns="_tmp")
        image_class_counts = full_grid.merge(image_class_counts, on=["image_key", "class_label"], how="left").fillna({"instance_count": 0})
        image_class_counts["instance_count"] = image_class_counts["instance_count"].astype(int)

    before_dist = (
        image_class_counts.groupby(["class_label", "instance_count"])["image_key"].nunique().rename("n_images_before").reset_index()
    )
    after_dist = (
        image_class_counts.loc[~image_class_counts["image_key"].isin(heldout_keys_set)]
        .groupby(["class_label", "instance_count"])["image_key"]
        .nunique()
        .rename("n_images_after")
        .reset_index()
    )

    count_dist_df = (
        before_dist.merge(after_dist, on=["class_label", "instance_count"], how="outer")
        .fillna(0)
        .sort_values(["class_label", "instance_count"])
        .reset_index(drop=True)
    )
    count_dist_df["n_images_before"] = count_dist_df["n_images_before"].astype(int)
    count_dist_df["n_images_after"] = count_dist_df["n_images_after"].astype(int)
    return count_dist_df


def build_dominant_group_summary(image_stats_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize dominant subgroup assignments."""

    dominant_frequency_df = (
        image_stats_df.loc[image_stats_df["total_object_count"] > 0, "dominant_subgroup"]
        .value_counts(dropna=False)
        .rename_axis("dominant_subgroup")
        .reset_index(name="n_images")
    )
    overview_df = pd.DataFrame(
        [
            {
                "avg_subgroup_memberships_per_image": float(image_stats_df["n_subgroups_present"].mean()),
                "prop_images_with_more_than_one_subgroup": float((image_stats_df["n_subgroups_present"] > 1).mean()),
                "prop_images_with_tied_dominant_count": float((image_stats_df["dominant_tie_count"] > 1).mean()),
            }
        ]
    )
    return dominant_frequency_df, overview_df


def build_feasibility_table(
    freq_df: pd.DataFrame,
    dominance_df: pd.DataFrame,
    image_subgroup_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    tau: float = 0.5,
) -> pd.DataFrame:
    """Compute the notebook's phase-1 feasibility summary."""

    rows: List[dict] = []
    for subgroup in freq_df["subgroup"].tolist():
        holdout_keys = get_holdout_image_keys(image_subgroup_df, subgroup, tau)
        _, collateral_summary = compute_collateral_damage(instance_df, image_subgroup_df, subgroup, tau)
        freq_row = freq_df.loc[freq_df["subgroup"] == subgroup].iloc[0]
        dom_row = dominance_df.loc[dominance_df["subgroup"] == subgroup].iloc[0]

        benchmark_feasible = (
            int(freq_row["n_instances"]) >= FEASIBILITY_RULES["min_instances"]
            and int(freq_row["n_images"]) >= FEASIBILITY_RULES["min_images"]
            and float(dom_row["median_dominance"]) >= FEASIBILITY_RULES["min_median_dominance"]
            and len(holdout_keys) >= FEASIBILITY_RULES["min_holdout_images_tau_0_5"]
            and float(collateral_summary["collateral_other_loss_frac"]) <= FEASIBILITY_RULES["max_collateral_other_loss_frac_tau_0_5"]
        )

        rows.append(
            {
                "subgroup": subgroup,
                "n_instances": int(freq_row["n_instances"]),
                "n_images": int(freq_row["n_images"]),
                "median_dominance": float(dom_row["median_dominance"]),
                "heldout_size_tau_0_5": int(len(holdout_keys)),
                "collateral_other_loss_frac_tau_0_5": float(collateral_summary["collateral_other_loss_frac"]),
                "benchmark_feasible": bool(benchmark_feasible),
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(["benchmark_feasible", "heldout_size_tau_0_5", "median_dominance", "n_images"], ascending=[False, False, False, False])
        .reset_index(drop=True)
    )


def build_selectable_groups(freq_df: pd.DataFrame, dominance_df: pd.DataFrame) -> pd.DataFrame:
    """Build structured subgroup options for the API and UI."""

    selectable_df = freq_df.merge(dominance_df[["subgroup", "median_dominance"]], on="subgroup", how="left")
    parsed_df = pd.DataFrame([parse_subgroup_label(label) for label in selectable_df["subgroup"].tolist()])
    selectable_df = pd.concat([parsed_df, selectable_df], axis=1)
    selectable_df = selectable_df.rename(columns={"subgroup": "subgroup_label"})
    return selectable_df


def build_dominance_histogram(
    image_subgroup_df: pd.DataFrame,
    subgroup: str,
    *,
    bins: int = DEFAULT_DOMINANCE_HISTOGRAM_BINS,
) -> pd.DataFrame:
    """Return chart-ready dominance histogram data for one subgroup."""

    subgroup_rows = image_subgroup_df.loc[image_subgroup_df["subgroup"] == subgroup].copy()
    values = subgroup_rows["dominance_ratio"].astype(float).to_numpy()
    if values.size == 0:
        return pd.DataFrame(columns=["bin_start", "bin_end", "count", "bin_label"])

    counts, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    rows: List[dict] = []
    for idx, count in enumerate(counts):
        rows.append(
            {
                "bin_start": float(edges[idx]),
                "bin_end": float(edges[idx + 1]),
                "count": int(count),
                "bin_label": f"{edges[idx]:.2f}-{edges[idx + 1]:.2f}",
            }
        )
    return pd.DataFrame(rows)


def _select_quantile_rows(df: pd.DataFrame, value_col: str, count: int) -> pd.DataFrame:
    """Select deterministic example rows spanning a value distribution."""

    if df.empty or count <= 0:
        return df.head(0).copy()

    unique_df = df.drop_duplicates("image_key").sort_values([value_col, "image_key"]).reset_index(drop=True)
    if len(unique_df) <= count:
        return unique_df.copy()

    if count == 1:
        quantiles = [0.5]
    else:
        quantiles = np.linspace(0.2, 0.8, count).tolist()

    targets = unique_df[value_col].quantile(quantiles).to_numpy()
    selected_indices: List[int] = []
    for target in targets:
        candidate_idx = (unique_df[value_col] - target).abs().sort_values().index
        chosen_idx = next((idx for idx in candidate_idx if idx not in selected_indices), int(candidate_idx[0]))
        selected_indices.append(chosen_idx)

    return unique_df.loc[selected_indices].copy().reset_index(drop=True)


def _build_class_level_example_rows(
    image_stats_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    class_label: str,
    *,
    excluded_keys: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Build per-image rows for same-class fallback example selection."""

    excluded = set(excluded_keys or [])
    class_counts = (
        instance_df.loc[instance_df["class_label"] == class_label]
        .groupby(["image_key", "split", "image_id"])
        .size()
        .rename("class_count")
        .reset_index()
    )
    if class_counts.empty:
        return class_counts

    out = class_counts.merge(image_stats_df[["image_key", "total_object_count", "dominant_ratio"]], on="image_key", how="left")
    out["class_ratio"] = out["class_count"] / out["total_object_count"].replace(0, np.nan)
    out["class_ratio"] = out["class_ratio"].fillna(0.0)
    if excluded:
        out = out.loc[~out["image_key"].isin(excluded)].copy()
    return out


def select_examples_for_group(
    image_stats_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    image_subgroup_df: pd.DataFrame,
    subgroup: str,
    tau: float,
    *,
    example_count: int = DEFAULT_EXAMPLE_COUNT,
) -> Dict[str, pd.DataFrame]:
    """Select held-out and retained examples for one subgroup."""

    heldout_keys = set(get_holdout_image_keys(image_subgroup_df, subgroup, tau))
    subgroup_rows = image_subgroup_df.loc[image_subgroup_df["subgroup"] == subgroup].copy()

    heldout_rows = subgroup_rows.loc[subgroup_rows["image_key"].isin(heldout_keys)].copy()
    heldout_examples = _select_quantile_rows(heldout_rows, "dominance_ratio", example_count)
    heldout_examples["partition"] = "held_out"
    heldout_examples["selection_source"] = "target_subgroup"

    retained_exact_rows = subgroup_rows.loc[~subgroup_rows["image_key"].isin(heldout_keys)].copy()
    retained_exact_examples = _select_quantile_rows(retained_exact_rows, "dominance_ratio", min(example_count, len(retained_exact_rows)))
    retained_exact_examples["partition"] = "train"
    retained_exact_examples["selection_source"] = "exact_subgroup"

    remaining_slots = max(example_count - len(retained_exact_examples), 0)
    retained_examples = retained_exact_examples.copy()
    fallback_examples = pd.DataFrame()

    if remaining_slots > 0:
        parsed = parse_subgroup_label(subgroup)
        class_label = parsed["class_label"]
        fallback_rows = _build_class_level_example_rows(
            image_stats_df,
            instance_df,
            class_label or "",
            excluded_keys=heldout_keys.union(set(retained_exact_examples["image_key"].tolist())),
        )
        fallback_examples = _select_quantile_rows(fallback_rows, "class_ratio", remaining_slots)
        if not fallback_examples.empty:
            fallback_examples["partition"] = "train"
            fallback_examples["selection_source"] = "same_class_fallback"
            fallback_examples["subgroup"] = subgroup
            fallback_examples["subgroup_count"] = 0
            fallback_examples["dominance_ratio"] = 0.0
        retained_examples = pd.concat([retained_examples, fallback_examples], ignore_index=True, sort=False)

    return {
        "held_out_examples_df": heldout_examples.reset_index(drop=True),
        "retained_examples_df": retained_examples.reset_index(drop=True),
    }


def build_example_boxes(
    instance_df: pd.DataFrame,
    image_key: str,
    subgroup: str,
) -> List[dict]:
    """Build annotation overlays for one image."""

    parsed = parse_subgroup_label(subgroup)
    target_class = parsed["class_label"]
    image_instances = instance_df.loc[instance_df["image_key"] == image_key].copy()
    rows: List[dict] = []
    for row in image_instances.itertuples(index=False):
        rows.append(
            {
                "ann_id": int(row.ann_id),
                "class_label": str(row.class_label),
                "subgroup_label": str(row.subgroup),
                "bbox_x": float(row.bbox_x),
                "bbox_y": float(row.bbox_y),
                "bbox_w": float(row.bbox_w),
                "bbox_h": float(row.bbox_h),
                "is_target_subgroup": bool(row.subgroup == subgroup),
                "is_target_class": bool(row.class_label == target_class),
            }
        )
    return rows


@dataclass
class PhaseAnalysisBundle:
    """All precomputed tables for one analysis phase."""

    phase: str
    include_position: bool
    position_mode: str
    size_bin_spec_df: pd.DataFrame
    instance_df: pd.DataFrame
    image_stats_df: pd.DataFrame
    image_subgroup_df: pd.DataFrame
    subgroup_frequency_df: pd.DataFrame
    dominance_summary_df: pd.DataFrame
    selectable_groups_df: pd.DataFrame
    dominant_group_frequency_df: pd.DataFrame
    dominant_group_overview_df: pd.DataFrame
    feasibility_df: pd.DataFrame
    default_subgroup_label: str
    example_subgroup_labels: List[str]


def build_phase_bundle(
    phase: str,
    image_df: pd.DataFrame,
    instance_df: pd.DataFrame,
    *,
    include_position: bool,
    position_mode: str = POSITION_MODE,
) -> PhaseAnalysisBundle:
    """Build all cached tables for one phase."""

    tables = build_analysis_tables(
        image_df,
        instance_df,
        size_bin_method=SIZE_BIN_METHOD,
        size_bin_labels=SIZE_BIN_LABELS,
        fixed_size_bins=FIXED_SIZE_BINS,
        include_position=include_position,
        position_mode=position_mode,
    )
    subgroup_frequency_df = build_subgroup_frequency_table(tables["instance_df"], tables["image_subgroup_df"])
    dominance_summary_df = build_dominance_summary(tables["image_subgroup_df"])
    selectable_groups_df = build_selectable_groups(subgroup_frequency_df, dominance_summary_df)
    dominant_group_frequency_df, dominant_group_overview_df = build_dominant_group_summary(tables["image_stats_df"])
    feasibility_df = build_feasibility_table(
        subgroup_frequency_df,
        dominance_summary_df,
        tables["image_subgroup_df"],
        tables["instance_df"],
        tau=0.5,
    )

    return PhaseAnalysisBundle(
        phase=phase,
        include_position=include_position,
        position_mode=position_mode,
        size_bin_spec_df=tables["size_bin_spec_df"],
        instance_df=tables["instance_df"],
        image_stats_df=tables["image_stats_df"],
        image_subgroup_df=tables["image_subgroup_df"],
        subgroup_frequency_df=subgroup_frequency_df,
        dominance_summary_df=dominance_summary_df,
        selectable_groups_df=selectable_groups_df,
        dominant_group_frequency_df=dominant_group_frequency_df,
        dominant_group_overview_df=dominant_group_overview_df,
        feasibility_df=feasibility_df,
        default_subgroup_label=choose_default_subgroup(subgroup_frequency_df),
        example_subgroup_labels=subgroup_frequency_df["subgroup"].head(MAX_EXAMPLE_SUBGROUPS).tolist(),
    )

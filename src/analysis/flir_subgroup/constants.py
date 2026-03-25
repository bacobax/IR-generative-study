"""Shared constants for the subgroup analysis app."""

from __future__ import annotations

from pathlib import Path

from src.core.paths import repo_root


REPO_ROOT = repo_root()
DEFAULT_DATASET_ID = "flir_private_proxy_alignment_v18"

PREFERRED_SPLITS = ("train", "val", "test")
ANALYSIS_SPLITS = ("train",)

SIZE_BIN_METHOD = "quantile"
SIZE_BIN_LABELS = ("small", "medium", "large")
FIXED_SIZE_BINS = None

POSITION_MODE = "horizontal"
POSITION_BIN_LABELS = ("left", "center", "right")
POSITION_BIN_EDGES = (0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0)
POSITION_GRID_LABELS = (
    "top_left",
    "top_center",
    "top_right",
    "middle_left",
    "middle_center",
    "middle_right",
    "bottom_left",
    "bottom_center",
    "bottom_right",
)

DOMINANCE_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7)
MAX_EXAMPLE_SUBGROUPS = 4
DEFAULT_EXAMPLE_COUNT = 3
DEFAULT_DOMINANCE_HISTOGRAM_BINS = 20

FEASIBILITY_RULES = {
    "min_instances": 100,
    "min_images": 40,
    "min_median_dominance": 0.50,
    "min_holdout_images_tau_0_5": 25,
    "max_collateral_other_loss_frac_tau_0_5": 0.35,
}

IMAGE_EXTENSIONS = {".npy", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

DEFAULT_FRONTEND_ORIGINS = (
    "http://127.0.0.1:3000",
    "http://localhost:3000",
    "http://127.0.0.1:4173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "http://localhost:5173",
)

"""Centralised constants for the flow-matching-trial repository.

Only *pure* constants that are duplicated across multiple scripts belong here.
Importing this module must have **zero** side-effects (no torch, no numpy at
module level) so it stays lightweight from any call-site.
"""

# ---------------------------------------------------------------------------
# Raw uint16 thermal image normalisation
# ---------------------------------------------------------------------------
# Computed once over the full v18 dataset: the p0.001 and p99.999 percentiles
# of the raw sensor values.  Every training and analysis script uses these
# values to map raw uint16 → [0, 1] → [-1, 1].

P0001_PERCENTILE_RAW_IMAGES: float = 11667.0
"""p0.001 percentile of the raw uint16 sensor distribution."""

P9999_PERCENTILE_RAW_IMAGES: float = 13944.0
"""p99.999 percentile of the raw uint16 sensor distribution."""

RAW_RANGE: float = P9999_PERCENTILE_RAW_IMAGES - P0001_PERCENTILE_RAW_IMAGES
"""B - A  (= 2277.0).  Denominator used in the percentile normalisation."""

# ---------------------------------------------------------------------------
# ImageNet normalisation (for DINOv2 / Inception pre-processing)
# ---------------------------------------------------------------------------

IMAGENET_MEAN = (0.485, 0.456, 0.406)
"""ImageNet channel means (RGB order)."""

IMAGENET_STD = (0.229, 0.224, 0.225)
"""ImageNet channel standard deviations (RGB order)."""

# ---------------------------------------------------------------------------
# Default spatial resolution
# ---------------------------------------------------------------------------

DEFAULT_IMAGE_SIZE: int = 256
"""Default H=W used for resizing thermal images before training / generation."""

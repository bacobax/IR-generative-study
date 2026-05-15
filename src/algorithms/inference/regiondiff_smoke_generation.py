"""Compatibility facade for RegionDiff synthetic generation helpers.

Canonical implementations live under ``src.algorithms.inference.regiondiff``.
This module preserves the historical import path used by scripts, tests, and notebooks.
"""

from __future__ import annotations

import numpy as np
import torch

from src.algorithms.inference.regiondiff.dataset_io import *
from src.algorithms.inference.regiondiff.backend_loaders import *
from src.algorithms.inference.regiondiff.generation_backends import *
from src.algorithms.inference.regiondiff.audit_filtering import *
from src.algorithms.inference.regiondiff.orchestration import *
from src.algorithms.training.yolo_experiment_b import load_full_train_samples

# Explicit registry aliases preserve mutable object identity for old callers.
from src.algorithms.inference.regiondiff.generation_backends import (
    GENERATOR_BACKENDS,
    STREAMING_GENERATOR_BACKENDS,
)

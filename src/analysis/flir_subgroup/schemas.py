"""Pydantic request schemas for the FLIR subgroup analysis API."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field

from src.analysis.flir_subgroup.analysis import canonical_subgroup_label


class Phase(str, Enum):
    """Supported analysis phases."""

    PHASE1 = "phase1"
    PHASE2 = "phase2"


class DatasetId(str, Enum):
    """Supported dataset ids."""

    FLIR_PRIVATE_PROXY_ALIGNMENT_V18 = "flir_private_proxy_alignment_v18"
    V18 = "v18"


class GroupSpec(BaseModel):
    """Structured subgroup selector used by the frontend."""

    class_label: str = Field(..., min_length=1)
    size_bin: str = Field(..., min_length=1)
    position_bin: Optional[str] = None

    def subgroup_label(self, phase: Phase) -> str:
        """Convert the structured selector to a canonical subgroup label."""

        if phase == Phase.PHASE1:
            return canonical_subgroup_label(self.class_label, self.size_bin)
        return canonical_subgroup_label(self.class_label, self.size_bin, self.position_bin)


class HoldoutCurvesRequest(BaseModel):
    """Request payload for hold-out curve data."""

    dataset: DatasetId
    phase: Phase
    groups: list[GroupSpec]
    thresholds: Optional[list[float]] = None


class CollateralRequest(BaseModel):
    """Request payload for collateral-damage data."""

    dataset: DatasetId
    phase: Phase
    groups: list[GroupSpec]
    tau: float = Field(..., ge=0.0, le=1.0)


class PartitionComparisonsRequest(BaseModel):
    """Request payload for union partition comparisons."""

    dataset: DatasetId
    phase: Phase
    groups: list[GroupSpec]
    tau: float = Field(..., ge=0.0, le=1.0)
    include_zero_counts: bool = False


class ExamplesRequest(BaseModel):
    """Request payload for held-out and retained examples."""

    dataset: DatasetId
    phase: Phase
    groups: list[GroupSpec]
    tau: float = Field(..., ge=0.0, le=1.0)
    example_count: int = Field(default=3, ge=1, le=12)

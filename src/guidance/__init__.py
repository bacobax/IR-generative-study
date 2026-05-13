"""Inference-time guidance modules."""

from src.guidance.base_guidance import BaseGuidance
from src.guidance.no_guidance import NoGuidance

_SCORE_EXPORTS = {
    "ScoreGuidanceConfig",
    "ScorePredictorGuidance",
    "run_sanity_check",
}


def __getattr__(name: str):
    if name in _SCORE_EXPORTS:
        from src.guidance import score_predictor_guidance

        return getattr(score_predictor_guidance, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "BaseGuidance",
    "NoGuidance",
    "ScoreGuidanceConfig",
    "ScorePredictorGuidance",
    "run_sanity_check",
]

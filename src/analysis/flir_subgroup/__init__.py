"""Subgroup analysis package exports."""

from __future__ import annotations

from src.analysis.flir_subgroup.context import (
    FlirSubgroupAnalysisContext,
    build_analysis_context,
    clear_analysis_context_cache,
    get_analysis_context,
)

__all__ = [
    "app",
    "create_app",
    "FlirSubgroupAnalysisContext",
    "build_analysis_context",
    "clear_analysis_context_cache",
    "get_analysis_context",
]


def __getattr__(name: str):
    """Load FastAPI app exports lazily so analysis utilities work without web deps."""

    if name in {"app", "create_app"}:
        from src.analysis.flir_subgroup.app import app, create_app

        exports = {"app": app, "create_app": create_app}
        return exports[name]
    raise AttributeError(name)

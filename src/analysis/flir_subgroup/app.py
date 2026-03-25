"""FastAPI application factory for the subgroup analysis API."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.analysis.flir_subgroup.api import create_router
from src.analysis.flir_subgroup.constants import DEFAULT_FRONTEND_ORIGINS
from src.analysis.flir_subgroup.datasets import DatasetConfig


def create_app(
    data_root: Path | None = None,
    *,
    dataset_registry: Mapping[str, DatasetConfig] | None = None,
) -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title="Subgroup Analysis API",
        version="0.1.0",
        description="Interactive API for subgroup hold-out analysis across FLIR-style datasets.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEFAULT_FRONTEND_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(create_router(data_root=data_root, dataset_registry=dataset_registry))

    @app.get("/health")
    def healthcheck() -> dict:
        return {"status": "ok"}

    return app


app = create_app()

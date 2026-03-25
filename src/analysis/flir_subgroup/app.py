"""FastAPI application factory for the FLIR subgroup analysis API."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.analysis.flir_subgroup.api import create_router
from src.analysis.flir_subgroup.constants import DEFAULT_FRONTEND_ORIGINS


def create_app(data_root: Path | None = None) -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title="FLIR Subgroup Analysis API",
        version="0.1.0",
        description="Interactive API for the FLIR subgroup split analysis notebook.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(DEFAULT_FRONTEND_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(create_router(data_root=data_root))

    @app.get("/health")
    def healthcheck() -> dict:
        return {"status": "ok"}

    return app


app = create_app()

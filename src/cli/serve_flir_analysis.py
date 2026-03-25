"""CLI entrypoint for the subgroup FastAPI service."""

from __future__ import annotations

import argparse

import uvicorn


def build_parser() -> argparse.ArgumentParser:
    """Build the server CLI parser."""

    parser = argparse.ArgumentParser(description="Serve the subgroup analysis FastAPI app")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload mode")
    parser.add_argument("--log-level", default="info", help="Uvicorn log level")
    return parser


def main() -> None:
    """Launch the FastAPI app with uvicorn."""

    args = build_parser().parse_args()
    if args.reload:
        uvicorn.run(
            "src.analysis.flir_subgroup.app:app",
            host=args.host,
            port=args.port,
            reload=True,
            log_level=args.log_level,
        )
        return

    from src.analysis.flir_subgroup.app import create_app

    uvicorn.run(create_app(), host=args.host, port=args.port, reload=False, log_level=args.log_level)


if __name__ == "__main__":
    main()

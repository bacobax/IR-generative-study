"""Quick dependency smoke test for the unconditional Stable Diffusion CLI.

Run with::

    python -m pytest tests/test_train_sd_uncond_imports.py -v
"""

from __future__ import annotations

import os
import sys
from importlib import import_module


_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def test_train_sd_uncond_cli_imports() -> None:
    """Ensure pip-installed runtime dependencies for the CLI are available.

    Importing the CLI module exercises the same transitive imports needed to
    start `src/cli/train_sd_uncond.py` before argument parsing and training.
    """

    try:
        import_module("src.cli.train_sd_uncond")
    except Exception as exc:  # pragma: no cover - surfaces missing deps directly
        raise AssertionError(
            "Importing `src.cli.train_sd_uncond` failed. "
            "A required runtime dependency is likely missing from the current "
            f"Python environment.\n{exc.__class__.__name__}: {exc}"
        ) from exc

"""Regression tests for diffusion-stack import compatibility helpers."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers is not installed")
def test_clip_text_model_import_survives_blocked_scipy() -> None:
    code = textwrap.dedent(
        """
        import sys
        from importlib.abc import MetaPathFinder

        from src.core.diffusers_compat import disable_diffusers_optional_scipy

        disable_diffusers_optional_scipy(lightweight_diffusers_imports=False)

        class BlockScipy(MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.startswith("scipy"):
                    raise ImportError(f"blocked {fullname}")
                return None

        sys.meta_path.insert(0, BlockScipy())

        from transformers import CLIPTextModel

        assert CLIPTextModel.__name__ == "CLIPTextModel"
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

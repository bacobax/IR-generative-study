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

        import transformers.utils.import_utils as import_utils

        import_utils._scipy_available = True

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


@pytest.mark.skipif(importlib.util.find_spec("transformers") is None, reason="transformers is not installed")
def test_sd_unet_imports_do_not_require_peft() -> None:
    code = textwrap.dedent(
        """
        import builtins
        import importlib.util

        real_find_spec = importlib.util.find_spec
        real_import = builtins.__import__

        def find_spec_without_peft(name, *args, **kwargs):
            if name == "peft" or name.startswith("peft."):
                return None
            return real_find_spec(name, *args, **kwargs)

        def import_without_peft(name, *args, **kwargs):
            if name == "peft" or name.startswith("peft."):
                raise ModuleNotFoundError("No module named 'peft'")
            return real_import(name, *args, **kwargs)

        importlib.util.find_spec = find_spec_without_peft
        builtins.__import__ = import_without_peft

        import src.algorithms.stable_diffusion.layout_models as layout_models
        import src.algorithms.stable_diffusion.training as training

        assert layout_models.STAGE2_UNET_WEIGHTS == "regiondiff_unet.safetensors"
        assert training.CHECKPOINT_METADATA_FILENAME == "training_state.json"
        """
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr

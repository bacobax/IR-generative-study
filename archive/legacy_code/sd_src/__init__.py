# Stable Diffusion 1.5 LoRA Fine-tuning Package
"""
**Backward-compatibility re-export layer.**

The source of truth has moved to ``src.algorithms.stable_diffusion``.
Import directly from the sub-modules, e.g.::

    from sd_src.config import TrainingConfig

or, preferably, from the new canonical location::

    from src.algorithms.stable_diffusion.config import TrainingConfig
"""

# Lazy: sub-modules are importable via sd_src.config, sd_src.models, etc.
# but we avoid eagerly loading heavy deps (diffusers, transformers, bitsandbytes).
import importlib as _importlib

def __getattr__(name):
    """Redirect attribute access to src.algorithms.stable_diffusion."""
    try:
        return _importlib.import_module(f"src.algorithms.stable_diffusion.{name}")
    except ModuleNotFoundError:
        raise AttributeError(f"module 'sd_src' has no attribute {name!r}")

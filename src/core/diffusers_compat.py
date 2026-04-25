"""Compatibility helpers for lightweight diffusers imports."""

from __future__ import annotations

import os
import sys
import types
from importlib.machinery import ModuleSpec


def disable_diffusers_optional_scipy() -> None:
    """Prevent diffusers' optional scheduler imports from loading SciPy.

    Some cluster SciPy builds can fail while importing BLAS-linked extension
    modules even when the current code path only needs diffusers model classes.
    Flow-matching training does not use diffusers schedulers, so treating SciPy
    as unavailable avoids importing optional scheduler code during lazy model
    resolution.
    """
    value = os.environ.get("FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY", "1")
    if value.strip().lower() in {"0", "false", "no", "off"}:
        return

    try:
        import diffusers.utils.import_utils as import_utils
    except ModuleNotFoundError:
        return

    import_utils._scipy_available = False
    import_utils._scipy_version = "unavailable"

    _install_lightweight_loaders_package()
    _install_lightweight_model_package("autoencoders")
    _install_lightweight_model_package("unets")
    _install_lightweight_model_package("transformers")


def _install_lightweight_loaders_package() -> None:
    """Avoid importing optional single-file and PEFT loader stacks."""
    package_name = "diffusers.loaders"
    if package_name in sys.modules:
        return

    loaders = types.ModuleType(package_name)
    loaders.__package__ = package_name
    loaders.__spec__ = ModuleSpec(package_name, loader=None, is_package=True)
    loaders.__path__ = []
    peft_mixin = type("PeftAdapterMixin", (), {})
    original_mixin = type("FromOriginalModelMixin", (), {})
    unet_condition_mixin = type("UNet2DConditionLoadersMixin", (), {})
    loaders.PeftAdapterMixin = peft_mixin
    loaders.FromOriginalModelMixin = original_mixin
    loaders.UNet2DConditionLoadersMixin = unet_condition_mixin
    sys.modules[package_name] = loaders

    single_file_name = f"{package_name}.single_file_model"
    single_file = types.ModuleType(single_file_name)
    single_file.__package__ = package_name
    single_file.__spec__ = ModuleSpec(single_file_name, loader=None)
    single_file.FromOriginalModelMixin = original_mixin
    sys.modules[single_file_name] = single_file


def _install_lightweight_model_package(package_leaf: str) -> None:
    """Avoid broad diffusers model package imports for UNet2DModel."""
    package_name = f"diffusers.models.{package_leaf}"
    existing = sys.modules.get(package_name)
    if existing is not None:
        return

    try:
        import diffusers
    except ModuleNotFoundError:
        return

    diffusers_root = os.path.dirname(getattr(diffusers, "__file__", ""))
    if not diffusers_root:
        return

    package_path = os.path.join(diffusers_root, "models", package_leaf)
    if not os.path.isdir(package_path):
        return

    package = types.ModuleType(package_name)
    package.__file__ = os.path.join(package_path, "__init__.py")
    package.__path__ = [package_path]
    package.__package__ = package_name
    package.__spec__ = ModuleSpec(package_name, loader=None, is_package=True)
    sys.modules[package_name] = package


def import_diffusers_attr(module_name: str, attr_name: str):
    """Import a diffusers attribute after applying local import safeguards."""
    disable_diffusers_optional_scipy()

    try:
        module = __import__(module_name, fromlist=[attr_name])
    except ModuleNotFoundError as exc:
        if exc.name == "diffusers":
            raise ModuleNotFoundError(
                "Diffusers is required for this model path. Install it in the "
                "active environment or switch to the repo's `diffusers-dev` "
                "environment."
            ) from exc
        raise
    return getattr(module, attr_name)

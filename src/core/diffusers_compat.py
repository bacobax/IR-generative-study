"""Compatibility helpers for lightweight diffusion-stack imports."""

from __future__ import annotations

import os
import sys
import types
import importlib
from importlib.machinery import ModuleSpec


def disable_diffusers_optional_scipy(*, lightweight_diffusers_imports: bool = True) -> None:
    """Prevent optional diffusion-stack imports from loading SciPy.

    Some cluster SciPy builds can fail while importing BLAS-linked extension
    modules even when the current code path only needs model classes.  Treating
    SciPy as unavailable avoids importing optional scheduler/loss code during
    lazy model resolution.

    ``lightweight_diffusers_imports`` also installs narrow Diffusers package
    shims used by unconditional/FM model imports.  Keep it disabled for Stable
    Diffusion pipeline code because those pipelines need the real loader mixins
    for LoRA loading and fusion.
    """
    value = os.environ.get("FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY", "1")
    if value.strip().lower() in {"0", "false", "no", "off"}:
        return

    _install_scipy_optimize_stub()
    _disable_transformers_optional_scipy()
    _disable_diffusers_optional_scipy(lightweight_imports=lightweight_diffusers_imports)


def restore_real_scipy_if_available() -> bool:
    """Replace the local SciPy stub with the real SciPy package when installed."""
    scipy_module = sys.modules.get("scipy")
    scipy_optimize = sys.modules.get("scipy.optimize")
    has_stubbed_scipy = bool(getattr(scipy_module, "_flow_matching_stub", False))
    has_stubbed_optimize = bool(getattr(scipy_optimize, "_flow_matching_stub", False))
    if not has_stubbed_scipy and not has_stubbed_optimize:
        return scipy_module is not None

    stubbed_names = [
        name
        for name, module in list(sys.modules.items())
        if name == "scipy" or name.startswith("scipy.")
        if getattr(module, "_flow_matching_stub", False)
    ]
    for name in stubbed_names:
        sys.modules.pop(name, None)

    try:
        importlib.import_module("scipy")
        return True
    except ModuleNotFoundError:
        for name in stubbed_names:
            sys.modules.pop(name, None)
        if has_stubbed_scipy:
            _install_scipy_optimize_stub()
        return False


def _disabled_linear_sum_assignment(*_args, **_kwargs):
    raise ImportError(
        "scipy.optimize.linear_sum_assignment is disabled by "
        "FLOW_MATCHING_DISABLE_DIFFUSERS_SCIPY."
    )


def _install_scipy_optimize_stub() -> None:
    """Provide the tiny SciPy surface imported by optional Transformers losses."""
    existing_scipy = sys.modules.get("scipy")
    if existing_scipy is not None and not getattr(existing_scipy, "_flow_matching_stub", False):
        return

    existing_optimize = sys.modules.get("scipy.optimize")
    if existing_optimize is not None and not getattr(existing_optimize, "_flow_matching_stub", False):
        return

    scipy_module = existing_scipy or types.ModuleType("scipy")
    scipy_module.__package__ = "scipy"
    scipy_module.__spec__ = ModuleSpec("scipy", loader=None, is_package=True)
    scipy_module.__path__ = []
    scipy_module._flow_matching_stub = True

    optimize_module = existing_optimize or types.ModuleType("scipy.optimize")
    optimize_module.__package__ = "scipy"
    optimize_module.__spec__ = ModuleSpec("scipy.optimize", loader=None)
    optimize_module._flow_matching_stub = True
    optimize_module.linear_sum_assignment = _disabled_linear_sum_assignment

    scipy_module.optimize = optimize_module
    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.optimize"] = optimize_module


def _disable_transformers_optional_scipy() -> None:
    try:
        import transformers.utils.import_utils as import_utils
    except ModuleNotFoundError:
        return

    import_utils._scipy_available = False
    # Transformers generation helpers treat sklearn as optional, but sklearn
    # imports SciPy at module import time.  Disable it with SciPy for CLIP-only
    # SD paths on clusters with broken SciPy shared-library links.
    import_utils._sklearn_available = False


def _disable_diffusers_optional_scipy(*, lightweight_imports: bool) -> None:
    try:
        import diffusers.utils.import_utils as import_utils
    except ModuleNotFoundError:
        return

    import_utils._scipy_available = False
    import_utils._scipy_version = "unavailable"

    if not lightweight_imports:
        _remove_lightweight_diffusers_shims()
        return

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
    loaders._flow_matching_lightweight_stub = True
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
    single_file._flow_matching_lightweight_stub = True
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
    package._flow_matching_lightweight_stub = True
    sys.modules[package_name] = package


def _remove_lightweight_diffusers_shims() -> None:
    """Remove local Diffusers shims before importing Stable Diffusion pipelines.

    FM model imports install tiny ``diffusers.loaders`` and model-package shims
    to avoid optional PEFT/single-file loader stacks.  SD pipeline imports need
    the real loader package, including ``FromSingleFileMixin``.  If the FM path
    runs first in the same process, remove only the shims created here and let
    Python resolve the real Diffusers modules on the next import.
    """
    shim_names = (
        "diffusers.loaders.single_file_model",
        "diffusers.loaders",
        "diffusers.models.autoencoders",
        "diffusers.models.unets",
        "diffusers.models.transformers",
    )
    for name in shim_names:
        module = sys.modules.get(name)
        if module is None or not getattr(module, "_flow_matching_lightweight_stub", False):
            continue
        sys.modules.pop(name, None)
        parent_name, _, attr_name = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None and getattr(parent, attr_name, None) is module:
            delattr(parent, attr_name)


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

"""Lightweight Hugging Face Diffusers model adapter skeleton."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Mapping, Sequence

from src.core.diffusers_compat import import_diffusers_attr
from src.core.registry import REGISTRIES
from src.models.adapters.base import ModelAdapterBase, ModelBuildRequest, ModelBundle


SUPPORTED_COMPONENTS = ("unet", "vae", "text_encoder", "tokenizer", "scheduler")

_DEFAULT_COMPONENT_CLASSES = {
    "unet": ("diffusers", "UNet2DConditionModel"),
    "vae": ("diffusers", "AutoencoderKL"),
    "scheduler": ("diffusers", "DDPMScheduler"),
    "text_encoder": ("transformers", "CLIPTextModel"),
    "tokenizer": ("transformers", "CLIPTokenizer"),
}


def _register_once(registry, name: str, value: Any) -> None:
    if name not in registry:
        registry.register(name)(value)


def _lazy_import(module_name: str, attr_name: str) -> Any:
    if module_name == "diffusers":
        return import_diffusers_attr(module_name, attr_name)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


class DiffusersModelAdapter(ModelAdapterBase):
    """Load selected diffusers/transformers components from a pretrained path."""

    supported_components = SUPPORTED_COMPONENTS

    def build(self, request: ModelBuildRequest) -> ModelBundle:
        config = dict(request.config or {})
        options = dict(request.options or {})
        pretrained_model_name_or_path = (
            config.get("pretrained_model_name_or_path")
            or options.get("pretrained_model_name_or_path")
            or request.name
        )
        if not pretrained_model_name_or_path:
            raise ValueError(
                "DiffusersModelAdapter requires pretrained_model_name_or_path."
            )

        revision = config.get("revision", options.get("revision"))
        variant = config.get("variant", options.get("variant"))
        component_names = _resolve_component_names(config, options)
        component_classes = dict(options.get("component_classes") or {})
        component_classes.update(config.get("component_classes") or {})
        subfolders = _resolve_subfolders(config, options)

        components: Dict[str, Any] = {}
        for component_name in component_names:
            if component_name not in SUPPORTED_COMPONENTS:
                raise ValueError(
                    f"Unsupported diffusers component {component_name!r}. "
                    f"Supported: {list(SUPPORTED_COMPONENTS)}"
                )
            cls = component_classes.get(component_name) or self._resolve_component_class(component_name)
            kwargs: Dict[str, Any] = {}
            subfolder = subfolders.get(component_name, component_name)
            if subfolder:
                kwargs["subfolder"] = subfolder
            if revision is not None:
                kwargs["revision"] = revision
            if variant is not None:
                kwargs["variant"] = variant
            component = cls.from_pretrained(pretrained_model_name_or_path, **kwargs)
            if request.device is not None and component_name != "tokenizer" and hasattr(component, "to"):
                component = component.to(request.device)
            components[component_name] = component

        self._apply_trainability_policy(components, config, options)
        primary_name = options.get("primary_component") or config.get("primary_component")
        if primary_name is None:
            primary_name = component_names[0] if component_names else "unet"
        primary = components.get(primary_name)
        if primary is None:
            raise ValueError(
                f"Primary component {primary_name!r} was not loaded. "
                f"Loaded components: {list(components)}"
            )

        return ModelBundle(
            primary=primary,
            components=components,
            adapter_name="diffusers",
            model_name=str(pretrained_model_name_or_path),
            metadata={
                "pretrained_model_name_or_path": pretrained_model_name_or_path,
                "revision": revision,
                "variant": variant,
                "components": list(component_names),
            },
        )

    def _resolve_component_class(self, component_name: str) -> Any:
        module_name, attr_name = _DEFAULT_COMPONENT_CLASSES[component_name]
        return _lazy_import(module_name, attr_name)

    def _apply_trainability_policy(
        self,
        components: Mapping[str, Any],
        config: Mapping[str, Any],
        options: Mapping[str, Any],
    ) -> None:
        raw_policy = options.get("trainability", config.get("trainability", "frozen"))
        partial_prefixes = options.get("partial_prefixes", config.get("partial_prefixes", ()))
        if isinstance(raw_policy, Mapping):
            partial_prefixes = raw_policy.get("partial_prefixes", partial_prefixes)
            policy = str(raw_policy.get("policy", raw_policy.get("mode", "frozen")))
        else:
            policy = str(raw_policy)

        if policy == "lora":
            raise NotImplementedError(
                "LoRA trainability for DiffusersModelAdapter is a placeholder for "
                "a future PEFT-backed migration."
            )
        if policy == "frozen":
            _set_requires_grad(components, False)
            return
        if policy == "full":
            _set_requires_grad(components, True)
            return
        if policy == "partial_prefixes":
            _set_requires_grad(components, False)
            _set_partial_prefixes_trainable(components, partial_prefixes)
            return
        raise ValueError(
            "Unsupported diffusers trainability policy "
            f"{policy!r}; expected frozen, full, partial_prefixes, or lora."
        )


def _resolve_component_names(
    config: Mapping[str, Any],
    options: Mapping[str, Any],
) -> list[str]:
    component_names = (
        options.get("component_names")
        or options.get("components")
        or config.get("component_names")
        or config.get("components")
        or ("unet",)
    )
    if isinstance(component_names, str):
        return [component_names]
    return [str(name) for name in component_names]


def _resolve_subfolders(
    config: Mapping[str, Any],
    options: Mapping[str, Any],
) -> Mapping[str, str | None]:
    subfolders = dict(config.get("subfolders") or config.get("component_subfolders") or {})
    subfolders.update(options.get("subfolders") or options.get("component_subfolders") or {})
    return subfolders


def _iter_parameters(module: Any):
    parameters = getattr(module, "parameters", None)
    if callable(parameters):
        yield from parameters()


def _iter_named_parameters(module: Any):
    named_parameters = getattr(module, "named_parameters", None)
    if callable(named_parameters):
        yield from named_parameters()


def _set_requires_grad(components: Mapping[str, Any], value: bool) -> None:
    for component in components.values():
        for param in _iter_parameters(component):
            param.requires_grad = value


def _prefixes_for_component(
    partial_prefixes: Mapping[str, Sequence[str]] | Sequence[str],
    component_name: str,
) -> Sequence[str]:
    if isinstance(partial_prefixes, Mapping):
        return tuple(str(prefix) for prefix in partial_prefixes.get(component_name, ()))
    return tuple(str(prefix) for prefix in partial_prefixes)


def _set_partial_prefixes_trainable(
    components: Mapping[str, Any],
    partial_prefixes: Mapping[str, Sequence[str]] | Sequence[str],
) -> None:
    for component_name, component in components.items():
        prefixes = _prefixes_for_component(partial_prefixes, component_name)
        if not prefixes:
            continue
        for name, param in _iter_named_parameters(component):
            if any(str(name).startswith(prefix) for prefix in prefixes):
                param.requires_grad = True


_register_once(REGISTRIES.model_adapter, "diffusers", DiffusersModelAdapter())


__all__ = ["DiffusersModelAdapter", "SUPPORTED_COMPONENTS"]

"""Default model adapter shims for existing builder patterns."""

from __future__ import annotations

from typing import Any, Callable, Mapping

from src.core.registry import REGISTRIES
from src.models.adapters.base import ModelAdapterBase, ModelBuildRequest, ModelBundle


def _register_once(registry, name: str, value: Any) -> None:
    if name not in registry:
        registry.register(name)(value)


class NativeModelAdapter(ModelAdapterBase):
    """Wrap a config-dict builder function in the adapter protocol."""

    def __init__(
        self,
        builder: Callable[..., Any],
        *,
        adapter_name: str | None = None,
        component_name: str = "model",
    ) -> None:
        self.builder = builder
        self.adapter_name = adapter_name or getattr(builder, "__name__", self.__class__.__name__)
        self.component_name = component_name

    def build(self, request: ModelBuildRequest) -> ModelBundle:
        config = dict(request.config or {})
        kwargs = dict(request.options or {})
        kwargs.pop("builder_name", None)
        kwargs.pop("model_builder_name", None)
        if request.device is not None and "device" not in kwargs:
            kwargs["device"] = request.device
        if request.dtype is not None and "dtype" not in kwargs:
            kwargs["dtype"] = request.dtype
        model = self.builder(config, **kwargs)
        components = dict(request.components or {})
        components.setdefault(self.component_name, model)
        return ModelBundle(
            primary=model,
            components=components,
            adapter_name=self.adapter_name,
            model_name=request.name,
            metadata=dict(request.metadata or {}),
        )


class RegistryModelBuilderAdapter(ModelAdapterBase):
    """Delegate model construction to the legacy ``model_builder`` registry."""

    def build(self, request: ModelBuildRequest) -> ModelBundle:
        options = dict(request.options or {})
        builder_name = (
            request.name
            or options.pop("builder_name", None)
            or options.pop("model_builder_name", None)
        )
        builder = REGISTRIES.model_builder.get(builder_name)
        config = dict(request.config or {})
        if request.device is not None and "device" not in options:
            options["device"] = request.device
        model = builder(config, **options)
        components = dict(request.components or {})
        components.setdefault("model", model)
        return ModelBundle(
            primary=model,
            components=components,
            adapter_name="registry_model_builder",
            model_name=builder_name,
            metadata=dict(request.metadata or {}),
        )


class ExternalModelWrapperAdapter(ModelAdapterBase):
    """Bundle an existing external model or create one with a factory callable."""

    def build(self, request: ModelBuildRequest) -> ModelBundle:
        options = dict(request.options or {})
        components = dict(request.components or {})
        primary = (
            options.pop("model", None)
            or options.pop("module", None)
            or components.get("model")
            or components.get("primary")
        )
        factory = options.pop("factory", None) or options.pop("wrapper_factory", None)
        if primary is None and callable(factory):
            primary = factory(request)
        elif primary is not None and callable(factory):
            primary = factory(primary)
        if primary is None:
            raise ValueError(
                "ExternalModelWrapperAdapter requires an existing model/module "
                "or a factory/wrapper_factory option."
            )
        components.setdefault("model", primary)
        return ModelBundle(
            primary=primary,
            components=components,
            adapter_name="external_wrapper",
            model_name=request.name,
            metadata=dict(request.metadata or {}),
        )


_register_once(REGISTRIES.model_adapter, "registry_model_builder", RegistryModelBuilderAdapter())
_register_once(REGISTRIES.model_adapter, "external_wrapper", ExternalModelWrapperAdapter())


__all__ = [
    "ExternalModelWrapperAdapter",
    "NativeModelAdapter",
    "RegistryModelBuilderAdapter",
]

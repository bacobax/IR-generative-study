from uuid import uuid4

from src.core.registry import REGISTRIES
from src.models.adapters import (
    ExternalModelWrapperAdapter,
    ModelBuildRequest,
    NativeModelAdapter,
    RegistryModelBuilderAdapter,
)


def test_registry_model_builder_adapter_delegates_to_legacy_registry() -> None:
    name = f"unit_builder_{uuid4().hex}"

    def builder(config, *, device=None):
        return {"config": dict(config), "device": device}

    REGISTRIES.model_builder.register(name)(builder)

    bundle = RegistryModelBuilderAdapter().build(
        ModelBuildRequest(name=name, config={"width": 4}, device="cpu")
    )

    assert bundle.primary == {"config": {"width": 4}, "device": "cpu"}
    assert bundle.components["model"] is bundle.primary
    assert bundle.adapter_name == "registry_model_builder"


def test_native_model_adapter_returns_model_bundle() -> None:
    def builder(config, *, device=None):
        return {"built": dict(config), "device": device}

    adapter = NativeModelAdapter(builder, adapter_name="native_unit", component_name="unet")
    bundle = adapter.build(ModelBuildRequest(config={"channels": 1}, device="cpu"))

    assert bundle.primary == {"built": {"channels": 1}, "device": "cpu"}
    assert bundle.components["unet"] is bundle.primary
    assert bundle.adapter_name == "native_unit"


def test_external_model_wrapper_adapter_accepts_existing_and_factory_models() -> None:
    existing = object()
    existing_bundle = ExternalModelWrapperAdapter().build(
        ModelBuildRequest(components={"model": existing})
    )
    assert existing_bundle.primary is existing

    factory_bundle = ExternalModelWrapperAdapter().build(
        ModelBuildRequest(options={"factory": lambda request: {"name": request.name}}, name="wrapped")
    )
    assert factory_bundle.primary == {"name": "wrapped"}

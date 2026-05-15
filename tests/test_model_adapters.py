from uuid import uuid4

import pytest
import torch

from src.core.registry import REGISTRIES
from src.models.adapters import (
    DiffusersModelAdapter,
    ExternalModelWrapperAdapter,
    FMModelAdapter,
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


class _FakeDataConfig:
    image_size = 64


class _FakeModelConfig:
    unet_config = "tiny_unet.json"
    vae_config = None
    vae_pretrained_model_name_or_path = None


class _FakeTrainConfig:
    data = _FakeDataConfig()
    model = _FakeModelConfig()

    def resolved_device(self):
        return "cpu"


class _FakeVAE(torch.nn.Module):
    def encode(self, x):
        return x + 1, torch.ones_like(x) * 0.5

    def sampling(self, z_mu, z_sigma):
        return z_mu + z_sigma

    def decode(self, z):
        return z - 1


def test_fm_adapter_builds_pixel_space_without_vae() -> None:
    built = {}

    def load_unet_config(_path):
        return {"sample_size": 8, "in_channels": 1}

    def build_unet(config, *, device=None):
        built["config"] = dict(config)
        built["device"] = device
        return {"unet": dict(config)}

    adapter = FMModelAdapter(
        load_unet_config_fn=load_unet_config,
        build_unet_fn=build_unet,
        resolve_vae_config_fn=lambda _model_config: None,
        build_vae_fn=lambda _config, *, device=None: None,
    )

    bundle = adapter.build_from_train_config(_FakeTrainConfig(), device="cpu")
    fm_adapter = bundle.components["fm_adapter"]

    assert bundle.primary == {"unet": {"sample_size": 64, "in_channels": 1}}
    assert bundle.components["unet"] is bundle.primary
    assert "vae" not in bundle.components
    assert fm_adapter.vae is None
    assert fm_adapter.unet_config["sample_size"] == 64
    assert built == {"config": {"sample_size": 64, "in_channels": 1}, "device": "cpu"}


def test_fm_adapter_builds_latent_space_with_vae_and_helpers() -> None:
    train_config = _FakeTrainConfig()
    train_config.model = type(
        "Model",
        (),
        {
            "unet_config": "tiny_unet.json",
            "vae_config": "tiny_vae.json",
            "vae_pretrained_model_name_or_path": None,
        },
    )()
    vae = _FakeVAE()

    adapter = FMModelAdapter(
        load_unet_config_fn=lambda _path: {"sample_size": 8, "in_channels": 4},
        build_unet_fn=lambda config, *, device=None: {"unet": dict(config), "device": device},
        resolve_vae_config_fn=lambda _model_config: {"num_channels": [16, 32, 64]},
        build_vae_fn=lambda config, *, device=None: vae,
    )

    bundle = adapter.build(ModelBuildRequest(config={"train_config": train_config}, device="cpu"))
    fm_adapter = bundle.components["fm_adapter"]
    x = torch.zeros(1, 1, 4, 4)

    assert fm_adapter.unet_config["sample_size"] == 16
    assert fm_adapter.vae_config == {"num_channels": [16, 32, 64]}
    assert bundle.components["vae"] is vae
    assert torch.allclose(fm_adapter.encode(x), torch.full_like(x, 1.5))
    assert torch.allclose(fm_adapter.decode(torch.ones_like(x)), torch.zeros_like(x))


class _FakeHFComponent(torch.nn.Module):
    calls = []

    def __init__(self, label):
        super().__init__()
        self.label = label
        self.keep = torch.nn.Parameter(torch.ones(1))
        self.train_me = torch.nn.Parameter(torch.ones(1))
        self.device_seen = None

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((cls.__name__, path, kwargs))
        return cls(cls.__name__)

    def to(self, device):
        self.device_seen = device
        return self


class _FakeTokenizer:
    calls = []

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        cls.calls.append((cls.__name__, path, kwargs))
        return cls()


def test_diffusers_adapter_passes_from_pretrained_kwargs_and_uses_fakes() -> None:
    _FakeHFComponent.calls = []
    _FakeTokenizer.calls = []
    adapter = DiffusersModelAdapter()

    bundle = adapter.build(
        ModelBuildRequest(
            config={
                "pretrained_model_name_or_path": "local/model",
                "revision": "abc123",
                "variant": "fp16",
                "components": ["unet", "vae", "tokenizer"],
            },
            device="cpu",
            options={
                "component_classes": {
                    "unet": _FakeHFComponent,
                    "vae": _FakeHFComponent,
                    "tokenizer": _FakeTokenizer,
                }
            },
        )
    )

    assert bundle.primary is bundle.components["unet"]
    assert _FakeHFComponent.calls == [
        (
            "_FakeHFComponent",
            "local/model",
            {"subfolder": "unet", "revision": "abc123", "variant": "fp16"},
        ),
        (
            "_FakeHFComponent",
            "local/model",
            {"subfolder": "vae", "revision": "abc123", "variant": "fp16"},
        ),
    ]
    assert _FakeTokenizer.calls == [
        (
            "_FakeTokenizer",
            "local/model",
            {"subfolder": "tokenizer", "revision": "abc123", "variant": "fp16"},
        )
    ]
    assert bundle.components["unet"].device_seen == "cpu"


def test_diffusers_adapter_trainability_policies() -> None:
    classes = {"unet": _FakeHFComponent}
    adapter = DiffusersModelAdapter()

    frozen = adapter.build(
        ModelBuildRequest(
            config={"pretrained_model_name_or_path": "local/model"},
            options={"component_classes": classes, "trainability": "frozen"},
        )
    )
    assert not any(param.requires_grad for param in frozen.components["unet"].parameters())

    full = adapter.build(
        ModelBuildRequest(
            config={"pretrained_model_name_or_path": "local/model"},
            options={"component_classes": classes, "trainability": "full"},
        )
    )
    assert all(param.requires_grad for param in full.components["unet"].parameters())

    partial = adapter.build(
        ModelBuildRequest(
            config={"pretrained_model_name_or_path": "local/model"},
            options={
                "component_classes": classes,
                "trainability": "partial_prefixes",
                "partial_prefixes": {"unet": ["train_me"]},
            },
        )
    )
    params = dict(partial.components["unet"].named_parameters())
    assert params["train_me"].requires_grad is True
    assert params["keep"].requires_grad is False


def test_diffusers_adapter_lora_policy_is_placeholder() -> None:
    with pytest.raises(NotImplementedError):
        DiffusersModelAdapter().build(
            ModelBuildRequest(
                config={"pretrained_model_name_or_path": "local/model"},
                options={
                    "component_classes": {"unet": _FakeHFComponent},
                    "trainability": "lora",
                },
            )
        )


def test_model_adapter_registry_includes_new_adapters() -> None:
    assert "fm" in REGISTRIES.model_adapter
    assert "native_fm" in REGISTRIES.model_adapter
    assert "diffusers" in REGISTRIES.model_adapter

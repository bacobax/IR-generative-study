from __future__ import annotations

from types import SimpleNamespace
from uuid import uuid4

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.cli.train_flow_matching import (
    _FLAT_TO_NESTED,
    _build_adapter_v1_non_layout_data,
    _build_adapter_v1_trainer,
    build_parser,
    run_training,
)
from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.fm_config import DataConfig, FMTrainConfig, OutputConfig
from src.core.data import DatasetBuildRequest, DatasetBundle
from src.core.data.training_data import NonLayoutTrainingData, ResolvedTrainingData
from src.core.registry import REGISTRIES


def test_fm_architecture_mode_defaults_to_legacy_and_cli_overrides() -> None:
    parser = build_parser()

    default_cfg = merge_config_and_cli(
        FMTrainConfig,
        None,
        parser,
        parser.parse_args([]),
        flat_to_nested=_FLAT_TO_NESTED,
        cli_argv=[],
    )
    adapter_cfg = merge_config_and_cli(
        FMTrainConfig,
        None,
        parser,
        parser.parse_args(["--architecture_mode", "adapter_v1"]),
        flat_to_nested=_FLAT_TO_NESTED,
        cli_argv=["--architecture_mode", "adapter_v1"],
    )

    assert default_cfg.architecture_mode == "legacy"
    assert adapter_cfg.architecture_mode == "adapter_v1"


def test_adapter_v1_rejects_layout_training() -> None:
    cfg = FMTrainConfig(architecture_mode="adapter_v1")
    cfg.layout_conditioning.enabled = True

    with pytest.raises(ValueError, match="only supported for FM non-layout"):
        run_training(cfg)


def test_adapter_v1_non_layout_data_resolves_dataset_adapter_before_existing_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_id = f"unit_adapter_{uuid4().hex}"
    calls: list[str] = []

    class _Adapter:
        def build(self, request: DatasetBuildRequest) -> DatasetBundle:
            calls.append(request.split)
            return DatasetBundle(adapter_name=dataset_id)

    REGISTRIES.dataset_adapter.register(dataset_id)(_Adapter())

    tiny = TensorDataset(torch.zeros(2, 1))
    loader = DataLoader(tiny, batch_size=1)
    expected = NonLayoutTrainingData(
        train_base_dataset=tiny,
        eval_base_dataset=tiny,
        train_dataset=tiny,
        eval_dataset=tiny,
        train_loader=loader,
        eval_loader=loader,
        use_annotation_ds=False,
    )

    def fake_build_non_layout_dataloaders(**kwargs):
        assert kwargs["data_config"].dataset_id == dataset_id
        return expected

    monkeypatch.setattr("src.cli.train_flow_matching.build_non_layout_dataloaders", fake_build_non_layout_dataloaders)
    cfg = FMTrainConfig(data=DataConfig(dataset_id=dataset_id))
    resolved = ResolvedTrainingData(
        train_dir="/tmp/train",
        val_dir="/tmp/val",
        train_annotations_path=None,
        val_annotations_path=None,
        normalization_mode="unit",
    )

    actual = _build_adapter_v1_non_layout_data(cfg, resolved_data=resolved, total_epochs=1)

    assert actual is expected
    assert calls == ["train", "val"]


def test_adapter_v1_trainer_builds_through_fm_model_adapter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.models.adapters.fm as fm_module

    class _TinyUNet(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(()))

    unet = _TinyUNet()

    class _FakeFMModelAdapter:
        def build_from_train_config(self, config, *, device=None):
            assert config.architecture_mode == "adapter_v1"
            assert device == "cpu"
            fm_adapter = SimpleNamespace(
                unet=unet,
                vae=None,
                unet_config={"sample_size": 4},
                vae_config=None,
            )
            return SimpleNamespace(components={"fm_adapter": fm_adapter})

    monkeypatch.setattr(fm_module, "FMModelAdapter", _FakeFMModelAdapter)
    cfg = FMTrainConfig(architecture_mode="adapter_v1", device="cpu", output=OutputConfig(model_dir="/tmp/unit"))

    trainer = _build_adapter_v1_trainer(cfg)

    assert trainer.unet is unet
    assert trainer.model_dir == "/tmp/unit"
    assert trainer.flow_task.train_target == cfg.training.train_target

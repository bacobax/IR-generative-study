from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.core.training_runtime import (
    build_ema,
    build_lr_scheduler,
    build_optimizer,
    set_epoch_for_dataloader,
    setup_precision,
)
from src.core.training_utils import (
    EMAState,
    TrainingProgressState,
    build_training_checkpoint,
    progress_state_from_checkpoint,
    restore_training_checkpoint,
    save_training_checkpoint,
)


class FakeScaler:
    def __init__(self, value: int = 0) -> None:
        self.value = value

    def state_dict(self) -> dict:
        return {"value": self.value}

    def load_state_dict(self, state_dict: dict) -> None:
        self.value = int(state_dict["value"])


def _stepped_model_and_optimizer() -> tuple[torch.nn.Linear, torch.optim.Adam]:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss = model(torch.ones(4, 2)).sum()
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return model, optimizer


def test_build_training_checkpoint_preserves_existing_payload_keys() -> None:
    model, optimizer = _stepped_model_and_optimizer()
    progress = TrainingProgressState(
        epoch=3,
        global_step=17,
        best_eval=0.25,
        best_epoch=2,
        bad_epochs=1,
    )

    payload = build_training_checkpoint(
        model_states={"unet_state": model},
        optimizer=optimizer,
        scheduler=None,
        scaler=None,
        ema=None,
        progress=progress,
        extra_metadata={
            "t_scale": 1000.0,
            "train_target": "velocity",
            "count_filter": {"seen_counts": [1, 2], "unseen_counts": [3]},
        },
        include_rng=True,
    )

    assert payload["epoch"] == 3
    assert payload["global_step"] == 17
    assert payload["best_eval"] == 0.25
    assert payload["best_epoch"] == 2
    assert payload["bad_epochs"] == 1
    assert set(payload) >= {
        "unet_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "ema_state",
        "rng_state",
        "t_scale",
        "train_target",
        "count_filter",
    }
    assert payload["scheduler_state"] is None
    assert payload["scaler_state"] is None
    assert payload["ema_state"] is None
    assert payload["count_filter"] == {"seen_counts": [1, 2], "unseen_counts": [3]}


def test_restore_training_checkpoint_loads_model_optimizer_and_moves_optimizer_state_to_device(tmp_path) -> None:
    model, optimizer = _stepped_model_and_optimizer()
    payload = build_training_checkpoint(
        model_states={"unet_state": model},
        optimizer=optimizer,
        progress=TrainingProgressState(epoch=1, global_step=5),
    )
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, payload, release_cache=False)

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.01)
    restore_training_checkpoint(
        path,
        device="cpu",
        model_states={"unet_state": restored_model},
        optimizer=restored_optimizer,
    )

    for expected, actual in zip(model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)
    for state in restored_optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.device.type == "cpu"


def test_restore_training_checkpoint_restores_scheduler_scaler_ema_when_present(tmp_path) -> None:
    model, optimizer = _stepped_model_and_optimizer()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: 0.5)
    scaler = FakeScaler(value=42)
    ema = EMAState(model, decay=0.9)
    progress = TrainingProgressState(epoch=2, global_step=8, best_eval=0.5)

    payload = build_training_checkpoint(
        model_states={"unet_state": model},
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        ema=ema,
        progress=progress,
    )
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, payload, release_cache=False)

    restored_model = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored_model.parameters(), lr=0.01)
    restored_scheduler = torch.optim.lr_scheduler.LambdaLR(
        restored_optimizer,
        lr_lambda=lambda step: 0.5,
    )
    restored_scaler = FakeScaler()
    restored_ema = EMAState(restored_model, decay=0.1)

    restore_training_checkpoint(
        path,
        device="cpu",
        model_states={"unet_state": restored_model},
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        scaler=restored_scaler,
        ema=restored_ema,
    )

    assert restored_scheduler.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]
    assert restored_scaler.value == 42
    assert restored_ema.decay == pytest.approx(0.9)
    for expected, actual in zip(ema.shadow_params, restored_ema.shadow_params):
        torch.testing.assert_close(actual, expected)


def test_restore_training_checkpoint_restores_rng_state_when_requested(tmp_path) -> None:
    model, optimizer = _stepped_model_and_optimizer()
    torch.manual_seed(123)
    payload = build_training_checkpoint(
        model_states={"unet_state": model},
        optimizer=optimizer,
        progress=TrainingProgressState(epoch=0, global_step=0),
        include_rng=True,
    )
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, payload, release_cache=False)

    torch.manual_seed(999)
    restore_training_checkpoint(
        path,
        device="cpu",
        model_states={"unet_state": model},
        optimizer=optimizer,
        restore_rng=True,
    )

    assert torch.equal(torch.random.get_rng_state(), payload["rng_state"])


def test_progress_state_roundtrip_uses_epoch_plus_one_start_epoch() -> None:
    checkpoint = {
        "epoch": 6,
        "global_step": 123,
        "best_eval": 1.25,
        "best_epoch": 4,
        "bad_epochs": 2,
    }

    start_epoch, progress = progress_state_from_checkpoint(checkpoint)

    assert start_epoch == 7
    assert progress == TrainingProgressState(
        epoch=6,
        global_step=123,
        best_eval=1.25,
        best_epoch=4,
        bad_epochs=2,
    )


def test_restore_training_checkpoint_validation_runs_before_state_mutation(tmp_path) -> None:
    model, optimizer = _stepped_model_and_optimizer()
    original_weight = model.weight.detach().clone()
    changed_model = torch.nn.Linear(2, 1)
    payload = build_training_checkpoint(
        model_states={"unet_state": changed_model},
        optimizer=optimizer,
        progress=TrainingProgressState(epoch=0, global_step=0),
        extra_metadata={"trainer_family": "sd_uncond"},
    )
    path = tmp_path / "checkpoint.pt"
    save_training_checkpoint(path, payload, release_cache=False)

    def reject_sd_uncond(checkpoint: dict, checkpoint_path: str) -> None:
        del checkpoint_path
        if checkpoint.get("trainer_family") == "sd_uncond":
            raise ValueError("rejected checkpoint family")

    with pytest.raises(ValueError, match="rejected checkpoint family"):
        restore_training_checkpoint(
            path,
            device="cpu",
            model_states={"unet_state": model},
            optimizer=optimizer,
            validate_checkpoint=reject_sd_uncond,
        )

    torch.testing.assert_close(model.weight, original_weight)


def test_runtime_build_optimizer_creates_adamw_with_expected_options() -> None:
    model = torch.nn.Linear(2, 1)

    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name="adamw",
        lr=0.02,
        weight_decay=0.03,
        beta1=0.8,
        beta2=0.9,
    )

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.02)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.03)
    assert optimizer.param_groups[0]["betas"] == (0.8, 0.9)


def test_runtime_build_optimizer_rejects_unknown_optimizer() -> None:
    model = torch.nn.Linear(2, 1)

    with pytest.raises(ValueError, match="Only 'adamw'"):
        build_optimizer(model.parameters(), optimizer_name="sgd")


def test_runtime_scheduler_delegates_warmup_behavior() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0)

    scheduler = build_lr_scheduler(
        optimizer,
        scheduler_name="constant_with_warmup",
        total_steps=4,
        warmup_ratio=0.5,
    )

    assert scheduler is not None
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.5)
    scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1.0)


def test_runtime_setup_precision_cpu_is_noop() -> None:
    precision, scaler = setup_precision("cpu", "auto")

    assert precision.mode == "no"
    assert precision.enabled is False
    assert scaler is None


def test_runtime_build_ema_respects_enabled_and_decay() -> None:
    model = torch.nn.Linear(2, 1)

    assert build_ema(model, enabled=False, decay=0.999) is None
    assert build_ema(model, enabled=True, decay=0.0) is None
    assert isinstance(build_ema(model, enabled=True, decay=0.9), EMAState)


class _EpochTransform:
    def __init__(self) -> None:
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(int(epoch))


class _EpochDataset(Dataset):
    def __init__(self, child: Dataset | None = None) -> None:
        self.dataset = child
        self.transform = _EpochTransform()
        self.epochs: list[int] = []

    def set_epoch(self, epoch: int) -> None:
        self.epochs.append(int(epoch))

    def __len__(self) -> int:
        return 1

    def __getitem__(self, index: int) -> torch.Tensor:
        del index
        return torch.zeros(1)


def test_runtime_set_epoch_reaches_nested_datasets_and_transforms() -> None:
    inner = _EpochDataset()
    outer = _EpochDataset(child=inner)
    loader = DataLoader(outer, batch_size=1)

    set_epoch_for_dataloader(loader, 7)

    assert outer.epochs == [7]
    assert outer.transform.epochs == [7]
    assert inner.epochs == [7]
    assert inner.transform.epochs == [7]

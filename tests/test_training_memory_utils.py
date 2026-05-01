"""Regression tests for memory-conscious training helpers."""

from __future__ import annotations

import torch

from src.core.training_utils import (
    EMAState,
    module_state_dict_cpu,
    optimizer_state_dict_cpu,
    tensor_tree_to_cpu,
)


def test_tensor_tree_to_cpu_detaches_nested_tensors() -> None:
    tensor = torch.randn(2, requires_grad=True)

    converted = tensor_tree_to_cpu({"items": [tensor], "pair": (tensor,)})

    assert converted["items"][0].device.type == "cpu"
    assert converted["items"][0].requires_grad is False
    assert converted["pair"][0].device.type == "cpu"
    assert converted["pair"][0].requires_grad is False


def test_module_and_optimizer_snapshots_are_cpu_tensors() -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss = model(torch.randn(4, 3)).square().mean()
    loss.backward()
    optimizer.step()

    module_state = module_state_dict_cpu(model)
    optimizer_state = optimizer_state_dict_cpu(optimizer)

    assert all(value.device.type == "cpu" for value in module_state.values())
    for state in optimizer_state["state"].values():
        for value in state.values():
            if torch.is_tensor(value):
                assert value.device.type == "cpu"
                assert value.requires_grad is False


def test_ema_average_parameters_restores_model_values() -> None:
    model = torch.nn.Linear(2, 2)
    original = [param.detach().clone() for param in model.parameters()]
    ema = EMAState(model, decay=0.9)

    with ema.average_parameters(model):
        for param in model.parameters():
            param.data.add_(1.0)

    for param, expected in zip(model.parameters(), original):
        assert torch.allclose(param, expected)

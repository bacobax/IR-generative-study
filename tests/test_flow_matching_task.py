from __future__ import annotations

import torch

from src.algorithms.tasks.flow_matching import FlowMatchingTask


def test_flow_matching_task_v_target_loss_matches_velocity_mse() -> None:
    task = FlowMatchingTask(train_target="v")
    state = {
        "t_expanded": torch.full((1, 1, 1, 1), 0.25),
        "zt": torch.tensor([[[[0.25, 0.75]]]]),
        "v_target": torch.tensor([[[[1.0, -1.0]]]]),
        "cond_kwargs": {},
    }
    prediction = torch.tensor([[[[0.5, -0.5]]]])

    loss = task.loss_from_prediction(prediction, state)

    expected = torch.mean((prediction - state["v_target"]) ** 2)
    torch.testing.assert_close(loss, expected)


def test_flow_matching_task_x0_target_loss_matches_derived_velocity_mse() -> None:
    task = FlowMatchingTask(train_target="x0")
    state = {
        "t_expanded": torch.full((1, 1, 1, 1), 0.25),
        "zt": torch.tensor([[[[0.25, 0.75]]]]),
        "v_target": torch.tensor([[[[1.0, -1.0]]]]),
        "cond_kwargs": {},
    }
    x0_prediction = torch.tensor([[[[1.0, 0.0]]]])

    loss = task.loss_from_prediction(x0_prediction, state)

    v_prediction = (x0_prediction - state["zt"]) / (1 - state["t_expanded"]).clamp(min=1e-5)
    expected = torch.mean((v_prediction - state["v_target"]) ** 2)
    torch.testing.assert_close(loss, expected)


def test_flow_matching_task_preserves_conditioning_permutation() -> None:
    cond_kwargs = {
        "boxes": torch.tensor([[0.0], [1.0], [2.0]]),
        "names": ["zero", "one", "two"],
        "scalar": torch.tensor(5.0),
    }
    permutation = torch.tensor([2, 0, 1])

    aligned = FlowMatchingTask.permute_conditioning_kwargs(
        cond_kwargs,
        permutation,
        batch_size=3,
    )

    assert aligned["boxes"].squeeze(-1).tolist() == [2.0, 0.0, 1.0]
    assert aligned["names"] == ["two", "zero", "one"]
    assert aligned["scalar"].item() == 5.0

from __future__ import annotations

import pytest
import torch

from src.models.simple_yolo import (
    SimpleYOLOConfig,
    SimpleYOLODetector,
    count_trainable_parameters,
)


def test_simple_yolo_output_shape() -> None:
    config = SimpleYOLOConfig(
        nc=3,
        base_channels=8,
        width_multiplier=0.5,
        channel_multipliers=[1, 2, 4],
        blocks_per_stage=[0, 1, 0],
        output_stride=8,
        boxes_per_cell=2,
    )
    model = SimpleYOLODetector(config)
    output = model(torch.zeros(2, 3, 64, 64))

    assert output.shape == (2, 2, 8, 8, 8)


def test_simple_yolo_width_multiplier_changes_parameter_count() -> None:
    small = SimpleYOLODetector(
        SimpleYOLOConfig(
            nc=1,
            base_channels=8,
            width_multiplier=0.5,
            channel_multipliers=[1, 2, 4],
            blocks_per_stage=[0, 0, 0],
            output_stride=8,
        )
    )
    larger = SimpleYOLODetector(
        SimpleYOLOConfig(
            nc=1,
            base_channels=8,
            width_multiplier=1.5,
            channel_multipliers=[1, 2, 4],
            blocks_per_stage=[0, 0, 0],
            output_stride=8,
        )
    )

    assert count_trainable_parameters(larger) > count_trainable_parameters(small)


def test_simple_yolo_rejects_mismatched_stage_lists() -> None:
    with pytest.raises(ValueError, match="equal length"):
        SimpleYOLODetector(
            SimpleYOLOConfig(
                nc=1,
                channel_multipliers=[1, 2, 4],
                blocks_per_stage=[1, 1],
            )
        )

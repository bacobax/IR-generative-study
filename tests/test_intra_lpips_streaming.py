from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

from src.evaluation import intra_lpips


class _TinyLPIPS(torch.nn.Module):
    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        self.net = net

    def forward(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        return (left - right).abs().mean(dim=(1, 2, 3), keepdim=True)


def _write_image(path: Path, value: int, *, shape: tuple[int, int] = (8, 8)) -> None:
    arr = np.full(shape, value, dtype=np.uint8)
    Image.fromarray(arr, mode="L").save(path)


def test_compute_intra_lpips_streams_batches_without_full_dataset_cat(tmp_path, monkeypatch) -> None:
    real_paths = []
    generated_paths = []
    for idx, value in enumerate([0, 32, 96, 160, 224]):
        path = tmp_path / f"real_{idx}.png"
        _write_image(path, value, shape=(7 + idx, 9 + idx))
        real_paths.append(path)
    for idx, value in enumerate([1, 2, 97, 98]):
        path = tmp_path / f"gen_{idx}.png"
        _write_image(path, value, shape=(12 + idx, 10 + idx))
        generated_paths.append(path)

    monkeypatch.setitem(__import__("sys").modules, "lpips", SimpleNamespace(LPIPS=_TinyLPIPS))

    result = intra_lpips.compute_intra_lpips(
        real_paths=real_paths,
        generated_paths=generated_paths,
        device="cpu",
        batch_size=2,
        resize_to=16,
    )

    assert result.num_real == 5
    assert result.num_generated == 4
    assert result.num_assigned_clusters == 2
    assert result.num_singleton_clusters == 0
    assert result.value > 0.0

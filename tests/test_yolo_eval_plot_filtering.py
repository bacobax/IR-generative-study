"""Tests for filtered YOLO evaluation confusion-matrix plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.cli.train_yolo import _save_filtered_confusion_matrix_plots


class _DummyConfusionMatrix:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix


class _DummyResults:
    def __init__(self) -> None:
        self.names = {0: "person", 1: "bike", 2: "car", 3: "dog"}
        self.nt_per_class = np.array([6, 0, 3, 0], dtype=int)
        self.confusion_matrix = _DummyConfusionMatrix(
            np.array(
                [
                    [5, 0, 0, 0, 0],  # pred person
                    [0, 0, 0, 0, 0],  # pred bike -> truly empty, should disappear
                    [0, 0, 3, 0, 0],  # pred car
                    [0, 0, 0, 0, 2],  # pred dog -> false positives only, should stay
                    [1, 0, 0, 0, 0],  # background row
                ],
                dtype=float,
            )
        )


def test_save_filtered_confusion_matrix_plots_drops_empty_classes(tmp_path: Path) -> None:
    results = _DummyResults()
    payload = _save_filtered_confusion_matrix_plots(results, analysis_dir=tmp_path)

    assert payload["filtered"] is True
    assert payload["kept_class_labels"] == ["person", "car", "dog"]
    assert payload["dropped_class_indices"] == [1]
    assert (tmp_path / "confusion_matrix.png").exists()
    assert (tmp_path / "confusion_matrix_normalized.png").exists()
    assert (tmp_path / "confusion_matrix_filtering.json").exists()
    assert (tmp_path / "eval_class_support.csv").exists()

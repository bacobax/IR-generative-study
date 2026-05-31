from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.core.normalization import SENTINEL2_REFLECTANCE, UINT8_LINEAR
from src.evaluation.feature_extractors import load_image_rgb


def test_load_image_rgb_accepts_sentinel2_tiff(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.tif"
    arr = np.array([[0, 5000], [10000, 12000]], dtype=np.uint16)
    Image.fromarray(arr).save(image_path)

    image = load_image_rgb(image_path, normalization_mode=SENTINEL2_REFLECTANCE)

    assert image.mode == "RGB"
    assert image.size == (2, 2)
    assert np.asarray(image)[0, 0].tolist() == [0, 0, 0]
    assert np.asarray(image)[1, 0].tolist() == [255, 255, 255]


def test_load_image_rgb_keeps_png_jpeg_behavior(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(image_path)

    image = load_image_rgb(image_path, normalization_mode=UINT8_LINEAR)

    assert image.mode == "RGB"
    assert image.size == (2, 2)

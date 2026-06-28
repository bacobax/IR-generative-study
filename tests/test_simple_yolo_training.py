from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.algorithms.training.simple_yolo_detector import (
    SimpleYoloLoss,
    SimpleYoloDataset,
    build_simple_yolo_targets,
    load_simple_yolo_checkpoint,
    render_detection_overlay,
    resolve_yolo_split_info,
)
from src.evaluation.detection_metrics import DetectionPrediction
from src.cli.train_yolo import (
    _FLAT_TO_NESTED,
    _validate_config_yaml_keys,
    build_parser,
    run_eval,
    run_train,
)
from src.core.configs.config_loader import merge_config_and_cli
from src.core.configs.yolo_experiment_config import YOLOExperimentConfig
from src.models.simple_yolo import SimpleYOLOConfig, SimpleYOLODetector


def _write_tiny_yolo_dataset(root: Path) -> Path:
    for split in ("train", "val", "test"):
        image_dir = root / split / "images" / split
        label_dir = root / split / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(2):
            arr = np.zeros((32, 32, 3), dtype=np.uint8)
            arr[8:20, 10:22, :] = 220
            stem = f"{split}_{idx}"
            Image.fromarray(arr).save(image_dir / f"{stem}.png")
            (label_dir / f"{stem}.txt").write_text("0 0.500000 0.500000 0.375000 0.375000\n", encoding="utf-8")
    yaml_path = root / "dataset.yaml"
    yaml_path.write_text(
        "\n".join(
            [
                f"path: {root.resolve()}",
                f"train: {(root / 'train' / 'images' / 'train').resolve()}",
                f"val: {(root / 'val' / 'images' / 'val').resolve()}",
                f"test: {(root / 'test' / 'images' / 'test').resolve()}",
                "names:",
                "  0: person",
                "nc: 1",
            ]
        ),
        encoding="utf-8",
    )
    return yaml_path


def _tiny_cfg(tmp_path: Path, dataset_yaml: Path) -> YOLOExperimentConfig:
    cfg = YOLOExperimentConfig()
    cfg.device = "cpu"
    cfg.data.dataset_yaml = str(dataset_yaml)
    cfg.data.test_dataset_yaml = str(dataset_yaml)
    cfg.data.batch_size = 2
    cfg.data.workers = 0
    cfg.data.image_size = 32
    cfg.model.backend = "simple_torch"
    cfg.model.weights = ""
    cfg.model.simple.base_channels = 8
    cfg.model.simple.width_multiplier = 0.5
    cfg.model.simple.channel_multipliers = [1, 2, 4]
    cfg.model.simple.blocks_per_stage = [0, 0, 0]
    cfg.model.simple.output_stride = 8
    cfg.model.simple.boxes_per_cell = 1
    cfg.training.epochs = 1
    cfg.training.lr0 = 0.001
    cfg.training.optimizer = "AdamW"
    cfg.training.patience = 5
    cfg.training.mixed_precision = "no"
    cfg.training.val_interval = 1
    cfg.training.tensorboard_image_interval = 1
    cfg.training.tensorboard_max_images = 2
    cfg.training.tensorboard_prediction_conf = 0.1
    cfg.evaluation.dataset_yaml = str(dataset_yaml)
    cfg.evaluation.split = "test"
    cfg.evaluation.conf = 0.001
    cfg.evaluation.iou = 0.5
    cfg.evaluation.per_slice_enabled = True
    cfg.evaluation.slice_threshold_dataset_yaml = str(dataset_yaml)
    cfg.output.experiment_name = "tiny_native"
    cfg.output.runs_root = str(tmp_path / "runs")
    cfg.output.checkpoints_root = str(tmp_path / "checkpoints")
    cfg.output.analysis_root = str(tmp_path / "analysis")
    return cfg


def test_build_targets_drops_same_cell_overflow() -> None:
    targets = build_simple_yolo_targets(
        boxes_xywh=[torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.51, 0.51, 0.1, 0.1]])],
        class_ids=[torch.tensor([0, 0])],
        batch_size=1,
        grid_h=4,
        grid_w=4,
        boxes_per_cell=1,
        nc=1,
        device=torch.device("cpu"),
    )

    assert targets.assigned_count == 1
    assert targets.dropped_count == 1


def test_simple_yolo_loss_is_finite_for_empty_labels() -> None:
    model = SimpleYOLODetector(
        SimpleYOLOConfig(
            nc=1,
            base_channels=8,
            width_multiplier=0.5,
            channel_multipliers=[1, 2, 4],
            blocks_per_stage=[0, 0, 0],
            output_stride=8,
            boxes_per_cell=1,
        )
    )
    criterion = SimpleYoloLoss(YOLOExperimentConfig().loss)
    output = model(torch.zeros(1, 3, 32, 32))
    loss, parts = criterion(output, boxes_xywh=[torch.zeros((0, 4))], class_ids=[torch.zeros(0, dtype=torch.long)])

    assert torch.isfinite(loss)
    assert parts["assigned_targets"] == 0.0


def test_render_detection_overlay_returns_rgb_tensor() -> None:
    image = torch.zeros(3, 32, 32)
    prediction = DetectionPrediction(
        image_id="img",
        boxes_xyxy=np.asarray([[0.2, 0.2, 0.7, 0.7]], dtype=np.float32),
        scores=np.asarray([0.9], dtype=np.float32),
        class_ids=np.asarray([0], dtype=np.int32),
    )

    rendered = render_detection_overlay(
        image,
        gt_boxes_xywh=torch.tensor([[0.5, 0.5, 0.4, 0.4]], dtype=torch.float32),
        gt_class_ids=torch.tensor([0], dtype=torch.long),
        prediction=prediction,
        names={0: "person"},
    )

    assert rendered.shape == (3, 32, 32)
    assert float(rendered.max()) > 0.0


def test_simple_yolo_cli_mapping_accepts_native_hparams() -> None:
    parser = build_parser()
    argv = [
        "--model_backend", "simple_torch",
        "--simple_channel_multipliers", "1,2,4",
        "--simple_blocks_per_stage", "0,1,0",
        "--box_weight", "7.0",
    ]
    args = parser.parse_args(argv)
    cfg = merge_config_and_cli(
        YOLOExperimentConfig,
        None,
        parser,
        args,
        flat_to_nested=_FLAT_TO_NESTED,
        cli_argv=argv,
    )

    assert cfg.model.backend == "simple_torch"
    assert cfg.model.simple.channel_multipliers == [1, 2, 4]
    assert cfg.model.simple.blocks_per_stage == [0, 1, 0]
    assert cfg.loss.box_weight == 7.0


def test_simple_yolo_config_yaml_validates() -> None:
    _validate_config_yaml_keys("configs/yolo/exp_v18_simple_yolo_tiny/small.yaml")


def test_simple_yolo_one_epoch_train_eval_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    dataset_yaml = _write_tiny_yolo_dataset(tmp_path / "dataset")
    cfg = _tiny_cfg(tmp_path, dataset_yaml)

    train_summary = run_train(cfg)
    eval_summary = run_eval(cfg, weights_path=train_summary["best_weights_path"])
    model, payload = load_simple_yolo_checkpoint(train_summary["best_weights_path"], map_location="cpu")

    assert Path(train_summary["best_weights_path"]).exists()
    assert Path(train_summary["loss_history_csv"]).exists()
    assert eval_summary["backend"] == "simple_torch"
    assert "map50" in eval_summary
    assert (cfg.output.analysis_dir() / "per_slice_metrics.csv").exists()
    assert payload["nc"] == 1
    assert model.config.nc == 1


def test_cache_images_produces_identical_output(tmp_path: Path) -> None:
    """Cached and uncached SimpleYoloDataset[i] must return identical tensors."""
    dataset_yaml = _write_tiny_yolo_dataset(tmp_path / "ds")
    split_info = resolve_yolo_split_info(str(dataset_yaml), split="train")

    ds_nocache = SimpleYoloDataset(split_info, image_size=32, input_channels=3, cache_images=False)
    ds_cached = SimpleYoloDataset(split_info, image_size=32, input_channels=3, cache_images=True)

    assert len(ds_nocache) == len(ds_cached)
    for i in range(len(ds_nocache)):
        item_a = ds_nocache[i]
        item_b = ds_cached[i]
        assert torch.allclose(item_a["image"], item_b["image"]), f"image mismatch at index {i}"
        assert torch.equal(item_a["boxes_xywh"], item_b["boxes_xywh"]), f"boxes mismatch at index {i}"
        assert torch.equal(item_a["class_ids"], item_b["class_ids"]), f"class_ids mismatch at index {i}"


def test_general_aug_produces_valid_boxes(tmp_path: Path) -> None:
    """General augmentation must keep boxes in [0,1] and non-degenerate."""
    import random as _random
    from src.core.configs.yolo_experiment_config import YOLOAugConfig

    dataset_yaml = _write_tiny_yolo_dataset(tmp_path / "ds")
    split_info = resolve_yolo_split_info(str(dataset_yaml), split="train")

    aug_cfg = YOLOAugConfig(enabled=True, fliplr=1.0, scale=0.3, translate=0.1, brightness=0.3)
    _random.seed(42)

    ds = SimpleYoloDataset(
        split_info,
        image_size=32,
        input_channels=3,
        augment=True,
        aug_cfg=aug_cfg,
        cache_images=True,
    )
    for i in range(len(ds)):
        item = ds[i]
        img = item["image"]
        boxes = item["boxes_xywh"]
        assert img.shape == (3, 32, 32), f"unexpected image shape {img.shape}"
        assert float(img.min()) >= 0.0 and float(img.max()) <= 1.0
        if len(boxes) > 0:
            cx = boxes[:, 0]; cy = boxes[:, 1]; w = boxes[:, 2]; h = boxes[:, 3]
            assert (cx >= 0).all() and (cx <= 1).all(), "cx out of range"
            assert (cy >= 0).all() and (cy <= 1).all(), "cy out of range"
            assert (w > 1e-4).all(), "degenerate box width"
            assert (h > 1e-4).all(), "degenerate box height"

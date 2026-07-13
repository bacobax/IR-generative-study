# Native Simple YOLO Detector Training

## Purpose
Load this for the in-repo PyTorch `simple_torch` YOLO backend: custom native
model architecture, training, evaluation, checkpointing, and v18 tiny detector
presets. This is separate from the default Ultralytics detector backend.

Do not load this for Experiment B generator changes, RegionDiff/STAY generation,
or Ultralytics-specific trainer hooks unless the request also touches those
systems.

## CLI & config
Entrypoint stays `src/cli/train_yolo.py`; select the native backend with:

```bash
python -m src.cli.train_yolo --action train --config configs/yolo/exp_v18_simple_yolo_tiny/small.yaml
python -m src.cli.train_yolo --action eval --config configs/yolo/exp_v18_simple_yolo_tiny/small.yaml
python -m src.cli.train_yolo --action eval_slices --config configs/yolo/exp_v18_simple_yolo_tiny/small.yaml
```

`model.backend: simple_torch` dispatches to native PyTorch code. Existing
Ultralytics configs keep `model.backend: ultralytics` by default.

Key hparams:
- `model.simple.base_channels`, `width_multiplier`, `channel_multipliers`,
  `blocks_per_stage` scale model size.
- `model.simple.output_stride` and `boxes_per_cell` control grid density and
  same-cell capacity. `data.image_size` must be divisible by `output_stride`.
- `loss.*` controls box, GIoU, objectness, no-object, and class loss weights.
- `training.mixed_precision`, `grad_clip_norm`, and `val_interval` control
  native trainer runtime behavior.
- `training.tensorboard_image_interval`, `tensorboard_max_images`, and
  `tensorboard_prediction_conf` control validation image overlays. Set the
  interval to `0` to disable image logging.

## Primary files
- `src/models/simple_yolo.py` — configurable single-scale detector.
- `src/algorithms/training/simple_yolo_detector.py` — YOLO YAML dataset loader,
  native loss, train/eval loop, checkpoint IO, prediction decoding.
- `src/evaluation/detection_metrics.py` — native IoU, NMS, AP, and prediction
  containers.
- `src/analysis/flir_subgroup/yolo_slice_eval.py` — per-slice metrics for both
  Ultralytics inference and native prediction arrays.
- `configs/yolo/exp_v18_simple_yolo_tiny/small.yaml` — first small v18 preset.

## Outputs
Native runs write under the existing YOLO roots:
`artifacts/runs/yolo/...`, `artifacts/checkpoints/yolo/...`, and
`artifacts/analysis/yolo/...`.

Expected artifacts include `best.pt`, `last.pt`, `resolved_config.json`,
`train_summary.json/csv`, `loss_history.csv`, `eval_summary.json/csv`,
`per_class_metrics.csv`, and optional per-slice files when
`evaluation.per_slice_enabled: true`.

TensorBoard logs are written to the run directory reported as
`tensorboard_log_dir`. Native training logs scalar losses/metrics and
`val/detection_overlays`, where green boxes are ground truth and red boxes are
model predictions.

## Modification guidance
Keep dataset format compatible with existing YOLO YAMLs and label folders.
Do not put native training logic into root wrappers. Add architecture changes to
`src/models/simple_yolo.py`, training/eval changes to
`simple_yolo_detector.py`, and metric-only changes to `detection_metrics.py` or
`yolo_slice_eval.py`.

When changing config keys, update `YOLOExperimentConfig`, CLI flat-to-nested
mapping, docs, and focused tests together.

## Validation guidance
Use focused CPU tests:

```bash
python -m pytest tests/test_simple_yolo_model.py tests/test_simple_yolo_training.py tests/test_simple_yolo_eval.py -v
python -m pytest tests/test_yolo_slice_eval.py tests/test_yolo_training_schedule.py tests/test_yolo_v18_scratch_configs.py -v
python scripts/checks/check_config_loading.py
```

Do not launch full v18 training, generation, or Slurm jobs unless explicitly
requested.

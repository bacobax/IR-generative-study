# YOLO Detector Training Routing

## Purpose
Load this for Ultralytics YOLO detector training, evaluation, and the two experiment
harnesses: **Experiment A** (balanced / unbalanced / full-train + rare-slice sampling
baselines) and **Experiment B** (synthetic-augmentation: FM / SD / RegionDiff / STAY generated
images mixed into full-train). Covers the `train_yolo` CLI, the `YOLOExperimentConfig`
dataclass tree, `configs/yolo/`, output/checkpoint/run roots, and the standalone/smoke
runners.

For the in-repo native PyTorch custom/simple YOLO backend (`model.backend:
simple_torch`), also load
`docs/agent_docs/training_objectives/simple_yolo_detector_training.md`.

Do not load this for the generative models that *produce* Experiment B augmentation images
(layout-conditioned FM, SD, RegionDiff) — route to
`docs/agent_docs/high_level_training_types/stay_cond_flow_matching.md` or `region_diff.md`.
Do not load for slice binning / class-rarity balance theory itself — route to
`docs/agent_docs/analysis/layout_slices_balance.md` (Experiment A's rare-slice baselines
consume those helpers).

## CLI & actions
Entrypoint: `src/cli/train_yolo.py` (config-driven Ultralytics wrapper; root wrapper kept
thin per `AGENTS.md`). `--action` choices:
- `train`, `eval` — single run (`run_train`, `run_eval`). `eval` optionally emits per-slice
  metrics when `evaluation.per_slice_enabled: true`.
- `eval_slices` — slice-only re-eval against an existing `best.pt` (skips `model.val()`).
- `run_exp_a`, `run_exp_a_all` — Experiment A launcher over balanced/unbalanced/full_train
  (`run_experiment_a`, `run_experiment_a_all`); emits `comparison_summary.csv`.
- `run_exp_b`, `run_exp_b_all` — Experiment B launcher (`run_experiment_b`,
  `run_experiment_b_all`).

`--baseline` choices `none|baseline_a|baseline_b` select rare-slice sampling mode;
`--experiment_b_mode` choices `plain|fm_aug|sd_aug|precomputed_aug`.

## Config schema
`src/core/configs/yolo_experiment_config.py` — `YOLOExperimentConfig` nests:
- `data` (`YOLODataConfig`) — `dataset_yaml`, `balanced/unbalanced/full_train/test_dataset_yaml`,
  `batch_size`, `workers`, `image_size`.
- `model` (`YOLOModelConfig`) — `weights` (e.g. `yolov8{n,m,l}.pt`), `task`,
  `backend` (`ultralytics` by default; `simple_torch` for the native detector),
  and native `simple` architecture hparams.
- `training` (`YOLOTrainConfig`) — `epochs`, `lr0`, `optimizer`, `seed`, `deterministic`,
  `patience`, `cos_lr`, backbone freeze/LR (`freeze_backbone_epochs`,
  `freeze_backbone_layers`, `backbone_lr_multiplier`, `backbone_param_prefixes`),
  plus native-backend runtime knobs (`mixed_precision`, `grad_clip_norm`,
  `val_interval`).
- `loss` (`YOLOLossConfig`) — native simple YOLO loss weights; ignored by the
  default Ultralytics backend.
- `baseline` (`YOLOBaselineConfig`) — rare-slice sampling: `mode`, `rarity_alpha`,
  `image_score_top_k`, `clip_weight_*`, `sampler_replacement`,
  `use_weighted_sampler` (default `True`; set `False` to apply targeted aug without
  oversampling), targeted-aug geometry knobs.
- `evaluation` (`YOLOEvalConfig`) — `dataset_yaml`, `split`, `save_json`, `save_hybrid`,
  `conf`, `iou`, `per_slice_enabled` (default `False`; set `True` to run per-slice mAP after
  `model.val()`), `slice_threshold_dataset_yaml` (path to training split YAML for frozen
  size-bin tertiles; falls back to `data.dataset_yaml` if `None`).
- `output` (`YOLOOutputConfig`), `launcher` (`YOLOLauncherConfig`), `experiment_b`
  (`YOLOExperimentBConfig` → nested `filter` / `fm` / `sd`).

## Primary files/directories to inspect
Training / orchestration:
- `src/cli/train_yolo.py` — actions, train/eval loops, comparison CSV, eval plots.
- `src/analysis/flir_subgroup/yolo_slice_eval.py` — per-slice mAP module:
  `compute_frozen_thresholds`, `assign_gt_slices`, `_assign_pred_slices`,
  `_build_per_slice_rows_from_entries`, `evaluate_per_slice`,
  `evaluate_per_slice_from_predictions`. TP/FN attributed to GT slice; FP
  attributed to predicted-box geometry. Emits `per_slice_metrics.csv` (27 rows
  + overall) and `per_slice_metrics.json`.
- `src/algorithms/training/yolo_slice_baselines.py` — Experiment A rare-slice sampling
  (`prepare_yolo_slice_baseline`, `build_weighted_train_dataloader`,
  `make_slice_aware_yolo_dataset_class`); see also `analysis/layout_slices_balance.md`.
- `src/algorithms/training/yolo_experiment_b.py` — Experiment B pipeline:
  `validate_experiment_b_config`, `prepare_experiment_b_dataset`, `generate_fm_candidates`,
  `generate_sd_candidates`, `audit_generated_candidates`, `classify_generated_image_rows`,
  `load_precomputed_generated_image_rows`, `build_instance_discard_summary`.
- `src/analysis/flir_subgroup/yolo_export.py` — YOLO-format export / eval support.

Configs (`configs/yolo/`):
- `exp_v18_scratch_yolo11n/` — three from-scratch yolo11n experiments on v18 dataset:
  `_base.yaml`, `default_aug.yaml` (Ultralytics defaults), `plain.yaml` (no aug),
  `rare_aug.yaml` (targeted aug on rare-slice images, no oversampling). All have
  `per_slice_enabled: true`.
- `exp_v18_simple_yolo_tiny/small.yaml` — native PyTorch simple YOLO tiny v18
  starting point (`model.backend: simple_torch`).
- `exp_a/{flir_yolov8n,flir_yolov8m,flir_yolov8l,v18}/` — per-model/dataset; each has
  `_base.yaml`, `exp_balanced.yaml`, `exp_unbalanced.yaml`, `exp_full_train*.yaml`,
  `run_exp_a*.yaml`.
- `exp_b/{flir_yolov8n,flir_yolov8m}/` — `exp_plain.yaml`, `exp_fm_aug.yaml`,
  `exp_sd_aug_{lora,unet}.yaml`, `exp_precomputed_{stay_fm_hflip,regiondiff_fm_ot_hflip}.yaml`,
  `run_exp_b_all.yaml`.
- `exp_b/smoked_e2e/`, `exp_b/synthetic_generation/default.yaml` — smoke + precompute presets.

Paths / outputs (`src/core/paths.py`): `yolo_test_ds_root`, `yolo_runs_root`,
`yolo_checkpoints_root`, `yolo_analysis_root`. Datasets at
`data/derived/yolo-test-ds/{balanced,unbalanced,full_train,test}.yaml`; outputs under
`artifacts/{runs,checkpoints,analysis}/yolo/`.

Standalone & smoke wrappers (delegate to `train_yolo`):
- `scripts/standalone/run_yolo_exp_a_parallel.py`
- `scripts/standalone/train_yolo_exp_b_with_synthetic_aug.py`
- `scripts/standalone/generate_yolo_exp_b_synthetic_counterparts.py`
- `scripts/smoke/run_smoked_e2e_pipeline.py`,
  `scripts/smoke/generate_smoked_regiondiff_dataset.py`.

Tests:
- `tests/test_yolo_experiment_b.py`, `tests/test_yolo_slice_baselines.py`,
  `tests/test_yolo_export.py`, `tests/test_yolo_config_extends_equivalence.py`,
  `tests/test_yolo_eval_plot_filtering.py`, `tests/test_yolo_training_schedule.py`,
  `tests/test_regiondiff_smoke_generation.py`, `tests/test_smoked_e2e_pipeline.py`.
- `tests/fixtures/yolo_config_effective_snapshots.json` — effective-config snapshots
  (guards `extends:` equivalence).

## Decision/routing notes
Start with:
`rg -n "YOLOExperimentConfig|run_exp_a|run_exp_b|experiment_b|yolov8|dataset_yaml" src/cli/train_yolo.py src/core/configs/yolo_experiment_config.py configs/yolo`

- Train/eval loop, action wiring, eval plots → `train_yolo.py`.
- Config keys / nested dataclasses → `yolo_experiment_config.py` (wire new keys through loader
  + defaults + snapshot fixture + tests together).
- Experiment A sampling/balance → `yolo_slice_baselines.py`.
- Experiment B generation/filter → `yolo_experiment_b.py`.
- New preset → matching `configs/yolo/exp_*/<model>/`; YAML holds only diffs from defaults.

## Modification guidance
Keep orchestration single-sourced in `src/cli/train_yolo.py`; standalone/smoke scripts stay
thin `subprocess` wrappers. Training-time helpers under `src/algorithms/training/`, export
under `src/analysis/flir_subgroup/`, presets under `configs/yolo/`. Default all output paths
into `artifacts/.../yolo/` via `src/core/paths.py` helpers. Do not add training loops or
config loading to root wrappers.

## Validation guidance
`python -m pytest tests/test_yolo_experiment_b.py tests/test_yolo_slice_baselines.py tests/test_yolo_export.py -v`.
When touching config keys or `extends:` layering, also run
`python -m pytest tests/test_yolo_config_extends_equivalence.py -v` (regenerate
`tests/fixtures/yolo_config_effective_snapshots.json` only if the change is intentional).
For smoke/e2e wiring: `python -m pytest tests/test_smoked_e2e_pipeline.py tests/test_regiondiff_smoke_generation.py -v`.
Run `scripts/checks/check_*.py` for path/config wiring. Do not launch real YOLO training,
synthetic dataset builds, or Slurm jobs as validation unless explicitly asked.

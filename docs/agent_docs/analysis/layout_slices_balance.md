# Layout Slices Analysis & Balance Routing

## Purpose
Load this for bbox/layout **slice analysis** and **balance** work: assigning detections to
size×position slices, building slice-count/rarity tables, rare-slice weighted sampling,
inverse-frequency class balancing, area-based spatial loss weighting, rare-layout generation,
and balanced-vs-unbalanced dataset experiments.

Do not load this for generation-side layout conditioning (layout tokenizer, region attention,
STAY UNet) — route to `docs/agent_docs/high_level_training_types/stay_cond_flow_matching.md`
or `region_diff.md` — unless the request also touches slice analysis or balancing. Do not load
for generic dataset loading/normalization (route to the `datasets/` docs).

## Concept
A **slice** is a detection subgroup keyed by **size bin × position bin**:
- Size bins `small/medium/large` come from global tertile thresholds `q33/q67`.
- Position bins are a 3×3 grid (`top/middle/bottom` × `left/center/right`) from normalized
  bbox centers.

**Balance** = correcting uneven slice/class occupancy via sampling weights (rare-slice and
inverse-frequency), spatial loss weights (area-inverse), targeted rare-layout generation, or
curated balanced manifests.

## Primary files/directories to inspect
Slice definition & stat tables:
- `src/analysis/flir_subgroup/yolo_slice_stats.py` — `YoloSliceThresholds`,
  `assign_bins_from_thresholds`, `add_position_bin_columns` (`POSITION_BIN_ORDER`,
  `SIZE_BIN_ORDER`), `YoloSliceDataset`, `load_yolo_slice_dataset`,
  `build_slice_counts_table`.

Sampling-time balance (rare-slice + rare-layout augmentation):
- `src/algorithms/training/yolo_slice_baselines.py` — `prepare_yolo_slice_baseline`,
  `_rarity_from_counts`, `_build_image_sampling_table`,
  `build_weighted_train_dataloader` (wraps `torch.utils.data.WeightedRandomSampler`),
  `make_slice_aware_yolo_dataset_class`, geometry helpers (`apply_translation`,
  `apply_center_scale`, `apply_horizontal_flip`, `apply_constrained_crop_resize`,
  xywh/xyxy converters). Emits `image_sampling_weights.csv`,
  `sampling_weight_summary.json`.

Class balance (inverse frequency):
- `src/core/data/foreground_background_dataset.py` — `build_balanced_sample_weights`,
  `MultiClassCropDataset` (per-sample `model_label_index`).

Loss-time balance (area-inverse spatial weighting):
- `src/models/regiondiffusion.py` — `build_area_weight_map`.
- `src/core/configs/fm_config.py` — `LayoutConditioningConfig.area_loss_*`
  (`area_loss_enabled`, `area_loss_alpha`, `area_loss_background_weight`,
  `area_loss_min_weight`, `area_loss_max_weight`).
- `src/algorithms/training/flow_matching_trainer.py` —
  `_uses_regiondiff_area_loss`, `_apply_regiondiff_area_loss_weights`.

Underlying bbox plumbing (referenced, owned by dataset/conditioning docs):
- `src/core/data/schema.py` — `LayoutFields`.
- `src/core/data/annotations.py` — `coco_bbox_to_xyxy`, `get_boxes_and_labels_for_image`.
- `src/core/data/layout_batching.py` — `collate_layout_batch`.

Configs:
- `configs/auxiliary/rare_layout_generation/presets/default.yaml` —
  `selection_mode: rare_first`, `rarity_aggregation`.
- `configs/yolo/exp_a/<model>/exp_balanced.yaml` vs `exp_unbalanced.yaml` —
  balanced-vs-imbalanced YOLO baselines.

Per-slice evaluation module (uses slice helpers above):
- `src/analysis/flir_subgroup/yolo_slice_eval.py` — `compute_frozen_thresholds`,
  `assign_gt_slices`, `evaluate_per_slice`, `_build_per_slice_rows_from_entries`.

Notebooks & tests:
- `docs/notebooks/flir_slice_subset_study.ipynb`,
  `docs/notebooks/flir_proxy_yolo_split_analysis.ipynb`,
  `docs/notebooks/rare_layout_flir_sampling_with_fgbg_filter.ipynb`.
- `tests/test_flir_subgroup_analysis.py`,
  `tests/test_flir_subgroup_notebook_parity.py`.

## Decision/routing notes
For per-slice mAP evaluation (AP50/AP50-95 per size×position cell, run after model.val()):
route to `src/analysis/flir_subgroup/yolo_slice_eval.py` (`compute_frozen_thresholds`,
`evaluate_per_slice`); controlled by `YOLOEvalConfig.per_slice_enabled` and surfaced via
`--action eval_slices` in `src/cli/train_yolo.py`.

Start with:
`rg -n "slice|rarity|sampling_weight|balanced|area_loss|position_bin|size_bin" src configs tests docs`

- Slice binning / stat tables → `yolo_slice_stats.py`.
- Rare-slice sampler, rarity scoring, weighted dataloader → `yolo_slice_baselines.py`.
- Inverse-frequency class weights → `foreground_background_dataset.py`.
- Area-inverse spatial loss weights → `build_area_weight_map` in `regiondiffusion.py` plus
  `area_loss_*` keys in `fm_config.py` and the apply hooks in `flow_matching_trainer.py`.
- Rare-layout generation behavior → `configs/auxiliary/rare_layout_generation/`.

Slice math must stay consistent between the analysis helpers and the notebooks — parity is
guarded by `tests/test_flir_subgroup_notebook_parity.py`.

## Modification guidance
Put slice/analysis helpers under `src/analysis/flir_subgroup/`, training-time sampling and
loss balancing under `src/algorithms/training/`, class-weight helpers under
`src/core/data/`, and config presets under `configs/auxiliary/rare_layout_generation/` or
`configs/yolo/`. Keep root wrappers thin. New config keys must be wired through the loader,
dataclass defaults, and tests together (see `AGENTS.md` config conventions).

## Validation guidance
Run `python -m pytest tests/test_flir_subgroup_analysis.py -v` and, when slice math or
notebook parity changes, `python -m pytest tests/test_flir_subgroup_notebook_parity.py -v`.
When touching config keys or paths, run the relevant `scripts/checks/check_*.py`
(e.g. `check_config_loading.py`, `check_repo_paths.py`). Do not run expensive data builds,
YOLO training, or Slurm jobs as validation unless explicitly asked.

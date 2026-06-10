# FLIR Dataset Routing

## Purpose
Load this for FLIR dataset requests, including FLIR private/proxy alignment, `flir_private_proxy_alignment_v18`, image serving, subgroup analysis, dataset selectors, annotations, transforms, normalization, generated FLIR subsets, and FLIR-specific training configs.

Do not load this for generic SDXL, RegionDiff, or STAY work unless the request also touches FLIR data, labels, splits, manifests, or analysis surfaces.

## Primary files/directories to inspect
- `src/core/data/`
- `src/core/data/dataset_targets.py`
- `src/core/data/training_data.py`
- `src/core/normalization.py`
- `src/core/paths.py`
- `src/analysis/flir_subgroup/`
- `src/analysis/flir_subgroup/datasets.py`
- `src/analysis/flir_subgroup/data.py`
- `src/analysis/flir_subgroup/api.py`
- `src/analysis/flir_subgroup/app.py`
- `src/cli/serve_flir_analysis.py`
- `frontend/flir-subgroup-analysis/src/`
- `configs/data/presets/flir_private_proxy_alignment_layout.yaml`
- `configs/datasets/flir/`
- `configs/eval/flir_stage1_single_runs/`
- `configs/eval/publication_single_runs/flir/`
- `configs/yolo/exp_a/flir_yolov8n/`
- `configs/yolo/exp_b/flir_yolov8n/`
- `scripts/standalone/build_flir_private_proxy_alignment_dataset.py`
- `scripts/standalone/build_flir_private_proxy_v18_dataset.py`
- `scripts/standalone/download_and_build_flir_full_thermal_dataset.py`
- `tests/test_flir_subgroup_analysis.py`
- `tests/test_flir_subgroup_notebook_parity.py`
- `tests/test_build_flir_private_proxy_alignment_dataset.py`

## Decision/routing notes
Use `rg -n "flir|flir_private_proxy_alignment|dataset_id|image_key|holdout" src configs tests frontend docs scripts` first. For API/frontend analysis behavior, route to `src/analysis/flir_subgroup/` and `frontend/flir-subgroup-analysis/`. For trainer dataloading, route to `src/core/data/` and `configs/datasets/flir/`. For image normalization, inspect `src/core/normalization.py`; FLIR-style uint16 paths usually use repo-native single-channel normalization.

## Modification guidance
Put API analysis changes under `src/analysis/flir_subgroup/`, UI changes under `frontend/flir-subgroup-analysis/`, dataset construction scripts under `scripts/standalone/`, and training config presets under `configs/datasets/flir/` or the relevant `configs/<objective>/` area. Keep `serve_flir_analysis.py` and other root wrappers thin.

## Validation guidance
For analysis/API changes run `python -m pytest tests/test_flir_subgroup_analysis.py -v` and, when notebook parity matters, `python -m pytest tests/test_flir_subgroup_notebook_parity.py -v`. For frontend changes run `cd frontend/flir-subgroup-analysis && npm run build`. For dataset construction changes run `python -m pytest tests/test_build_flir_private_proxy_alignment_dataset.py -v` plus targeted path/config checks. Do not run expensive data builds or Slurm jobs unless requested.

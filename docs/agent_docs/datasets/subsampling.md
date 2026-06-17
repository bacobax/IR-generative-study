# Subsampling And Manifest Routing

## Purpose
Load this for subset manifests, sampling ratios, train-size variants, class/domain balancing, train/val split filters, holdouts, checkpoint-selection subsets, synthetic/derived dataset manifests, or deterministic debug subsets.

Do not load objective or high-level training docs unless the sampling request changes trainer behavior, loss behavior, or generated sample semantics.

## Primary files/directories to inspect
- `src/core/data/subset_manifest.py`
- `src/core/data/datasets.py`
- `src/core/data/training_data.py`
- `src/core/data/dataset_targets.py`
- `scripts/datasets/create_subset_manifest.py`
- `src/evaluation/checkpoint_selection/`
- `scripts/select_best_checkpoint_and_compute_metrics.py`
- `scripts/recover_checkpoint_selection_publication.py`
- `scripts/recover_all_checkpoint_selection_publication.sh`
- `configs/datasets/flir/`
- `configs/datasets/bigearthnet_s2_b08_5x5_stride3/`
- `configs/eval/`
- `slurm/fir/flir/checkpoint_selection/`
- `slurm/fir/bigearthnet_s2_b08_5x5_stride3/checkpoint_selection/`
- `slurm/killarney/flir/checkpoint_selection_publication/`
- `tests/test_subset_manifest.py`
- `tests/test_create_subset_manifest.py`
- `tests/test_checkpoint_selection_pipeline.py`
- `tests/test_checkpoint_selection_viewer.py`

## Decision/routing notes
Use `rg -n "subset_manifest|manifest|holdout|train_2000|train_2040|train_5000|train_5100|checkpoint_selection" src configs scripts slurm tests` first. If the request is about which samples are included, stay in manifest/config code. If it is about how selected samples are trained, combine this with the appropriate training regime/objective doc.

## Modification guidance
Manifest parsing/filtering belongs in `src/core/data/subset_manifest.py` and dataset consumers in `src/core/data/`. Manifest creation belongs in `scripts/datasets/create_subset_manifest.py`. Selection config changes belong under `configs/eval/` or dataset-specific `configs/datasets/`. Do not hard-code subset policy in root wrappers or Slurm launchers.

## Validation guidance
Run `python -m pytest tests/test_subset_manifest.py -v`, `python -m pytest tests/test_create_subset_manifest.py -v`, and checkpoint-selection tests only when touched. Run `python scripts/checks/check_config_loading.py` for YAML changes. Avoid launching selection sweeps unless requested.

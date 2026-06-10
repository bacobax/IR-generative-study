# BigEarthNet S2 B08 5x5 Stride3 Dataset Routing

## Purpose
Load this for BigEarthNet/Sentinel-2/B08/5x5/stride3 dataset requests: mosaic generation, manifests, split compliance, Sentinel-2 normalization, transforms, dataset configs, subset train sizes, or BigEarthNet checkpoint-selection wiring.

Do not load this for FLIR, v18, STAY, or RegionDiff unless the user explicitly combines those with BigEarthNet.

## Primary files/directories to inspect
- `docs/bigearthnet_s2_b08_5x5_protocol.md`
- `docs/notebooks/bigearthnet_s2_b08_5x5_creation.ipynb`
- `scripts/datasets/explore_bigearthnet_s2_mosaics.py`
- `src/core/data/`
- `src/core/data/datasets.py`
- `src/core/data/dataset_targets.py`
- `src/core/data/subset_manifest.py`
- `src/core/normalization.py`
- `src/core/paths.py`
- `src/core/configs/`
- `configs/datasets/bigearthnet_s2_b08_5x5_stride3/`
- `configs/eval/bigearthnet_s2_b08_5x5_stride3_stage1_single_runs/`
- `configs/eval/publication_single_runs/bigearthnet_s2_b08_5x5_stride3/`
- `configs/sdxl/train/presets/bigearthnet_s2_b08_5x5_stride3_lora_stage1_r8.yaml`
- `slurm/fir/bigearthnet_s2_b08_5x5_stride3/`
- `slurm/killarney/bigearthnet_s2_b08_5x5_stride3/`
- `tests/test_dataset_adapters.py`
- `tests/test_create_subset_manifest.py`

## Decision/routing notes
Use `rg -n "bigearthnet|s2_b08|sentinel2|SENTINEL2_REFLECTANCE|manifest|train_2040|train_5100" src configs scripts tests docs` first. For preprocessing and split policy, start with the protocol doc and `scripts/datasets/explore_bigearthnet_s2_mosaics.py`. For training, start from `configs/datasets/bigearthnet_s2_b08_5x5_stride3/` and then route to the objective/regime docs only if training behavior changes.

## Modification guidance
Put dataset generation or exploration changes in `scripts/datasets/`, shared loading in `src/core/data/`, normalization changes in `src/core/normalization.py`, and presets under `configs/datasets/bigearthnet_s2_b08_5x5_stride3/`. Keep cluster runtime details in Slurm files and experiment behavior in YAML. Do not edit vendored `src/diffusers/`.

## Validation guidance
Prefer `python -m pytest tests/test_dataset_adapters.py -v`, `python -m pytest tests/test_create_subset_manifest.py -v`, `python scripts/checks/check_named_dataset_loader.py`, and `python scripts/checks/check_config_loading.py`. Do not generate mosaics, train models, or submit Slurm jobs unless explicitly asked.

# SDXL Adaptation Routing

## Purpose
Load this for Stable Diffusion XL adaptation as a high-level task: SDXL LoRA, stage-1 manifests, SDXL generation, dataset-specific SDXL presets, and SDXL Slurm launchers.

Do not load SD15 or STAY docs unless the user explicitly mentions them.

## Primary files/directories to inspect
- `docs/agent_docs/training_regime/finetuning_sdxl.md`
- `docs/agent_docs/training_objectives/diffusion.md`
- `src/cli/train_sdxl.py`
- `src/algorithms/stable_diffusion_xl/`
- `configs/sdxl/train/`
- `configs/sdxl/generate/`
- `configs/datasets/flir/sdxl_adaptation/`
- `configs/datasets/bigearthnet_s2_b08_5x5_stride3/sdxl_adaptation/`
- `scripts/train/sdxl_lora_r8.sh`
- `scripts/generate/sdxl_r8.sh`
- `slurm/resume_interrupted_sdxl_lora_runs.sh`
- `tests/test_sdxl_stage1.py`

## Decision/routing notes
Combine with the relevant dataset doc only when dataset handling is part of the request. If RegionDiff on top of SDXL is requested, route to `docs/agent_docs/high_level_training_types/region_diff.md`; otherwise keep SDXL separate from SD15.

## Modification guidance
SDXL changes usually belong in `src/algorithms/stable_diffusion_xl/`, `src/cli/train_sdxl.py`, and `configs/sdxl/` or dataset-specific SDXL configs. Root `train_sdxl.py` stays thin.

## Validation guidance
Run `python -m pytest tests/test_sdxl_stage1.py -v` and config/launcher checks if touched. Avoid SDXL training or generation unless requested.

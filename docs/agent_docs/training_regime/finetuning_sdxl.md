# SDXL Fine-Tuning Regime Routing

## Purpose
Load this for Stable Diffusion XL adaptation/fine-tuning requests: SDXL LoRA configs, stage-1 manifests, SDXL data pipelines, checkpoint loading, generation, validation, and SDXL launcher wiring.

Do not load this for SD 1.5 adaptation; route SD15 requests to `training_regime/finetuning_sd15.md`.

## Primary files/directories to inspect
- `src/cli/train_sdxl.py`
- `src/algorithms/stable_diffusion_xl/config.py`
- `src/algorithms/stable_diffusion_xl/data.py`
- `src/algorithms/stable_diffusion_xl/models.py`
- `src/algorithms/stable_diffusion_xl/training.py`
- `src/cli/generate.py`
- `configs/sdxl/train/default.yaml`
- `configs/sdxl/train/presets/`
- `configs/sdxl/generate/presets/`
- `configs/datasets/flir/sdxl_adaptation/`
- `configs/datasets/bigearthnet_s2_b08_5x5_stride3/sdxl_adaptation/`
- `scripts/train/sdxl_lora_r8.sh`
- `scripts/generate/sdxl_r8.sh`
- `slurm/resume_interrupted_sdxl_lora_runs.sh`
- `slurm/fir/flir/sd_adaptation/`
- `slurm/fir/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/`
- `tests/test_sdxl_stage1.py`

## Decision/routing notes
Use `rg -n "sdxl|stage1_manifest|pooled|time_ids|lora_target_modules" src configs scripts slurm tests` first. Dataset-specific SDXL requests may need only the relevant dataset doc plus SDXL config files. RegionDiff requests on top of SDXL should first load `docs/agent_docs/high_level_training_types/region_diff.md`.

## Modification guidance
Put SDXL behavior in `src/algorithms/stable_diffusion_xl/`, CLI wiring in `src/cli/train_sdxl.py`, and presets in `configs/sdxl/`, `configs/datasets/flir/sdxl_adaptation/`, or `configs/datasets/bigearthnet_s2_b08_5x5_stride3/sdxl_adaptation/`. Keep root `train_sdxl.py` thin. Do not mix SD15-only config fields into SDXL config unless compatibility code already supports them.

## Validation guidance
Run `python -m pytest tests/test_sdxl_stage1.py -v`. For config/launcher changes run `python scripts/checks/check_config_loading.py`, `python scripts/checks/check_shell_launchers.py`, or `python scripts/checks/check_slurm_launchers.py` as appropriate. Do not run SDXL training or generation unless explicitly requested.

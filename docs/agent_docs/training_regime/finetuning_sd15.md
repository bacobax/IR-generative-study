# SD15 Fine-Tuning Regime Routing

## Purpose
Load this for Stable Diffusion 1.5 adaptation/fine-tuning requests: LoRA, full/partial UNet adaptation, DomainStudio configs, stage-1 checkpoint paths, SD15 generation presets, validation, and SD15 launcher wiring.

Do not load this for SDXL fine-tuning; route SDXL requests to `training_regime/finetuning_sdxl.md`.

## Primary files/directories to inspect
- `src/cli/adapt_stable_diffusion.py`
- `src/cli/adapt_stable_diffusion_stage1.py`
- `src/algorithms/stable_diffusion/config.py`
- `src/algorithms/stable_diffusion/data.py`
- `src/algorithms/stable_diffusion/domainstudio.py`
- `src/algorithms/stable_diffusion/models.py`
- `src/algorithms/stable_diffusion/training.py`
- `src/algorithms/stable_diffusion/helpers.py`
- `src/cli/generate.py`
- `configs/sd/train/`
- `configs/sd/generate/`
- `configs/datasets/flir/sd_adaptation/`
- `configs/datasets/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/`
- `configs/models/sd/sd15.yaml`
- `scripts/train/sd_flir_lora_stage1.sh`
- `scripts/train/sd_flir_unet_full_stage1.sh`
- `scripts/train/sd_lora_r8.sh`
- `scripts/generate/sd_r8.sh`
- `tests/test_sd_ir_baselines.py`
- `tests/test_sd_domainstudio.py`
- `tests/test_sd_stage_chain_launcher.py`

## Decision/routing notes
Use `rg -n "lora|stage1|sd15|domainstudio|pretrained_model_name_or_path" src configs scripts tests` first. If the request is about RegionDiff stage 2 on top of SD15, also load `docs/agent_docs/high_level_training_types/region_diff.md`. If the request is only dataset paths or subset sizes, load the relevant dataset doc instead.

## Modification guidance
Put SD15 training behavior in `src/algorithms/stable_diffusion/`, CLI/config wiring in `src/cli/adapt_stable_diffusion.py` and `src/algorithms/stable_diffusion/config.py`, and presets under `configs/sd/`, `configs/datasets/flir/sd_adaptation/`, or `configs/datasets/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/`. Keep `adapt_stable_diffusion.py` at repo root thin.

## Validation guidance
Run targeted SD15 tests such as `python -m pytest tests/test_sd_ir_baselines.py -v`, `python -m pytest tests/test_sd_domainstudio.py -v`, and `python -m pytest tests/test_sd_stage_chain_launcher.py -v`. For config/launcher changes run `python scripts/checks/check_config_loading.py`, `python scripts/checks/check_sd_modular_imports.py`, and relevant launcher checks. Avoid model downloads and training unless requested.

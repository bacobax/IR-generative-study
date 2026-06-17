# Diffusion Objective Routing

## Purpose
Load this for diffusion objective logic: unconditional latent diffusion, SD15 adaptation, SDXL adaptation, schedulers/noise prediction, trainer loops, generation, configs, and tests. For RegionDiff-specific requests, also load `docs/agent_docs/high_level_training_types/region_diff.md`.

Do not load flow matching or STAY docs unless the request explicitly compares objectives or mentions STAY-conditioned flow matching.

## Primary files/directories to inspect
- `src/cli/train_latent_diffusion.py`
- `src/cli/adapt_stable_diffusion.py`
- `src/cli/train_sdxl.py`
- `src/cli/generate.py`
- `src/algorithms/training/unconditional_sd_trainer.py`
- `src/algorithms/inference/unconditional_sd_sampler.py`
- `src/algorithms/stable_diffusion/`
- `src/algorithms/stable_diffusion_xl/`
- `src/core/configs/sd_uncond_config.py`
- `configs/sd_uncond/train/`
- `configs/sd/train/`
- `configs/sd/generate/`
- `configs/sdxl/train/`
- `configs/sdxl/generate/`
- `scripts/train/uncond_latent_flir_sd15_512.sh`
- `scripts/train/sd_flir_lora_stage1.sh`
- `scripts/train/sdxl_lora_r8.sh`
- `scripts/generate/qcmp_uncond_sd_hflip.sh`
- `tests/test_sd_uncond.py`
- `tests/test_train_sd_uncond_imports.py`
- `tests/test_sd_ir_baselines.py`
- `tests/test_sdxl_stage1.py`

## Decision/routing notes
Use `rg -n "noise_scheduler|prediction_type|epsilon|v_prediction|latents|diffusion|sd_uncond|lora|sdxl" src configs scripts tests` first. For plain from-scratch latent diffusion, combine with `training_regime/from_scratch.md`. For SD15 or SDXL adaptation, prefer the corresponding high-level training type doc.

## Modification guidance
Unconditional diffusion changes belong in `src/algorithms/training/unconditional_sd_trainer.py`, `src/algorithms/inference/unconditional_sd_sampler.py`, and `src/core/configs/sd_uncond_config.py`. SD15 changes belong under `src/algorithms/stable_diffusion/`; SDXL changes under `src/algorithms/stable_diffusion_xl/`. Root wrappers stay thin and configs stay under the matching `configs/` area.

## Validation guidance
Run `python -m pytest tests/test_sd_uncond.py -v`, `python -m pytest tests/test_train_sd_uncond_imports.py -v`, `python -m pytest tests/test_sd_ir_baselines.py -v`, or `python -m pytest tests/test_sdxl_stage1.py -v` depending on touched files. Run `python scripts/checks/check_train_cli_sd_uncond.py` and `python scripts/checks/check_config_loading.py` for CLI/config changes. Avoid model downloads and training unless requested.

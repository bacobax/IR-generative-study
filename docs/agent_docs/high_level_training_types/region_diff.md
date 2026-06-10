# RegionDiff Routing

## Purpose
Load this for RegionDiff, region-level layout diffusion, stage-2 region adaptation, RegionDiff synthetic generation, attention distillation, or comparing RegionDiff variants.

Do not load STAY docs unless the user explicitly asks about STAY-conditioned flow matching.

## Primary files/directories to inspect
- `src/models/regiondiffusion.py`
- `src/models/regiondiffusion_factory.py`
- `src/algorithms/inference/regiondiff/`
- `src/algorithms/inference/regiondiff_smoke_generation.py`
- `src/algorithms/training/regiondiff_attention_distillation.py`
- `scripts/smoke/generate_smoked_regiondiff_dataset.py`
- `scripts/smoke/run_smoked_e2e_pipeline.py`
- `scripts/debug_regiondiff_attention_distillation.py`
- `configs/fm/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_b64.yaml`
- `configs/fm/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_ot_b64_hflip.yaml`
- `configs/fm/train/presets/regiondiff_attention_kd_latent_flir_sd15_512_l005.yaml`
- `configs/sd_uncond/train/presets/regiondiff.yaml`
- `configs/sd_uncond/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_b64.yaml`
- `configs/sd_uncond/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_b64_hflip.yaml`
- `configs/sd_layout/train/`
- `configs/yolo/exp_b/flir_yolov8n/exp_precomputed_regiondiff_fm_ot_hflip.yaml`
- `tests/test_regiondiff_generalized.py`
- `tests/test_regiondiff_smoke_generation.py`
- `tests/test_regiondiff_attention_distillation.py`

## Decision/routing notes
For RegionDiff on SD 1.5 adaptation, combine with `high_level_training_types/sd15_adaptation.md`, `training_regime/finetuning_sd15.md`, and inspect `src/algorithms/stable_diffusion/layout_data.py`, `layout_models.py`, `layout_training.py`, `src/core/configs/sd_layout_config.py`, and `configs/sd_layout/train/presets/`.

For RegionDiff on SDXL adaptation, combine with `high_level_training_types/sdxl_adaptation.md` and `training_regime/finetuning_sdxl.md`; first confirm current SDXL RegionDiff support with `rg -n "sdxl.*regiondiff|regiondiff.*sdxl" src configs tests` because SDXL stage-2 may not mirror SD15.

For RegionDiff on a latent diffusion model, combine with `training_objectives/diffusion.md` and inspect `src/cli/train_latent_diffusion.py`, `src/algorithms/training/unconditional_sd_trainer.py`, `src/core/configs/sd_uncond_config.py`, `configs/sd_uncond/train/presets/regiondiff.yaml`, and `configs/sd_uncond/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_b64.yaml`.

For RegionDiff on a latent flow matching model, combine with `training_objectives/flow_matching.md` and inspect `src/cli/train_flow_matching.py`, `src/algorithms/training/flow_matching_trainer.py`, `src/core/configs/fm_config.py`, `configs/fm/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_b64.yaml`, and `configs/fm/train/presets/regiondiff_latent_flir_sd15_512_from_uncond_ot_b64_hflip.yaml`.

## Modification guidance
Keep shared RegionDiff model wrappers in `src/models/`, generation/orchestration in `src/algorithms/inference/regiondiff/`, attention distillation in `src/algorithms/training/regiondiff_attention_distillation.py`, SD15 stage-2 behavior in `src/algorithms/stable_diffusion/layout_data.py`, `src/algorithms/stable_diffusion/layout_models.py`, and `src/algorithms/stable_diffusion/layout_training.py`, and objective-specific RegionDiff hooks in the matching trainer/config. Do not route STAY layout FM edits through RegionDiff files unless explicitly requested.

## Validation guidance
Run `python -m pytest tests/test_regiondiff_generalized.py -v`, `python -m pytest tests/test_regiondiff_smoke_generation.py -v`, `python -m pytest tests/test_regiondiff_attention_distillation.py -v`, or `python -m pytest tests/test_sd_layout_regiondiff.py -v` based on touched files. For configs, run `python scripts/checks/check_config_loading.py`. Avoid smoke E2E generation and Slurm jobs unless explicitly asked.

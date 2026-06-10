# SD15 Adaptation Routing

## Purpose
Load this for Stable Diffusion 1.5 adaptation as a high-level task: SD15 LoRA, full/partial UNet fine-tuning, generation from SD15 adapters, stage chaining, and SD15 dataset presets.

Do not load SDXL or STAY docs unless the user explicitly mentions them.

## Primary files/directories to inspect
- `docs/agent_docs/training_regime/finetuning_sd15.md`
- `docs/agent_docs/training_objectives/diffusion.md`
- `src/cli/adapt_stable_diffusion.py`
- `src/algorithms/stable_diffusion/`
- `configs/sd/train/`
- `configs/sd/generate/`
- `configs/datasets/flir/sd_adaptation/`
- `configs/datasets/bigearthnet_s2_b08_5x5_stride3/sd_adaptation/`
- `scripts/train/sd_flir_lora_stage1.sh`
- `scripts/train/sd_flir_unet_full_stage1.sh`
- `scripts/generate/sd_r8.sh`
- `tests/test_sd_ir_baselines.py`
- `tests/test_sd_domainstudio.py`

## Decision/routing notes
Combine with the relevant dataset doc only when dataset handling is part of the request. If RegionDiff stage 2 on top of SD15 is requested, route to `docs/agent_docs/high_level_training_types/region_diff.md` after this doc.

## Modification guidance
SD15 changes usually belong in `src/algorithms/stable_diffusion/`, `src/cli/adapt_stable_diffusion.py`, and `configs/sd/` or dataset-specific SD adaptation configs. Root `adapt_stable_diffusion.py` stays thin.

## Validation guidance
Run `python -m pytest tests/test_sd_ir_baselines.py -v`, `python -m pytest tests/test_sd_domainstudio.py -v`, and config/launcher checks if touched. Avoid training or generation unless requested.

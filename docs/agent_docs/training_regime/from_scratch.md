# From-Scratch Training Regime Routing

## Purpose
Load this when the user asks to train or configure models from scratch: unconditional flow matching, unconditional latent diffusion, VAE training, model config creation, optimizer/scheduler/runtime wiring, outputs, launchers, or tests.

Do not load SD15 or SDXL fine-tuning docs for from-scratch UNet/DiT/VAE work unless the request explicitly compares or reuses fine-tuned Stable Diffusion assets.

## Primary files/directories to inspect
- `src/cli/train_flow_matching.py`
- `src/cli/train_latent_diffusion.py`
- `src/cli/train_vae.py`
- `src/algorithms/training/flow_matching_trainer.py`
- `src/algorithms/training/unconditional_sd_trainer.py`
- `src/algorithms/tasks/flow_matching.py`
- `src/algorithms/inference/flow_matching_sampler.py`
- `src/algorithms/inference/unconditional_sd_sampler.py`
- `src/models/fm_unet.py`
- `src/models/dit.py`
- `src/models/vae.py`
- `src/core/configs/fm_config.py`
- `src/core/configs/sd_uncond_config.py`
- `configs/fm/train/`
- `configs/sd_uncond/train/`
- `configs/vae/train/`
- `configs/models/fm/`
- `scripts/train/stable_fm.sh`
- `scripts/train/uncond_fm_latent_flir_sd15_512.sh`
- `scripts/train/uncond_latent_flir_sd15_512.sh`
- `scripts/train/vae_4x.sh`
- `scripts/train/vae_8x.sh`
- `tests/test_flow_matching_task.py`
- `tests/test_sd_uncond.py`
- `tests/test_vae.py`
- `tests/test_training_runtime_utils.py`

## Decision/routing notes
For objective math, combine with `docs/agent_docs/training_objectives/flow_matching.md` or `docs/agent_docs/training_objectives/diffusion.md`. For dataset-only changes, load a dataset doc instead. For launcher-only changes, inspect `scripts/lib/common.sh`, the relevant shell wrapper, and Slurm file before source code.

## Modification guidance
Put CLI argument/config wiring in `src/cli/` and `src/core/configs/`, trainer behavior in `src/algorithms/training/`, model definitions in `src/models/`, and presets under `configs/fm/`, `configs/sd_uncond/`, or `configs/vae/`. Root scripts remain compatibility wrappers. Default outputs should stay under `artifacts/` or `logs/`.

## Validation guidance
Use focused tests such as `python -m pytest tests/test_flow_matching_task.py -v`, `python -m pytest tests/test_sd_uncond.py -v`, `python -m pytest tests/test_vae.py -v`, and cheap checks like `python scripts/checks/check_train_cli_fm.py`, `python scripts/checks/check_train_cli_sd_uncond.py`, or `python scripts/checks/check_config_loading.py`. Do not run training jobs unless explicitly asked.

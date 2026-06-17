# Unconditional From-Scratch Flow And Diffusion Routing

## Purpose
Load this for high-level requests spanning unconditional flow matching and unconditional diffusion from scratch, including comparing FM vs diffusion configs, shared launchers, checkpoint-selection first-stage runs, or dataset-specific unconditional presets.

Do not load SD15/SDXL adaptation, STAY, or RegionDiff docs unless explicitly mentioned.

## Primary files/directories to inspect
- `docs/agent_docs/training_regime/from_scratch.md`
- `docs/agent_docs/training_objectives/flow_matching.md` for flow matching requests
- `docs/agent_docs/training_objectives/diffusion.md` for diffusion requests
- `src/cli/train_flow_matching.py`
- `src/cli/train_latent_diffusion.py`
- `configs/fm/train/`
- `configs/sd_uncond/train/`
- `configs/datasets/flir/flow_matching/`
- `configs/datasets/flir/diffusion/`
- `configs/datasets/bigearthnet_s2_b08_5x5_stride3/flow_matching/`
- `configs/datasets/bigearthnet_s2_b08_5x5_stride3/diffusion/`
- `scripts/train/stable_fm.sh`
- `scripts/train/uncond_fm_latent_flir_sd15_512.sh`
- `scripts/train/uncond_latent_flir_sd15_512.sh`
- `slurm/fir/flir/flow_matching/`
- `slurm/fir/flir/diffusion/`
- `tests/test_flow_matching_task.py`
- `tests/test_sd_uncond.py`

## Decision/routing notes
Combine `training_regime/from_scratch.md` with exactly one objective doc unless the user explicitly asks to compare FM and diffusion. Load a dataset doc only when the user mentions dataset loading, normalization, splits, manifests, or dataset-specific config paths.

## Modification guidance
Keep shared runtime/config behavior in `src/core/` or the specific CLI/config module. Keep objective-specific logic in the objective trainer/task/sampler. Do not cross-wire SD15/SDXL fine-tuning assumptions into from-scratch trainers.

## Validation guidance
Run the objective-specific tests plus `python scripts/checks/check_config_loading.py` for YAML changes. Use launcher checks only when shell or Slurm files change. Do not launch training jobs.

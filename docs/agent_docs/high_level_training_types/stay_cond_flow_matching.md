# STAY-Conditioned Flow Matching Routing

## Purpose
Load this for STAY-conditioned or layout-conditioned flow matching requests: STAY/layout conditioners, pixel or latent layout FM, layout batching, layout sampler/generation, STAY configs, and generated layout audits.

Do not load `region_diff.md` unless the user explicitly mentions RegionDiff or region-level diffusion distillation.

## Primary files/directories to inspect
- `docs/agent_docs/training_objectives/flow_matching.md`
- `src/conditioning/`
- `src/core/data/layout_batching.py`
- `src/core/data/annotation_dataset.py`
- `src/core/data/latent_cache.py`
- `src/algorithms/training/layout_flow_matching_trainer.py`
- `src/algorithms/inference/layout_flow_matching_sampler.py`
- `src/models/layout_conditioned_unet.py`
- `src/models/stay_layout_conditioned_unet.py`
- `src/core/configs/fm_config.py`
- `configs/fm/train/presets/stay_layout_latent_flir_sd15_512.yaml`
- `configs/fm/train/presets/stay_layout_latent_flir_sd15_512_b64.yaml`
- `configs/fm/train/presets/stay_layout_latent_flir_x4_512.yaml`
- `configs/fm/train/presets/stay_layout_pixel_flir_v2.yaml`
- `configs/fm/train/presets/stay_layout_pixel_flir_v2_smoke200.yaml`
- `configs/data/presets/flir_private_proxy_alignment_layout.yaml`
- `configs/auxiliary/rare_layout_generation/`
- `configs/auxiliary/generated_layout_audit/`
- `scripts/train/stay_layout_latent_flir_sd15_512.sh`
- `scripts/train/stay_layout_pixel_flir_v2.sh`
- `scripts/generate/qcmp_stay_layout_fm_hflip.sh`
- `scripts/standalone/generate_rare_layout_dataset.py`
- `scripts/standalone/filter_generated_layout_dataset.py`
- `tests/test_stay_layout_fm.py`
- `tests/test_layout_fm.py`
- `tests/test_generated_layout_dataset_scripts.py`
- `tests/test_annotation_layout_dataset.py`

## Decision/routing notes
Use `rg -n "STAY|stay|layout|condition|layout_conditioning|collate_layout|stay_layout" src configs scripts tests docs` first. For target/label extraction, add the relevant dataset doc. For objective changes, combine with `training_objectives/flow_matching.md`. Keep RegionDiff out unless explicitly requested.

## Latent caching
STAY/layout latent FM supports a disk latent cache (`latent_cache:` block, see `training_objectives/flow_matching.md`). When enabled, `src/cli/train_flow_matching.py` builds the base `AnnotationLayoutDataset` without the probabilistic flip, materialises hflip into a fixed pool, encodes once via `src/core/data/latent_cache.py`, and wraps with `LatentCacheDataset`. `collate_layout_batch` carries `latent_mu`/`latent_sigma` through; the trainer samples from them in `fm_input_from_batch` and skips the VAE encode. Enabled in `configs/fm/train/presets/stay_layout_latent_v18_sd15ft_x8_256.yaml`.

## Modification guidance
Put layout/STAY data collation in `src/core/data/`, conditioning contracts in `src/conditioning/`, STAY/layout trainers in `src/algorithms/training/layout_flow_matching_trainer.py`, samplers in `src/algorithms/inference/layout_flow_matching_sampler.py`, and model changes in `src/models/layout_conditioned_unet.py` or `src/models/stay_layout_conditioned_unet.py`. Configs belong under `configs/fm/train/presets/` or `configs/data/presets/`. Keep wrappers thin.

## Validation guidance
Run `python -m pytest tests/test_stay_layout_fm.py -v`, `python -m pytest tests/test_layout_fm.py -v`, `python -m pytest tests/test_annotation_layout_dataset.py -v`, and `python scripts/checks/check_conditioning_integration.py` when conditioning wiring changes. Do not run layout generation sweeps unless requested.

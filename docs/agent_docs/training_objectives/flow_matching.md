# Flow Matching Objective Routing

## Purpose
Load this for flow matching objective logic: target construction, time/noise schedules, OT toggles, trainers, samplers, inference, conditioner inputs, FM configs, and tests. For STAY-conditioned flow matching, also load `docs/agent_docs/high_level_training_types/stay_cond_flow_matching.md`.

Do not load diffusion or RegionDiff docs unless the request explicitly compares objectives or mentions RegionDiff.

## Primary files/directories to inspect
- `src/algorithms/tasks/flow_matching.py`
- `src/algorithms/training/flow_matching_trainer.py`
- `src/algorithms/training/layout_flow_matching_trainer.py`
- `src/algorithms/inference/flow_matching_sampler.py`
- `src/algorithms/inference/cfg_flow_matching_sampler.py`
- `src/algorithms/inference/layout_flow_matching_sampler.py`
- `src/cli/train_flow_matching.py`
- `src/cli/sample.py`
- `src/cli/sample_text_fm.py`
- `src/core/configs/fm_config.py`
- `src/core/configs/text_fm_config.py`
- `src/conditioning/`
- `src/guidance/`
- `src/models/fm_unet.py`
- `src/models/fm_text_unet.py`
- `configs/fm/train/`
- `configs/fm/sample/`
- `configs/fm/generate/`
- `scripts/checks/check_fm_trainer_sampler_split.py`
- `tests/test_flow_matching_task.py`
- `tests/test_layout_fm.py`
- `tests/test_text_fm_cfg.py`
- `tests/test_sample_text_fm.py`

## Decision/routing notes
Use `rg -n "FlowMatchingTask|target_type|time|noise|ot|sampler|velocity|x0|conditioning" src configs tests` first. If the request concerns dataset targets, load the relevant dataset doc. If it concerns RegionDiff distillation or regional generation from FM checkpoints, load `high_level_training_types/region_diff.md`.

## Modification guidance
Objective math belongs in `src/algorithms/tasks/flow_matching.py` or trainer/sampler modules. Config fields belong in `src/core/configs/fm_config.py` and YAML presets. Conditioning belongs in `src/conditioning/` or STAY/layout-specific modules, not in generic objective code unless the generic contract changes. Keep root wrappers thin.

## Validation guidance
Run `python -m pytest tests/test_flow_matching_task.py -v` for objective math, `python -m pytest tests/test_layout_fm.py -v` for layout FM, and `python scripts/checks/check_train_cli_fm.py` or `python scripts/checks/check_fm_trainer_sampler_split.py` for CLI/trainer wiring. Avoid long sampling/generation runs unless requested.

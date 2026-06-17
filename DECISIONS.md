# Repository Decisions for AI Coding Agents

This file records architectural and workflow decisions that are already present in the repository. It is not a feature roadmap. If this file conflicts with executable code, tests, or configs, trust the executable evidence and update this file only after verifying the new behavior.

## 1. Active Source of Truth Lives in Top-Level `src/`

Decision: active Python code belongs under the top-level `src/` package. CLI modules live in `src/cli/`; reusable training and inference behavior lives under `src/algorithms/`; shared config, path, data, registry, artifact, and normalization utilities live under `src/core/`; model code lives under `src/models/`; analysis backend code lives under `src/analysis/`.

Consequences:
- Put new training, inference, data, model, evaluation, and analysis behavior in the appropriate `src/` package.
- Root Python files should stay as compatibility entry points or wrappers.
- Do not treat nested checkout-like trees as active source unless the user explicitly asks.

Evidence:
- `README.md:7` to `README.md:19` describes `src/`, `configs/`, and model config ownership.
- `README.md:50` to `README.md:62` maps root wrappers to `src/cli/*`.
- `scripts/checks/check_wrappers_only.py:127` to `scripts/checks/check_wrappers_only.py:133` verifies source-of-truth files exist.

## 2. Root Entry Points Are Thin Compatibility Wrappers

Decision: root scripts such as `train_flow_matching.py`, `adapt_stable_diffusion.py`, `train_latent_diffusion.py`, `train_vae.py`, `train_controlnet.py`, `generate_datasets.py`, `serve_flir_analysis.py`, and `train_sdxl.py` delegate to canonical modules and should not contain core logic.

Consequences:
- Add or change CLI behavior in `src/cli/*`, not in root wrappers.
- Keep root wrappers short, with imports and `main()` dispatch only.
- Standalone experiment tools live under `scripts/standalone/`, with older root paths kept as compatibility wrappers where needed.

Evidence:
- `train_flow_matching.py:1` to `train_flow_matching.py:6` imports and calls `src.cli.train_flow_matching.main`.
- `generate_datasets.py:2` to `generate_datasets.py:12` documents the compatibility wrapper pattern.
- `scripts/checks/check_wrappers_only.py:33` to `scripts/checks/check_wrappers_only.py:68` defines wrapper mappings and forbidden core-logic patterns.
- `scripts/checks/check_wrappers_only.py:201` to `scripts/checks/check_wrappers_only.py:217` requires source-of-truth modules to expose `main()`.

## 3. Experiments Are Config-First

Decision: experiment behavior is driven by YAML or JSON under `configs/`, grouped by area and often by `presets/`. CLIs accept `--config`, then explicit CLI flags override config values.

Consequences:
- Prefer adding descriptive presets under `configs/<area>/<action>/presets/`.
- Keep shell and Slurm launchers pointed at configs rather than hard-coding experiment behavior.
- When schema or argument mappings change, update the loader, dataclasses, defaults, tests, and check scripts together.

Evidence:
- `README.md:182` to `README.md:205` describes shell wrappers as config-preset launchers.
- `src/core/configs/config_loader.py:1` to `src/core/configs/config_loader.py:9` documents merge order: dataclass defaults, YAML, then CLI overrides.
- `src/core/configs/config_loader.py:29` to `src/core/configs/config_loader.py:77` implements relative `extends` inheritance and deep merge.
- `src/core/configs/config_loader.py:249` to `src/core/configs/config_loader.py:301` implements defaults plus YAML plus CLI override merging.
- `src/cli/train_flow_matching.py:87` to `src/cli/train_flow_matching.py:92` exposes `--config` and an explicit architecture mode flag.
- `src/cli/train_flow_matching.py:260` to `src/cli/train_flow_matching.py:354` maps flat CLI options into nested config fields.
- `tests/test_config_loader_extends.py:37` to `tests/test_config_loader_extends.py:183` covers YAML inheritance and CLI precedence.

## 4. Paths, Data, and Artifacts Have Canonical Homes

Decision: well-known repository paths are centralized in `src/core/paths.py`. Datasets and caches belong under `data/`; generated outputs, analysis, checkpoints, debug samples, and runs belong under `artifacts/` or `logs/`.

Consequences:
- Use `src.core.paths` helpers when adding reusable code that needs known paths.
- Make new output paths configurable and default them under `artifacts/` or `logs/`.
- Do not commit weights, checkpoints, generated datasets, event files, or local secrets.

Evidence:
- `src/core/paths.py:1` to `src/core/paths.py:8` declares the module as the path source of truth.
- `src/core/paths.py:27` to `src/core/paths.py:64` defines data, artifacts, archive, legacy, and config roots.
- `src/core/paths.py:138` to `src/core/paths.py:193` defines generated, debug, analysis, and checkpoint roots.
- `.gitignore:63` to `.gitignore:87` ignores heavy data and artifact contents while preserving `.gitkeep` structure.
- `scripts/checks/check_repo_paths.py:66` to `scripts/checks/check_repo_paths.py:119` validates canonical path helpers.
- `scripts/checks/check_dataset_locations.py:32` to `scripts/checks/check_dataset_locations.py:79` enforces dataset and cache locations.

## 5. Active vs. Legacy Boundaries Are Deliberate

Decision: `archive/legacy_code/`, `ControlNet/`, `src/diffusers/`, and nested checkout-like trees such as `src/flow-matching-trial/` are not normal active-change targets. Active runtime code should not depend on retired `fm_src` or `sd_src` modules.

Consequences:
- Do not import retired `fm_src` or `sd_src` from active `src/`, `scripts/`, or `tests`.
- Do not reintroduce retired guidance/config keys into active presets.
- Treat vendored or nested repositories as external reference material unless the user explicitly asks to touch them.

Evidence:
- `README.md:81` to `README.md:84` identifies retired guidance and meta/MoE code under `archive/legacy_code/`.
- `scripts/checks/check_no_legacy_runtime_dependency.py:1` to `scripts/checks/check_no_legacy_runtime_dependency.py:7` states the no-runtime-dependency rule.
- `scripts/checks/check_no_legacy_runtime_dependency.py:37` to `scripts/checks/check_no_legacy_runtime_dependency.py:45` excludes `src/diffusers` and `src/flow-matching-trial` from active scans.
- `scripts/checks/check_no_legacy_runtime_dependency.py:50` to `scripts/checks/check_no_legacy_runtime_dependency.py:148` checks forbidden old-path literals/imports and archived legacy roots.
- `tests/test_fm_generation_presets.py:10` to `tests/test_fm_generation_presets.py:18` names retired FM guidance keys, and `tests/test_fm_generation_presets.py:41` to `tests/test_fm_generation_presets.py:47` prevents them in active FM generation presets.

## 6. Slurm Launchers Are Self-Contained; Simple Shell Wrappers May Use `scripts/lib/common.sh`

Decision: `.slurm` launchers spell out runtime setup, config path resolution, diagnostics, and timed execution directly. They must not depend on `slurm/lib/common.sh` or custom `slurm_*` helper calls. Ordinary shell wrappers under `scripts/train`, `scripts/generate`, and `scripts/analyze` are checked against `scripts/lib/common.sh` unless explicitly bespoke.

Consequences:
- In `.slurm` files, keep plain Bash/Slurm syntax: `CONFIG_REL`, direct Conda activation, `grep`, `nvidia-smi`, `/usr/bin/time`, and command lines inline.
- Preserve cluster-specific SBATCH headers, accounts, GPU requests, and log paths in Slurm files.
- For non-Slurm shell wrappers, follow `scripts/lib/common.sh` and the launcher mapping checks.

Evidence:
- `scripts/checks/check_slurm_launchers.py:394` to `scripts/checks/check_slurm_launchers.py:424` checks no Slurm common helper, unchanged launcher set, preserved headers, self-contained launchers, no `slurm_` calls, strict shell mode, direct Conda activation, and valid config references.
- `slurm/killarney/train_stay_layout_fm_hflip_kl.slurm:11` to `slurm/killarney/train_stay_layout_fm_hflip_kl.slurm:88` shows inline strict mode, `CONFIG_REL`, Conda activation, config checks, `grep`, `nvidia-smi`, `/usr/bin/time`, and explicit exit handling.
- `scripts/lib/common.sh:1` to `scripts/lib/common.sh:54` defines helpers for simple shell launchers.
- `scripts/checks/check_shell_launchers.py:72` to `scripts/checks/check_shell_launchers.py:103` checks simple shell wrappers source `common.sh`, enter the repo root, check configs, preserve CLI passthrough, and reference expected configs.

## 7. Registries Are Explicit and Import-Driven

Decision: component lookup uses lightweight named registries. There is no dynamic import magic; modules must be imported so decorators register components before lookup.

Consequences:
- Register new builders, trainers, samplers, guidance, conditioning, adapters, datasets, tasks, and artifact loaders through `REGISTRIES`.
- Ensure the module that registers a component is imported by the CLI or package path that needs it.
- Do not assume adding a file is enough to make a component discoverable.

Evidence:
- `src/core/registry.py:1` to `src/core/registry.py:25` documents explicit decorator registration and no dynamic imports.
- `src/core/registry.py:103` to `src/core/registry.py:137` defines registries for model builders, trainers, samplers, guidance, conditioning, model adapters, dataset adapters, task adapters, and artifact loaders.
- `src/cli/train_flow_matching.py:42` to `src/cli/train_flow_matching.py:49` imports default modules to register components.
- `tests/test_registry_adapter_extensions.py:4` to `tests/test_registry_adapter_extensions.py:43` verifies existing and adapter registries and registry behavior.

## 8. Config Validation Has Legacy and Experimental Modes

Decision: legacy dataclass config loading is permissive about unknown keys, while the experimental config shape is strict and reports dotted paths for unknown keys.

Consequences:
- Do not make legacy config loading strict without updating tests and existing presets.
- Use the experimental config path only when the YAML declares `kind: experimental_training_config`.

Evidence:
- `src/core/configs/config_loader.py:108` to `src/core/configs/config_loader.py:126` ignores unknown keys for legacy dataclass loading.
- `src/core/configs/config_loader.py:129` to `src/core/configs/config_loader.py:153` implements strict dataclass loading with dotted unknown-key errors.
- `src/core/configs/config_loader.py:156` to `src/core/configs/config_loader.py:173` recognizes and loads experimental configs.
- `tests/test_config_loader_extends.py:235` to `tests/test_config_loader_extends.py:253` confirms legacy unknown keys are ignored.
- `tests/test_config_loader_extends.py:256` to `tests/test_config_loader_extends.py:293` confirms experimental configs are strict.

## 9. Model Adapter Abstraction Is Additive and Opt-In

Decision: model adapters provide a structured construction/wrapping layer without replacing current trainers, CLIs, or the existing `REGISTRIES.model_builder` path. FM `architecture_mode` defaults to `legacy`; `adapter_v1` is an opt-in path for non-layout FM.

Consequences:
- Do not rewrite existing training flows to adapters unless the requested change needs that path.
- Do not use `adapter_v1` for layout-conditioned FM; it is explicitly rejected.
- Do not assume LoRA trainability is implemented in `DiffusersModelAdapter`; tests mark it as a placeholder.

Evidence:
- `src/models/adapters/base.py:1` to `src/models/adapters/base.py:6` states adapters are intentionally additive.
- `src/cli/train_flow_matching.py:90` to `src/cli/train_flow_matching.py:92` sets `architecture_mode` choices and default.
- `src/cli/train_flow_matching.py:522` to `src/cli/train_flow_matching.py:547` builds the adapter-v1 trainer through `FMModelAdapter`.
- `src/cli/train_flow_matching.py:563` to `src/cli/train_flow_matching.py:570` rejects `adapter_v1` with layout conditioning.
- `src/models/adapters/fm.py:84` to `src/models/adapters/fm.py:167` builds native FM UNet plus optional VAE and returns an `fm_adapter` bundle.
- `tests/test_fm_adapter_v1_cli.py:24` to `tests/test_fm_adapter_v1_cli.py:53` verifies the default and the layout rejection.
- `tests/test_model_adapters.py:258` to `tests/test_model_adapters.py:275` verifies LoRA trainability is not implemented and adapter registry keys exist.

## 10. FM Task Semantics Are Centralized

Decision: core flow-matching target construction, path coupling, conditioning permutation, and loss computation live in `FlowMatchingTask`, not scattered through launchers.

Consequences:
- Change FM target semantics in task/trainer code, with focused tests.
- Keep conditioning tensors/lists aligned with target permutations when using OT path matching.

Evidence:
- `src/algorithms/tasks/flow_matching.py:17` to `src/algorithms/tasks/flow_matching.py:31` defines the task and supported targets.
- `src/algorithms/tasks/flow_matching.py:32` to `src/algorithms/tasks/flow_matching.py:75` handles conditioning permutation and target matching.
- `src/algorithms/tasks/flow_matching.py:87` to `src/algorithms/tasks/flow_matching.py:131` samples FM states and computes `v` or `x0` loss.
- `tests/test_flow_matching_task.py:8` to `tests/test_flow_matching_task.py:57` covers velocity loss, `x0`-derived velocity loss, and conditioning permutation.

## 11. Evaluation and Checkpoint Selection Are Standalone, Configured Pipelines

Decision: post-training checkpoint selection and generative metrics run through `scripts/select_best_checkpoint_and_compute_metrics.py` with YAML configs under `configs/eval/`. The script supports native FM, native SD-unconditional, SD LoRA, and SDXL LoRA checkpoint discovery. Publication-style clean-FID selection is a configured pipeline mode, not the default.

Consequences:
- Add checkpoint-selection behavior in the standalone pipeline and eval configs, not in training CLIs.
- Keep fallback metrics explicit: if ranking falls back from requested `clean_fid`, record the effective metric as `inception_fid_fallback`.
- Preserve manifests, verification, and cleanup metadata when changing evaluation outputs.

Evidence:
- `scripts/select_best_checkpoint_and_compute_metrics.py:96` to `scripts/select_best_checkpoint_and_compute_metrics.py:129` defines the checkpoint-selection CLI.
- `scripts/select_best_checkpoint_and_compute_metrics.py:349` and `scripts/select_best_checkpoint_and_compute_metrics.py:516` to `scripts/select_best_checkpoint_and_compute_metrics.py:584` discover checkpoints and resolve model/run metadata.
- `configs/eval/checkpoint_selection_clean_fid_publication_flir.yaml:1` to `configs/eval/checkpoint_selection_clean_fid_publication_flir.yaml:19` documents publication mode, selection images, final images, and metric intent.
- `configs/eval/checkpoint_selection_clean_fid_publication_flir.yaml:39` to `configs/eval/checkpoint_selection_clean_fid_publication_flir.yaml:60` configures selection, final reporting, and metrics.
- `tests/test_checkpoint_selection_pipeline.py:32` to `tests/test_checkpoint_selection_pipeline.py:135` covers checkpoint discovery across model families.
- `tests/test_checkpoint_selection_pipeline.py:156` to `tests/test_checkpoint_selection_pipeline.py:158` confirms the default mode is `legacy_staged_kid_fid`.
- `tests/test_checkpoint_selection_pipeline.py:219` to `tests/test_checkpoint_selection_pipeline.py:232` confirms clean-FID fallback ranking records requested and effective metrics separately.

## 12. Analysis Backend and React Frontend Are Separate

Decision: subgroup analysis and checkpoint-selection viewer behavior lives in Python/FastAPI under `src/analysis/`; the React app under `frontend/flir-subgroup-analysis/` is a client that calls typed API helpers.

Consequences:
- Keep filesystem scanning, dataset resolution, analysis computation, and image rendering in backend modules.
- Keep UI state, charts, controls, and API request wiring in the frontend package.
- Validate backend changes with Python tests; validate frontend changes with `npm run build`.

Evidence:
- `README.md:115` to `README.md:180` describes the backend routes, frontend setup, and checkpoint-selection view.
- `src/cli/serve_flir_analysis.py:21` to `src/cli/serve_flir_analysis.py:37` launches the FastAPI app via uvicorn.
- `src/analysis/flir_subgroup/app.py:17` to `src/analysis/flir_subgroup/app.py:43` creates the FastAPI app and includes subgroup and checkpoint-selection routers.
- `src/analysis/flir_subgroup/api.py:168` to `src/analysis/flir_subgroup/api.py:276` creates dataset-aware API routes.
- `frontend/flir-subgroup-analysis/src/api.ts:17` to `frontend/flir-subgroup-analysis/src/api.ts:125` defines the API base URL and typed request functions.
- `frontend/flir-subgroup-analysis/src/CheckpointSelectionView.tsx:29` to `frontend/flir-subgroup-analysis/src/CheckpointSelectionView.tsx:44` sets the default checkpoint-selection root and viewer constants.
- `frontend/flir-subgroup-analysis/README.md:42` to `frontend/flir-subgroup-analysis/README.md:44` says the backend scans roots read-only and serves previews only below the selected root.

## 13. Tests Are Fast, Focused, and Usually Synthetic

Decision: tests live in `tests/`, target narrow behavior, and commonly use temporary directories, tiny tensors, and synthetic datasets rather than expensive training or downloads.

Consequences:
- Add tests beside related coverage with `test_<feature>.py` and `test_<behavior>()` names.
- Use targeted pytest and check scripts for validation.
- Do not launch expensive training, generation, or Slurm jobs unless explicitly asked.

Evidence:
- `README.md:207` to `README.md:223` lists repo check scripts and targeted subgroup/frontend validation.
- `pyproject.toml:5` to `pyproject.toml:20` declares Python 3.10+ and optional web dependencies.
- `tests/test_flow_matching_task.py:8` to `tests/test_flow_matching_task.py:57` uses small tensors for FM task behavior.
- `tests/test_flir_subgroup_analysis.py:25` to `tests/test_flir_subgroup_analysis.py:155` builds synthetic datasets in temporary paths.
- The top-level `tests/` directory contains focused coverage for configs, adapters, checkpoint selection, datasets, training utilities, analysis, YOLO, SD, VAE, and FM behavior.

## Common Decision Pitfalls

- Putting real behavior in root wrappers instead of `src/cli/*`.
- Adding configs without updating parser-to-dataclass mappings, loader tests, or launcher/check-script references.
- Hard-coding output paths in the repo root instead of making them configurable under `artifacts/` or `logs/`.
- Importing or modifying retired `fm_src`, `sd_src`, `src/diffusers`, `ControlNet`, or nested checkout trees for active behavior without explicit user direction.
- Adding `slurm_*` helper calls or `slurm/lib/common.sh` dependencies to `.slurm` launchers.
- Forgetting that simple non-Slurm shell wrappers are expected to use `scripts/lib/common.sh`.
- Registering a component but failing to import the registration module before lookup.
- Treating legacy YAML loading as strict, or treating experimental config validation as permissive.
- Assuming `architecture_mode="adapter_v1"` supports layout-conditioned FM.
- Assuming `DiffusersModelAdapter` implements LoRA trainability.
- Writing clean-FID fallback values under the `clean_fid` metric name instead of preserving `inception_fid_fallback` as the effective metric.
- Moving backend analysis, filesystem scans, or preview serving into the React frontend.
- Running heavy training/generation/Slurm jobs as validation without an explicit request.

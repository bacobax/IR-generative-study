# Development Guide for AI Coding Agents

This repository is config-first and experiment-heavy. Before editing, inspect the
existing source, configs, tests, and check scripts for the path you are touching.
Do not infer behavior from names alone, and do not move or clean unrelated files
in a dirty worktree.

## Repository Map

- `src/` is the active Python source tree. CLI entry points live in `src/cli/`,
  reusable training and inference code in `src/algorithms/`, shared path/config
  and data utilities in `src/core/`, conditioning in `src/conditioning/`, models
  in `src/models/`, guidance in `src/guidance/`, evaluation in `src/evaluation/`,
  and FLIR analysis in `src/analysis/`.
- Root Python files such as `train_flow_matching.py`,
  `adapt_stable_diffusion.py`, `train_latent_diffusion.py`, `train_sdxl.py`,
  `train_vae.py`, `train_controlnet.py`, `generate_datasets.py`, and
  `serve_flir_analysis.py` are compatibility wrappers. The source of truth is in
  `src/cli/`; see `scripts/checks/check_wrappers_only.py`.
- Experiment presets live under `configs/`, grouped by area: `configs/fm/`,
  `configs/sd/`, `configs/sd_uncond/`, `configs/sd_layout/`, `configs/sdxl/`,
  `configs/vae/`, `configs/controlnet/`, `configs/auxiliary/`, `configs/eval/`,
  `configs/models/`, `configs/datasets/`, and `configs/yolo/`.
- Shell launchers live under `scripts/train/`, `scripts/generate/`, and
  `scripts/analyze/`; most source `scripts/lib/common.sh`.
- Slurm launchers live under `slurm/<cluster>/`, for example
  `slurm/killarney/`, `slurm/fir/`, and `slurm/tamia/`.
- Tests live in `tests/`. Repo health checks live in `scripts/checks/`.
- The FLIR subgroup React app lives in `frontend/flir-subgroup-analysis/`; the
  FastAPI backend is launched by `serve_flir_analysis.py` and implemented under
  `src/analysis/flir_subgroup/`.
- Treat `archive/`, `ControlNet/`, `src/diffusers/`, and checkout-like nested
  trees as legacy or external reference material unless a task explicitly asks
  to touch them.

## Setup

Use Python 3.10 or newer. The package metadata in `pyproject.toml` declares the
local package plus optional extras:

```bash
python -m pip install -e .
python -m pip install -e .[web]
python -m pip install -e .[yolo]
```

For a fuller GPU-oriented environment, use the checked-in requirement snapshots:

```bash
python -m pip install -r requirements-pip-clean.txt
python -m pip install -r requirements-pip-clean-noindex.txt
python -m pip install -r requirements-cc.txt
```

The requirement files include CUDA/PyTorch, Diffusers, Accelerate, PEFT,
Ultralytics, FastAPI, pytest, and analysis dependencies. Cluster launchers often
activate a Conda environment named `diffusers-dev`; see Slurm files such as
`slurm/killarney/flir/flow_matching/train_stable_fm_hflip_ot_kl.slurm`.

## Config-First Workflows

Prefer changing YAML presets or config dataclasses over hard-coded behavior.
The config loader in `src/core/configs/config_loader.py` merges in this order:
dataclass defaults, YAML, then explicitly provided CLI args. CLI defaults should
not overwrite YAML values.

YAML `extends:` is supported by `load_yaml()`:

- Parent paths are relative to the child YAML file.
- Absolute parent paths are rejected.
- Parent files load first; child values deep-merge on top.
- Cycles raise a clear error.

Useful config-backed commands:

```bash
python train_flow_matching.py --config configs/fm/train/default.yaml
python adapt_stable_diffusion.py --config configs/sd/train/default.yaml
python train_latent_diffusion.py --config configs/sd_uncond/train/default.yaml
python train_sdxl.py --config configs/sdxl/train/default.yaml
python train_vae.py --config configs/vae/train/presets/vae_4x.yaml
python train_controlnet.py --config configs/controlnet/train/default.yaml
python generate_datasets.py --mode fm --max_samples 100
```

When adding a preset, put it under the matching `configs/<area>/<action>/presets/`
directory when that pattern exists. If a schema or registry changes, update the
loader/config dataclass, defaults, tests, and check scripts together. Relevant
files include `src/core/configs/fm_config.py`,
`src/core/configs/config_loader.py`, `tests/test_config_loader_extends.py`, and
`scripts/checks/check_config_loading.py`.

## Wrappers and Launchers

Root wrappers must stay thin: import `main` from the canonical module and call
it. `scripts/checks/check_wrappers_only.py` enforces short wrappers, correct
imports, `main()` availability, and no training/generation logic in root files.

Simple shell launchers should use `scripts/lib/common.sh` unless they are one of
the explicitly bespoke wrappers in `scripts/checks/check_shell_launchers.py`.
The migrated pattern is:

```bash
#!/usr/bin/env bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../lib/common.sh"
enter_repo_root "${SCRIPT_DIR}"
run_python_module_config src.cli.train_flow_matching configs/fm/train/presets/stable_latent.yaml "$@"
```

Preserve `"$@"` so agents and users can pass runtime overrides. Make wrappers
point at existing YAML files; `scripts/checks/check_shell_launchers.py` and
`scripts/checks/check_script_config_mapping.py` inspect this.

## Slurm Conventions

Slurm launchers are intentionally self-contained Bash scripts. Do not add
`slurm/lib/common.sh` or helper-style calls such as `slurm_grep_config_keys`,
`slurm_print_gpu_diagnostics`, or `slurm_run_timed`. Inline the relevant `echo`,
`grep`, `if [[ ! -f ... ]]`, `nvidia-smi`, `/usr/bin/time`, and Python command.

A typical launcher, such as
`slurm/killarney/flir/flow_matching/train_stable_fm_hflip_ot_kl.slurm`, does the
following:

- declares `#SBATCH` account, resources, GPU, and log paths;
- enables `set -euo pipefail`;
- resolves `PROJECT_ROOT`, `ENV_NAME`, `CONFIG_REL`, and `CONFIG`;
- changes to the project root and sets `PYTHONPATH`;
- activates Conda directly;
- prints host, time, config, Python, and GPU diagnostics;
- checks the config path before launching;
- calls a `src.cli` module with `--config`.

Keep cluster-specific paths, accounts, GPU requests, and log locations in Slurm
files. Keep experiment behavior in YAML configs. If adding, removing, or
renaming Slurm files, inspect `scripts/checks/check_slurm_launchers.py`; it
encodes expected launcher sets, headers, config references, and the no-helper
policy.

## Artifacts, Data, and Paths

Use `src/core/paths.py` for canonical paths instead of hard-coded root-relative
strings. Current canonical roots include:

- datasets and caches: `data/raw/`, `data/derived/`, `data/cache/`;
- checkpoints: `artifacts/checkpoints/flow_matching/`,
  `artifacts/checkpoints/vae/`, `artifacts/checkpoints/stable_diffusion/`,
  `artifacts/checkpoints/count_adapter/`, `artifacts/checkpoints/yolo/`;
- generated data: `artifacts/generated/`;
- analysis outputs: `artifacts/analysis/`;
- debug outputs: `artifacts/debug/`;
- logs/runs: `artifacts/runs/`.

Do not write new generated datasets, checkpoints, TensorBoard events, model
weights, local `.env` files, or private data into tracked source. `.gitignore`
keeps heavy contents out while preserving `.gitkeep` structure. Checks such as
`scripts/checks/check_checkpoint_roots.py`,
`scripts/checks/check_generated_paths.py`, `scripts/checks/check_run_log_paths.py`,
`scripts/checks/check_dataset_locations.py`, and
`scripts/checks/check_canonical_layout_only.py` look for stale root-level paths
like `./generated/`, `./debug_samples/`, `./runs_test/`, `./serious_runs/`,
`./vae_runs/`, and `./stable_diffusion_15_out/`.

## Frontend and Analysis Service

Install the backend web extras from the repository root:

```bash
python -m pip install -e .[web]
python serve_flir_analysis.py --host 127.0.0.1 --port 8000
```

`src/cli/serve_flir_analysis.py` launches `src.analysis.flir_subgroup.app`.
Backend behavior should stay in `src/analysis/flir_subgroup/`; UI behavior
should stay in `frontend/flir-subgroup-analysis/`.

Run the frontend:

```bash
cd frontend/flir-subgroup-analysis
npm install
npm run dev
```

The frontend defaults to `http://127.0.0.1:8000`. Override with:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Build validation:

```bash
cd frontend/flir-subgroup-analysis && npm run build
```

Relevant tests include `tests/test_flir_subgroup_analysis.py`,
`tests/test_flir_subgroup_notebook_parity.py`, and
`tests/test_checkpoint_selection_viewer.py`.

## Tests and Check Scripts

Run the narrowest meaningful tests for the change. Examples:

```bash
python -m pytest tests/test_text_fm_cfg.py -v
python -m pytest tests/test_config_loader_extends.py -v
python -m pytest tests/test_flir_subgroup_analysis.py -v
python scripts/checks/check_repo_paths.py
python scripts/checks/check_wrappers_only.py
python scripts/checks/check_config_loading.py
```

For broad repo checks in the expected Conda environment:

```bash
for f in scripts/checks/check_*.py; do
  conda run -n diffusers-dev python "$f"
done
```

Use targeted checks when touching their area:

- wrappers or CLI entry points: `scripts/checks/check_wrappers_only.py`;
- config loading, YAML defaults, or CLI override mapping:
  `scripts/checks/check_config_loading.py`;
- path helpers or artifact roots: `scripts/checks/check_repo_paths.py`,
  `scripts/checks/check_checkpoint_roots.py`,
  `scripts/checks/check_generated_paths.py`, `scripts/checks/check_run_log_paths.py`;
- data locations: `scripts/checks/check_dataset_locations.py`;
- shell launchers: `scripts/checks/check_shell_launchers.py`;
- Slurm launchers: `scripts/checks/check_slurm_launchers.py`;
- legacy boundaries: `scripts/checks/check_no_legacy_runtime_dependency.py`.

Avoid expensive training, generation, downloads, or Slurm submissions as
validation unless the user explicitly asks.

## Common Pitfalls Encoded in Checks

- Root wrappers are not a place for logic. Put implementation in `src/cli/` or
  the relevant `src/` package.
- CLI parsers must preserve YAML config values. Be careful with argparse
  defaults; use the existing merge helpers so only explicit CLI args override.
- New CLI flags in `src/cli/train_flow_matching.py` need `_FLAT_TO_NESTED`
  coverage, or config merging can drift.
- `dict_to_dataclass()` ignores unknown YAML keys, but
  `load_experimental_config()` is strict for `kind: experimental_training_config`.
- `src/core/paths.py` should not grow fallback logic to old root-level folders.
- Active `src/`, `scripts/`, and `tests/` code must not import `fm_src` or
  `sd_src`; retired code belongs under `archive/legacy_code/`.
- `src/diffusers/` and `src/flow-matching-trial/` are excluded legacy or nested
  trees in legacy-dependency checks; do not use them as active code targets
  without an explicit request.
- Shell launchers and YAML presets are mapped by static checks. If a preset is
  intentionally unwrapped or a wrapper is bespoke, update/check the allowlists
  deliberately.
- Slurm launchers use plain Bash directly, unlike local shell launchers.
- Some checks validate physical private-data directories such as
  `data/raw/v18/`, `data/raw/flir_private_proxy_alignment_v18/`, and
  `data/cache/dino_cache/`. If they fail because private data is absent, report
  that clearly instead of inventing placeholder data.
- The root cleanliness check has a whitelist and may flag unrelated local files.
  Do not delete or move user/agent work just to satisfy it unless asked.

## Safe Contribution Practices

- Work from the repository root.
- Keep changes scoped to the requested files and active source/config/test trees.
- Preserve existing dirty worktree changes made by other agents or the user.
- Prefer existing helpers in `src/core/` for paths, configs, registries,
  datasets, normalization, and artifact handling.
- Add fast, local tests with small tensors, temp directories, or mocks when a
  behavior change needs coverage.
- Document validation performed. If a relevant check cannot run because of
  missing private data, unavailable GPU dependencies, or unrelated workspace
  state, say so directly.

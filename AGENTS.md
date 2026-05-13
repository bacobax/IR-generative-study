# Repository Guidelines

## Project Shape & Ownership
Work from the repository root. `src/` is the source of truth for active Python code: CLI entry points live in `src/cli/`, reusable training and inference logic in `src/algorithms/`, shared path/config/data utilities in `src/core/`, conditioning logic in `src/conditioning/`, model definitions in `src/models/`, guidance code in `src/guidance/`, evaluation in `src/evaluation/`, and analysis code in `src/analysis/`.

Root scripts such as `train_sfm.py`, `train_sd.py`, `train_sd_uncond.py`, `train_vae.py`, `train_controlnet.py`, and `generate_datasets.py` are thin wrappers around `src/cli/`. Do not add core logic to root wrappers; add or update the corresponding `src/cli/*` module and keep wrappers minimal.

Keep experiment configuration under `configs/`, grouped by task such as `fm`, `sd`, `sd_uncond`, `sd_layout`, `vae`, `controlnet`, `auxiliary`, `analysis`, `eval`, `models`, and `yolo`. Put shell automation in `scripts/`, Slurm launchers in `slurm/<cluster>/`, tests in `tests/`, docs and notebooks in `docs/`, and web UI code in `frontend/flir-subgroup-analysis/`.

Treat `archive/`, `ControlNet/`, `src/diffusers/`, and nested checkout-like trees such as `src/flow-matching-trial/` as external or legacy reference material unless the user explicitly asks to touch them. Active changes should normally land in the top-level `src/`, `configs/`, `scripts/`, `slurm/`, `tests/`, or `frontend/` trees.

## Data, Artifacts & Large Files
Keep large or generated outputs out of tracked source. Use `data/raw/`, `data/derived/`, and `data/cache/` for datasets and preprocessing caches. Use `artifacts/checkpoints/`, `artifacts/runs/`, `artifacts/generated/`, `artifacts/analysis/`, `artifacts/evaluations/`, `artifacts/cache/`, and `logs/` for experiment outputs. Do not commit weights, checkpoints, TensorBoard event files, generated datasets, or local `.env` files.

When adding a new training or generation flow, make output paths configurable and default them into `artifacts/` or `logs/`, not the repository root. Preserve `.gitkeep` directory structure when present.

## Development Commands
Use Python 3.10+. The project is config-first, so prefer exercising CLIs through YAML presets.

```bash
python train_sfm.py --config configs/fm/train/default.yaml
python train_sd.py --config configs/sd/train/default.yaml
python train_sd_uncond.py --config configs/sd_uncond/train/default.yaml
python train_vae.py --config configs/vae/train/presets/vae_4x.yaml
python generate_datasets.py --mode fm --max_samples 100
python -m pytest tests -v
```

For the subgroup analysis service:

```bash
python -m pip install -e .[web]
python serve_flir_analysis.py --host 127.0.0.1 --port 8000
cd frontend/flir-subgroup-analysis && npm install && npm run dev
```

Cluster jobs use Slurm wrappers such as `slurm/killarney/*.slurm` and usually activate the `diffusers-dev` Conda environment. Keep cluster-specific paths, accounts, GPU requests, and log locations in Slurm files; keep experiment behavior in YAML configs.

## Config & Launcher Conventions
Prefer adding YAML presets under the matching `configs/<area>/<action>/presets/` directory rather than hard-coding behavior. Keep names descriptive and stable, for example `flir_unet_full_domainstudio_512.yaml` or `regiondiff_latent_flir_sd15_512_from_uncond_b64_hflip.yaml`.

When a shell or Slurm wrapper launches an experiment, it should point at a config via a variable such as `CONFIG_REL` or a clear command-line argument. If you change a config schema or registry mapping, update the matching loader, defaults, tests, and check scripts together.

## Coding Style
Follow the existing Python style: 4-space indentation, `snake_case` for files/functions/config keys, `PascalCase` for classes, and type hints where they clarify contracts. Add module docstrings for nontrivial entry points. Keep comments useful and sparse.

Use existing utilities in `src/core/` for paths, configs, registries, datasets, and normalization before adding new helpers. Avoid unsanctioned runtime dependencies on retired `fm_src` or `sd_src`; legacy code belongs under `archive/legacy_code/`.

## Testing & Validation
Add tests beside related coverage in `tests/` using `test_<feature>.py` and `test_<behavior>()` names. Favor fast tests with small tensors, temporary directories, and mocked conditioners/tokenizers over tests that download models or require long GPU runs.

Run the narrowest meaningful validation while iterating:

```bash
python -m pytest tests/test_text_fm_cfg.py -v
python scripts/checks/check_repo_paths.py
python scripts/checks/check_wrappers_only.py
python scripts/checks/check_config_loading.py
```

When touching launcher wiring, paths, registries, config loading, or legacy boundaries, run the relevant `scripts/checks/check_*.py` script. For broader local validation in the expected Conda environment:

```bash
for f in scripts/checks/check_*.py; do conda run -n diffusers-dev python "$f"; done
```

Do not launch expensive training, generation, or Slurm jobs as a validation step unless the user explicitly asks. If a check fails because the local workspace has intentional extra files or missing private data, report that clearly instead of reshaping unrelated files.

## Frontend Notes
The React dashboard lives in `frontend/flir-subgroup-analysis/` and talks to the FastAPI service from `serve_flir_analysis.py`. Keep API-facing analysis behavior in `src/analysis/flir_subgroup/` and UI behavior in the frontend package. Validate frontend changes with:

```bash
cd frontend/flir-subgroup-analysis && npm run build
```

## Commit & PR Guidance
Recent history uses short imperative subjects, often with Conventional Commit prefixes such as `feat:`, `fix:`, or `test:`. Keep commits focused. PRs should summarize the behavior change, list touched configs/scripts, note validation performed, and include sample outputs or screenshots only when generated artifacts or analysis UI behavior changed.

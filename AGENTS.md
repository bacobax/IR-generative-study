# Repository Guidelines

## Project Structure & Module Organization
`src/` is the source of truth for active code. Put CLI entry points in `src/cli/`, reusable training and inference code in `src/algorithms/`, shared utilities in `src/core/`, and model definitions in `src/models/` and `src/conditioning/`. Root scripts such as `train_sfm.py` and `train_sd.py` are thin wrappers around `src/cli/`. Keep experiment presets under `configs/`, lightweight automation under `scripts/`, tests in `tests/`, and large outputs in `artifacts/`. Treat `archive/` as read-only legacy material.

## Build, Test, and Development Commands
Use Python 3.10+.

```bash
python train_sfm.py --config configs/fm/train/default.yaml
python train_sd.py
python generate_datasets.py --mode fm --max_samples 100
python -m pytest tests -v
python scripts/checks/check_repo_paths.py
for f in scripts/checks/check_*.py; do conda run -n diffusers-dev python "$f"; done
```

The training and generation commands exercise the config-driven launchers. `pytest` covers unit and smoke tests. `scripts/checks/check_*.py` validates repository layout, path helpers, registries, and CLI wiring.

## Coding Style & Naming Conventions
Follow existing Python style: 4-space indentation, type hints where useful, and module-level docstrings for nontrivial entry points. Use `snake_case` for files, functions, and config keys; use `PascalCase` for classes. Prefer extending code in `src/` instead of adding logic to root wrappers. Match the repository’s config-first pattern by adding YAML presets under the relevant `configs/<area>/.../presets/` directory.

## Testing Guidelines
Add tests beside related coverage in `tests/` using `test_<feature>.py` and `test_<behavior>()` names. Favor fast smoke tests with small tensors, temporary directories, and mock conditioners/tokenizers instead of real model downloads. Run targeted tests during iteration, for example:

```bash
python -m pytest tests/test_text_fm_cfg.py -v
```

When changing path, registry, or launcher behavior, run the matching script in `scripts/checks/`.

## Commit & Pull Request Guidelines
Recent history uses short, imperative subjects, often with Conventional Commit prefixes such as `feat:`. Keep commits focused and descriptive, for example `feat: add meta FM resume preset`. PRs should summarize the behavior change, list touched configs or scripts, note validation performed, and include sample outputs or screenshots only when a generation or analysis change affects user-facing artifacts.

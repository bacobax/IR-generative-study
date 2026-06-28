# Repository Guidelines

## Agent Doc Routing
Start here, then load only the smallest useful docs under `docs/agent_docs/`. Load at most one high-level training type doc when the request clearly maps to one. Load at most one training regime doc and at most one training objective doc unless the user explicitly spans multiple types. Load dataset docs only for dataset loading, normalization, preprocessing, targets, annotations, splits, manifests, analysis datasets, or dataset-specific configs. Do not load unrelated docs just in case. Use `rg` first to confirm exact paths before editing. Keep root wrappers thin. When making a refactor, update the affected `docs/agent_docs/` routing docs in the same change so future agents follow the new structure. Run narrow validation only.

| User asks about... | Load these docs | Do not load unless explicitly mentioned |
|---|---|---|
| v18 loading/normalization/splits | `docs/agent_docs/datasets/v18.md` | RegionDiff/STAY docs |
| FLIR subgroup/data analysis | `docs/agent_docs/datasets/flir.md` | SDXL/RegionDiff docs |
| BigEarthNet S2 B08 5x5 stride3 | `docs/agent_docs/datasets/bigearthnet_s2_b08_5x5_stride3.md` | FLIR/v18 docs |
| subsampling/manifests/holdouts | `docs/agent_docs/datasets/subsampling.md` | Training objective docs unless training behavior changes |
| layout slices / bbox subgroup analysis / class-area-rarity balancing | `docs/agent_docs/analysis/layout_slices_balance.md` | STAY/RegionDiff generation docs unless conditioning is involved |
| native/custom/simple YOLO detector architecture/training/eval | `docs/agent_docs/training_objectives/simple_yolo_detector_training.md` | Ultralytics Experiment A/B docs unless those harnesses change |
| YOLO detector training/eval, Experiment A/B, synthetic-aug detector runs | `docs/agent_docs/training_objectives/yolo_detector_training.md` | FM/SD/RegionDiff generation docs unless the generators themselves change |
| flow matching objective | `docs/agent_docs/training_objectives/flow_matching.md` | diffusion/RegionDiff docs |
| diffusion objective | `docs/agent_docs/training_objectives/diffusion.md` | flow/STAY docs |
| from-scratch training | `docs/agent_docs/training_regime/from_scratch.md` | SD15/SDXL fine-tuning docs |
| SD15 adaptation | `docs/agent_docs/high_level_training_types/sd15_adaptation.md` | SDXL/STAY docs |
| SDXL adaptation | `docs/agent_docs/high_level_training_types/sdxl_adaptation.md` | SD15/STAY docs |
| FLUX QLoRA adaptation | `docs/agent_docs/high_level_training_types/flux_adaptation.md` | SDXL/SD15/STAY docs |
| STAY flow matching | `docs/agent_docs/high_level_training_types/stay_cond_flow_matching.md` | `region_diff.md` |
| RegionDiff | `docs/agent_docs/high_level_training_types/region_diff.md` | STAY docs |

### Sub-agent usage
Spawn sub-agents only when independent discovery reduces main-context load. Use them for reads like "inspect dataset routing files" or "inspect SDXL training configs" when results can be summarized. Do not spawn multiple agents to edit the same files. Give each sub-agent a narrow path allowlist and a concise output contract: relevant files, key findings, recommended edits, validation suggestions. Sub-agents must not load unrelated agent docs. The main agent owns final edits, consistency, and validation. For STAY flow-matching tasks, do not spawn a RegionDiff sub-agent unless RegionDiff is explicit. For RegionDiff tasks, do not spawn a STAY sub-agent unless STAY is explicit. Prefer parallel sub-agents only for independent reads; sequence work when one result determines the next step.

## Project Shape & Ownership
Work from the repository root. `src/` is the source of truth for active Python code: CLI entry points live in `src/cli/`, reusable training and inference logic in `src/algorithms/`, shared path/config/data utilities in `src/core/`, conditioning logic in `src/conditioning/`, model definitions in `src/models/`, guidance code in `src/guidance/`, evaluation in `src/evaluation/`, and analysis code in `src/analysis/`. The subgroup-analysis API logic is in `src/analysis/flir_subgroup/` and is served through `src/cli/serve_flir_analysis.py`.

Root scripts such as `train_flow_matching.py`, `adapt_stable_diffusion.py`, `train_latent_diffusion.py`, `train_vae.py`, `train_controlnet.py`, `train_sdxl.py`, `generate_datasets.py`, and `serve_flir_analysis.py` are compatibility wrappers around `src/cli/` modules. `analyze_distribution_shift.py` and `train_count_adapter.py` wrap `scripts/standalone/` modules. Do not add core logic, argparse handling, training loops, pipeline classes, or config loading to these wrappers.

Keep experiment configuration under `configs/`, grouped by task such as `fm`, `sd`, `sd_uncond`, `sdxl`, `sd_layout`, `vae`, `controlnet`, `auxiliary`, `analysis`, `eval`, `models`, `datasets`, and `yolo`. Put reusable shell automation in `scripts/`, Slurm launchers in `slurm/<cluster>/`, tests in `tests/`, docs and notebooks in `docs/`, and web UI code in `frontend/flir-subgroup-analysis/`.

Treat `archive/`, `ControlNet/`, `src/diffusers/`, and nested checkout-like trees such as `src/flow-matching-trial/` as external or legacy reference material unless explicitly asked to touch them. Active changes should normally land in top-level `src/`, `configs/`, `scripts/`, `slurm/`, `tests/`, `docs/`, or `frontend/`.

## Data, Artifacts & Large Files
Keep large or generated outputs out of tracked source. Use `data/raw/` for raw datasets and `data/derived/` or `data/cache/` for preprocessing outputs. Use `artifacts/checkpoints/`, `artifacts/runs/`, `artifacts/generated/`, `artifacts/analysis/`, `artifacts/evaluations/`, `artifacts/cache/`, and `logs/` for experiment outputs.

Do not commit weights, checkpoints, TensorBoard event files, generated datasets, local `.env` files, or new root-level output dumps. When adding training, generation, or evaluation flows, make output paths configurable and default them into `artifacts/` or `logs/`. Preserve `.gitkeep` directory structure when present.

## Development Commands
Use Python 3.10+. The project is config-first, so prefer exercising CLIs through YAML presets.

```bash
python -m pip install -e .
python -m pip install -e .[web]
python train_flow_matching.py --config configs/fm/train/default.yaml
python adapt_stable_diffusion.py --config configs/sd/train/default.yaml
python train_latent_diffusion.py --config configs/sd_uncond/train/default.yaml
python train_vae.py --config configs/vae/train/presets/vae_4x.yaml
python generate_datasets.py --mode fm --max_samples 100
python -m src.cli.train_flow_matching --config configs/fm/train/presets/stable_latent.yaml
python -m pytest tests -v
```

For the subgroup analysis service:

```bash
python -m pip install -e .[web]
python serve_flir_analysis.py --host 127.0.0.1 --port 8000
cd frontend/flir-subgroup-analysis && npm install && npm run dev
cd frontend/flir-subgroup-analysis && npm run build
```

Cluster jobs use Slurm wrappers such as `slurm/killarney/*.slurm` and usually activate the `diffusers-dev` Conda environment. Do not launch expensive training, generation, or Slurm jobs as validation unless explicitly asked.

## Config & Launcher Conventions
Prefer adding YAML presets under the matching `configs/<area>/<action>/presets/` directory rather than hard-coding behavior. Keep names descriptive and stable, for example `flir_unet_full_domainstudio_512.yaml` or `regiondiff_latent_flir_sd15_512_from_uncond_b64_hflip.yaml`.

Config precedence is `CLI flag > YAML preset > Python/dataclass default`. `src/core/configs/config_loader.py` implements `load_yaml()` with relative `extends:` support, `apply_yaml_defaults()` for argparse scripts, and `merge_config_and_cli()` for nested dataclass configs. YAML presets should contain only values that differ from Python defaults, and new config keys must be wired through the loader, defaults, registry or mapping code, tests, and checks together.

Regular shell wrappers in `scripts/train/` and `scripts/generate/` normally source `scripts/lib/common.sh`, enter the repo root, and call `run_python_module_config` or `run_python_script_config` with a repo-relative config path. When a shell or Slurm wrapper launches an experiment, keep the config path visible via `CONFIG_REL` or a clear `--config` argument.

Slurm launchers must use plain Bash/Slurm syntax directly. Keep cluster-specific paths, accounts, GPU requests, Conda activation, log locations, diagnostics, and `/usr/bin/time` calls in the `.slurm` file; keep experiment behavior in YAML configs. Do not add macro, wrapper, helper, or function calls for Slurm runtime setup, diagnostics, config inspection, path checks, GPU checks, or timed execution. In particular, do not use `slurm_*` helpers such as `slurm_grep_config_keys`, `slurm_print_gpu_diagnostics`, or `slurm_run_timed`; write the `echo`, `grep`, conditionals, `nvidia-smi`, `/usr/bin/time`, and command lines inline.

## Coding Style
Follow the existing Python style: 4-space indentation, `snake_case` for files/functions/config keys, `PascalCase` for classes, and type hints where they clarify contracts. Add module docstrings for nontrivial entry points. Keep comments useful and sparse.

Use existing utilities in `src/core/` for paths, configs, registries, datasets, normalization, GPU handling, and training runtime before adding new helpers. Avoid unsanctioned runtime dependencies on retired `fm_src` or `sd_src`; legacy code belongs under `archive/legacy_code/`, and active code should not import from it unless a check documents a sanctioned transition.

## Testing & Validation
Add tests beside related coverage in `tests/` using `test_<feature>.py` and `test_<behavior>()` names. Favor fast tests with small tensors, temporary directories, and mocked conditioners/tokenizers over tests that download models or require long GPU runs.

Run the narrowest meaningful validation while iterating:

```bash
python -m pytest tests/test_text_fm_cfg.py -v
python scripts/checks/check_repo_paths.py
python scripts/checks/check_wrappers_only.py
python scripts/checks/check_config_loading.py
```

When touching launcher wiring, paths, registries, config loading, generated-output locations, or legacy boundaries, run the relevant `scripts/checks/check_*.py` script. Common targeted checks include `check_shell_launchers.py`, `check_slurm_launchers.py`, `check_script_config_mapping.py`, `check_no_legacy_runtime_dependency.py`, `check_checkpoint_roots.py`, `check_generated_paths.py`, and `check_run_log_paths.py`. For broader local validation in the expected Conda environment:

```bash
for f in scripts/checks/check_*.py; do conda run -n diffusers-dev python "$f"; done
```

If a check fails because the local workspace has intentional extra files, missing private data, or unrelated worktree changes, report that clearly instead of reshaping unrelated files.

## Frontend Notes
The React dashboard lives in `frontend/flir-subgroup-analysis/` and talks to the FastAPI service from `serve_flir_analysis.py`. Keep API-facing analysis behavior in `src/analysis/flir_subgroup/` and UI behavior in the frontend package. Validate frontend changes with:

```bash
cd frontend/flir-subgroup-analysis && npm run build
```

## AI-Agent Pitfalls
Assume multiple agents or the user may be editing in parallel. Inspect `git status --short` before editing, keep write scope tightly tied to the request, and never revert or clean up changes you did not make. If unrelated files are already modified or deleted, leave them alone.

Use `rg`/`rg --files` for discovery and inspect existing code, configs, tests, and checks before documenting or changing behavior. Do not invent features or workflows in guidance; extract them from the repo.

Keep generated files, pyc caches, notebooks, PDFs, weights, and outputs out of normal code changes. Do not touch legacy or external trees for convenience, even if a search result lands there first.

## Commit & PR Guidance
Recent history uses short imperative subjects, often with Conventional Commit prefixes such as `feat:`, `fix:`, or `test:`. Keep commits focused. PRs should summarize the behavior change, list touched configs/scripts, note validation performed, and include sample outputs or screenshots only when generated artifacts or analysis UI behavior changed.

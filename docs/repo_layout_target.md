# Repository Layout — Final Implemented State

This document describes the **current** directory structure of the
flow-matching-trial repository after the full cleanup migration.

---

## Root overview

```
flow_matching_trial/
├── src/                  # Source of truth for all runnable code
├── configs/              # YAML / JSON experiment configs
├── scripts/              # Shell helpers, standalone Python scripts & checks
├── docs/                 # Documentation, notebooks, prompt files
├── data/                 # Datasets (gitignored heavy contents, .gitkeep tracked)
├── artifacts/            # All training outputs (gitignored heavy contents)
├── archive/              # Retired code & outputs
├── tests/                # Automated tests
├── train_sfm.py          # Thin wrapper → src.cli.train
├── train_sd.py           # Thin wrapper → src.cli.train_sd
├── train_vae.py          # Thin wrapper → src.cli.train_vae
├── train_controlnet.py   # Thin wrapper → src.cli.train_controlnet
├── generate_datasets.py  # Thin wrapper → src.cli.generate
├── pyproject.toml
├── README.md
└── .gitignore
```

---

## `src/` — Source of truth

| Sub-path | Purpose |
|---|---|
| `src/cli/` | CLI entrypoints (train, sample, generate, train_sd, train_vae, train_controlnet) |
| `src/core/` | Shared utilities: paths, configs, constants, normalization, registry, gpu_utils |
| `src/core/configs/` | Dataclass configs + config_loader (YAML/CLI merge) |
| `src/algorithms/training/` | FM trainer |
| `src/algorithms/inference/` | FM sampler |
| `src/algorithms/stable_diffusion/` | SD LoRA training & modules |
| `src/models/` | Model construction helpers (UNet, VAE) |
| `src/guidance/` | Guidance ABC + implementations (surprise, GMM, combo) |
| `src/conditioning/` | Conditioning ABC + implementations |
| `src/data/` | Datasets and transforms |

---

## `configs/`

```
configs/
├── fm/
│   ├── train/            # FM training YAML defaults & experiment overrides
│   └── sample/           # FM sampling YAML defaults
├── sd/
│   └── train/            # SD LoRA training YAML defaults
└── models/
    ├── fm/               # UNet & VAE architecture JSON/YAML
    └── sd/               # SD model references
```

---

## `data/`

All dataset files.  **Heavy contents ignored by git** (.gitkeep tracked).

| Sub-path | Contents |
|---|---|
| `data/raw/v18/` | Original IR images (`train/`, `val/` splits) |
| `data/derived/surprise_pred_dataset/` | Preprocessed surprise prediction dataset |
| `data/cache/dino_cache/` | Cached DINOv2 features |

---

## `artifacts/`

All outputs produced by training, sampling, and analysis scripts.
**Heavy contents ignored by git** (.gitkeep tracked).

| Sub-path | Contents |
|---|---|
| `artifacts/checkpoints/flow_matching/serious_runs/` | FM UNet + VAE weights |
| `artifacts/checkpoints/vae/vae_runs/` | Standalone VAE training outputs |
| `artifacts/checkpoints/stable_diffusion/lora_runs/` | SD LoRA checkpoints |
| `artifacts/checkpoints/count_adapter/runs/` | Count-adapter runs |
| `artifacts/checkpoints/legacy/` | Legacy pipeline_model, UNET, VAE copies |
| `artifacts/runs/main/` | TensorBoard / experiment-tracker logs (primary) |
| `artifacts/runs/test/` | Test experiment logs |
| `artifacts/generated/main/` | Generated image datasets (primary) |
| `artifacts/generated/test/` | Test-time generated datasets |
| `artifacts/generated/old/` | Archived older generated datasets |
| `artifacts/analysis/main/` | Distribution-shift analysis results (primary) |
| `artifacts/analysis/test/` | Test analysis results |
| `artifacts/debug/debug_samples/` | Debug sample images |

---

## `archive/`

Retired material that is no longer on the active execution path but
preserved for reference.

| Sub-path | Contents |
|---|---|
| `archive/legacy_code/fm_src/` | Old monolithic FM pipeline code. **Tracked by git.** |
| `archive/legacy_code/sd_src/` | Old monolithic SD pipeline code. **Tracked by git.** |
| `archive/legacy_outputs/` | Old output folders no longer needed. **Ignored by git.** |

---

## `scripts/`

Shell launcher scripts (`*.sh`), standalone Python scripts, and verification
check scripts (`check_*.py`).  Not part of the importable `src` package.

| Category | Examples |
|---|---|
| Shell launchers | `train_stable_fm.sh`, `generate_dataset_fm.sh`, `analysis.sh`, … |
| Standalone Python | `train_fm.py`, `train_count_adapter.py`, `analyze_distribution_shift.py`, … |
| Check scripts | `check_repo_paths.py`, `check_canonical_layout_only.py`, … |

---

## `docs/`

| Sub-path | Contents |
|---|---|
| `docs/notebooks/` | Jupyter analysis notebooks (cluster study, DINO features, guided sampling) |
| `docs/repo_layout_target.md` | This file |
| `docs/refactor_plan.md` | Original refactor planning document |

---

## `tests/`

Automated tests (unit, integration, smoke).  Currently empty — will be
populated as the codebase stabilises.  The `scripts/check_*.py` scripts
serve as interim structural-verification tests.

---

## Canonical paths

All path resolution is centralized in `src/core/paths.py`.  This module
provides 40+ helpers that resolve relative to the repository root.  Code
should import from `src.core.paths` rather than hard-coding paths.

# Incremental Refactor Plan

## Status: In Progress

This project is entering an **incremental refactor** toward a modular `src/` package architecture.

## Current State

- All existing entry points (`train_sfm.py`, `train_sd.py`, `train_vae.py`, `train_controlnet.py`, `generate_datasets.py`, etc.) remain the **source of truth** and are unchanged.
- A new `src/` package tree has been created with empty sub-packages as scaffolding for future code migration.

## Target Package Layout

```
src/
├── core/          # Core utilities, configuration, shared helpers
│   └── data/      # Dataset loading, preprocessing, data pipelines
├── models/        # Model definitions (VAE, UNet, ControlNet, adapters)
├── algorithms/
│   ├── training/  # Training loops, schedulers, optimization
│   └── inference/ # Sampling, generation, inference pipelines
├── cli/           # CLI entry points and argument parsing
└── analysis/      # Analysis, evaluation, visualization
```

## Migration Rules

1. **No breaking changes** — old root scripts must keep working at every step.
2. Code is moved **one module at a time**, with the root script updated to re-export or import from `src/`.
3. Each migration step must pass existing tests / manual verification before proceeding.
4. Existing `fm_src/` and `sd_src/` directories will be absorbed into `src/` in later phases.

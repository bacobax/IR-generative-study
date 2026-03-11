# Config-Driven Launcher Workflow

## Overview

Experiment settings live in **YAML config presets** under `configs/`.
Shell scripts under `scripts/` are **thin wrappers** that pair a CLI
entry-point with the right config file.  This keeps experiment parameters
in one place (YAML) and avoids duplicating values across shell, YAML,
and Python defaults.

## Architecture

```
  shell wrapper          YAML preset             Python CLI
┌────────────────┐   ┌──────────────────┐   ┌──────────────────────┐
│ scripts/train/ │──▶│ configs/fm/train/ │──▶│ src/cli/train.py     │
│ fm_stable.sh   │   │ stable_latent.yaml│   │ (FMTrainConfig)      │
└────────────────┘   └──────────────────┘   └──────────────────────┘
```

1. **Shell wrapper** (`scripts/<category>/<name>.sh`) — sets up `cd $ROOT`,
   then runs `python -m src.cli.<module> --config <preset> "$@"`.
   Passes any extra CLI flags through `"$@"`.

2. **YAML preset** (`configs/<domain>/<stage>/presets/<name>.yaml`) — contains
   only the parameter values that differ from the Python defaults.
   All other settings fall back to the Python-level defaults.

3. **Python CLI** (`src/cli/` or `scripts/standalone/`) — defines
   `argparse` arguments with sensible defaults.  Loads the YAML preset
   via `--config`, merges with CLI overrides (CLI wins).

## Precedence

```
CLI flag  >  YAML preset  >  Python default
```

- If a flag is given on the command line, it always wins.
- If not, the YAML preset value is used.
- If neither specifies a value, the Python default applies.

This is implemented by a two-pass parsing pattern:

```python
preliminary, _ = parser.parse_known_args()
if preliminary.config:
    apply_yaml_defaults(parser, preliminary.config)
args = parser.parse_args()
```

The `apply_yaml_defaults()` function (in `src/core/configs/config_loader.py`)
loads the YAML file, filters keys to known `argparse` destinations, and
calls `parser.set_defaults()` before the final parse.

For nested dataclass configs (e.g. `FMTrainConfig`), the equivalent is
`merge_config_and_cli()` which deep-merges YAML with the dataclass defaults.

## Directory Layout

### Shell wrappers (`scripts/`)

| Directory           | Purpose                                      |
|---------------------|----------------------------------------------|
| `scripts/train/`    | Training launchers (FM, VAE, SD, ControlNet, cluster, etc.) |
| `scripts/generate/` | Generation / sampling launchers              |
| `scripts/analyze/`  | Analysis launchers                           |
| `scripts/checks/`   | Repo health check scripts                    |
| `scripts/standalone/`| Python entry-point scripts (not CLI modules) |

### Config presets (`configs/`)

| Directory                              | What it configures              |
|----------------------------------------|---------------------------------|
| `configs/fm/train/presets/`            | FM training experiments         |
| `configs/fm/generate/presets/`         | FM generation experiments       |
| `configs/vae/train/presets/`           | VAE training                    |
| `configs/sd/train/presets/`            | SD LoRA training                |
| `configs/sd/generate/presets/`         | SD generation                   |
| `configs/controlnet/train/presets/`    | ControlNet training             |
| `configs/auxiliary/*/presets/`         | Cluster recon, surprise pred, count adapter |
| `configs/analysis/presets/`            | Analysis scripts                |

## YAML Preset Conventions

1. **Only override non-defaults.** Presets contain only values that differ
   from the Python defaults.  A comment at the top notes this:
   ```yaml
   # Only values that differ from train_vae.py defaults are listed.
   ```

2. **Header comments** include a description and the launcher path:
   ```yaml
   # VAE training: 8x compression
   # Launcher: scripts/train/vae_8x.sh
   ```

3. **Flat key format** for `argparse`-based scripts (keys match `--flag` dest):
   ```yaml
   epochs: 300
   train_target: x0
   ```

4. **Nested key format** for dataclass-based configs:
   ```yaml
   training:
     epochs: 1
   output:
     model_dir: ./artifacts/runs/test/test_fm_dataset
   ```

## Running Experiments

### Via shell wrapper (recommended)

```bash
# Train FM (stable latent)
bash scripts/train/fm_stable.sh

# Override a single parameter
bash scripts/train/fm_stable.sh --training.epochs 50

# Generate FM guided samples
bash scripts/generate/fm_guided_combo_maxmin.sh
```

### Via Python directly

```bash
# With a config preset
python -m src.cli.train --config configs/fm/train/presets/stable_latent.yaml

# With inline overrides
python -m src.cli.generate \
  --config configs/fm/generate/presets/plain_100_steps.yaml \
  --max_samples 50

# Without a config (all defaults + flags)
python -m src.cli.train_vae --train-dir ./data/raw/v18/train --val-dir ./data/raw/v18/val
```

### Via root thin wrappers (legacy convenience)

```bash
python train_sfm.py --config configs/fm/train/presets/stable_latent.yaml
python generate_datasets.py --config configs/fm/generate/presets/plain_100_steps.yaml
```

## Adding a New Experiment

1. Create a YAML preset under the appropriate `configs/.../presets/` directory.
   Include only values that differ from the Python defaults.

2. Create a shell wrapper under the appropriate `scripts/` subdirectory:
   ```bash
   #!/usr/bin/env bash
   set -euo pipefail
   SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
   ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null || echo "$SCRIPT_DIR/../..")"
   cd "$ROOT_DIR"
   python -m src.cli.<module> --config configs/<path>/presets/<name>.yaml "$@"
   ```

3. Run it:
   ```bash
   bash scripts/train/my_new_experiment.sh
   ```

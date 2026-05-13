# Flow Matching Trial

## Repository Layout

```
flow_matching_trial/
├── src/                 # Source of truth — all active Python code
│   ├── cli/             # CLI entry-points (train, sample, generate, …)
│   ├── core/            # Shared utilities: paths, configs, registry, normalization
│   ├── algorithms/      # Training & inference logic (flow matching, SD, guidance, …)
│   └── data/            # Datasets and transforms
├── configs/             # Experiment configuration (YAML / JSON)
│   ├── fm/              # Flow-matching train & generate presets
│   ├── sd/              # Stable Diffusion train & generate presets
│   ├── vae/             # VAE training presets
│   ├── controlnet/      # ControlNet training presets
│   ├── auxiliary/       # Cluster recon, surprise predictor, count adapter presets
│   ├── analysis/        # Analysis script presets
│   └── models/          # Architecture JSONs (UNet, VAE)
├── scripts/             # Shell launch wrappers & standalone Python utilities
│   ├── train/           # Training shell wrappers
│   ├── generate/        # Generation shell wrappers
│   ├── analyze/         # Analysis shell wrappers
│   ├── standalone/      # Standalone Python entry-point scripts
│   └── checks/          # Repo health check scripts
├── data/                # Datasets & caches (gitignored heavy contents)
│   ├── raw/             # Original / unprocessed datasets (v18, …)
│   ├── derived/         # Preprocessed / feature datasets
│   └── cache/           # Ephemeral caches (DINO features, …)
├── artifacts/           # All training outputs (gitignored heavy contents)
│   ├── checkpoints/     # Model checkpoints (flow_matching, vae, stable_diffusion, …)
│   ├── runs/            # TensorBoard / experiment logs
│   ├── generated/       # Synthetic image datasets
│   ├── analysis/        # Distribution-shift analysis results
│   └── debug/           # Debug sample images
├── archive/             # Inactive / retired material
│   └── legacy_code/     # Old fm_src & sd_src trees
├── docs/                # Documentation, notebooks, prompt files
│   ├── notebooks/       # Jupyter analysis notebooks
│   ├── launcher_workflow.md  # Config-driven launcher architecture
│   └── repo_layout_target.md
├── frontend/            # Web frontends
│   └── flir-subgroup-analysis/  # React dashboard for subgroup analysis
├── tests/               # Test suite
├── README.md
├── pyproject.toml
└── .gitignore
```

Root-level Python scripts (`train_sfm.py`, `train_sd.py`, `train_vae.py`,
`train_controlnet.py`, `generate_datasets.py`, `serve_flir_analysis.py`) are
**thin wrappers** that delegate to the corresponding module inside `src/cli/`.

| Root wrapper           | Source of truth              | Purpose                                  |
| ---------------------- | ---------------------------- | ---------------------------------------- |
| `train_sfm.py`         | `src/cli/train.py`           | Flow-Matching training (YAML config)     |
| `train_sd.py`          | `src/cli/train_sd.py`        | Stable Diffusion 1.5 LoRA fine-tuning    |
| `train_vae.py`         | `src/cli/train_vae.py`       | VAE (KL) training                        |
| `train_controlnet.py`  | `src/cli/train_controlnet.py`| ControlNet training                      |
| `generate_datasets.py` | `src/cli/generate.py`        | Synthetic dataset generation (SD / FM)   |
| `serve_flir_analysis.py` | `src/cli/serve_flir_analysis.py` | FastAPI subgroup analysis service |

### Other CLI entry-points under `src/cli/`

| Module                | Purpose                          |
| --------------------- | -------------------------------- |
| `src/cli/sample.py`   | FM sampling / inference          |

### Standalone scripts under `scripts/standalone/`

These files are the source of truth for standalone experiment tools. Older
root-level invocation paths are kept as thin compatibility wrappers where used.

| Script                                             | Purpose                                          |
| -------------------------------------------------- | ------------------------------------------------ |
| `scripts/standalone/train_fm.py`                   | Pixel-space flow matching training (no VAE)      |
| `scripts/standalone/train_count_adapter.py`        | Count-conditioned DINO adapter training          |
| `scripts/standalone/train_cluster_reconstruction.py`| Masked cluster reconstruction training          |
| `scripts/standalone/train_surprise_predictor.py`   | Surprise/GMM multi-task predictor training       |
| `scripts/standalone/build_surprise_pred_dataset.py`| Build surprise prediction dataset                |
| `scripts/standalone/analyze_distribution_shift.py` | Distribution shift analysis (FID, MMD, etc.)     |
| `scripts/standalone/analyze_fm_subsampling_coverage.py` | Coverage/diversity analysis for FM samples  |
| `scripts/standalone/sample_guided_fm.py`           | Guided FM sampling script                        |

## Quick start

```bash
# Train flow-matching model (config-driven)
python train_sfm.py --config configs/fm/train/default.yaml

# Train SD 1.5 LoRA
python train_sd.py  # edit src/cli/train_sd.py defaults or pass flags

# Train VAE
python train_vae.py --train-dir ./data/raw/v18/train --val-dir ./data/raw/v18/val

# Generate FM samples
python generate_datasets.py --mode fm \
  --fm_pipeline_dir ./artifacts/checkpoints/flow_matching/serious_runs/stable_training_t_scaled \
  --max_samples 100

# Generate SD 1.5 LoRA samples
python generate_datasets.py --mode sd15 \
  --lora_dir ./artifacts/checkpoints/stable_diffusion/lora_runs/.../checkpoint-32000
```

## Multi-dataset subgroup analysis web app

The notebook logic from `docs/notebooks/v18_scene_graph_score_analysis_flir.py`
now lives in reusable modules under `src/analysis/flir_subgroup/`.

### Backend

Install the web dependencies in your Python environment:

```bash
python -m pip install -e .[web]
```

Run the FastAPI API:

```bash
python serve_flir_analysis.py --host 127.0.0.1 --port 8000
```

Main routes:

- `GET /api/flir-analysis/datasets`
- `GET /api/flir-analysis/options?dataset=<dataset_id>`
- `POST /api/flir-analysis/holdout-curves`
- `POST /api/flir-analysis/collateral`
- `POST /api/flir-analysis/partition-comparisons`
- `POST /api/flir-analysis/examples`
- `GET /api/flir-analysis/images/{dataset_id}/{image_key}`

The web app now supports both discovered dataset roots:

- `flir_private_proxy_alignment_v18`
- `v18`

All POST analysis payloads require a `dataset` field, and the frontend lets you
switch datasets directly from the control panel. The dedicated bin-explainer
section and all image/chart helpers are dataset-specific.

### Frontend

```bash
cd frontend/flir-subgroup-analysis
npm install
npm run dev
```

The frontend defaults to `http://127.0.0.1:8000`. Override the backend if
needed:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Build the production bundle with:

```bash
npm run build
```

## Shell scripts

Shell wrappers live in `scripts/` organized by category.  Each is a thin
wrapper that pairs a Python CLI with a YAML config preset:

```bash
# Training
bash scripts/train/fm_stable.sh          # FM stable latent training
bash scripts/train/sd_lora_r4.sh         # SD LoRA rank-4 training
bash scripts/train/vae_4x.sh             # VAE 4x compression training

# Generation
bash scripts/generate/fm_plain.sh        # FM plain Euler generation
bash scripts/generate/fm_guided_combo_maxmin.sh  # FM guided combo generation
bash scripts/generate/sd_r4.sh           # SD LoRA generation

# Analysis
bash scripts/analyze/distribution_shift.sh
bash scripts/analyze/fm_subsampling.sh

# Override a parameter from the command line
bash scripts/train/fm_stable.sh --training.epochs 50
```

See [docs/launcher_workflow.md](docs/launcher_workflow.md) for the full
config-driven launcher architecture.

## Verification

Run all check scripts to validate the modular architecture:

```bash
for f in scripts/checks/check_*.py; do
  echo "=== $f ==="
  conda run -n diffusers-dev python "$f"
done
```

Targeted validation for the subgroup analysis app:

```bash
conda run -n diffusers-dev python -m pytest tests/test_flir_subgroup_analysis.py -v
conda run -n diffusers-dev python -m pytest tests/test_flir_subgroup_notebook_parity.py -v
cd frontend/flir-subgroup-analysis && npm run build
```

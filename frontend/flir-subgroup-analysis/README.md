# FLIR Analysis Frontend

This Vite + React app visualizes the FLIR subgroup analysis API and the
checkpoint-selection result viewer.

## Backend

Install the Python web extras from the repository root:

```bash
python -m pip install -e .[web]
```

Run the FastAPI service:

```bash
python serve_flir_analysis.py --host 127.0.0.1 --port 8000
```

## Run

```bash
cd frontend/flir-subgroup-analysis
npm install
npm run dev
```

By default the app talks to `http://127.0.0.1:8000`. Override that with:

```bash
VITE_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

Use the top-level view switch to open **Checkpoint Selection**. Enter a ROOT
folder such as:

```text
artifacts/generated/checkpoint_selection
/scratch/bacobax2/killarney_scratch
```

The backend scans that ROOT read-only, detects direct run folders and
`ROOT/subroot/run` layouts, and serves only preview image files below the
selected ROOT.

## Build

```bash
npm run build
```

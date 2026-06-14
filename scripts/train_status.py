#!/usr/bin/env python3
"""Track LoRA training runs: squeue + log parsing + on-disk checkpoint state.

Covers all LoRA training output dirs:
  artifacts/checkpoints/stable_diffusion/lora_runs/*      (SD1.5)
  artifacts/checkpoints/stable_diffusion_xl/lora_runs/*   (SDXL)

For SDXL runs the corresponding killarney slurm script is discovered automatically,
enabling --relaunch. SD1.5 runs (all done) are reported from disk only.

States:
  RUNNING      in squeue state R  (+ tqdm step progress + time left)
  PENDING      in squeue state PD (+ scheduler reason)
  DONE         pytorch_lora_weights.safetensors exists at output root, or
               latest .out log contains "End time:"
  INTERRUPTED  latest .err has "DUE TO TIME LIMIT" (timeout) or a Python
               traceback (crash + exception class) — and not in squeue
  INCOMPLETE   a checkpoint-<N>/ dir exists but no done marker and not in squeue
  NO RUN       output dir exists but no checkpoints and no logs yet

Usage:
  python scripts/train_status.py                  # print status table
  python scripts/train_status.py --json           # machine-readable
  python scripts/train_status.py --relaunch       # resubmit interrupted/incomplete
  python scripts/train_status.py --relaunch --dry-run
  python scripts/train_status.py --relaunch --include-errors  # also crashes
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

# ---------------------------------------------------------------------------
# Shared helpers from eval_status (same scripts/ dir)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.realpath(__file__)))
from eval_status import (  # noqa: E402
    PROJECT_ROOT, LOGS,
    read, tail, count_files,
    squeue_map, latest_log,
)

# ---------------------------------------------------------------------------
# Regexes
# ---------------------------------------------------------------------------
JOBNAME_RE = re.compile(r"#SBATCH\s+--job-name=(\S+)")
OUTPUT_REL_RE = re.compile(r'^OUTPUT_REL="([^"]+)"', re.M)
# tqdm: "Steps:  96%|███| 256543/268600 [89:27:23<4:14:04, 1.26s/it, ...]"
STEP_RE = re.compile(
    r"Steps:\s+\d+%\|[^|]*\|\s*(\d+)/(\d+)\s+\[([^<]*)<([0-9:]+)"
)
EXC_RE = re.compile(r"^(\w+(?:Error|Exception)):", re.M)

# ---------------------------------------------------------------------------
# Cluster detection (mirrors resume_interrupted_sdxl_lora_runs.sh)
# ---------------------------------------------------------------------------

def detect_cluster() -> str:
    h = os.uname().nodename.lower()
    if "fir" in h:
        return "fir"
    return "killarney"


def cluster_suffix(cluster: str) -> str:
    return "_kl" if cluster == "killarney" else "_fir"


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def _size_from_name(name: str) -> str:
    """Derive size label from output dir basename."""
    if re.search(r"_train_(2000|2040)$", name):
        return "2k"
    if re.search(r"_train_(5000|5100)$", name):
        return "5k"
    return "full"


def _dataset_from_name(name: str) -> str:
    if "flir" in name:
        return "flir"
    if "bigearthnet" in name:
        return "bigearthnet"
    return name


def _family_from_path(abs_path: str) -> str:
    if "stable_diffusion_xl" in abs_path:
        return "sdxl"
    return "sd15"


def build_slurm_map(cluster: str) -> dict[str, dict]:
    """Return {output_rel: {job_name, worker}} for slurm-backed runs."""
    suf = cluster_suffix(cluster)
    pattern = os.path.join(
        PROJECT_ROOT, "slurm", cluster, "*", "sd_adaptation",
        f"train_*lora*{suf}.slurm",
    )
    m: dict[str, dict] = {}
    for path in glob.glob(pattern):
        text = read(path)
        jn = JOBNAME_RE.search(text)
        out = OUTPUT_REL_RE.search(text)
        if not jn or not out:
            continue
        output_rel = out.group(1).strip()
        m[output_rel] = {
            "job_name": jn.group(1),
            "worker": path,
        }
    return m


def parse_registry(cluster: str) -> list[dict]:
    """Discover all LoRA output dirs + enrich with slurm metadata."""
    slurm_map = build_slurm_map(cluster)

    seen_output_rels: set[str] = set()
    runs: list[dict] = []

    # 1. Disk backbone
    for pattern in (
        os.path.join(PROJECT_ROOT, "artifacts/checkpoints/stable_diffusion/lora_runs/*"),
        os.path.join(PROJECT_ROOT, "artifacts/checkpoints/stable_diffusion_xl/lora_runs/*"),
    ):
        for out_abs in sorted(glob.glob(pattern)):
            if not os.path.isdir(out_abs):
                continue
            # Compute output_rel (relative to PROJECT_ROOT)
            out_rel = os.path.relpath(out_abs, PROJECT_ROOT)
            seen_output_rels.add(out_rel)
            name = os.path.basename(out_abs)
            sm = slurm_map.get(out_rel, {})
            runs.append({
                "name": sm.get("job_name") or name,
                "display_name": sm.get("job_name") or name,
                "output_rel": out_rel,
                "output_abs": out_abs,
                "job_name": sm.get("job_name"),    # None for SD1.5
                "worker": sm.get("worker"),
                "family": _family_from_path(out_abs),
                "dataset": _dataset_from_name(name),
                "size": _size_from_name(name),
            })

    # 2. Slurm-declared runs whose output dir doesn't exist yet
    for out_rel, sm in slurm_map.items():
        if out_rel in seen_output_rels:
            continue
        out_abs = os.path.join(PROJECT_ROOT, out_rel)
        name = os.path.basename(out_rel)
        runs.append({
            "name": sm["job_name"],
            "display_name": sm["job_name"],
            "output_rel": out_rel,
            "output_abs": out_abs,
            "job_name": sm["job_name"],
            "worker": sm["worker"],
            "family": _family_from_path(out_abs),
            "dataset": _dataset_from_name(name),
            "size": _size_from_name(name),
        })

    return runs


# ---------------------------------------------------------------------------
# Progress / disk state helpers
# ---------------------------------------------------------------------------

def train_progress(job_name: str) -> tuple[str | None, str | None]:
    """Return (progress_str, eta_str) parsed from latest .err tqdm bar, or (None, None)."""
    err, _ = latest_log(job_name, "err")
    if not err:
        return None, None
    matches = STEP_RE.findall(tail(err))
    if not matches:
        return None, None
    cur, total, elapsed, eta = matches[-1]
    pct = int(100 * int(cur) / int(total)) if int(total) else 0
    return f"step {cur}/{total} ({pct}%)", eta


def latest_checkpoint_step(out_abs: str) -> int | None:
    """Return highest N from checkpoint-<N>/ dirs under out_abs, or None."""
    best: int | None = None
    for entry in os.scandir(out_abs) if os.path.isdir(out_abs) else []:
        m = re.match(r"checkpoint-(\d+)$", entry.name)
        if m and entry.is_dir():
            n = int(m.group(1))
            if best is None or n > best:
                best = n
    return best


def disk_detail(run: dict) -> str | None:
    """Short on-disk summary: checkpoint step and/or weights-saved."""
    out = run["output_abs"]
    if not os.path.isdir(out):
        return None
    parts = []
    ckpt = latest_checkpoint_step(out)
    if ckpt is not None:
        parts.append(f"ckpt-{ckpt}")
    if os.path.exists(os.path.join(out, "pytorch_lora_weights.safetensors")):
        parts.append("weights saved")
    return " | ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

ORDER = {"RUNNING": 0, "PENDING": 1, "INTERRUPTED": 2,
         "INCOMPLETE": 3, "NO RUN": 4, "DONE": 5}


def classify(run: dict, sq: dict) -> dict:
    info: dict = {"state": "NO RUN", "detail": "", "jobid": None, "kind": None}
    out = run["output_abs"]
    jn = run["job_name"]

    # 1. squeue state (only if job_name is known)
    if jn and jn in sq:
        q = sq[jn]
        info["jobid"] = q["jobid"]
        if q["state"] == "R":
            info["state"] = "RUNNING"
            prog, eta = train_progress(jn)
            detail = prog or "started"
            if eta:
                detail += f" | ETA {eta}"
            detail += f" | {q['timeleft']} left"
            info["detail"] = detail
        elif q["state"] == "PD":
            info["state"] = "PENDING"
            info["detail"] = q["reason"].strip("()")
        else:
            info["state"] = q["state"]
        return info

    # 2. Done marker (disk first, then log sentinel)
    weights = os.path.join(out, "pytorch_lora_weights.safetensors")
    if os.path.exists(weights):
        info["state"] = "DONE"
        return info

    out_p, out_id = latest_log(jn, "out") if jn else (None, -1)
    if out_p and "End time:" in tail(out_p):
        info["state"] = "DONE"
        info["jobid"] = str(out_id) if out_id >= 0 else None
        return info

    if not jn:
        # SD1.5 without a job_name: check on disk only
        if os.path.isdir(out) and latest_checkpoint_step(out) is not None:
            info["state"] = "INCOMPLETE"
        return info

    # 3. Log analysis (only for runs with known job_name)
    err_p, err_id = latest_log(jn, "err")
    err_txt = tail(err_p) if err_p else ""
    info["jobid"] = str(out_id if out_id >= 0 else err_id) if (out_id >= 0 or err_id >= 0) else None

    if "DUE TO TIME LIMIT" in err_txt:
        info["state"] = "INTERRUPTED"
        info["detail"] = "timeout"
        info["kind"] = "timeout"
    elif "Traceback (most recent call last)" in err_txt:
        info["state"] = "INTERRUPTED"
        exc = EXC_RE.findall(err_txt)
        info["detail"] = f"crash: {exc[-1]}" if exc else "crash"
        info["kind"] = "error"
    elif os.path.isdir(out) and latest_checkpoint_step(out) is not None:
        info["state"] = "INCOMPLETE"
        ckpt = latest_checkpoint_step(out)
        info["detail"] = f"last ckpt-{ckpt}, no done marker"
    elif not os.path.isdir(out):
        info["state"] = "NO RUN"
    else:
        info["state"] = "NO RUN"

    return info


# ---------------------------------------------------------------------------
# Relaunch
# ---------------------------------------------------------------------------

def do_relaunch(run: dict, dry_run: bool) -> None:
    worker = run.get("worker")
    if not worker:
        print(f"  {run['display_name']}: no worker slurm — relaunch manually")
        return
    cmd = [
        "sbatch",
        f"--chdir={PROJECT_ROOT}",
        f"--export=ALL,PROJECT_ROOT={PROJECT_ROOT},RESUME_FROM_CHECKPOINT=latest",
        worker,
    ]
    print(f"  $ {' '.join(cmd)}")
    if dry_run:
        return
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    print("   " + (r.stdout or r.stderr).strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--relaunch", action="store_true",
                    help="resubmit interrupted/incomplete runs")
    ap.add_argument("--include-errors", action="store_true",
                    help="with --relaunch, also resubmit crashed runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --relaunch, print sbatch commands only")
    ap.add_argument("--cluster", choices=["killarney", "fir"],
                    help="override cluster auto-detection")
    args = ap.parse_args()

    cluster = args.cluster or detect_cluster()
    runs = parse_registry(cluster)
    sq = squeue_map()

    for r in runs:
        r.update(classify(r, sq))
        r["disk"] = disk_detail(r)

    if args.json:
        print(json.dumps(runs, indent=2))
        return 0

    counts: dict[str, int] = {}
    # Group: family → dataset
    for family in ("sdxl", "sd15"):
        for dataset in ("bigearthnet", "flir"):
            group = [r for r in runs
                     if r["family"] == family and r["dataset"] == dataset]
            if not group:
                continue
            print(f"\n=== {family} / {dataset} ===")
            for r in sorted(group, key=lambda x: (ORDER.get(x["state"], 9), x["size"])):
                counts[r["state"]] = counts.get(r["state"], 0) + 1
                jid = r.get("jobid") or "-"
                print(f"  {r['display_name']:<24} {r['state']:<12} "
                      f"{jid:>8}  {r.get('detail', '')}")
                if r["disk"]:
                    print(f"      {r['disk']}")

    print("\nsummary: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if args.relaunch:
        targets = [
            r for r in runs
            if r["state"] in ("INTERRUPTED", "INCOMPLETE")
            and (r.get("kind") != "error" or args.include_errors)
            and r.get("job_name") not in sq   # never double-submit a live job
        ]
        if not targets:
            print("\nnothing to relaunch.")
            return 0
        print(f"\nrelaunching {len(targets)} run(s)"
              f"{' [dry-run]' if args.dry_run else ''}:")
        for r in targets:
            print(f"- {r['display_name']} ({r.get('detail', r['state'])})")
            do_relaunch(r, args.dry_run)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Track publication checkpoint-selection eval runs: squeue + log parsing.

Source of truth for "what was launched" is the submit wrapper
(slurm/killarney/submit_publication_checkpoint_selection_single_runs_kl.sh):
every uncommented `sbatch ... <file>.slurm` line is one run.

For each run we resolve job-name + config (-> run_identifier, output_root) and
classify its state:

  RUNNING      in squeue, state R (+ live progress parsed from the .err tqdm bar)
  PENDING      in squeue, state PD (+ scheduler reason)
  DONE         latest .out contains "End time:"  (the slurm script's success
               sentinel: it only prints after python exits 0 under set -e)
  INTERRUPTED  latest .err has "DUE TO TIME LIMIT" (timeout) or a Python
               traceback (crash, + the exception class)
  NO LOG       launched name has no matching log yet / never ran

Usage:
  python scripts/eval_status.py                 # print status table
  python scripts/eval_status.py --json          # machine-readable
  python scripts/eval_status.py --relaunch      # resubmit timed-out runs
  python scripts/eval_status.py --relaunch --include-errors   # also crashes
  python scripts/eval_status.py --relaunch --dry-run          # print only
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import subprocess
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
WRAPPER = os.path.join(
    PROJECT_ROOT,
    "slurm/killarney/submit_publication_checkpoint_selection_single_runs_kl.sh",
)
LOGS = os.path.join(PROJECT_ROOT, "logs")

JOBNAME_RE = re.compile(r"#SBATCH\s+--job-name=(\S+)")
CONFIG_RE = re.compile(r'CONFIG_REL="\$\{CONFIG_REL:-([^"}]+)\}"')
RUNID_RE = re.compile(r'^\s*-?\s*run_identifier:\s*"?([^"\n]+?)"?\s*$', re.M)
OUTROOT_RE = re.compile(r'^\s*output_root:\s*"?([^"\n]+?)"?\s*$', re.M)
SELNUM_RE = re.compile(r'^\s*selection_num_images:\s*(\d+)', re.M)
FINALNUM_RE = re.compile(r'^\s*final_total_images:\s*(\d+)', re.M)
# tqdm: "Generating final:  53%|####  | 15873/30000 [..]"
PROG_RE = re.compile(r"([A-Za-z][^:|\n]*?):\s*\d+%\|[^|\n]*\|\s*(\d+)/(\d+)")
EXC_RE = re.compile(r"^(\w+(?:Error|Exception)):", re.M)


def read(path: str) -> str:
    try:
        with open(path, "r", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def tail(path: str, nbytes: int = 32768) -> str:
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > nbytes:
                f.seek(size - nbytes)
            data = f.read()
        return data.decode("utf-8", errors="replace").replace("\r", "\n")
    except OSError:
        return ""


def parse_registry() -> list[dict]:
    """Each uncommented sbatch line in the wrapper -> one run dict."""
    runs = []
    for line in read(WRAPPER).splitlines():
        s = line.strip()
        if s.startswith("#") or "sbatch" not in s:
            continue
        m = re.search(r"(\S+\.slurm)", s)
        if not m:
            continue
        slurm_rel = m.group(1)
        slurm_abs = os.path.join(PROJECT_ROOT, slurm_rel)
        text = read(slurm_abs)
        jn = JOBNAME_RE.search(text)
        cfg = CONFIG_RE.search(text)
        job_name = jn.group(1) if jn else os.path.basename(slurm_rel)
        cfg_text = read(os.path.join(PROJECT_ROOT, cfg.group(1))) if cfg else ""
        rid = RUNID_RE.search(cfg_text)
        out = OUTROOT_RE.search(cfg_text)
        runs.append(
            {
                "job_name": job_name,
                "slurm_rel": slurm_rel,
                "run_identifier": rid.group(1) if rid else None,
                "output_root": out.group(1) if out else None,
                "sel_target": int(sn.group(1)) if (sn := SELNUM_RE.search(cfg_text)) else 10000,
                "final_target": int(fn.group(1)) if (fn := FINALNUM_RE.search(cfg_text)) else 30000,
                "dataset": "flir" if "/flir/" in slurm_rel else "bigearthnet",
            }
        )
    return runs


def squeue_map() -> dict:
    """job_name -> {jobid, state, reason, timeleft} for current user."""
    try:
        out = subprocess.run(
            ["squeue", "-u", os.environ.get("USER", ""), "-h",
             "-o", "%i|%j|%t|%R|%L"],
            capture_output=True, text=True, timeout=30,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    m = {}
    for line in out.splitlines():
        parts = line.split("|")
        if len(parts) < 5:
            continue
        jid, name, st, reason, left = parts[:5]
        # keep the highest jobid if a name appears twice
        if name not in m or int(jid) > int(m[name]["jobid"]):
            m[name] = {"jobid": jid, "state": st,
                       "reason": reason, "timeleft": left}
    return m


def latest_log(job_name: str, ext: str) -> tuple[str | None, int]:
    """Return (path, jobid) of the highest-jobid <job_name>-<jobid>.<ext>."""
    best, best_id = None, -1
    for p in glob.glob(os.path.join(LOGS, f"{job_name}-*.{ext}")):
        m = re.search(rf"{re.escape(job_name)}-(\d+)\.{ext}$", p)
        if m and int(m.group(1)) > best_id:
            best, best_id = p, int(m.group(1))
    return best, best_id


def progress(job_name: str) -> str | None:
    err, _ = latest_log(job_name, "err")
    if not err:
        return None
    matches = PROG_RE.findall(tail(err))
    if not matches:
        return None
    desc, cur, tot = matches[-1]
    pct = 100 * int(cur) / int(tot) if int(tot) else 0
    return f"{desc.strip()} {cur}/{tot} ({pct:.0f}%)"


def count_files(d: str) -> int:
    """Count entries in directory d without allocating a list."""
    try:
        return sum(1 for _ in os.scandir(d))
    except OSError:
        return 0


def disk_status(run: dict) -> dict | None:
    """Read on-disk progress for *run* from its output folder.

    Returns None if the run has no usable output directory.
    Otherwise returns a dict with keys:
      total, evaluated, phase (None|"selecting"|"final"), ckpt, gen, target
    """
    rid = run.get("run_identifier")
    root = run.get("output_root")
    if not rid or not root:
        return None
    run_dir = os.path.join(root, rid)
    if not os.path.isdir(run_dir):
        return None

    disc = os.path.join(run_dir, "checkpoint_discovery.json")
    if not os.path.exists(disc):
        return {"note": "discovery pending", "total": None, "evaluated": None,
                "phase": None, "ckpt": None, "gen": None, "target": None}

    try:
        with open(disc) as f:
            candidates = [c["checkpoint_identifier"]
                          for c in json.load(f).get("candidates", [])]
    except (OSError, ValueError, KeyError):
        return {"note": "discovery unreadable", "total": None, "evaluated": None,
                "phase": None, "ckpt": None, "gen": None, "target": None}

    total = len(candidates)
    evaluated = sum(
        os.path.exists(os.path.join(run_dir, c, "selection", "selection_metrics.json"))
        for c in candidates
    )

    # -- Phase detection --
    # 1. Final phase: any candidate has a non-empty final/generated_npy_images dir
    winner, gen_count = None, 0
    for c in candidates:
        final_npy = os.path.join(run_dir, c, "final", "generated_npy_images")
        if os.path.isdir(final_npy):
            n = count_files(final_npy)
            if n > 0 or winner is None:
                # prefer the one that actually has images (in case multiple exist)
                if n > gen_count:
                    winner, gen_count = c, n
    if winner is not None:
        return {
            "total": total, "evaluated": evaluated,
            "phase": "final", "ckpt": winner,
            "gen": gen_count, "target": run["final_target"],
            "note": None,
        }

    # 2. Selection phase: candidate with npy images but no metrics yet
    for c in candidates:
        metrics = os.path.join(run_dir, c, "selection", "selection_metrics.json")
        sel_npy = os.path.join(run_dir, c, "selection", "generated_npy_images")
        if not os.path.exists(metrics) and os.path.isdir(sel_npy):
            n = count_files(sel_npy)
            if n > 0:
                return {
                    "total": total, "evaluated": evaluated,
                    "phase": "selecting", "ckpt": c,
                    "gen": n, "target": run["sel_target"],
                    "note": None,
                }

    # 3. No active generation detected (between phases or fully done)
    return {
        "total": total, "evaluated": evaluated,
        "phase": None, "ckpt": None, "gen": None, "target": None,
        "note": None,
    }


def disk_summary_line(ds: dict) -> str:
    """One-line summary of on-disk checkpoint/generation progress."""
    if ds.get("note"):
        return ds["note"]
    total, evaluated = ds["total"], ds["evaluated"]
    base = f"ckpts {evaluated}/{total} eval"
    phase, ckpt, gen, target = ds["phase"], ds["ckpt"], ds["gen"], ds["target"]
    if phase == "final":
        pct = int(100 * gen / target) if target else 0
        return f"{base} | final {ckpt} {gen}/{target} ({pct}%)"
    if phase == "selecting":
        pct = int(100 * gen / target) if target else 0
        return f"{base} | selecting {ckpt} {gen}/{target} ({pct}%)"
    if evaluated == total:
        return f"{base} | all done"
    return base


def classify(run: dict, sq: dict) -> dict:
    name = run["job_name"]
    info = {"detail": "", "jobid": None}
    q = sq.get(name)
    if q:
        info["jobid"] = q["jobid"]
        if q["state"] == "R":
            info["state"] = "RUNNING"
            p = progress(name)
            info["detail"] = (p or "started") + f" | {q['timeleft']} left"
        elif q["state"] == "PD":
            info["state"] = "PENDING"
            info["detail"] = q["reason"].strip("()")
        else:
            info["state"] = q["state"]
        return info

    out_p, out_id = latest_log(name, "out")
    err_p, _ = latest_log(name, "err")
    if not out_p and not err_p:
        info["state"] = "NO LOG"
        return info
    info["jobid"] = str(out_id) if out_id >= 0 else None
    err_txt = tail(err_p) if err_p else ""
    if out_p and "End time:" in tail(out_p):
        info["state"] = "DONE"
    elif "DUE TO TIME LIMIT" in err_txt:
        info["state"] = "INTERRUPTED"
        info["detail"] = "timeout"
        info["kind"] = "timeout"
    elif "Traceback (most recent call last)" in err_txt:
        info["state"] = "INTERRUPTED"
        exc = EXC_RE.findall(err_txt)
        info["detail"] = f"crash: {exc[-1]}" if exc else "crash"
        info["kind"] = "error"
    else:
        info["state"] = "UNKNOWN"
        info["detail"] = "no End time / no cancel marker (node failure?)"
    return info


def relaunch(run: dict) -> None:
    cmd = ["sbatch", f"--chdir={PROJECT_ROOT}",
           os.path.join(PROJECT_ROOT, run["slurm_rel"])]
    print(f"  $ {' '.join(cmd)}")
    if DRY_RUN:
        return
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)
    print("   " + (r.stdout or r.stderr).strip())


ORDER = {"RUNNING": 0, "PENDING": 1, "INTERRUPTED": 2, "UNKNOWN": 3,
         "NO LOG": 4, "DONE": 5}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--relaunch", action="store_true",
                    help="resubmit timed-out runs via sbatch")
    ap.add_argument("--include-errors", action="store_true",
                    help="with --relaunch, also resubmit crashed runs")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --relaunch, print sbatch commands only")
    args = ap.parse_args()
    global DRY_RUN
    DRY_RUN = args.dry_run

    runs = parse_registry()
    sq = squeue_map()
    for r in runs:
        r.update(classify(r, sq))
        r["disk"] = disk_status(r)

    if args.json:
        print(json.dumps(runs, indent=2))
        return 0

    counts: dict[str, int] = {}
    for ds in ("bigearthnet", "flir"):
        group = [r for r in runs if r["dataset"] == ds]
        if not group:
            continue
        print(f"\n=== {ds} ===")
        for r in sorted(group, key=lambda x: (ORDER.get(x["state"], 9),
                                              x["job_name"])):
            counts[r["state"]] = counts.get(r["state"], 0) + 1
            jid = r["jobid"] or "-"
            print(f"  {r['job_name']:<22} {r['state']:<12} "
                  f"{jid:>8}  {r['detail']}")
            if r["disk"] is not None:
                print(f"      {disk_summary_line(r['disk'])}")

    print("\nsummary: " +
          ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    if args.relaunch:
        targets = [r for r in runs if r["state"] == "INTERRUPTED" and
                   (r.get("kind") == "timeout" or args.include_errors)]
        if not targets:
            print("\nnothing to relaunch.")
            return 0
        print(f"\nrelaunching {len(targets)} interrupted run(s)"
              f"{' [dry-run]' if DRY_RUN else ''}:")
        for r in targets:
            print(f"- {r['job_name']} ({r['detail']})")
            relaunch(r)
    return 0


if __name__ == "__main__":
    sys.exit(main())

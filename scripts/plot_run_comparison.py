"""
Compare 4 minmax training runs:
  1. 640M overfit (no augmentation)
  2. 3L overfit (no augmentation)
  3. 3L + regularization (aug + dropout)
  4. 640M + regularization (aug + dropout)  [current run]
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path

BASE = Path(".")
RUNS_DIR = BASE / "artifacts/runs/main/flow_matching"
CKPT_DIR = BASE / "artifacts/checkpoints/flow_matching/serious_runs"

RUNS = {
    "640M overfit":    "stay_layout_latent_v18_sd15ft_x8_256_minmax",
    "3L overfit":      "stay_layout_latent_v18_sd15ft_x8_256_minmax_3L",
    "3L + reg":        "stay_layout_latent_v18_sd15ft_x8_256_minmax_3L_reg",
    "640M + reg":      "stay_layout_latent_v18_sd15ft_x8_256_minmax_reg",
}

COLORS = {
    "640M overfit": "#e74c3c",
    "3L overfit":   "#e67e22",
    "3L + reg":     "#2ecc71",
    "640M + reg":   "#3498db",
}

STYLES = {
    "640M overfit": "--",
    "3L overfit":   "--",
    "3L + reg":     "-",
    "640M + reg":   "-",
}


def load_metrics(run_key):
    name = RUNS[run_key]
    path = RUNS_DIR / name / "metrics_epoch.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


def load_ckpt_eval(run_key):
    name = RUNS[run_key]
    path = CKPT_DIR / name / "checkpoint_eval" / "checkpoint_eval_metrics.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df


fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("LayoutFM minmax runs — training comparison", fontsize=14, fontweight="bold", y=0.98)

ax_train   = axes[0, 0]
ax_eval    = axes[0, 1]
ax_fid     = axes[1, 0]
ax_kid     = axes[1, 1]
ax_mmd     = axes[1, 2]
ax_unused  = axes[0, 2]

# ── legend / info panel ───────────────────────────────────────────────────────
ax_unused.axis("off")
legend_text = [
    ("640M overfit", "640M UNet, no aug/dropout\n(640M params, U-curve ~ep72)"),
    ("3L overfit",   "3-level UNet (~59M), no aug/dropout\n(also overfits, similar U-curve)"),
    ("3L + reg",     "3L + bbox-aug + dropout 0.1\nResumed→ep400, best FID=289.9@ep360"),
    ("640M + reg",   "640M + bbox-aug + dropout 0.1\nStopped ep524, best FID=218.3@ep330"),
]
y = 0.92
for key, desc in legend_text:
    ax_unused.plot([], [], color=COLORS[key], ls=STYLES[key], lw=2.5, label=key)
    ax_unused.text(0.02, y, f"  {key}", color=COLORS[key], fontsize=10,
                   fontweight="bold", transform=ax_unused.transAxes, va="top")
    ax_unused.text(0.02, y - 0.06, f"  {desc}", color="#555555", fontsize=8,
                   transform=ax_unused.transAxes, va="top")
    y -= 0.22

# ── train fm-loss ─────────────────────────────────────────────────────────────
for key in RUNS:
    df = load_metrics(key)
    if df is None:
        continue
    ax_train.plot(df["epoch"], df["train_fm_loss"], color=COLORS[key],
                  ls=STYLES[key], lw=1.5, alpha=0.9, label=key)

ax_train.set_title("Train FM-loss", fontweight="bold")
ax_train.set_xlabel("Epoch")
ax_train.set_ylabel("FM loss")
ax_train.legend(fontsize=8)
ax_train.grid(True, alpha=0.3)
ax_train.set_ylim(bottom=0)

# ── eval fm-loss ──────────────────────────────────────────────────────────────
for key in RUNS:
    df = load_metrics(key)
    if df is None:
        continue
    eval_df = df[df["eval_fm_loss"].notna()].copy()
    ax_eval.plot(eval_df["epoch"], eval_df["eval_fm_loss"], color=COLORS[key],
                 ls=STYLES[key], lw=1.8, alpha=0.9, label=key)
    # mark best eval point
    best_row = eval_df.loc[eval_df["eval_fm_loss"].idxmin()]
    ax_eval.scatter(best_row["epoch"], best_row["eval_fm_loss"],
                    color=COLORS[key], s=60, zorder=5, marker="*")

ax_eval.set_title("Eval FM-loss  (★ = best)", fontweight="bold")
ax_eval.set_xlabel("Epoch")
ax_eval.set_ylabel("FM loss")
ax_eval.legend(fontsize=8)
ax_eval.grid(True, alpha=0.3)
ax_eval.set_ylim(bottom=0)

# ── generative metrics (only reg runs have checkpoint_eval) ───────────────────
CKPT_RUNS = ["3L + reg", "640M + reg"]

for key in CKPT_RUNS:
    df = load_ckpt_eval(key)
    if df is None:
        continue
    ax_fid.plot(df["epoch"], df["FID"], color=COLORS[key], ls=STYLES[key],
                lw=2, marker="o", ms=5, label=key)
    ax_kid.plot(df["epoch"], df["KID"], color=COLORS[key], ls=STYLES[key],
                lw=2, marker="o", ms=5, label=key)
    ax_mmd.plot(df["epoch"], df["MMD"], color=COLORS[key], ls=STYLES[key],
                lw=2, marker="o", ms=5, label=key)

for ax, title, ylabel in [
    (ax_fid, "FID (DINOv2) ↓", "FID"),
    (ax_kid, "KID (DINOv2) ↓", "KID"),
    (ax_mmd, "MMD (DINOv2) ↓", "MMD"),
]:
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

plt.tight_layout(rect=[0, 0, 1, 0.97])
out = BASE / "artifacts/plots/run_comparison_minmax.png"
out.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

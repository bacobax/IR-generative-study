import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import importlib
    import math
    import random
    import sys
    from pathlib import Path
    from typing import Any, Dict, List, Optional

    import numpy as np
    import pandas as pd
    import torch
    from PIL import Image, ImageDraw

    def _resolve_repo_root() -> Path:
        start = Path.cwd().resolve()
        for candidate in [start, *start.parents]:
            if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
                return candidate
        here = Path(__file__).resolve()
        for candidate in [here, *here.parents]:
            if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
                return candidate
        raise RuntimeError("Could not locate repository root.")

    REPO_ROOT = _resolve_repo_root()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    def _optional_import(name: str):
        try:
            return importlib.import_module(name), None
        except Exception as exc:
            return None, str(exc)

    plotly_express, _px_err = _optional_import("plotly.express")
    plotly_go, _go_err     = _optional_import("plotly.graph_objects")
    ultralytics_mod, _     = _optional_import("ultralytics")

    from src.algorithms.inference.rare_layout_dataset_tools import (
        load_sampler_from_pipeline,
        sample_layout_batch,
    )
    from src.analysis.flir_subgroup.yolo_slice_stats import (
        POSITION_BIN_ORDER,
        SIZE_BIN_ORDER,
        build_slice_counts_table,
        load_yolo_slice_dataset,
    )
    from src.core.data.layout_batching import collate_layout_batch
    from src.core.visualization.layout_debug import (
        draw_bbox_overlays,
        ensure_rgb,
        render_class_layout,
    )

    _RUN = "stay_layout_latent_v18_sd15ft_x8_256_minmax_reg_v2"
    FM_CHECKPOINT = (
        REPO_ROOT / "artifacts/checkpoints/flow_matching/serious_runs"
        / _RUN / "UNET/unet_fm_epoch_570.pt"
    )
    FM_PRESET     = (
        REPO_ROOT / "artifacts/checkpoints/flow_matching/serious_runs"
        / _RUN / "effective_config.yaml"
    )
    PIPELINE_DIR  = FM_CHECKPOINT.parent.parent
    YOLO_WEIGHTS  = REPO_ROOT / "artifacts/checkpoints/yolo/exp_v18_scratch_yolo11n/default_aug/best.pt"
    DATASET_YAML  = REPO_ROOT / "data/derived/yolo-test-ds_v18/full_train.yaml"

    # ── display helpers ────────────────────────────────────────────────────────

    def tensor_to_pil(t: torch.Tensor) -> Image.Image:
        """(C, H, W) float [0, 1] → RGB PIL Image."""
        arr = ensure_rgb(t.unsqueeze(0))[0].permute(1, 2, 0).numpy()
        return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))

    def generated_to_01(t: torch.Tensor) -> torch.Tensor:
        """VAE-decoded (B, C, H, W) in [-1, 1] → [0, 1]."""
        return ((t + 1.0) / 2.0).clamp(0.0, 1.0)

    def to_uint8_rgb(t: torch.Tensor) -> np.ndarray:
        """(C, H, W) float [0, 1] → HxWx3 uint8 for YOLO."""
        arr = ensure_rgb(t.unsqueeze(0))[0].permute(1, 2, 0).numpy()
        return (arr * 255).clip(0, 255).astype(np.uint8)

    def draw_pred_boxes(pil_img: Image.Image, pred_xyxy: np.ndarray) -> Image.Image:
        """Overlay YOLO predictions (red) on a PIL image."""
        img = pil_img.copy()
        drw = ImageDraw.Draw(img)
        for x1, y1, x2, y2 in pred_xyxy:
            drw.rectangle([float(x1), float(y1), float(x2), float(y2)],
                          outline=(220, 50, 50), width=2)
        return img

    def yolo_predict(yolo_model, img_01: torch.Tensor, device: str, conf: float) -> np.ndarray:
        """(C, H, W) [0, 1] → run YOLO → (N, 4) xyxy pixel array."""
        res = yolo_model.predict(source=[to_uint8_rgb(img_01)],
                                 conf=conf, device=device, verbose=False)
        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            return res[0].boxes.xyxy.detach().cpu().numpy()
        return np.zeros((0, 4), dtype=np.float32)

    def make_count_sample(instance_df: pd.DataFrame, count: int,
                          image_size: int, seed: int) -> Dict[str, Any]:
        """Novel layout: `count` persons, real bbox sizes, random positions."""
        rng = np.random.default_rng(seed)
        rows = instance_df.sample(n=count, replace=True, random_state=int(seed))
        boxes: List[List[float]] = []
        for _, r in rows.iterrows():
            w = max(2.0, float(r["bbox_w_norm"]) * image_size)
            h = max(2.0, float(r["bbox_h_norm"]) * image_size)
            cx = float(rng.uniform(w / 2, max(w / 2 + 1, image_size - w / 2)))
            cy = float(rng.uniform(h / 2, max(h / 2 + 1, image_size - h / 2)))
            boxes.append([max(0., cx - w/2), max(0., cy - h/2),
                          min(float(image_size), cx + w/2),
                          min(float(image_size), cy + h/2)])
        return {
            "pixel_values": torch.zeros(3, image_size, image_size),
            "boxes_xyxy":   torch.tensor(boxes, dtype=torch.float32),
            "labels":       torch.zeros(count, dtype=torch.long),
            "n_objects":    count,
            "file_name":    f"count_{count}",
            "image_id":     count,
            "label_names":  ["person"] * count,
        }

    def make_slice_sample(instance_df: pd.DataFrame, size_bin: str,
                          position_bin: str, image_size: int,
                          seed: int) -> Optional[Dict[str, Any]]:
        """Single-box layout for a specific (size_bin, position_bin) slice."""
        mask = (
            (instance_df["class_label"] == "person") &
            (instance_df["size_bin"].astype(str) == size_bin) &
            (instance_df["position_bin"].astype(str) == position_bin)
        )
        if not mask.any():
            return None
        rng = random.Random(int(seed))
        r = instance_df[mask].sample(n=1, random_state=int(seed)).iloc[0]
        cx, cy = float(r["bbox_center_x_norm"]), float(r["bbox_center_y_norm"])
        w,  h  = float(r["bbox_w_norm"]),         float(r["bbox_h_norm"])
        # jitter within the 3×3 cell
        idx = list(POSITION_BIN_ORDER).index(position_bin)
        ri, ci = idx // 3, idx % 3
        x_lo, x_hi = ci / 3.0, (ci + 1) / 3.0
        y_lo, y_hi = ri / 3.0, (ri + 1) / 3.0
        span = 0.1 / 3.0
        cx = min(max(cx + rng.uniform(-span, span), x_lo, w / 2), x_hi, 1.0 - w / 2)
        cy = min(max(cy + rng.uniform(-span, span), y_lo, h / 2), y_hi, 1.0 - h / 2)
        wp, hp = w * image_size, h * image_size
        cxp, cyp = cx * image_size, cy * image_size
        return {
            "pixel_values": torch.zeros(3, image_size, image_size),
            "boxes_xyxy":   torch.tensor([[cxp-wp/2, cyp-hp/2, cxp+wp/2, cyp+hp/2]]),
            "labels":       torch.zeros(1, dtype=torch.long),
            "n_objects":    1,
            "file_name":    f"slice_{size_bin}_{position_bin}",
            "image_id":     0,
            "label_names":  ["person"],
        }

    def generate_and_detect(sampler, yolo_model, sample_dicts, steps, seed, device, conf):
        """Generate one FM image per sample dict, run YOLO, return result list."""
        results = []
        for i, sd in enumerate(sample_dicts):
            batch = collate_layout_batch([sd])
            gen   = sample_layout_batch(sampler, batch, steps=steps, seed=seed + i)
            gen01 = generated_to_01(gen)
            sketch = render_class_layout(
                boxes_xyxy=batch["boxes_xyxy"],
                labels=batch["labels"],
                object_mask=batch["object_mask"],
                image_size=gen.shape[-1],
            )
            gen_gt = draw_bbox_overlays(
                gen01,
                boxes_xyxy=batch["boxes_xyxy"],
                object_mask=batch["object_mask"],
                labels=batch["labels"],
                line_width=2,
            )
            pred  = yolo_predict(yolo_model, gen01[0], device, conf)
            results.append({
                "sketch":     tensor_to_pil(sketch[0]),
                "gen_gt":     tensor_to_pil(gen_gt[0]),
                "gen_yolo":   draw_pred_boxes(tensor_to_pil(gen01[0]), pred),
                "n_detected": int(pred.shape[0]),
                "meta":       sd.get("label", sd["file_name"]),
            })
            print(f"  [{i+1}/{len(sample_dicts)}] {sd['file_name']}  →  {pred.shape[0]} detected")
        return results

    def render_panels(mo, results, title_key="meta"):
        rows = []
        for r in results:
            rows.append(mo.vstack([
                mo.md(f"**{r[title_key]}** — YOLO: {r['n_detected']} box(es)"),
                mo.hstack([
                    mo.vstack([mo.md("_layout_"),            mo.image(r["sketch"])]),
                    mo.vstack([mo.md("_generated + GT_"),    mo.image(r["gen_gt"])]),
                    mo.vstack([mo.md("_YOLO detections_"),   mo.image(r["gen_yolo"])]),
                ], gap=1),
            ]))
        return mo.vstack(rows, gap=2)

    ds = load_yolo_slice_dataset(str(DATASET_YAML))
    return (
        FM_CHECKPOINT,
        FM_PRESET,
        PIPELINE_DIR,
        POSITION_BIN_ORDER,
        SIZE_BIN_ORDER,
        YOLO_WEIGHTS,
        build_slice_counts_table,
        ds,
        generate_and_detect,
        load_sampler_from_pipeline,
        make_count_sample,
        make_slice_sample,
        pd,
        plotly_express,
        plotly_go,
        render_panels,
        ultralytics_mod,
    )


@app.cell
def _():
    DEVICE    = "cuda:0"
    FM_STEPS  = 50
    SEED      = 42
    MAX_COUNT = 8      # sweep person counts 1..MAX_COUNT
    YOLO_CONF = 0.25
    N_RARE    = 9      # number of rarest slices to test in Part 2
    IMAGE_SIZE = 256
    return DEVICE, FM_STEPS, IMAGE_SIZE, MAX_COUNT, N_RARE, SEED, YOLO_CONF


@app.cell
def _(mo):
    mo.md("""
    # FM Conditioning Quality — `stay_layout_latent_v18_sd15ft_x8_256_minmax_reg_v2`

    Visual test of layout-conditioning fidelity on the single-class v18 dataset (`person`).

    **Part 1** — Person-count sweep: does the generator render 1…N persons?
    Layouts use novel random positions sampled from the real bbox-size distribution.

    **Part 2** — Slice sweep: does the generator respond to conditioning on
    underrepresented `(size_bin, position_bin)` cells?

    Each row: **layout sketch** | **generated + GT boxes (green)** | **YOLOv11n detections (red)**.

    > Edit the **Config** cell above to change device, steps, counts, etc.
    """)
    return


@app.cell
def _(
    DEVICE,
    FM_CHECKPOINT,
    FM_PRESET,
    PIPELINE_DIR,
    YOLO_WEIGHTS,
    load_sampler_from_pipeline,
    mo,
    ultralytics_mod,
):
    mo.stop(
        not FM_CHECKPOINT.exists(),
        mo.callout(mo.md(f"Checkpoint not found: `{FM_CHECKPOINT}`"), kind="danger"),
    )
    _preset, _meta, _unet, _vae, sampler = load_sampler_from_pipeline(
        PIPELINE_DIR, FM_PRESET, FM_CHECKPOINT.name, DEVICE,
    )
    _YOLO    = getattr(ultralytics_mod, "YOLO")
    yolo_model = _YOLO(str(YOLO_WEIGHTS))
    mo.callout(mo.md(f"Models loaded on **{DEVICE}**."), kind="success")
    return sampler, yolo_model


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 1 — Person Count Sweep
    """)
    return


@app.cell
def _(ds, mo, pd, plotly_express):
    _idf = ds.instance_df
    _counts_per_img = (
        _idf.groupby("image_index").size()
        .reindex(range(len(ds.image_df)), fill_value=0)
    )
    _summary = pd.Series(_counts_per_img).value_counts().sort_index()
    _fig = plotly_express.bar(
        x=_summary.index.tolist(),
        y=_summary.values.tolist(),
        labels={"x": "persons per image", "y": "image count"},
        title="v18 full_train — person count distribution",
    )
    _fig.update_layout(bargap=0.15, height=320)
    mo.ui.plotly(_fig)
    return


@app.cell
def _(
    DEVICE,
    FM_STEPS,
    IMAGE_SIZE,
    MAX_COUNT,
    SEED,
    YOLO_CONF,
    ds,
    generate_and_detect,
    make_count_sample,
    mo,
    sampler,
    yolo_model,
):
    mo.stop(sampler is None, mo.md("Model not loaded."))
    print(f"Generating {MAX_COUNT} images (counts 1–{MAX_COUNT}) …")
    _samples_p1 = [
        {**make_count_sample(ds.instance_df, n, IMAGE_SIZE, SEED + n), "label": f"N = {n}"}
        for n in range(1, MAX_COUNT + 1)
    ]
    p1_results = generate_and_detect(
        sampler, yolo_model, _samples_p1, FM_STEPS, SEED, DEVICE, YOLO_CONF,
    )
    print("Part 1 done.")
    return (p1_results,)


@app.cell
def _(mo, p1_results, render_panels):
    render_panels(mo, p1_results)
    return


@app.cell
def _(mo):
    mo.md("""
    ---
    ## Part 2 — Slice Distribution Sweep
    """)
    return


@app.cell
def _(
    POSITION_BIN_ORDER,
    SIZE_BIN_ORDER,
    build_slice_counts_table,
    ds,
    mo,
    plotly_go,
):
    _idf    = ds.instance_df
    _labels = sorted(_idf["class_label"].astype(str).unique().tolist())
    slice_counts = build_slice_counts_table(_idf, _labels)

    _person = (
        slice_counts[slice_counts["class_label"] == "person"]
        .assign(size_bin=lambda d: d["size_bin"].astype(str),
                position_bin=lambda d: d["position_bin"].astype(str))
    )
    _pivot  = (
        _person
        .pivot(index="size_bin", columns="position_bin", values="count")
        .reindex(index=list(SIZE_BIN_ORDER), columns=list(POSITION_BIN_ORDER), fill_value=0)
    )
    _fig = plotly_go.Figure(plotly_go.Heatmap(
        z=_pivot.values.tolist(),
        x=list(POSITION_BIN_ORDER),
        y=list(SIZE_BIN_ORDER),
        colorscale="Viridis",
        text=_pivot.values.tolist(),
        texttemplate="%{text}",
    ))
    _fig.update_layout(
        title="v18 full_train — person slice distribution (size × position)",
        xaxis_title="position_bin", yaxis_title="size_bin", height=300,
    )
    mo.ui.plotly(_fig)
    return (slice_counts,)


@app.cell
def _(
    DEVICE,
    FM_STEPS,
    IMAGE_SIZE,
    N_RARE,
    SEED,
    YOLO_CONF,
    ds,
    generate_and_detect,
    make_slice_sample,
    mo,
    sampler,
    slice_counts,
    yolo_model,
):
    mo.stop(sampler is None, mo.md("Model not loaded."))

    _person_slices = (
        slice_counts[slice_counts["class_label"] == "person"]
        .assign(size_bin=lambda d: d["size_bin"].astype(str),
                position_bin=lambda d: d["position_bin"].astype(str))
        .sort_values("count")
        .head(N_RARE)
    )

    _samples_p2 = []
    for _i, (_, _row) in enumerate(_person_slices.iterrows()):
        _sd = make_slice_sample(
            ds.instance_df, str(_row["size_bin"]), str(_row["position_bin"]),
            IMAGE_SIZE, SEED + _i,
        )
        if _sd is not None:
            _sd = {**_sd, "label": f"{_row['size_bin']} | {_row['position_bin']}  (n={int(_row['count'])})"}
            _samples_p2.append(_sd)

    print(f"Generating {len(_samples_p2)} rare-slice images …")
    p2_results = generate_and_detect(
        sampler, yolo_model, _samples_p2, FM_STEPS, SEED, DEVICE, YOLO_CONF,
    )
    print("Part 2 done.")
    return (p2_results,)


@app.cell
def _(mo, p2_results, render_panels):
    render_panels(mo, p2_results)
    return


if __name__ == "__main__":
    app.run()

import marimo

__generated_with = "0.23.9"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md("""
    # SD1.5 VAE Posterior Analysis

    This notebook compares the original Stable Diffusion 1.5 VAE posterior
    against finetuned VAE checkpoints. It focuses on whether the latent
    posterior remains suitable for downstream latent Flow Matching training:
    posterior drift, variance collapse, Gaussianity, covariance structure,
    reconstruction quality, random prior decoding, and interpolation
    smoothness.

    The defaults target the `v18_sd15_vae_x8_256_minmax` run. All paths and
    analysis limits are editable below.
    """)
    return


@app.cell
def _():
    import contextlib
    import gc
    import hashlib
    import importlib
    import json
    import math
    import os
    import random
    import re
    import sys
    import time
    import traceback
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Any, Iterable, Optional

    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from tqdm.auto import tqdm

    NOTEBOOK_VERSION = "vae_posterior_analysis_v1"
    SEED = 17

    def _resolve_repo_root() -> Path:
        start = Path.cwd().resolve()
        for candidate in [start, *start.parents]:
            if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
                return candidate
        here = Path(__file__).resolve()
        for candidate in [here, *here.parents]:
            if (candidate / "pyproject.toml").exists() and (candidate / "src").exists():
                return candidate
        raise RuntimeError("Could not resolve repository root.")

    REPO_ROOT = _resolve_repo_root()
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    def _optional_import(module_name: str):
        try:
            return importlib.import_module(module_name), None
        except Exception as exc:
            return None, f"{type(exc).__name__}: {exc}"

    plotly_express, _plotly_express_error = _optional_import("plotly.express")
    plotly_go, _plotly_go_error = _optional_import("plotly.graph_objects")
    scipy_stats, _scipy_error = _optional_import("scipy.stats")
    sklearn_decomposition, _sklearn_decomposition_error = _optional_import("sklearn.decomposition")
    sklearn_preprocessing, _sklearn_preprocessing_error = _optional_import("sklearn.preprocessing")
    umap_module, _umap_error = _optional_import("umap")
    lpips_module, _lpips_error = _optional_import("lpips")
    diffusers_module, _diffusers_error = _optional_import("diffusers")

    from src.core.data.datasets import load_single_channel_tensor
    from src.core.normalization import (
        PER_IMAGE_MINMAX,
        RAW_UINT16_PERCENTILE,
        SENTINEL2_REFLECTANCE,
        UINT8_LINEAR,
        denorm_for_display,
        resize_and_normalize,
    )
    try:
        from src.models.vae import (
            DiffusersAutoencoderAdapter,
            build_vae_from_config,
            load_vae_config,
            load_vae_weights,
        )

        _repo_vae_error = None
    except Exception as exc:
        DiffusersAutoencoderAdapter = None
        build_vae_from_config = None
        load_vae_config = None
        load_vae_weights = None
        _repo_vae_error = f"{type(exc).__name__}: {exc}"

    DEFAULT_DATASET_ROOT = REPO_ROOT / "data" / "raw" / "v18" / "val"
    DEFAULT_OUTPUT_ROOT = REPO_ROOT / "artifacts" / "analysis" / "vae_posterior_analysis"
    DEFAULT_CACHE_ROOT = REPO_ROOT / "data" / "cache" / "vae_posterior_analysis"
    DEFAULT_VAE_ROOT = (
        REPO_ROOT
        / "artifacts"
        / "checkpoints"
        / "vae"
        / "vae_runs"
        / "v18_sd15_vae_x8_256_minmax"
        / "VAE"
    )

    DEFAULT_CHECKPOINT_SPEC = "\n".join(
        [
            f"epoch_10, {DEFAULT_VAE_ROOT / 'vae_epoch_10.pt'}",
            f"epoch_20, {DEFAULT_VAE_ROOT / 'vae_epoch_20.pt'}",
            f"epoch_50, {DEFAULT_VAE_ROOT / 'vae_epoch_50.pt'}",
            f"epoch_100, {DEFAULT_VAE_ROOT / 'vae_epoch_100.pt'}",
            f"epoch_150, {DEFAULT_VAE_ROOT / 'vae_epoch_150.pt'}",
            f"best, {DEFAULT_VAE_ROOT / 'vae_best.pt'}",
        ]
    )

    NORMALIZATION_CHOICES = [
        PER_IMAGE_MINMAX,
        RAW_UINT16_PERCENTILE,
        UINT8_LINEAR,
        SENTINEL2_REFLECTANCE,
    ]
    PRECISION_CHOICES = ["auto", "fp32", "fp16", "bf16"]

    def dependency_status() -> pd.DataFrame:
        rows = [
            ("marimo", True, "required to open this notebook"),
            ("torch", True, torch.__version__),
            ("pandas", True, pd.__version__),
            ("Pillow", True, Image.__version__),
            ("diffusers", diffusers_module is not None, _diffusers_error or "required for AutoencoderKL"),
            ("repo VAE helpers", _repo_vae_error is None, _repo_vae_error or "training-compatible checkpoint loading"),
            ("plotly", plotly_express is not None and plotly_go is not None, _plotly_express_error or _plotly_go_error or "interactive plots"),
            ("scipy", scipy_stats is not None, _scipy_error or "Gaussianity tests"),
            ("scikit-learn", sklearn_decomposition is not None and sklearn_preprocessing is not None, _sklearn_decomposition_error or _sklearn_preprocessing_error or "PCA and scaling"),
            ("umap-learn", umap_module is not None, _umap_error or "optional UMAP"),
            ("lpips", lpips_module is not None, _lpips_error or "optional LPIPS"),
        ]
        return pd.DataFrame(rows, columns=["package", "available", "notes"])

    def install_hint() -> str:
        return (
            "python -m pip install marimo plotly diffusers scikit-learn "
            "umap-learn torchvision lpips scipy pillow pandas tqdm"
        )

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    return (
        Any,
        DEFAULT_CACHE_ROOT,
        DEFAULT_CHECKPOINT_SPEC,
        DEFAULT_DATASET_ROOT,
        DEFAULT_OUTPUT_ROOT,
        DEFAULT_VAE_ROOT,
        DataLoader,
        Dataset,
        DiffusersAutoencoderAdapter,
        F,
        Image,
        NORMALIZATION_CHOICES,
        NOTEBOOK_VERSION,
        Optional,
        PRECISION_CHOICES,
        Path,
        SEED,
        build_vae_from_config,
        contextlib,
        dataclass,
        denorm_for_display,
        dependency_status,
        diffusers_module,
        gc,
        hashlib,
        install_hint,
        json,
        load_single_channel_tensor,
        load_vae_config,
        load_vae_weights,
        lpips_module,
        math,
        np,
        pd,
        plotly_express,
        plotly_go,
        re,
        resize_and_normalize,
        scipy_stats,
        sklearn_decomposition,
        sklearn_preprocessing,
        time,
        torch,
        tqdm,
        traceback,
        umap_module,
    )


@app.cell
def _(dependency_status, install_hint, mo):
    deps = dependency_status()
    missing = deps.loc[~deps["available"], "package"].tolist()
    dep_message = "All optional analysis dependencies are available."
    if missing:
        dep_message = (
            "Missing packages: "
            + ", ".join(missing)
            + "\n\nInstall in the active environment with:\n\n"
            + f"`{install_hint()}`"
        )
    mo.vstack([mo.md("## Runtime dependencies"), mo.ui.table(deps), mo.md(dep_message)])
    return


@app.cell
def _(
    DEFAULT_CACHE_ROOT,
    DEFAULT_CHECKPOINT_SPEC,
    DEFAULT_DATASET_ROOT,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VAE_ROOT,
    NORMALIZATION_CHOICES,
    PRECISION_CHOICES,
    mo,
):
    dataset_root_ui = mo.ui.text(
        value=str(DEFAULT_DATASET_ROOT),
        label="Dataset root",
        full_width=True,
    )
    normalization_ui = mo.ui.dropdown(
        options=NORMALIZATION_CHOICES,
        value="per_image_minmax",
        label="Normalization",
    )
    image_size_ui = mo.ui.slider(64, 1024, value=256, step=32, label="Image resolution")
    batch_size_ui = mo.ui.slider(1, 64, value=4, step=1, label="Batch size")
    max_images_ui = mo.ui.slider(1, 5000, value=256, step=1, label="Max images")
    num_workers_ui = mo.ui.slider(0, 0, value=0, step=1, label="DataLoader workers")

    original_model_ui = mo.ui.text(
        value="runwayml/stable-diffusion-v1-5",
        label="Original SD1.5 model/path",
        full_width=True,
    )
    original_subfolder_ui = mo.ui.text(value="vae", label="Original VAE subfolder")
    finetuned_config_ui = mo.ui.text(
        value=str(DEFAULT_VAE_ROOT / "config.json"),
        label="Finetuned VAE config.json",
        full_width=True,
    )
    checkpoint_spec_ui = mo.ui.text_area(
        value=DEFAULT_CHECKPOINT_SPEC,
        label="Finetuned checkpoints: one 'label, path' per line",
        full_width=True,
    )

    device_ui = mo.ui.text(value="auto", label="Device: auto, cpu, cuda, cuda:0, ...")
    precision_ui = mo.ui.dropdown(options=PRECISION_CHOICES, value="auto", label="Precision")
    cache_root_ui = mo.ui.text(value=str(DEFAULT_CACHE_ROOT), label="Cache root", full_width=True)
    output_root_ui = mo.ui.text(value=str(DEFAULT_OUTPUT_ROOT), label="Output root", full_width=True)
    use_cache_ui = mo.ui.checkbox(value=True, label="Use cache")
    force_recompute_ui = mo.ui.checkbox(value=False, label="Force recompute")

    flat_sample_cap_ui = mo.ui.slider(1000, 2_000_000, value=250_000, step=1000, label="Flat value sample cap")
    latent_vector_cap_ui = mo.ui.slider(16, 5000, value=512, step=16, label="Latent vector cap")
    feature_cap_ui = mo.ui.slider(64, 4096, value=768, step=64, label="Covariance feature cap")
    shapiro_cap_ui = mo.ui.slider(100, 5000, value=2000, step=100, label="Shapiro sample cap")
    random_grid_ui = mo.ui.slider(0, 32, value=8, step=1, label="Random decode images")
    interp_pairs_ui = mo.ui.slider(0, 8, value=2, step=1, label="Interpolation pairs")
    interp_steps_ui = mo.ui.slider(3, 12, value=7, step=1, label="Interpolation steps")

    enable_lpips_ui = mo.ui.checkbox(value=False, label="Compute LPIPS if available")
    enable_umap_ui = mo.ui.checkbox(value=True, label="Compute UMAP if available")
    save_grids_ui = mo.ui.checkbox(value=True, label="Save reconstruction, interpolation, and random grids")
    run_analysis_ui = mo.ui.run_button(label="Run posterior analysis")
    export_ui = mo.ui.run_button(label="Export summary/report")

    mo.vstack(
        [
            mo.md("## Inputs"),
            mo.hstack([dataset_root_ui, normalization_ui]),
            mo.hstack([image_size_ui, batch_size_ui, max_images_ui, num_workers_ui]),
            mo.md("## VAE checkpoints"),
            mo.hstack([original_model_ui, original_subfolder_ui]),
            finetuned_config_ui,
            checkpoint_spec_ui,
            mo.md("## Runtime"),
            mo.hstack([device_ui, precision_ui, use_cache_ui, force_recompute_ui]),
            mo.hstack([cache_root_ui, output_root_ui]),
            mo.md("## Analysis limits"),
            mo.hstack([flat_sample_cap_ui, latent_vector_cap_ui, feature_cap_ui, shapiro_cap_ui]),
            mo.hstack([random_grid_ui, interp_pairs_ui, interp_steps_ui]),
            mo.hstack([enable_lpips_ui, enable_umap_ui, save_grids_ui]),
            mo.hstack([run_analysis_ui, export_ui]),
        ]
    )
    return (
        batch_size_ui,
        cache_root_ui,
        checkpoint_spec_ui,
        dataset_root_ui,
        device_ui,
        enable_lpips_ui,
        enable_umap_ui,
        export_ui,
        feature_cap_ui,
        finetuned_config_ui,
        flat_sample_cap_ui,
        force_recompute_ui,
        image_size_ui,
        interp_pairs_ui,
        interp_steps_ui,
        latent_vector_cap_ui,
        max_images_ui,
        normalization_ui,
        original_model_ui,
        original_subfolder_ui,
        output_root_ui,
        precision_ui,
        random_grid_ui,
        run_analysis_ui,
        save_grids_ui,
        shapiro_cap_ui,
        use_cache_ui,
    )


@app.cell
def _(
    Any,
    DataLoader,
    Dataset,
    DiffusersAutoencoderAdapter,
    F,
    Image,
    Optional,
    Path,
    SEED,
    build_vae_from_config,
    contextlib,
    dataclass,
    denorm_for_display,
    diffusers_module,
    gc,
    hashlib,
    json,
    load_single_channel_tensor,
    load_vae_config,
    load_vae_weights,
    lpips_module,
    math,
    np,
    pd,
    re,
    resize_and_normalize,
    scipy_stats,
    sklearn_decomposition,
    sklearn_preprocessing,
    time,
    torch,
    tqdm,
    umap_module,
):
    @dataclass(frozen=True)
    class CheckpointSpec:
        label: str
        path: str
        role: str = "finetuned"

    class PosteriorImageDataset(Dataset):
        def __init__(self, root: str | Path, image_size: int, normalization_mode: str, max_images: int):
            self.root = Path(root).expanduser()
            self.image_size = int(image_size)
            self.normalization_mode = str(normalization_mode)
            if not self.root.exists():
                raise FileNotFoundError(f"Dataset root does not exist: {self.root}")
            exts = {".npy", ".tif", ".tiff"}
            self.paths = sorted(p for p in self.root.rglob("*") if p.suffix.lower() in exts)
            if max_images and max_images > 0:
                self.paths = self.paths[: int(max_images)]
            if not self.paths:
                raise RuntimeError(f"No .npy/.tif/.tiff images found in {self.root}")

        def __len__(self) -> int:
            return len(self.paths)

        def __getitem__(self, idx: int) -> dict[str, Any]:
            path = self.paths[idx]
            raw = load_single_channel_tensor(path)
            normalized = resize_and_normalize(
                raw,
                image_size=self.image_size,
                normalization_mode=self.normalization_mode,
            )
            raw_resized = F.interpolate(
                raw.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
            return {
                "pixel_values": normalized,
                "raw_resized": raw_resized,
                "path": str(path),
            }

    def parse_checkpoint_spec(text: str) -> list[CheckpointSpec]:
        specs: list[CheckpointSpec] = []
        for line_number, raw_line in enumerate(str(text).splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                label, path = [part.strip() for part in line.split(",", 1)]
            else:
                parts = re.split(r"\s+", line, maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Checkpoint line {line_number} must be 'label, path' or 'label path'."
                    )
                label, path = parts
            if not label or not path:
                raise ValueError(f"Checkpoint line {line_number} has an empty label or path.")
            specs.append(CheckpointSpec(label=label, path=str(Path(path).expanduser())))
        return specs

    def resolve_device(device_text: str) -> torch.device:
        text = str(device_text or "auto").strip().lower()
        if text == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(text)

    def resolve_dtype(precision: str, device: torch.device) -> Optional[torch.dtype]:
        precision = str(precision or "auto").lower()
        if device.type != "cuda":
            return torch.float32
        if precision == "fp16":
            return torch.float16
        if precision == "bf16":
            return torch.bfloat16
        if precision == "fp32":
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def autocast_context(device: torch.device, dtype: Optional[torch.dtype]):
        if device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}:
            return torch.autocast(device_type="cuda", dtype=dtype)
        return contextlib.nullcontext()

    def checkpoint_signature(specs: list[CheckpointSpec]) -> list[dict[str, Any]]:
        rows = []
        for spec in specs:
            path = Path(spec.path)
            rows.append(
                {
                    "label": spec.label,
                    "path": str(path),
                    "role": spec.role,
                    "exists": path.exists(),
                    "mtime": path.stat().st_mtime if path.exists() else None,
                    "size": path.stat().st_size if path.exists() else None,
                }
            )
        return rows

    def cache_key(payload: dict[str, Any]) -> str:
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:24]

    def _match_channel_count(x: torch.Tensor, expected_channels: int) -> torch.Tensor:
        if int(x.shape[1]) == int(expected_channels):
            return x
        if int(x.shape[1]) == 1 and int(expected_channels) == 3:
            return x.expand(-1, 3, -1, -1)
        if int(x.shape[1]) == 3 and int(expected_channels) == 1:
            return x.mean(dim=1, keepdim=True)
        raise ValueError(f"Cannot adapt {x.shape[1]} input channels to {expected_channels}.")

    def posterior_stddev(posterior: Any) -> torch.Tensor:
        for name in ("stddev", "std"):
            value = getattr(posterior, name, None)
            if value is not None:
                return value
        logvar = getattr(posterior, "logvar", None)
        if logvar is not None:
            return torch.exp(0.5 * logvar)
        var = getattr(posterior, "var", None)
        if var is not None:
            return torch.sqrt(var)
        raise AttributeError("Could not read posterior stddev/std/logvar/var.")

    def posterior_mode(posterior: Any) -> torch.Tensor:
        mode = getattr(posterior, "mode", None)
        if callable(mode):
            return mode()
        mean = getattr(posterior, "mean", None)
        if mean is None:
            raise AttributeError("Could not read posterior mode or mean.")
        return mean

    def scaling_factor(vae: Any) -> float:
        config = getattr(vae, "config", None)
        return float(getattr(config, "scaling_factor", 1.0) or 1.0)

    def load_original_vae(model_path: str, subfolder: str, device: torch.device, dtype: Optional[torch.dtype]):
        if diffusers_module is None:
            raise ImportError("diffusers is required to load the original SD1.5 VAE.")
        AutoencoderKL = getattr(diffusers_module, "AutoencoderKL")
        kwargs: dict[str, Any] = {}
        if subfolder:
            kwargs["subfolder"] = subfolder
        if dtype is not None and dtype != torch.float32:
            kwargs["torch_dtype"] = dtype
        vae = AutoencoderKL.from_pretrained(str(model_path), **kwargs)
        vae.to(device)
        vae.eval()
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        for parameter in vae.parameters():
            parameter.requires_grad_(False)
        return vae

    def load_finetuned_vae(config_path: str, weights_path: str, device: torch.device, dtype: Optional[torch.dtype]):
        if load_vae_config is None or build_vae_from_config is None or load_vae_weights is None:
            raise ImportError(
                "Repo VAE helpers are unavailable. Check the dependency table above; "
                "the training-compatible loader could not be imported."
            )
        config = load_vae_config(str(Path(config_path).expanduser()))
        wrapped = build_vae_from_config(config, device=device)
        wrapped = load_vae_weights(wrapped, str(Path(weights_path).expanduser()), map_location=device)
        vae = (
            wrapped.autoencoder
            if DiffusersAutoencoderAdapter is not None and isinstance(wrapped, DiffusersAutoencoderAdapter)
            else wrapped
        )
        vae.to(device)
        if dtype is not None and dtype != torch.float32:
            vae.to(dtype=dtype)
        vae.eval()
        if hasattr(vae, "enable_slicing"):
            vae.enable_slicing()
        if hasattr(vae, "enable_tiling"):
            vae.enable_tiling()
        for parameter in vae.parameters():
            parameter.requires_grad_(False)
        return vae

    def sample_flat(tensor: torch.Tensor, cap: int, rng: np.random.Generator) -> np.ndarray:
        flat = tensor.detach().float().cpu().reshape(-1).numpy()
        cap = int(cap)
        if cap <= 0 or flat.size <= cap:
            return flat
        idx = rng.choice(flat.size, size=cap, replace=False)
        return flat[idx]

    def sample_rows(array: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
        if array.shape[0] <= cap:
            return array
        idx = rng.choice(array.shape[0], size=int(cap), replace=False)
        return array[idx]

    def describe_values(values: np.ndarray, prefix: str, include_abs: bool = False) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            keys = ["mean", "std", "min", "max", "p01", "p05", "p25", "p50", "p75", "p95", "p99"]
            out = {f"{prefix}_{key}": float("nan") for key in keys}
            if include_abs:
                out[f"{prefix}_abs_mean"] = float("nan")
            return out
        percentiles = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
        out = {
            f"{prefix}_mean": float(arr.mean()),
            f"{prefix}_std": float(arr.std(ddof=0)),
            f"{prefix}_min": float(arr.min()),
            f"{prefix}_max": float(arr.max()),
            f"{prefix}_p01": float(percentiles[0]),
            f"{prefix}_p05": float(percentiles[1]),
            f"{prefix}_p25": float(percentiles[2]),
            f"{prefix}_p50": float(percentiles[3]),
            f"{prefix}_p75": float(percentiles[4]),
            f"{prefix}_p95": float(percentiles[5]),
            f"{prefix}_p99": float(percentiles[6]),
        }
        if include_abs:
            out[f"{prefix}_abs_mean"] = float(np.abs(arr).mean())
        return out

    def compute_reconstruction_metrics(x: torch.Tensor, recon: torch.Tensor) -> dict[str, np.ndarray]:
        x_f = x.detach().float()
        recon_f = recon.detach().float()
        err = recon_f - x_f
        mse = err.pow(2).mean(dim=(1, 2, 3)).cpu().numpy()
        l1 = err.abs().mean(dim=(1, 2, 3)).cpu().numpy()
        psnr = (10.0 * torch.log10(torch.tensor(4.0, device=x.device) / err.pow(2).mean(dim=(1, 2, 3)).clamp_min(1e-8))).cpu().numpy()
        ssim = compute_ssim(x_f, recon_f).cpu().numpy()
        return {"recon_mse": mse, "recon_l1": l1, "recon_psnr": psnr, "recon_ssim": ssim}

    def compute_ssim(x: torch.Tensor, y: torch.Tensor, data_range: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
        mu_x = x.mean(dim=(2, 3), keepdim=True)
        mu_y = y.mean(dim=(2, 3), keepdim=True)
        sigma_x = ((x - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma_y = ((y - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=(2, 3), keepdim=True)
        c1 = (0.01 * float(data_range)) ** 2
        c2 = (0.03 * float(data_range)) ** 2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2) + eps
        )
        return ssim.mean(dim=(1, 2, 3))

    def make_lpips_model(device: torch.device, enabled: bool):
        if not enabled or lpips_module is None:
            return None
        model = lpips_module.LPIPS(net="alex").to(device).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model

    def compute_lpips_batch(model: Any, x: torch.Tensor, recon: torch.Tensor) -> Optional[np.ndarray]:
        if model is None:
            return None
        x3 = x if x.shape[1] == 3 else x.expand(-1, 3, -1, -1)
        r3 = recon if recon.shape[1] == 3 else recon.expand(-1, 3, -1, -1)
        with torch.no_grad():
            return model(x3.float(), r3.float()).reshape(-1).detach().cpu().numpy()

    def encode_batch(vae: Any, x: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype]):
        expected_channels = int(getattr(getattr(vae, "config", None), "in_channels", x.shape[1]))
        x = _match_channel_count(x.to(device), expected_channels)
        scale = scaling_factor(vae)
        with torch.no_grad(), autocast_context(device, dtype):
            posterior = vae.encode(x).latent_dist
            mu = posterior.mean * scale
            sigma = posterior_stddev(posterior) * scale
            sample = posterior.sample() * scale
            mode = posterior_mode(posterior) * scale
        return x, mu, sigma, sample, mode

    def decode_scaled_latents(vae: Any, z: torch.Tensor, device: torch.device, dtype: Optional[torch.dtype]) -> torch.Tensor:
        scale = scaling_factor(vae)
        with torch.no_grad(), autocast_context(device, dtype):
            decoded = vae.decode(z.to(device) / scale).sample
        return decoded

    def analyze_checkpoint(
        spec: CheckpointSpec,
        *,
        config_path: Optional[str],
        original_model: str,
        original_subfolder: str,
        loader: DataLoader,
        device: torch.device,
        dtype: Optional[torch.dtype],
        flat_sample_cap: int,
        latent_vector_cap: int,
        feature_cap: int,
        shapiro_cap: int,
        enable_lpips: bool,
    ) -> dict[str, Any]:
        rng = np.random.default_rng(SEED)
        t0 = time.time()
        if spec.role == "baseline":
            vae = load_original_vae(original_model, original_subfolder, device, dtype)
        else:
            if config_path is None:
                raise ValueError("config_path is required for finetuned checkpoints.")
            vae = load_finetuned_vae(config_path, spec.path, device, dtype)

        lpips_model = make_lpips_model(device, enable_lpips)
        mu_values: list[np.ndarray] = []
        sigma_values: list[np.ndarray] = []
        z_values: list[np.ndarray] = []
        latent_vectors: list[np.ndarray] = []
        latent_norms: list[np.ndarray] = []
        metric_values: dict[str, list[np.ndarray]] = {
            "recon_mse": [],
            "recon_l1": [],
            "recon_psnr": [],
            "recon_ssim": [],
        }
        lpips_values: list[np.ndarray] = []
        channel_sum = channel_sq_sum = None
        channel_count = None
        image_paths: list[str] = []
        total_images = 0
        latent_shape = None

        for batch in tqdm(loader, desc=f"Analyze {spec.label}", leave=False):
            x_in = batch["pixel_values"]
            paths = list(batch["path"])
            x, mu, sigma, z, mode = encode_batch(vae, x_in, device, dtype)
            recon = decode_scaled_latents(vae, mode, device, dtype)
            recon = _match_channel_count(recon, int(x.shape[1]))

            metrics = compute_reconstruction_metrics(x, recon)
            for key, value in metrics.items():
                metric_values[key].append(value)
            lp = compute_lpips_batch(lpips_model, x, recon)
            if lp is not None:
                lpips_values.append(lp)

            bs = x.shape[0]
            total_images += int(bs)
            image_paths.extend(paths)
            latent_shape = tuple(int(v) for v in z.shape[1:])

            mu_values.append(sample_flat(mu, max(1, flat_sample_cap // 3), rng))
            sigma_values.append(sample_flat(sigma, max(1, flat_sample_cap // 3), rng))
            z_values.append(sample_flat(z, max(1, flat_sample_cap // 3), rng))

            z_float = z.detach().float()
            dims = tuple(range(0, z_float.dim()))
            channel_dims = (0, 2, 3)
            channel_batch_sum = z_float.sum(dim=channel_dims).cpu().numpy()
            channel_batch_sq_sum = z_float.pow(2).sum(dim=channel_dims).cpu().numpy()
            channel_batch_count = int(z_float.shape[0] * z_float.shape[2] * z_float.shape[3])
            if channel_sum is None:
                channel_sum = channel_batch_sum
                channel_sq_sum = channel_batch_sq_sum
                channel_count = channel_batch_count
            else:
                channel_sum += channel_batch_sum
                channel_sq_sum += channel_batch_sq_sum
                channel_count += channel_batch_count

            flat_z = z_float.reshape(bs, -1)
            latent_norms.append(torch.linalg.vector_norm(flat_z, dim=1).cpu().numpy())
            latent_vectors.append(flat_z.cpu().numpy())

            del x, mu, sigma, z, mode, recon

        mu_arr = sample_rows(np.concatenate(mu_values), flat_sample_cap, rng)
        sigma_arr = sample_rows(np.concatenate(sigma_values), flat_sample_cap, rng)
        z_arr = sample_rows(np.concatenate(z_values), flat_sample_cap, rng)
        latent_norm_arr = np.concatenate(latent_norms)
        vectors = sample_rows(np.concatenate(latent_vectors, axis=0), latent_vector_cap, rng)
        metrics_np = {key: np.concatenate(values) for key, values in metric_values.items() if values}
        if lpips_values:
            metrics_np["recon_lpips"] = np.concatenate(lpips_values)

        summary: dict[str, Any] = {
            "checkpoint": spec.label,
            "role": spec.role,
            "path": spec.path,
            "num_images": total_images,
            "latent_shape": str(latent_shape),
            "scaling_factor": scaling_factor(vae),
            "runtime_sec": time.time() - t0,
            "error": "",
        }
        summary.update(describe_values(mu_arr, "mu", include_abs=True))
        summary.update(describe_values(sigma_arr, "sigma", include_abs=False))
        summary.update(describe_values(z_arr, "z", include_abs=True))
        summary.update(describe_values(latent_norm_arr, "latent_norm", include_abs=False))
        for key, arr in metrics_np.items():
            summary.update(describe_values(arr, key, include_abs=False))

        var = np.square(np.clip(sigma_arr.astype(np.float64), 1e-12, None))
        kl_mu = 0.5 * np.square(mu_arr.astype(np.float64))
        kl_sigma = 0.5 * (var - np.log(var) - 1.0)
        summary["kl_mu_mean"] = float(np.mean(kl_mu))
        summary["kl_sigma_mean"] = float(np.mean(kl_sigma))
        summary["kl_total_mean"] = float(np.mean(kl_mu + kl_sigma))

        if channel_sum is not None and channel_sq_sum is not None and channel_count:
            ch_mean = channel_sum / channel_count
            ch_var = np.maximum(channel_sq_sum / channel_count - np.square(ch_mean), 0.0)
            for idx, (mean, std) in enumerate(zip(ch_mean, np.sqrt(ch_var))):
                summary[f"z_channel_{idx}_mean"] = float(mean)
                summary[f"z_channel_{idx}_std"] = float(std)

        covariance = covariance_summary(vectors, feature_cap)
        summary.update(covariance)
        gaussian = gaussian_summary(z_arr, shapiro_cap)
        summary.update(gaussian)

        arrays = {
            "mu": mu_arr.astype(np.float32),
            "sigma": sigma_arr.astype(np.float32),
            "z": z_arr.astype(np.float32),
            "latent_norm": latent_norm_arr.astype(np.float32),
            "latent_vectors": vectors.astype(np.float32),
            "image_paths": np.array(image_paths, dtype=object),
            "metrics": {key: arr.astype(np.float32) for key, arr in metrics_np.items()},
        }

        del vae, lpips_model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return {"summary": summary, "arrays": arrays}

    def covariance_summary(vectors: np.ndarray, feature_cap: int) -> dict[str, float]:
        out = {
            "corr_offdiag_abs_mean": float("nan"),
            "effective_rank": float("nan"),
            "pca_explained_var_1": float("nan"),
            "pca_explained_var_2": float("nan"),
            "pca_explained_var_3": float("nan"),
        }
        if vectors.ndim != 2 or min(vectors.shape) < 3:
            return out
        rng = np.random.default_rng(SEED)
        x = vectors.astype(np.float64)
        if x.shape[1] > int(feature_cap):
            cols = rng.choice(x.shape[1], size=int(feature_cap), replace=False)
            x = x[:, cols]
        x = x - x.mean(axis=0, keepdims=True)
        std = x.std(axis=0, keepdims=True)
        keep = std.reshape(-1) > 1e-8
        x = x[:, keep]
        if x.shape[1] < 2:
            return out
        xs = x / (x.std(axis=0, keepdims=True) + 1e-8)
        corr = np.corrcoef(xs, rowvar=False)
        mask = ~np.eye(corr.shape[0], dtype=bool)
        out["corr_offdiag_abs_mean"] = float(np.nanmean(np.abs(corr[mask])))
        if sklearn_decomposition is None:
            return out
        n_components = min(20, x.shape[0], x.shape[1])
        if n_components < 2:
            return out
        pca = sklearn_decomposition.PCA(n_components=n_components, svd_solver="randomized", random_state=SEED)
        pca.fit(x)
        ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        probs = ratios / max(float(ratios.sum()), 1e-12)
        out["effective_rank"] = float(np.exp(-np.sum(probs * np.log(np.clip(probs, 1e-12, None)))))
        for idx in range(min(3, ratios.size)):
            out[f"pca_explained_var_{idx + 1}"] = float(ratios[idx])
        for idx in range(min(10, pca.explained_variance_.size)):
            out[f"cov_eigenvalue_{idx + 1}"] = float(pca.explained_variance_[idx])
        return out

    def gaussian_summary(values: np.ndarray, shapiro_cap: int) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0 or scipy_stats is None:
            return {
                "z_skewness": float("nan"),
                "z_kurtosis": float("nan"),
                "z_jarque_bera_stat": float("nan"),
                "z_jarque_bera_p": float("nan"),
                "z_shapiro_stat": float("nan"),
                "z_shapiro_p": float("nan"),
            }
        rng = np.random.default_rng(SEED)
        jb = scipy_stats.jarque_bera(arr)
        shapiro_arr = sample_rows(arr.reshape(-1, 1), min(int(shapiro_cap), 5000), rng).reshape(-1)
        shapiro = scipy_stats.shapiro(shapiro_arr) if shapiro_arr.size >= 3 else (float("nan"), float("nan"))
        return {
            "z_skewness": float(scipy_stats.skew(arr)),
            "z_kurtosis": float(scipy_stats.kurtosis(arr, fisher=True)),
            "z_jarque_bera_stat": float(jb.statistic),
            "z_jarque_bera_p": float(jb.pvalue),
            "z_shapiro_stat": float(shapiro.statistic if hasattr(shapiro, "statistic") else shapiro[0]),
            "z_shapiro_p": float(shapiro.pvalue if hasattr(shapiro, "pvalue") else shapiro[1]),
        }

    def compute_baseline_deltas(summary_df: pd.DataFrame) -> pd.DataFrame:
        df = summary_df.copy()
        baseline_rows = df[df["role"] == "baseline"]
        if baseline_rows.empty:
            return df
        base = baseline_rows.iloc[0]
        for key in ["mu_mean", "mu_std", "sigma_mean", "z_std"]:
            if key in df.columns:
                df[f"delta_{key}"] = df[key] - float(base[key])
        distance_terms = []
        for key in ["mu_mean", "mu_std", "sigma_mean", "z_std", "kl_total_mean", "corr_offdiag_abs_mean"]:
            if key in df.columns and pd.notna(base.get(key, np.nan)):
                denom = abs(float(base[key])) + 1e-6
                term = ((df[key] - float(base[key])) / denom).astype(float).pow(2)
                distance_terms.append(term)
        df["baseline_distance"] = np.sqrt(sum(distance_terms)) if distance_terms else np.nan
        return df

    def readiness_score(summary_df: pd.DataFrame) -> pd.DataFrame:
        df = summary_df.copy()
        if df.empty:
            return df
        ft_mask = df["role"] != "baseline"
        if not ft_mask.any():
            df["fm_readiness_score"] = np.nan
            df["fm_readiness_reason"] = ""
            return df
        def normalized_good_low(series: pd.Series) -> pd.Series:
            s = series.astype(float)
            lo, hi = np.nanmin(s), np.nanmax(s)
            if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
                return pd.Series(0.5, index=s.index)
            return 1.0 - (s - lo) / (hi - lo)
        def normalized_good_high(series: pd.Series) -> pd.Series:
            s = series.astype(float)
            lo, hi = np.nanmin(s), np.nanmax(s)
            if not np.isfinite(lo) or not np.isfinite(hi) or abs(hi - lo) < 1e-12:
                return pd.Series(0.5, index=s.index)
            return (s - lo) / (hi - lo)

        closeness = normalized_good_low(df.get("baseline_distance", pd.Series(np.nan, index=df.index)))
        std_stability = normalized_good_low((df.get("z_std", 0.0) - 1.0).abs())
        kl_behavior = normalized_good_low(df.get("kl_total_mean", pd.Series(np.nan, index=df.index)))
        recon_quality = normalized_good_low(df.get("recon_mse_mean", pd.Series(np.nan, index=df.index)))
        random_health = normalized_good_high(df.get("random_decode_contrast", pd.Series(0.5, index=df.index)))
        score = (
            0.35 * closeness
            + 0.20 * std_stability
            + 0.15 * kl_behavior
            + 0.20 * recon_quality
            + 0.10 * random_health
        ) * 100.0
        df["fm_readiness_score"] = score
        reasons = []
        for _, row in df.iterrows():
            if row["role"] == "baseline":
                reasons.append("Baseline reference; not ranked as a finetuned candidate.")
                continue
            chunks = []
            chunks.append(f"baseline distance={row.get('baseline_distance', np.nan):.3g}")
            chunks.append(f"z std={row.get('z_std', np.nan):.3g}")
            chunks.append(f"KL={row.get('kl_total_mean', np.nan):.3g}")
            chunks.append(f"MSE={row.get('recon_mse_mean', np.nan):.3g}")
            if row.get("sigma_mean", np.nan) < 0.05:
                chunks.append("posterior variance is very small")
            reasons.append("; ".join(chunks))
        df["fm_readiness_reason"] = reasons
        return df

    def arrays_to_distribution_frame(results: dict[str, Any], key: str, max_per_checkpoint: int = 50000) -> pd.DataFrame:
        rows = []
        rng = np.random.default_rng(SEED)
        for label, result in results.items():
            arr = np.asarray(result["arrays"][key]).reshape(-1)
            arr = sample_rows(arr.reshape(-1, 1), min(max_per_checkpoint, arr.size), rng).reshape(-1)
            rows.append(pd.DataFrame({"checkpoint": label, "value": arr, "stat": key}))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["checkpoint", "value", "stat"])

    def combined_latent_vectors(results: dict[str, Any], max_per_checkpoint: int = 512):
        rng = np.random.default_rng(SEED)
        xs, labels = [], []
        for label, result in results.items():
            vectors = np.asarray(result["arrays"]["latent_vectors"], dtype=np.float32)
            vectors = sample_rows(vectors, min(max_per_checkpoint, vectors.shape[0]), rng)
            xs.append(vectors)
            labels.extend([label] * vectors.shape[0])
        if not xs:
            return np.empty((0, 0), dtype=np.float32), []
        return np.concatenate(xs, axis=0), labels

    def compute_embeddings(results: dict[str, Any], enable_umap: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        x, labels = combined_latent_vectors(results)
        if x.size == 0 or x.shape[0] < 3 or sklearn_decomposition is None or sklearn_preprocessing is None:
            return pd.DataFrame(), pd.DataFrame()
        scaler = sklearn_preprocessing.StandardScaler(with_mean=True, with_std=True)
        xs = scaler.fit_transform(x)
        n_components = min(3, xs.shape[0], xs.shape[1])
        pca = sklearn_decomposition.PCA(n_components=n_components, random_state=SEED)
        xp = pca.fit_transform(xs)
        pca_df = pd.DataFrame(
            {
                "checkpoint": labels,
                "pc1": xp[:, 0],
                "pc2": xp[:, 1] if n_components > 1 else 0.0,
                "pc3": xp[:, 2] if n_components > 2 else 0.0,
            }
        )
        umap_df = pd.DataFrame()
        if enable_umap and umap_module is not None and xs.shape[0] >= 10:
            reducer = umap_module.UMAP(n_components=2, random_state=SEED, n_neighbors=min(15, xs.shape[0] - 1))
            xu = reducer.fit_transform(xs)
            umap_df = pd.DataFrame({"checkpoint": labels, "umap1": xu[:, 0], "umap2": xu[:, 1]})
        return pca_df, umap_df

    def tensor_to_display_array(tensor: torch.Tensor, normalization_mode: str) -> np.ndarray:
        disp = denorm_for_display(tensor.detach().cpu(), normalization_mode=normalization_mode)
        arr = disp.float().clamp(0, 1).numpy()
        return arr

    def save_grid(images: np.ndarray, path: str | Path, nrow: int = 4, pad: int = 2) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if images.ndim != 4:
            raise ValueError(f"Expected BCHW images, got shape {images.shape}")
        b, c, h, w = images.shape
        c = int(c)
        nrow = max(1, min(int(nrow), b))
        ncol = int(math.ceil(b / nrow))
        canvas = np.ones((ncol * h + pad * (ncol - 1), nrow * w + pad * (nrow - 1), 3), dtype=np.float32)
        for idx in range(b):
            y = (idx // nrow) * (h + pad)
            x = (idx % nrow) * (w + pad)
            img = images[idx]
            if c == 1:
                img = np.repeat(img, 3, axis=0)
            elif c > 3:
                img = img[:3]
            img = np.moveaxis(img, 0, -1)
            canvas[y : y + h, x : x + w] = img
        Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8)).save(path)
        return path

    def compute_random_decode_health(images: np.ndarray) -> dict[str, float]:
        if images.size == 0:
            return {"random_decode_contrast": float("nan"), "random_decode_collapse": float("nan")}
        flat = images.reshape(images.shape[0], -1)
        contrast = flat.std(axis=1)
        mean_contrast = float(np.mean(contrast))
        collapsed = float(mean_contrast < 0.03)
        return {"random_decode_contrast": mean_contrast, "random_decode_collapse": collapsed}

    def generate_image_grids(
        specs: list[CheckpointSpec],
        *,
        config_path: str,
        original_model: str,
        original_subfolder: str,
        dataset: PosteriorImageDataset,
        output_root: Path,
        normalization_mode: str,
        device: torch.device,
        dtype: Optional[torch.dtype],
        random_grid_count: int,
        interp_pairs: int,
        interp_steps: int,
    ) -> tuple[dict[str, dict[str, float]], list[str]]:
        output_root.mkdir(parents=True, exist_ok=True)
        loader = DataLoader(dataset, batch_size=min(8, max(1, len(dataset))), shuffle=False, num_workers=0)
        first_batch = next(iter(loader))
        x0 = first_batch["pixel_values"].to(device)
        grid_rows = []
        health: dict[str, dict[str, float]] = {}

        for spec in specs:
            try:
                if spec.role == "baseline":
                    vae = load_original_vae(original_model, original_subfolder, device, dtype)
                else:
                    vae = load_finetuned_vae(config_path, spec.path, device, dtype)
                x, _mu, _sigma, _sample, mode = encode_batch(vae, x0, device, dtype)
                recon = decode_scaled_latents(vae, mode, device, dtype)
                recon = _match_channel_count(recon, int(x.shape[1]))
                display_pair = torch.cat([x[:4].detach().cpu(), recon[:4].detach().cpu()], dim=0)
                pair_arr = tensor_to_display_array(display_pair, normalization_mode)
                path = save_grid(pair_arr, output_root / f"reconstruction_{spec.label}.png", nrow=4)
                grid_rows.append(str(path))

                if random_grid_count > 0:
                    latent_shape = tuple(mode.shape[1:])
                    z = torch.randn((int(random_grid_count), *latent_shape), device=device, dtype=mode.dtype)
                    decoded = decode_scaled_latents(vae, z, device, dtype)
                    decoded = _match_channel_count(decoded, int(x.shape[1]))
                    rand_arr = tensor_to_display_array(decoded.detach().cpu(), normalization_mode)
                    path = save_grid(rand_arr, output_root / f"random_decode_{spec.label}.png", nrow=4)
                    grid_rows.append(str(path))
                    health[spec.label] = compute_random_decode_health(rand_arr)

                if interp_pairs > 0 and len(dataset) >= 2:
                    imgs = []
                    pair_count = min(int(interp_pairs), len(dataset) // 2)
                    for pair_idx in range(pair_count):
                        a = dataset[pair_idx * 2]["pixel_values"].unsqueeze(0)
                        b = dataset[pair_idx * 2 + 1]["pixel_values"].unsqueeze(0)
                        xa, _mua, _sa, _za, za = encode_batch(vae, a, device, dtype)
                        _xb, _mub, _sb, _zb, zb = encode_batch(vae, b, device, dtype)
                        for alpha in np.linspace(0.0, 1.0, int(interp_steps)):
                            zi = (1.0 - float(alpha)) * za + float(alpha) * zb
                            dec = decode_scaled_latents(vae, zi, device, dtype)
                            dec = _match_channel_count(dec, int(xa.shape[1]))
                            imgs.append(dec.detach().cpu())
                    if imgs:
                        interp_tensor = torch.cat(imgs, dim=0)
                        interp_arr = tensor_to_display_array(interp_tensor, normalization_mode)
                        path = save_grid(interp_arr, output_root / f"interpolation_{spec.label}.png", nrow=int(interp_steps))
                        grid_rows.append(str(path))
                del vae
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    health[spec.label] = {"random_decode_contrast": float("nan"), "random_decode_collapse": float("nan")}
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    grid_rows.append(f"{spec.label}: CUDA OOM during grid generation")
                else:
                    raise
        return health, grid_rows

    def export_report(
        summary_df: pd.DataFrame,
        output_root: Path,
        config: dict[str, Any],
        grid_paths: list[str],
    ) -> dict[str, str]:
        output_root.mkdir(parents=True, exist_ok=True)
        csv_path = output_root / "summary.csv"
        json_path = output_root / "summary.json"
        md_path = output_root / "research_report.md"
        summary_df.to_csv(csv_path, index=False)
        json_path.write_text(
            json.dumps(
                {"config": config, "summary": summary_df.replace({np.nan: None}).to_dict(orient="records")},
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
        ranked = summary_df[summary_df["role"] != "baseline"].sort_values("fm_readiness_score", ascending=False)
        best = ranked.iloc[0] if not ranked.empty else None
        lines = [
            "# VAE Posterior Analysis Report",
            "",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Conclusion",
        ]
        if best is not None:
            lines.extend(
                [
                    f"- Most likely checkpoint for downstream latent Flow Matching: **{best['checkpoint']}**.",
                    f"- Flow Matching readiness score: {best['fm_readiness_score']:.2f}.",
                    f"- Reasoning: {best['fm_readiness_reason']}",
                ]
            )
        else:
            lines.append("- No finetuned checkpoint was available to rank.")
        if "sigma_mean" in summary_df:
            sigma_min = summary_df["sigma_mean"].min()
            lines.append(f"- Lowest average posterior sigma: {sigma_min:.6g}. Very small values indicate deterministic collapse risk.")
        if "baseline_distance" in summary_df:
            drift = summary_df.loc[summary_df["role"] != "baseline", "baseline_distance"].max()
            lines.append(f"- Maximum baseline latent distribution distance: {drift:.6g}.")
        lines.extend(
            [
                "",
                "## Summary Table",
                "",
                summary_df.to_markdown(index=False),
                "",
                "## Saved Grids",
                "",
            ]
        )
        if grid_paths:
            lines.extend(f"- {path}" for path in grid_paths)
        else:
            lines.append("- No image grids were generated.")
        lines.extend(
            [
                "",
                "## Method",
                "",
                "- VAEs were loaded with Diffusers AutoencoderKL, matching the training architecture.",
                "- Posterior statistics use SD latent scaling for downstream latent Flow Matching relevance.",
                "- KL is decomposed into mean and sigma contributions.",
                "- Percentiles and distribution tests are computed from capped samples when needed.",
            ]
        )
        md_path.write_text("\n".join(lines), encoding="utf-8")
        return {"csv": str(csv_path), "json": str(json_path), "markdown": str(md_path)}

    return (
        CheckpointSpec,
        PosteriorImageDataset,
        analyze_checkpoint,
        arrays_to_distribution_frame,
        cache_key,
        checkpoint_signature,
        compute_baseline_deltas,
        compute_embeddings,
        export_report,
        generate_image_grids,
        parse_checkpoint_spec,
        readiness_score,
        resolve_device,
        resolve_dtype,
    )


@app.cell
def _(
    CheckpointSpec,
    Path,
    batch_size_ui,
    cache_key,
    cache_root_ui,
    checkpoint_signature,
    checkpoint_spec_ui,
    dataset_root_ui,
    device_ui,
    enable_lpips_ui,
    enable_umap_ui,
    feature_cap_ui,
    finetuned_config_ui,
    flat_sample_cap_ui,
    force_recompute_ui,
    image_size_ui,
    interp_pairs_ui,
    interp_steps_ui,
    latent_vector_cap_ui,
    max_images_ui,
    normalization_ui,
    original_model_ui,
    original_subfolder_ui,
    output_root_ui,
    parse_checkpoint_spec,
    pd,
    precision_ui,
    random_grid_ui,
    save_grids_ui,
    shapiro_cap_ui,
    use_cache_ui,
):
    finetuned_specs = parse_checkpoint_spec(checkpoint_spec_ui.value)
    all_specs = [
        CheckpointSpec(label="sd15_vae", path=original_model_ui.value, role="baseline"),
        *finetuned_specs,
    ]
    analysis_config = {
        "dataset_root": str(Path(dataset_root_ui.value).expanduser()),
        "normalization_mode": normalization_ui.value,
        "image_size": int(image_size_ui.value),
        "batch_size": int(batch_size_ui.value),
        "max_images": int(max_images_ui.value),
        "num_workers": 0,
        "original_model": original_model_ui.value,
        "original_subfolder": original_subfolder_ui.value,
        "finetuned_config": str(Path(finetuned_config_ui.value).expanduser()),
        "device": device_ui.value,
        "precision": precision_ui.value,
        "cache_root": str(Path(cache_root_ui.value).expanduser()),
        "output_root": str(Path(output_root_ui.value).expanduser()),
        "use_cache": bool(use_cache_ui.value),
        "force_recompute": bool(force_recompute_ui.value),
        "flat_sample_cap": int(flat_sample_cap_ui.value),
        "latent_vector_cap": int(latent_vector_cap_ui.value),
        "feature_cap": int(feature_cap_ui.value),
        "shapiro_cap": int(shapiro_cap_ui.value),
        "random_grid_count": int(random_grid_ui.value),
        "interp_pairs": int(interp_pairs_ui.value),
        "interp_steps": int(interp_steps_ui.value),
        "enable_lpips": bool(enable_lpips_ui.value),
        "enable_umap": bool(enable_umap_ui.value),
        "save_grids": bool(save_grids_ui.value),
        "checkpoints": checkpoint_signature(all_specs),
    }
    key = cache_key(analysis_config)
    checkpoint_table = pd.DataFrame(analysis_config["checkpoints"])
    return all_specs, analysis_config, checkpoint_table, key


@app.cell
def _(checkpoint_table, key, mo):
    mo.vstack(
        [
            mo.md("## Resolved run configuration"),
            mo.md(f"Cache key: `{key}`"),
            mo.ui.table(checkpoint_table),
            mo.md(
                "Missing finetuned checkpoint paths will be reported as errors during analysis. "
                "The baseline row is a HuggingFace model id/path and does not need to exist locally."
            ),
        ]
    )
    return


@app.cell
def _(
    Any,
    DataLoader,
    NOTEBOOK_VERSION,
    Path,
    PosteriorImageDataset,
    all_specs,
    analysis_config,
    analyze_checkpoint,
    compute_baseline_deltas,
    compute_embeddings,
    generate_image_grids,
    key,
    mo,
    pd,
    readiness_score,
    resolve_device,
    resolve_dtype,
    run_analysis_ui,
    torch,
    traceback,
):
    if not run_analysis_ui.value:
        analysis_state: dict[str, Any] = {
            "ready": False,
            "message": "Press 'Run posterior analysis' to compute statistics.",
            "results": {},
            "summary": pd.DataFrame(),
            "pca": pd.DataFrame(),
            "umap": pd.DataFrame(),
            "grid_paths": [],
        }
    else:
        cache_root = Path(analysis_config["cache_root"])
        cache_root.mkdir(parents=True, exist_ok=True)
        cache_path = cache_root / f"{key}.pt"
        if analysis_config["use_cache"] and cache_path.exists() and not analysis_config["force_recompute"]:
            analysis_state = torch.load(cache_path, map_location="cpu", weights_only=False)
            analysis_state["message"] = f"Loaded cached analysis from {cache_path}"
        else:
            device = resolve_device(analysis_config["device"])
            dtype = resolve_dtype(analysis_config["precision"], device)
            dataset = PosteriorImageDataset(
                analysis_config["dataset_root"],
                analysis_config["image_size"],
                analysis_config["normalization_mode"],
                analysis_config["max_images"],
            )
            loader = DataLoader(
                dataset,
                batch_size=analysis_config["batch_size"],
                shuffle=False,
                num_workers=0,
                pin_memory=device.type == "cuda",
            )
            results = {}
            summaries = []
            for spec in all_specs:
                try:
                    if spec.role != "baseline" and not Path(spec.path).exists():
                        raise FileNotFoundError(f"Checkpoint does not exist: {spec.path}")
                    result = analyze_checkpoint(
                        spec,
                        config_path=analysis_config["finetuned_config"],
                        original_model=analysis_config["original_model"],
                        original_subfolder=analysis_config["original_subfolder"],
                        loader=loader,
                        device=device,
                        dtype=dtype,
                        flat_sample_cap=analysis_config["flat_sample_cap"],
                        latent_vector_cap=analysis_config["latent_vector_cap"],
                        feature_cap=analysis_config["feature_cap"],
                        shapiro_cap=analysis_config["shapiro_cap"],
                        enable_lpips=analysis_config["enable_lpips"],
                    )
                except RuntimeError as exc:
                    if "out of memory" in str(exc).lower():
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                        result = {
                            "summary": {
                                "checkpoint": spec.label,
                                "role": spec.role,
                                "path": spec.path,
                                "error": f"CUDA OOM: {exc}",
                            },
                            "arrays": {},
                        }
                    else:
                        result = {
                            "summary": {
                                "checkpoint": spec.label,
                                "role": spec.role,
                                "path": spec.path,
                                "error": traceback.format_exc(limit=4),
                            },
                            "arrays": {},
                        }
                except Exception:
                    result = {
                        "summary": {
                            "checkpoint": spec.label,
                            "role": spec.role,
                            "path": spec.path,
                            "error": traceback.format_exc(limit=4),
                        },
                        "arrays": {},
                    }
                results[spec.label] = result
                summaries.append(result["summary"])

            summary_df = pd.DataFrame(summaries)
            summary_df = compute_baseline_deltas(summary_df)

            grid_paths = []
            if analysis_config["save_grids"]:
                try:
                    health, grid_paths = generate_image_grids(
                        all_specs,
                        config_path=analysis_config["finetuned_config"],
                        original_model=analysis_config["original_model"],
                        original_subfolder=analysis_config["original_subfolder"],
                        dataset=dataset,
                        output_root=Path(analysis_config["output_root"]) / "grids",
                        normalization_mode=analysis_config["normalization_mode"],
                        device=device,
                        dtype=dtype,
                        random_grid_count=analysis_config["random_grid_count"],
                        interp_pairs=analysis_config["interp_pairs"],
                        interp_steps=analysis_config["interp_steps"],
                    )
                    for label, row in health.items():
                        mask = summary_df["checkpoint"] == label
                        for col, value in row.items():
                            summary_df.loc[mask, col] = value
                except Exception as exc:
                    grid_paths = [f"Grid generation failed: {type(exc).__name__}: {exc}"]

            summary_df = readiness_score(summary_df)
            pca_df, umap_df = compute_embeddings(
                {label: result for label, result in results.items() if result.get("arrays", {}).get("latent_vectors") is not None},
                analysis_config["enable_umap"],
            )
            analysis_state = {
                "ready": True,
                "version": NOTEBOOK_VERSION,
                "config": analysis_config,
                "results": results,
                "summary": summary_df,
                "pca": pca_df,
                "umap": umap_df,
                "grid_paths": grid_paths,
                "message": f"Computed analysis for {len(results)} checkpoints.",
            }
            if analysis_config["use_cache"]:
                torch.save(analysis_state, cache_path)
                analysis_state["message"] += f" Cached to {cache_path}."
    mo.md(analysis_state["message"])
    return (analysis_state,)


@app.cell
def _(analysis_state: "dict[str, Any]", mo):
    summary_df = analysis_state["summary"]
    if summary_df.empty:
        mo.md("No summary table yet.")
    else:
        visible_cols = [
            col
            for col in [
                "checkpoint",
                "role",
                "error",
                "num_images",
                "mu_mean",
                "mu_std",
                "mu_abs_mean",
                "sigma_mean",
                "sigma_std",
                "z_mean",
                "z_std",
                "z_abs_mean",
                "kl_total_mean",
                "kl_mu_mean",
                "kl_sigma_mean",
                "baseline_distance",
                "recon_mse_mean",
                "recon_l1_mean",
                "recon_psnr_mean",
                "recon_ssim_mean",
                "recon_lpips_mean",
                "random_decode_contrast",
                "random_decode_collapse",
                "fm_readiness_score",
                "fm_readiness_reason",
            ]
            if col in summary_df.columns
        ]
        mo.vstack([mo.md("## Summary and Flow Matching readiness"), mo.ui.table(summary_df[visible_cols])])
    return (summary_df,)


@app.cell
def _(
    analysis_state: "dict[str, Any]",
    arrays_to_distribution_frame,
    mo,
    pd,
    plotly_express,
):
    dist_df = pd.DataFrame()
    stat_selector = None
    if not analysis_state["ready"] or plotly_express is None:
        mo.md("Interactive Plotly distribution plots are unavailable until analysis is ready and Plotly is installed.")
    else:
        dist_frames = []
        for key_name in ["mu", "sigma", "z", "latent_norm"]:
            frame = arrays_to_distribution_frame(analysis_state["results"], key_name)
            dist_frames.append(frame)
        dist_df = pd.concat(dist_frames, ignore_index=True)
        stat_selector = mo.ui.dropdown(options=sorted(dist_df["stat"].unique()), value="mu", label="Distribution")
        mo.vstack([mo.md("## Distribution diagnostics"), stat_selector])
    return dist_df, stat_selector


@app.cell
def _(dist_df, mo, plotly_express, stat_selector):
    if stat_selector is None or dist_df.empty or plotly_express is None:
        mo.md("No distribution plot selected.")
    else:
        subset = dist_df[dist_df["stat"] == stat_selector.value]
        fig = plotly_express.histogram(
            subset,
            x="value",
            color="checkpoint",
            barmode="overlay",
            nbins=120,
            opacity=0.55,
            marginal="box",
            title=f"{stat_selector.value} distribution by checkpoint",
        )
        fig.update_layout(height=520)
        mo.ui.plotly(fig)
    return


@app.cell
def _(mo, plotly_express, summary_df):
    if summary_df.empty or plotly_express is None:
        mo.md("No trend plots available.")
    else:
        trend_cols = [
            col
            for col in [
                "mu_abs_mean",
                "sigma_mean",
                "z_std",
                "kl_total_mean",
                "kl_mu_mean",
                "kl_sigma_mean",
                "baseline_distance",
                "recon_mse_mean",
                "recon_ssim_mean",
                "fm_readiness_score",
            ]
            if col in summary_df.columns
        ]
        if not trend_cols:
            mo.md("Summary does not contain plottable trend columns.")
        else:
            trend_long = summary_df.melt(
                id_vars=["checkpoint", "role"],
                value_vars=trend_cols,
                var_name="metric",
                value_name="value",
            )
            fig = plotly_express.line(
                trend_long,
                x="checkpoint",
                y="value",
                color="metric",
                markers=True,
                title="Checkpoint trends",
            )
            fig.update_layout(height=520)
            mo.vstack([mo.md("## Checkpoint trends"), mo.ui.plotly(fig)])
    return


@app.cell
def _(analysis_state: "dict[str, Any]", mo, plotly_go, scipy_stats):
    qq_selector = None
    if not analysis_state["ready"] or plotly_go is None or scipy_stats is None:
        mo.md("QQ plots require completed analysis, Plotly, and SciPy.")
    else:
        labels = [label for label, result in analysis_state["results"].items() if "z" in result.get("arrays", {})]
        qq_selector = mo.ui.dropdown(options=labels, value=labels[0] if labels else None, label="QQ checkpoint")
        mo.vstack([mo.md("## Gaussianity QQ plot"), qq_selector])
    return (qq_selector,)


@app.cell
def _(
    analysis_state: "dict[str, Any]",
    mo,
    np,
    plotly_go,
    qq_selector,
    scipy_stats,
):
    if qq_selector is None or qq_selector.value is None or plotly_go is None or scipy_stats is None:
        mo.md("No QQ plot selected.")
    else:
        values = np.asarray(analysis_state["results"][qq_selector.value]["arrays"]["z"], dtype=np.float64)
        values = values[np.isfinite(values)]
        values = np.sort(values)
        if values.size > 5000:
            idx = np.linspace(0, values.size - 1, 5000).astype(int)
            values = values[idx]
        probs = (np.arange(values.size) + 0.5) / max(values.size, 1)
        normal_q = scipy_stats.norm.ppf(probs)
        fig = plotly_go.Figure()
        fig.add_trace(plotly_go.Scatter(x=normal_q, y=values, mode="markers", name=qq_selector.value))
        lo = float(min(normal_q.min(), values.min()))
        hi = float(max(normal_q.max(), values.max()))
        fig.add_trace(plotly_go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="N(0,1)"))
        fig.update_layout(title=f"QQ plot for z: {qq_selector.value}", xaxis_title="Normal quantile", yaxis_title="Observed z quantile", height=520)
        mo.ui.plotly(fig)
    return


@app.cell
def _(analysis_state: "dict[str, Any]", mo, plotly_express):
    pca_df = analysis_state["pca"]
    umap_df = analysis_state["umap"]
    if pca_df.empty or plotly_express is None:
        mo.md("PCA/UMAP plots are unavailable until analysis is ready and scikit-learn/Plotly are installed.")
    else:
        fig2 = plotly_express.scatter(
            pca_df,
            x="pc1",
            y="pc2",
            color="checkpoint",
            title="2D PCA of latent vectors",
        )
        fig2.update_layout(height=520)
        fig3 = plotly_express.scatter_3d(
            pca_df,
            x="pc1",
            y="pc2",
            z="pc3",
            color="checkpoint",
            title="3D PCA of latent vectors",
        )
        fig3.update_layout(height=620)
        plots = [mo.ui.plotly(fig2), mo.ui.plotly(fig3)]
        if not umap_df.empty:
            figu = plotly_express.scatter(
                umap_df,
                x="umap1",
                y="umap2",
                color="checkpoint",
                title="UMAP of latent vectors",
            )
            figu.update_layout(height=520)
            plots.append(mo.ui.plotly(figu))
        mo.vstack([mo.md("## Latent manifold analysis"), *plots])
    return


@app.cell
def _(analysis_state: "dict[str, Any]", mo):
    grid_paths = analysis_state.get("grid_paths", [])
    if not grid_paths:
        mo.md("No image grids saved yet.")
    else:
        mo.vstack([mo.md("## Saved image grids"), mo.md("\n".join(f"- `{path}`" for path in grid_paths))])
    return


@app.cell
def _(
    Path,
    analysis_config,
    analysis_state: "dict[str, Any]",
    export_report,
    export_ui,
    mo,
):
    if not export_ui.value:
        mo.md("Press 'Export summary/report' after analysis to write CSV, JSON, and Markdown outputs.")
    elif not analysis_state["ready"]:
        mo.md("Run analysis before exporting.")
    else:
        paths = export_report(
            analysis_state["summary"],
            Path(analysis_config["output_root"]),
            analysis_config,
            analysis_state.get("grid_paths", []),
        )
        mo.md(
            "## Exported outputs\n\n"
            + "\n".join(f"- **{name}**: `{path}`" for name, path in paths.items())
        )
    return


if __name__ == "__main__":
    app.run()

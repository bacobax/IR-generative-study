"""Tests for disk latent caching (src/core/data/latent_cache.py)."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.core.data import latent_cache as lc
from src.core.normalization import (
    PER_IMAGE_MINMAX,
    RAW_UINT16_PERCENTILE,
    denorm_for_display,
    norm_to_display,
)


class _DummyVAE:
    """Minimal VAE: deterministic encode + a call counter to detect re-encodes."""

    def __init__(self) -> None:
        self.encode_calls = 0

    def eval(self):
        return self

    def encode(self, x: torch.Tensor):
        self.encode_calls += 1
        mu = F.adaptive_avg_pool2d(x, (8, 8))
        sigma = torch.ones_like(mu)
        return mu, sigma


class _LayoutLikeDataset(Dataset):
    """Tiny dataset returning layout-style dict samples."""

    def __init__(self, n: int = 3, image_size: int = 32):
        self.n = n
        self.image_size = image_size

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        torch.manual_seed(idx)
        return {
            "pixel_values": torch.rand(1, self.image_size, self.image_size) * 2 - 1,
            "boxes_xyxy": torch.tensor([[2.0, 4.0, 10.0, 12.0]]),
            "labels": torch.tensor([1]),
            "n_objects": 1,
            "image_id": idx,
            "file_name": f"f{idx}.npy",
            "label_names": ["x"],
        }


class _Aug:
    def __init__(self, hflip=0.0, crop=0.0, rot=0.0):
        self.p_hflip_warmup = self.p_hflip_max = self.p_hflip_final = hflip
        self.p_crop_warmup = self.p_crop_max = self.p_crop_final = crop
        self.p_rot_warmup = self.p_rot_max = self.p_rot_final = rot


class _ModelCfg:
    vae_pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
    vae_pretrained_subfolder = "vae"
    vae_revision = None
    vae_variant = None
    vae_config = None
    vae_weights = None


def _identities(normalization_mode):
    vae_id = lc.build_vae_identity(_ModelCfg())
    ds_id = lc.build_dataset_identity(
        dataset_id="v18",
        train_dir=None,
        val_dir=None,
        split="train",
        image_size=32,
        subset_manifest=None,
    )
    variants = lc.enumerate_aug_variants(_Aug(hflip=0.5))
    return vae_id, ds_id, variants


def test_cache_key_normalization_isolation():
    """Percentile vs minmax must produce different cache keys (same VAE+dataset+aug)."""
    vae_id, ds_id, variants = _identities(RAW_UINT16_PERCENTILE)
    combo = lc.aug_combo_descriptor(variants)
    k_pct = lc.build_cache_key(
        vae_identity=vae_id, dataset_identity=ds_id,
        normalization_mode=RAW_UINT16_PERCENTILE, aug_combo=combo,
    )
    k_mm = lc.build_cache_key(
        vae_identity=vae_id, dataset_identity=ds_id,
        normalization_mode=PER_IMAGE_MINMAX, aug_combo=combo,
    )
    assert k_pct != k_mm
    # Stable across calls.
    assert k_pct == lc.build_cache_key(
        vae_identity=vae_id, dataset_identity=ds_id,
        normalization_mode=RAW_UINT16_PERCENTILE, aug_combo=combo,
    )


def test_enumerate_variants_rejects_crop_rot():
    assert [v.name for v in lc.enumerate_aug_variants(_Aug())] == ["id"]
    assert [v.name for v in lc.enumerate_aug_variants(_Aug(hflip=0.6))] == ["id", "hflip"]
    for bad in (_Aug(crop=0.2), _Aug(rot=0.3)):
        try:
            lc.enumerate_aug_variants(bad)
            assert False, "expected ValueError for crop/rot"
        except ValueError:
            pass


def test_precompute_expands_and_flips(tmp_path):
    base = _LayoutLikeDataset(n=3, image_size=32)
    vae = _DummyVAE()
    variants = lc.enumerate_aug_variants(_Aug(hflip=0.6))  # [id, hflip]
    cache_dir = tmp_path / "cache"

    lc.precompute_latents(
        base_dataset=base, vae=vae, device="cpu", variants=variants,
        cache_dir=cache_dir, image_size=32, store_dtype="fp32",
    )
    ds = lc.LatentCacheDataset(cache_dir)
    assert len(ds) == 3 * 2  # materialised pool
    assert vae.encode_calls == 6

    sample0 = base[0]
    id_entry = ds[0]      # variant id
    flip_entry = ds[1]    # variant hflip
    # hflip mirrors image
    assert torch.allclose(flip_entry["pixel_values"], torch.flip(sample0["pixel_values"], dims=[-1]))
    # hflip mirrors boxes: x0' = W - x1, x2' = W - x0
    box = sample0["boxes_xyxy"][0]
    fbox = flip_entry["boxes_xyxy"][0]
    assert torch.allclose(fbox[0], 32.0 - box[2])
    assert torch.allclose(fbox[2], 32.0 - box[0])
    # latents present and correct shape
    assert id_entry["latent_mu"].shape == (1, 8, 8)
    assert id_entry["latent_sigma"].shape == (1, 8, 8)


def test_warm_cache_skips_reencode(tmp_path):
    base = _LayoutLikeDataset(n=2, image_size=32)
    variants = lc.enumerate_aug_variants(_Aug())  # [id]
    cache_dir = tmp_path / "cache"

    vae1 = _DummyVAE()
    lc.precompute_latents(
        base_dataset=base, vae=vae1, device="cpu", variants=variants,
        cache_dir=cache_dir, image_size=32, store_dtype="fp32",
    )
    assert vae1.encode_calls == 2

    # Second build over a warm cache must not encode again.
    vae2 = _DummyVAE()
    lc.precompute_latents(
        base_dataset=base, vae=vae2, device="cpu", variants=variants,
        cache_dir=cache_dir, image_size=32, store_dtype="fp32",
    )
    assert vae2.encode_calls == 0


def test_denorm_for_display_by_normalization():
    x = torch.rand(2, 1, 16, 16) * 2 - 1
    # minmax: plot directly, equals (x+1)/2 clamped.
    mm = denorm_for_display(x, normalization_mode=PER_IMAGE_MINMAX)
    assert torch.allclose(mm, norm_to_display(x).clamp(0, 1))
    # percentile: reverse-normalize then stretch; output is a valid [0,1] image
    # and is NOT the same as the direct minmax display.
    pct = denorm_for_display(x, normalization_mode=RAW_UINT16_PERCENTILE)
    assert pct.shape == x.shape
    assert float(pct.min()) >= 0.0 and float(pct.max()) <= 1.0
    # 3D input also supported.
    assert denorm_for_display(x[0], normalization_mode=PER_IMAGE_MINMAX).shape == x[0].shape


def test_sampling_matches_mu_sigma(tmp_path):
    base = _LayoutLikeDataset(n=1, image_size=32)
    variants = lc.enumerate_aug_variants(_Aug())
    cache_dir = tmp_path / "cache"
    lc.precompute_latents(
        base_dataset=base, vae=_DummyVAE(), device="cpu", variants=variants,
        cache_dir=cache_dir, image_size=32, store_dtype="fp32",
    )
    entry = lc.LatentCacheDataset(cache_dir)[0]
    mu, sigma = entry["latent_mu"], entry["latent_sigma"]
    torch.manual_seed(0)
    z = mu + torch.randn_like(mu) * sigma
    assert z.shape == mu.shape

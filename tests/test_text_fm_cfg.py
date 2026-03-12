"""Smoke tests for the text-conditioned FM pipeline with CFG.

Validates that:
1. Text-conditioned UNet builds and runs a forward pass
2. TextConditioner encodes text and applies conditioning dropout
3. TextFMTrainer runs one training step with gradients
4. CFGFlowMatchingSampler runs CFG sampling at all guidance scales
5. Config system loads YAML presets and merges CLI overrides
6. Existing unconditional FM path still works (backward compat)
7. TextImageDataset loads text+image pairs

Run with:
    python -m pytest tests/test_text_fm_cfg.py -v
    # or:
    python tests/test_text_fm_cfg.py
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


# ── Mock text encoder/tokenizer (avoids downloading CLIP weights) ────────

class _MockEncoder:
    """Mimics CLIPTextModel interface with tiny random weights."""

    class config:
        hidden_size = 32

    def __call__(self, input_ids, attention_mask):
        class _Out:
            last_hidden_state = torch.randn(
                input_ids.shape[0], input_ids.shape[1], 32,
            )
        return _Out()

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):
        return iter([])


class _MockTokenizer:
    """Mimics CLIPTokenizer interface."""

    def __call__(self, texts, **kw):
        B = len(texts)
        L = kw.get("max_length", 10)
        return {
            "input_ids": torch.ones(B, L, dtype=torch.long),
            "attention_mask": torch.ones(B, L, dtype=torch.long),
        }


def _mock_conditioner(cond_drop_prob: float = 0.0):
    from src.conditioning.text_conditioner import TextConditioner

    cond = TextConditioner.__new__(TextConditioner)
    cond.cond_drop_prob = cond_drop_prob
    cond.max_length = 10
    cond.device = "cpu"
    cond._null_embedding = None
    cond.encoder_name = "mock"
    cond.text_encoder = _MockEncoder()
    cond.tokenizer = _MockTokenizer()
    return cond


def _small_unet():
    from src.models.fm_text_unet import build_text_fm_unet

    return build_text_fm_unet(
        {
            "sample_size": 16,
            "in_channels": 1,
            "out_channels": 1,
            "block_out_channels": [32, 64],
            "down_block_types": ["CrossAttnDownBlock2D", "DownBlock2D"],
            "up_block_types": ["UpBlock2D", "CrossAttnUpBlock2D"],
            "mid_block_type": "UNetMidBlock2DCrossAttn",
            "layers_per_block": 1,
            "cross_attention_dim": 32,
            "attention_head_dim": 8,
            "norm_num_groups": 16,
        },
        device="cpu",
    )


# ════════════════════════════════════════════════════════════════════════════
# Tests
# ════════════════════════════════════════════════════════════════════════════

def test_text_unet_builds():
    unet = _small_unet()
    x = torch.randn(2, 1, 16, 16)
    t = torch.tensor([0.5, 0.3])
    enc_hs = torch.randn(2, 5, 32)
    out = unet(x, t, encoder_hidden_states=enc_hs).sample
    assert out.shape == (2, 1, 16, 16)


def test_text_conditioner():
    cond = _mock_conditioner(cond_drop_prob=0.5)

    emb = cond.encode_text(["hello", "world"], "cpu")
    assert emb.shape == (2, 10, 32)

    null = cond.null_embedding(4, "cpu")
    assert null.shape == (4, 10, 32)

    batch = {"text": ["a", "b"], "pixel_values": torch.randn(2, 1, 16, 16)}
    kw = cond.prepare_for_training(batch, "cpu")
    assert "encoder_hidden_states" in kw
    assert kw["encoder_hidden_states"].shape == (2, 10, 32)

    kw_s = cond.prepare_for_sampling(3, "cpu")
    assert kw_s["encoder_hidden_states"].shape == (3, 10, 32)

    c, u = cond.prepare_cfg_pair(["a", "b"], "cpu")
    assert c["encoder_hidden_states"].shape == (2, 10, 32)
    assert u["encoder_hidden_states"].shape == (2, 10, 32)


def test_trainer_step():
    from src.algorithms.training.text_fm_trainer import TextFMTrainer

    unet = _small_unet()
    cond = _mock_conditioner(cond_drop_prob=0.1)
    trainer = TextFMTrainer(
        unet,
        conditioner=cond,
        device="cpu",
        t_scale=1.0,
        model_dir="/tmp/test_text_fm_trainer",
    )
    x_fm = torch.randn(2, 1, 16, 16)
    batch = {"text": ["a", "b"], "pixel_values": torch.randn(2, 1, 16, 16)}
    cond_kw = cond.prepare_for_training(batch, "cpu")
    loss = trainer.flow_matching_step(x_fm, cond_kw)
    assert loss.item() > 0
    loss.backward()
    grad_norms = [p.grad.norm().item() for p in unet.parameters() if p.grad is not None]
    assert sum(grad_norms) > 0


def test_cfg_sampler():
    from src.algorithms.inference.cfg_flow_matching_sampler import CFGFlowMatchingSampler

    unet = _small_unet()
    cond = _mock_conditioner()
    sampler = CFGFlowMatchingSampler(
        unet, conditioner=cond, device="cpu", t_scale=1.0,
    )
    shape = (1, 16, 16)

    # Unconditional (inherited)
    z = sampler.sample_euler(steps=3, batch_size=2, sample_shape=shape)
    assert z.shape == (2, 1, 16, 16)

    # CFG with scale > 1
    z_cfg = sampler.sample_euler_cfg(["a", "b"], steps=3, guidance_scale=7.5, sample_shape=shape)
    assert z_cfg.shape == (2, 1, 16, 16)

    # No CFG (scale=1)
    z_1 = sampler.sample_euler_cfg(["a", "b"], steps=3, guidance_scale=1.0, sample_shape=shape)
    assert z_1.shape == (2, 1, 16, 16)

    # Fully unconditional (scale=0)
    z_0 = sampler.sample_euler_cfg(["a", "b"], steps=3, guidance_scale=0.0, sample_shape=shape)
    assert z_0.shape == (2, 1, 16, 16)


def test_config_system():
    from src.core.configs.text_fm_config import TextFMTrainConfig, TextFMSampleConfig
    from src.core.configs.config_loader import merge_config_and_cli, load_yaml
    from src.cli.train_text_fm import build_parser, _FLAT_TO_NESTED

    cfg = TextFMTrainConfig()
    assert cfg.trainer_name == "text_fm_cfg"
    assert cfg.conditioning.cond_drop_prob == 0.1

    yaml_data = load_yaml("configs/fm/train/presets/text_cfg.yaml")
    assert "conditioning" in yaml_data

    parser = build_parser()
    args = parser.parse_args(["--config", "configs/fm/train/presets/text_cfg.yaml", "--epochs", "5"])
    merged = merge_config_and_cli(TextFMTrainConfig, args.config, parser, args, flat_to_nested=_FLAT_TO_NESTED)
    assert merged.training.epochs == 5
    assert merged.conditioning.cond_drop_prob == 0.1

    scfg = TextFMSampleConfig()
    assert scfg.guidance_scale == 7.5


def test_existing_unconditional_fm():
    """Backward compat: existing unconditional FM path still works."""
    from src.models.fm_unet import build_fm_unet_from_config
    from src.algorithms.training.flow_matching_trainer import FlowMatchingTrainer
    from src.algorithms.inference.flow_matching_sampler import FlowMatchingSampler

    config = {
        "sample_size": 16, "in_channels": 1, "out_channels": 1,
        "block_out_channels": [32, 64], "layers_per_block": 1,
        "down_block_types": ["DownBlock2D", "AttnDownBlock2D"],
        "up_block_types": ["AttnUpBlock2D", "UpBlock2D"],
    }
    unet = build_fm_unet_from_config(config, device="cpu")
    trainer = FlowMatchingTrainer(unet, device="cpu", t_scale=1.0)
    loss = trainer.flow_matching_step(torch.randn(2, 1, 16, 16))
    assert loss.item() > 0

    sampler = FlowMatchingSampler(unet, device="cpu", t_scale=1.0)
    z = sampler.sample_euler(steps=3, batch_size=2, sample_shape=(1, 16, 16))
    assert z.shape == (2, 1, 16, 16)


def test_text_image_dataset():
    from src.core.data.datasets import TextImageDataset

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy data
        for i in range(3):
            np.save(os.path.join(tmpdir, f"img_{i:03d}.npy"), np.random.rand(16, 16).astype(np.float32))

        # With fallback text
        ds = TextImageDataset(tmpdir, fallback_text="test caption")
        assert len(ds) == 3
        sample = ds[0]
        assert "pixel_values" in sample
        assert "text" in sample
        assert sample["text"] == "test caption"
        assert sample["pixel_values"].shape == (1, 16, 16)

        # With JSON annotations
        annot = {"img_000": "first", "img_001": "second", "img_002": "third"}
        annot_path = os.path.join(tmpdir, "captions.json")
        with open(annot_path, "w") as f:
            json.dump(annot, f)

        ds_json = TextImageDataset(tmpdir, text_annotations=annot_path)
        s = ds_json[0]
        assert s["text"] == "first"

        # With companion .txt files
        with open(os.path.join(tmpdir, "img_000.txt"), "w") as f:
            f.write("from txt file")
        ds_txt = TextImageDataset(tmpdir)
        s_txt = ds_txt[0]
        assert s_txt["text"] == "from txt file"


def test_registry_entries():
    from src.core.registry import REGISTRIES
    # Import to register
    import src.conditioning.text_conditioner  # noqa: F401
    import src.conditioning.no_conditioner  # noqa: F401
    import src.models.fm_text_unet  # noqa: F401
    import src.models.fm_unet  # noqa: F401
    import src.algorithms.training.text_fm_trainer  # noqa: F401
    import src.algorithms.training.flow_matching_trainer  # noqa: F401
    import src.algorithms.inference.cfg_flow_matching_sampler  # noqa: F401
    import src.algorithms.inference.flow_matching_sampler  # noqa: F401

    assert "text_cfg" in REGISTRIES.conditioning
    assert "text_fm_unet" in REGISTRIES.model_builder
    assert "text_fm_cfg" in REGISTRIES.trainer
    assert "cfg_fm" in REGISTRIES.sampler

    # Existing still present
    assert "no_conditioning" in REGISTRIES.conditioning
    assert "default_unet" in REGISTRIES.model_builder
    assert "default_fm" in REGISTRIES.trainer


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tests = [
        test_text_unet_builds,
        test_text_conditioner,
        test_trainer_step,
        test_cfg_sampler,
        test_config_system,
        test_existing_unconditional_fm,
        test_text_image_dataset,
        test_registry_entries,
    ]
    for t in tests:
        print(f"Running {t.__name__} ... ", end="", flush=True)
        t()
        print("PASS")
    print(f"\nAll {len(tests)} tests passed.")

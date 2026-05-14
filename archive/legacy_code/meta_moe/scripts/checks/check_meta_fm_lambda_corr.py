#!/usr/bin/env python3
"""Smoke-check phase-dependent lambda_corr wiring for Meta FM."""

import os
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

import torch
import torch.nn as nn

from src.algorithms.training.meta_fm_trainer import MetaFMTrainer
from src.models.moe_text_unet import build_text_moe_unet

passed = 0
failed = 0


def check(label, cond):
    global passed, failed
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {label}")
    if cond:
        passed += 1
    else:
        failed += 1


class _MockEncoder:
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
    def __call__(self, texts, **kw):
        batch = len(texts)
        length = kw.get("max_length", 10)
        return {
            "input_ids": torch.ones(batch, length, dtype=torch.long),
            "attention_mask": torch.ones(batch, length, dtype=torch.long),
        }


def _mock_conditioner():
    from src.conditioning.text_conditioner import TextConditioner

    cond = TextConditioner.__new__(TextConditioner)
    cond.cond_drop_prob = 0.0
    cond.max_length = 10
    cond.device = "cpu"
    cond._null_embedding = None
    cond.encoder_name = "mock"
    cond.text_encoder = _MockEncoder()
    cond.tokenizer = _MockTokenizer()
    return cond


class _IdentityAdapter(nn.Module):
    def forward(self, hidden_state, weights=None):
        return hidden_state


class _ConstantCorrection(nn.Module):
    def forward(self, hidden_state, mixture_residual, pooled_text_embeds):
        batch, channels = hidden_state.shape[:2]
        return torch.ones(batch, channels, 1, 1, dtype=hidden_state.dtype, device=hidden_state.device)


def _small_moe_unet():
    return build_text_moe_unet(
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


print("=== A. Direct correction scaling ===")
model = _small_moe_unet()
model.mid_adapter = _IdentityAdapter()
model.gated_correction = _ConstantCorrection()

hidden = torch.zeros(2, 64, 4, 4)
model._active_router_weights = torch.ones(2, model.num_experts) / model.num_experts
model._active_pooled_text_embeds = torch.randn(2, 32)

model.set_lambda_corr(0.05)
small = model._apply_adapter_and_correction(hidden)
model.set_lambda_corr(1.0)
full = model._apply_adapter_and_correction(hidden)

check("lambda_corr=0.05 scales correction amplitude", torch.allclose(small, full * 0.05))


print("\n=== B. Trainer phase application ===")
with tempfile.TemporaryDirectory() as tmpdir:
    trainer = MetaFMTrainer(
        _small_moe_unet(),
        conditioner=_mock_conditioner(),
        device="cpu",
        t_scale=1.0,
        train_target="v",
        model_dir=tmpdir,
    )
    trainer._apply_phase_trainability(
        type(
            "PhaseCfg",
            (),
            {
                "mlp_trainable": True,
                "router_trainable": True,
                "moe_trainable": False,
                "unet_trainable": False,
                "unfreeze_unet_policy": "none",
                "lambda_corr": 0.05,
            },
        )(),
    )
    check(
        "phase config pushes lambda_corr onto model",
        torch.isclose(trainer._moe_unet().lambda_corr, torch.tensor(0.05)),
    )

    trainer._apply_phase_trainability(
        type(
            "PhaseCfg",
            (),
            {
                "mlp_trainable": True,
                "router_trainable": True,
                "moe_trainable": True,
                "unet_trainable": False,
                "unfreeze_unet_policy": "none",
                "lambda_corr": 1.0,
            },
        )(),
    )
    check(
        "later phase can fully restore correction weight",
        torch.isclose(trainer._moe_unet().lambda_corr, torch.tensor(1.0)),
    )


print("\n=== Summary ===")
print(f"  {passed}/{passed + failed} checks passed")
if failed:
    sys.exit(1)
print("  All OK!")

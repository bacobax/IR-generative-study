from __future__ import annotations

import types

import pytest
import torch

from src.algorithms.inference.cfg_flow_matching_sampler import CFGFlowMatchingSampler
from src.conditioning.expert_subset_policy import ExpertSubsetPolicy
from src.models.moe_text_unet import build_text_moe_unet


def _small_moe_unet(num_experts: int = 3):
    return build_text_moe_unet(
        {
            "UNET": {
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
            "MOE": {
                "num_experts": num_experts,
            },
        },
        device="cpu",
    )


class _MockCfgConditioner:
    def prepare_cfg_pair(self, prompts, device="cpu"):
        batch = len(prompts)
        seq = torch.randn(batch, 5, 32, device=device)
        pooled = torch.stack(
            [torch.linspace(0.1 + i, 0.9 + i, 32, device=device) for i in range(batch)],
            dim=0,
        )
        cond = {
            "encoder_hidden_states": seq,
            "pooled_text_embeds": pooled,
        }
        uncond = {
            "encoder_hidden_states": torch.zeros_like(seq),
            "pooled_text_embeds": torch.zeros_like(pooled),
        }
        return cond, uncond


def test_subset_policy_rejects_incremental_experts_not_seen_in_base():
    policy = ExpertSubsetPolicy(
        num_experts=3,
        enabled=True,
        configured_subsets={
            1: [0, 1],
            2: [2],
        },
    )

    with pytest.raises(ValueError, match="do not appear in any base subset"):
        policy.validate_training_conditions(base_conditions=[1], incremental_conditions=[2])


def test_subset_policy_uses_configured_subset_and_top1_empty_fallback():
    policy = ExpertSubsetPolicy(
        num_experts=3,
        enabled=True,
        configured_subsets={1: [0, 2]},
        unseen_policy="router_threshold",
        threshold=0.95,
        min_experts=1,
        empty_fallback="top1",
    )
    raw_weights = torch.tensor(
        [
            [0.20, 0.50, 0.30],
            [0.70, 0.20, 0.10],
        ],
        dtype=torch.float32,
    )

    masks = policy.build_masks([1, 99], raw_weights=raw_weights, device=raw_weights.device)

    assert torch.equal(masks[0], torch.tensor([True, False, True]))
    assert torch.equal(masks[1], torch.tensor([True, False, False]))


def test_moe_unet_normalizes_masked_weights():
    model = _small_moe_unet(num_experts=3)
    raw_weights = torch.tensor(
        [
            [0.2, 0.5, 0.3],
            [0.7, 0.2, 0.1],
        ],
        dtype=torch.float32,
    )
    expert_mask = torch.tensor(
        [
            [True, False, True],
            [False, True, True],
        ],
        dtype=torch.bool,
    )

    masked = model.normalize_masked_weights(raw_weights, expert_mask)

    assert torch.allclose(masked.sum(dim=-1), torch.ones(2))
    assert torch.equal(masked[:, 1] == 0, torch.tensor([True, False]))
    assert torch.equal(masked[:, 0] == 0, torch.tensor([False, True]))


def test_cfg_sampler_applies_condition_subset_to_both_cfg_branches():
    torch.manual_seed(0)
    model = _small_moe_unet(num_experts=3)
    conditioner = _MockCfgConditioner()
    subset_policy = ExpertSubsetPolicy(
        num_experts=3,
        enabled=True,
        configured_subsets={1: [0, 1]},
    )
    sampler = CFGFlowMatchingSampler(
        model,
        conditioner=conditioner,
        subset_policy=subset_policy,
        device="cpu",
        t_scale=1.0,
    )

    recorded_router_weights = []
    original_forward = model.forward

    def _wrapped_forward(self, sample, timestep, **kwargs):
        weights = kwargs.get("router_weights")
        recorded_router_weights.append(None if weights is None else weights.detach().clone())
        return original_forward(sample, timestep, **kwargs)

    model.forward = types.MethodType(_wrapped_forward, model)
    try:
        z = sampler.sample_euler_cfg(
            ["IR image with 1 person"],
            steps=1,
            guidance_scale=7.5,
            sample_shape=(1, 16, 16),
            condition_ids=[1],
        )
    finally:
        model.forward = original_forward

    assert z.shape == (1, 1, 16, 16)
    assert len(recorded_router_weights) == 2
    for weights in recorded_router_weights:
        assert weights is not None
        assert torch.allclose(weights.sum(dim=-1), torch.ones(1))
        assert torch.allclose(weights[:, 2], torch.zeros(1))


def test_cfg_sampler_reuses_unseen_subset_mask_for_null_branch():
    torch.manual_seed(0)
    model = _small_moe_unet(num_experts=3)
    conditioner = _MockCfgConditioner()
    subset_policy = ExpertSubsetPolicy(
        num_experts=3,
        enabled=True,
        configured_subsets={},
        unseen_policy="router_topk",
        top_k=1,
    )
    sampler = CFGFlowMatchingSampler(
        model,
        conditioner=conditioner,
        subset_policy=subset_policy,
        device="cpu",
        t_scale=1.0,
    )

    recorded_router_weights = []
    original_forward = model.forward
    original_compute_raw_router_weights = model.compute_raw_router_weights

    def _mock_compute_raw_router_weights(self, pooled_text_embeds):
        if torch.allclose(pooled_text_embeds, torch.zeros_like(pooled_text_embeds)):
            return torch.tensor([[0.1, 0.2, 0.7]], dtype=pooled_text_embeds.dtype)
        return torch.tensor([[0.8, 0.1, 0.1]], dtype=pooled_text_embeds.dtype)

    def _wrapped_forward(self, sample, timestep, **kwargs):
        weights = kwargs.get("router_weights")
        recorded_router_weights.append(None if weights is None else weights.detach().clone())
        return original_forward(sample, timestep, **kwargs)

    model.compute_raw_router_weights = types.MethodType(
        _mock_compute_raw_router_weights,
        model,
    )
    model.forward = types.MethodType(_wrapped_forward, model)
    try:
        z = sampler.sample_euler_cfg(
            ["IR image with 5 people"],
            steps=1,
            guidance_scale=7.5,
            sample_shape=(1, 16, 16),
            condition_ids=[5],
        )
    finally:
        model.forward = original_forward
        model.compute_raw_router_weights = original_compute_raw_router_weights

    assert z.shape == (1, 1, 16, 16)
    assert len(recorded_router_weights) == 2
    cond_mask = recorded_router_weights[0] > 0
    uncond_mask = recorded_router_weights[1] > 0
    assert torch.equal(cond_mask, torch.tensor([[True, False, False]]))
    assert torch.equal(uncond_mask, cond_mask)

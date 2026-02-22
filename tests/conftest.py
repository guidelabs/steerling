"""Shared test fixtures."""

import pytest
import torch

from steerling.configs.causal_diffusion import CausalDiffusionConfig
from steerling.configs.concept import ConceptConfig
from steerling.data.tokenizer import SteerlingTokenizer


@pytest.fixture
def tiny_config() -> CausalDiffusionConfig:
    """Minimal config for fast CPU tests (2 layers, 128 dim)."""
    return CausalDiffusionConfig(
        n_layers=2,
        n_head=4,
        n_embd=128,
        block_size=256,
        n_kv_heads=2,
        diff_block_size=16,
        use_rms_norm=True,
        norm_order="post",
        use_qk_norm=True,
        use_rope=True,
        rope_base=500000.0,
        mlp_type="swiglu",
        use_bias=False,
        clip_qkv=10.0,
        weight_sharing=True,
    )


@pytest.fixture
def tiny_concept_config() -> ConceptConfig:
    """Minimal concept config for fast CPU tests."""
    return ConceptConfig(
        n_concepts=32,
        n_unknown_concepts=64,
        concept_dim=128,
        use_attention_known=False,
        use_attention_unknown=False,
        topk_known=4,
        topk_known_features=4,
        unknown_topk=8,
        use_unknown=True,
        factorize_unknown=False,
        use_epsilon_correction=True,
        block_size=256,
        pad_multiple=16,
        apply_topk_to_unknown=True,
        inject_layer=1,
    )


@pytest.fixture
def tokenizer() -> SteerlingTokenizer:
    return SteerlingTokenizer()


@pytest.fixture
def device() -> torch.device:
    return torch.device("cpu")

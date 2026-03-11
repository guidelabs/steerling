"""
CausalDiffusionLM-specific layer implementations.

Contains:
- BlockCausalAttention: Block-causal self-attention with FlexAttention + GQA
- CausalDiffusionBlock: Transformer block (pre/post-norm)
- Block-causal mask utilities
"""

from __future__ import annotations

import logging
import os
from functools import partial
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .primitives import MLP, RMSNorm, RotaryEmbedding

logger = logging.getLogger(__name__)

# FlexAttention imports (with fallback to SDPA)
# flex_attention requires Triton compilation which may fail on some GPUs
# (e.g. insufficient shared memory). Set STEERLING_USE_FLEX_ATTN=1 to enable.
try:
    from torch.nn.attention.flex_attention import (
        BlockMask,
        _dense_to_ordered,
        flex_attention,
    )

    _FLEX_ATTN_AVAILABLE = True
except ImportError:
    _FLEX_ATTN_AVAILABLE = False
    BlockMask = None
    flex_attention = None
    _dense_to_ordered = None

if os.environ.get("STEERLING_USE_FLEX_ATTN", "0") != "1":
    _FLEX_ATTN_AVAILABLE = False

if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask as BlockMaskType

    from steerling.configs.causal_diffusion import CausalDiffusionConfig

if torch.cuda.is_available() and _FLEX_ATTN_AVAILABLE:
    compiled_flex_attention = torch.compile(flex_attention, fullgraph=True)
else:
    compiled_flex_attention = flex_attention


# Block-causal mask utilities
def block_causal_mask_mod(
    b: Any,
    h: Any,
    q_idx: torch.Tensor,
    kv_idx: torch.Tensor,
    *,
    block_size: int,
) -> torch.Tensor:
    """Block-causal mask: causal across blocks, bidirectional within blocks."""
    return q_idx // block_size >= kv_idx // block_size


def fast_create_block_causal_mask(
    attn_block_size: int,
    seq_length: int,
    mask_block_size: int,
    device: torch.device,
) -> BlockMaskType:
    """
    Fast block-causal mask creation for flex_attention.

    Analytically computes the sparse block structure instead of evaluating
    the mask function at every position.
    """

    if not _FLEX_ATTN_AVAILABLE or _dense_to_ordered is None or BlockMask is None:
        raise RuntimeError("flex_attention not available")

    num_mask_blocks = -(-seq_length // mask_block_size)
    attn_blocks_per_mask_block, rem = divmod(mask_block_size, attn_block_size)

    if rem != 0:
        raise ValueError(
            f"mask_block_size ({mask_block_size}) must be divisible by attn_block_size ({attn_block_size})"
        )

    num_attn_blocks = num_mask_blocks * attn_blocks_per_mask_block
    lowres_attn_mask = torch.tril(
        torch.ones(num_attn_blocks, num_attn_blocks, dtype=torch.bool, device=device)
    )
    block_attn_count = (
        lowres_attn_mask.reshape(
            num_mask_blocks,
            attn_blocks_per_mask_block,
            num_mask_blocks,
            attn_blocks_per_mask_block,
        )
        .permute(0, 2, 1, 3)
        .sum(dim=[-2, -1])
    )
    max_count = attn_blocks_per_mask_block * attn_blocks_per_mask_block

    full_block_mask = block_attn_count == max_count
    if seq_length % mask_block_size > 0:
        full_block_mask[-1, :] = False

    normal_block_mask = (block_attn_count > 0) & (~full_block_mask)

    kv_num_blocks, kv_indices = _dense_to_ordered(normal_block_mask)
    full_kv_num_blocks, full_kv_indices = _dense_to_ordered(full_block_mask)
    q_num_blocks, q_indices = _dense_to_ordered(normal_block_mask.transpose(-2, -1))
    full_q_num_blocks, full_q_indices = _dense_to_ordered(full_block_mask.transpose(-2, -1))

    return BlockMask(
        seq_lengths=(seq_length, seq_length),
        kv_num_blocks=kv_num_blocks[None, None, ...],
        kv_indices=kv_indices[None, None, ...],
        full_kv_num_blocks=full_kv_num_blocks[None, None, ...],
        full_kv_indices=full_kv_indices[None, None, ...],
        q_num_blocks=q_num_blocks[None, None, ...],
        q_indices=q_indices[None, None, ...],
        full_q_num_blocks=full_q_num_blocks[None, None, ...],
        full_q_indices=full_q_indices[None, None, ...],
        mask_mod=partial(block_causal_mask_mod, block_size=attn_block_size),
        BLOCK_SIZE=(mask_block_size, mask_block_size),
    )


def sdpa_with_block_causal_mask(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    diff_block_size: int,
    mask_cache: dict[str, torch.Tensor],
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Fallback using SDPA with dense mask when flex_attention unavailable."""

    B, H, T, D = q.shape
    device = q.device
    dtype = q.dtype

    cache_key = f"sdpa_{T}_{device}_{dtype}"
    if cache_key not in mask_cache:
        q_idx = torch.arange(T, device=device).unsqueeze(1)
        kv_idx = torch.arange(T, device=device).unsqueeze(0)
        bool_mask = q_idx // diff_block_size >= kv_idx // diff_block_size
        attn_mask = torch.zeros(T, T, device=device, dtype=dtype)
        attn_mask.masked_fill_(~bool_mask, float("-inf"))
        mask_cache[cache_key] = attn_mask

    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask_cache[cache_key],
        dropout_p=0.0,
        is_causal=False,
        enable_gqa=enable_gqa,
    )


# Block-causal self-attention
class BlockCausalAttention(nn.Module):
    """Block-causal self-attention with FlexAttention and optional GQA."""

    FLEX_MASK_BLOCK_SIZE = 128

    def __init__(self, config: CausalDiffusionConfig) -> None:
        super().__init__()

        if not hasattr(config, "diff_block_size"):
            raise ValueError("BlockCausalAttention requires 'diff_block_size' in config.")

        assert config.n_embd % config.n_head == 0

        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        n_kv = getattr(config, "n_kv_heads", None)
        self.n_kv_heads = self.n_head if n_kv is None else int(n_kv)

        if self.n_kv_heads <= 0:
            raise ValueError(f"n_kv_heads must be >= 1 (got {self.n_kv_heads})")
        if self.n_head % self.n_kv_heads != 0:
            raise ValueError(f"n_head ({self.n_head}) must be divisible by n_kv_heads ({self.n_kv_heads})")

        self.kv_repeat = self.n_head // self.n_kv_heads
        use_bias = getattr(config, "use_bias", False)

        kv_out = self.n_kv_heads * self.head_dim
        attn_out = self.n_embd + 2 * kv_out

        self.c_attn = nn.Linear(config.n_embd, attn_out, bias=use_bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=use_bias)
        self.c_proj.SCALE_INIT = 1

        # QK Norm
        if getattr(config, "use_qk_norm", False):
            if getattr(config, "use_rms_norm", True):
                self.q_norm: nn.Module | None = RMSNorm(config, size=self.head_dim)
                self.k_norm: nn.Module | None = RMSNorm(config, size=self.head_dim)
            else:
                self.q_norm = nn.LayerNorm(self.head_dim)
                self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        # RoPE
        if getattr(config, "use_rope", True):
            self.rope: RotaryEmbedding | None = RotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=config.block_size,
                base=getattr(config, "rope_base", 500000.0),
                rope_full_precision=getattr(config, "rope_full_precision", True),
            )
        else:
            self.rope = None

        self._mask_cache: dict = {}
        self._sdpa_mask_cache: dict[str, torch.Tensor] = {}
        self._logged_attention_mode = False

    def _get_block_mask(self, T: int, device: torch.device):
        cache_key = f"flex_{T}_{device}"
        if cache_key not in self._mask_cache:
            diff_block_size = self.config.diff_block_size
            mask_block_size = self.FLEX_MASK_BLOCK_SIZE
            if mask_block_size % diff_block_size != 0:
                mask_block_size = diff_block_size * (mask_block_size // diff_block_size)
                if mask_block_size == 0:
                    mask_block_size = diff_block_size
            self._mask_cache[cache_key] = fast_create_block_causal_mask(
                attn_block_size=diff_block_size,
                seq_length=T,
                mask_block_size=mask_block_size,
                device=device,
            )
        return self._mask_cache[cache_key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        device = x.device
        use_flex = _FLEX_ATTN_AVAILABLE and x.is_cuda and flex_attention is not None

        if not self._logged_attention_mode:
            self._logged_attention_mode = True
            mode = "flex_attention" if use_flex else "SDPA fallback"
            logger.debug(
                f"[CausalDiffusion] Using {mode} with GQA "
                f"(n_head={self.n_head}, n_kv_heads={self.n_kv_heads})"
            )

        qkv = self.c_attn(x)
        clip_qkv = getattr(self.config, "clip_qkv", None)
        if clip_qkv is not None:
            qkv = qkv.clamp(min=-clip_qkv, max=clip_qkv)

        kv_dim = self.n_kv_heads * self.head_dim
        q, k, v = qkv.split([self.n_embd, kv_dim, kv_dim], dim=2)

        q = q.reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.q_norm is not None and self.k_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)

        if self.rope is not None:
            q, k = self.rope(q, k)

        if use_flex:
            block_mask = self._get_block_mask(T, device)
            if q.is_cuda:
                y = compiled_flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)  # type: ignore
            else:
                y = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)  # type: ignore
        else:
            y = sdpa_with_block_causal_mask(
                q,
                k,
                v,
                diff_block_size=self.config.diff_block_size,
                mask_cache=self._sdpa_mask_cache,
                enable_gqa=True,
            )

        y = y.transpose(1, 2).reshape(B, T, C)  # type: ignore
        y = self.c_proj(y)
        return y


# Transformer block
class CausalDiffusionBlock(nn.Module):
    """Transformer block for CausalDiffusionLM (block-causal attention + MLP)."""

    def __init__(self, config: CausalDiffusionConfig) -> None:
        super().__init__()

        use_rms_norm = getattr(config, "use_rms_norm", True)
        if use_rms_norm:
            self.ln_1: nn.Module = RMSNorm(config)
            self.ln_2: nn.Module = RMSNorm(config)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.ln_2 = nn.LayerNorm(config.n_embd)

        self.norm_order = getattr(config, "norm_order", "post")
        self.attn = BlockCausalAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_order == "pre":
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.ln_1(self.attn(x))
            x = x + self.ln_2(self.mlp(x))
        return x

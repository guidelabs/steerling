#!/usr/bin/env python3
"""Generate HF-compatible files for Steerling.

Writes self-contained HuggingFace model files into hf/ from static templates.
These files have no dependency on the steerling package — all layers are inlined.

Usage:
    python scripts/build_hf_files.py
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "hf"

# ===================================================================
# JSON templates
# ===================================================================

CONFIG_JSON = {
    "model_type": "steerling",
    "auto_map": {
        "AutoConfig": "configuration_steerling.SteerlingConfig",
        "AutoModel": "modeling_steerling.SteerlingForCausalLM",
        "AutoModelForCausalLM": "modeling_steerling.SteerlingForCausalLM",
        "AutoTokenizer": ["tokenization_steerling.SteerlingTokenizer", None],
    },
    "architectures": ["SteerlingForCausalLM"],
    "vocab_size": 100281,
    "n_layers": 32,
    "n_head": 32,
    "n_embd": 4096,
    "n_kv_heads": 4,
    "block_size": 4096,
    "diff_block_size": 64,
    "use_rms_norm": True,
    "norm_eps": 1e-05,
    "norm_order": "post",
    "use_qk_norm": True,
    "use_rope": True,
    "rope_base": 500000.0,
    "rope_full_precision": True,
    "clip_qkv": 10.0,
    "mlp_type": "swiglu",
    "activation": "gelu",
    "mlp_ratio": 4,
    "intermediate_size": None,
    "use_bias": False,
    "weight_sharing": True,
    "pad_token_id": 100277,
    "bos_token_id": 100278,
    "eos_token_id": 100257,
    "mask_token_id": 100280,
    "endofchunk_token_id": 100279,
    "n_concepts": 33732,
    "n_unknown_concepts": 101196,
    "concept_dim": 4096,
    "use_attention_known": False,
    "use_attention_unknown": False,
    "topk_known": 16,
    "topk_known_features": 32,
    "unknown_topk": 128,
    "use_unknown": True,
    "apply_topk_to_unknown": True,
    "topk_on_logits": False,
    "factorize_unknown": True,
    "factorize_rank": 256,
    "use_epsilon_correction": True,
    "concept_block_size": 4096,
    "pad_multiple": 16,
    "store_unknown_weights": False,
    "inject_layer": 16,
    "inject_alpha": 1.0,
    "torch_dtype": "bfloat16",
    "transformers_version": "4.48.0",
}

TOKENIZER_CONFIG_JSON = {
    "tokenizer_class": "SteerlingTokenizer",
    "auto_map": {
        "AutoTokenizer": ["tokenization_steerling.SteerlingTokenizer", None],
    },
    "pad_token": "<|pad|>",
    "bos_token": "<|bos|>",
    "eos_token": "<|endoftext|>",
    "additional_special_tokens": ["<|endofchunk|>", "<|mask|>"],
    "encoding_name": "cl100k_base",
    "pad_token_id": 100277,
    "bos_token_id": 100278,
    "eos_token_id": 100257,
    "endofchunk_token_id": 100279,
    "mask_token_id": 100280,
}

# ===================================================================
# Python file templates
# ===================================================================

CONFIGURATION_TEMPLATE = '''\
"""Steerling model configuration for HuggingFace integration."""

from transformers import PretrainedConfig


class SteerlingConfig(PretrainedConfig):
    """
    Configuration for Steerling-8B: an interpretable causal diffusion language model.

    Steerling uses block-causal attention (bidirectional within blocks, causal across
    blocks) with concept decomposition heads for interpretability and steering.
    """

    model_type = "steerling"

    def __init__(
        self,
        # Architecture
        vocab_size=100281,
        n_layers=32,
        n_head=32,
        n_embd=4096,
        n_kv_heads=4,
        block_size=4096,
        diff_block_size=64,
        # Normalization
        use_rms_norm=True,
        norm_eps=1e-5,
        norm_order="post",
        # Attention
        use_qk_norm=True,
        use_rope=True,
        rope_base=500000.0,
        rope_full_precision=True,
        clip_qkv=10.0,
        # MLP
        mlp_type="swiglu",
        activation="gelu",
        mlp_ratio=4,
        intermediate_size=None,
        use_bias=False,
        # Weight sharing
        weight_sharing=True,
        # Special tokens
        mask_token_id=100280,
        endofchunk_token_id=100279,
        # Concept decomposition
        n_concepts=33732,
        n_unknown_concepts=101196,
        concept_dim=4096,
        use_attention_known=False,
        use_attention_unknown=False,
        topk_known=16,
        topk_known_features=32,
        unknown_topk=128,
        use_unknown=True,
        apply_topk_to_unknown=True,
        topk_on_logits=False,
        factorize_unknown=True,
        factorize_rank=256,
        use_epsilon_correction=True,
        concept_block_size=4096,
        pad_multiple=16,
        store_unknown_weights=False,
        # Steering
        inject_layer=16,
        inject_alpha=1.0,
        **kwargs,
    ):
        self.n_layers = n_layers
        self.n_head = n_head
        self.n_embd = n_embd
        self.n_kv_heads = n_kv_heads
        self.block_size = block_size
        self.diff_block_size = diff_block_size
        self.use_rms_norm = use_rms_norm
        self.norm_eps = norm_eps
        self.norm_order = norm_order
        self.use_qk_norm = use_qk_norm
        self.use_rope = use_rope
        self.rope_base = rope_base
        self.rope_full_precision = rope_full_precision
        self.clip_qkv = clip_qkv
        self.mlp_type = mlp_type
        self.activation = activation
        self.mlp_ratio = mlp_ratio
        self.intermediate_size = intermediate_size
        self.use_bias = use_bias
        self.weight_sharing = weight_sharing
        self.mask_token_id = mask_token_id
        self.endofchunk_token_id = endofchunk_token_id
        self.n_concepts = n_concepts
        self.n_unknown_concepts = n_unknown_concepts
        self.concept_dim = concept_dim
        self.use_attention_known = use_attention_known
        self.use_attention_unknown = use_attention_unknown
        self.topk_known = topk_known
        self.topk_known_features = topk_known_features
        self.unknown_topk = unknown_topk
        self.use_unknown = use_unknown
        self.apply_topk_to_unknown = apply_topk_to_unknown
        self.topk_on_logits = topk_on_logits
        self.factorize_unknown = factorize_unknown
        self.factorize_rank = factorize_rank
        self.use_epsilon_correction = use_epsilon_correction
        self.concept_block_size = concept_block_size
        self.pad_multiple = pad_multiple
        self.store_unknown_weights = store_unknown_weights
        self.inject_layer = inject_layer
        self.inject_alpha = inject_alpha

        super().__init__(
            vocab_size=vocab_size,
            pad_token_id=kwargs.pop("pad_token_id", 100277),
            bos_token_id=kwargs.pop("bos_token_id", 100278),
            eos_token_id=kwargs.pop("eos_token_id", 100257),
            **kwargs,
        )
'''

TOKENIZATION_TEMPLATE = '''\
from __future__ import annotations

from typing import Any

import tiktoken
from transformers import PreTrainedTokenizer


class SteerlingTokenizer(PreTrainedTokenizer):
    """
    Tokenizer for Steerling models based on tiktoken cl100k_base.

    Wraps tiktoken's cl100k_base encoding and adds four custom special tokens:
    - <|pad|> (100277): Padding token
    - <|bos|> (100278): Beginning of sequence
    - <|endofchunk|> (100279): End of chunk delimiter
    - <|mask|> (100280): Mask token for diffusion
    """

    vocab_files_names: dict[str, str] = {}
    model_input_names = ["input_ids", "attention_mask"]

    PAD_TOKEN = "<|pad|>"
    BOS_TOKEN = "<|bos|>"
    EOS_TOKEN = "<|endoftext|>"
    ENDOFCHUNK_TOKEN = "<|endofchunk|>"
    MASK_TOKEN = "<|mask|>"

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
        pad_token_id: int = 100277,
        bos_token_id: int = 100278,
        eos_token_id: int = 100257,
        endofchunk_token_id: int = 100279,
        mask_token_id: int = 100280,
        **kwargs: Any,
    ):
        base_enc = tiktoken.get_encoding(encoding_name)
        base_vocab = base_enc.n_vocab

        assert pad_token_id == base_vocab, f"pad_token_id should be {base_vocab}, got {pad_token_id}"
        assert bos_token_id == base_vocab + 1
        assert endofchunk_token_id == base_vocab + 2
        assert mask_token_id == base_vocab + 3

        self._tokenizer = tiktoken.Encoding(
            name=f"{encoding_name}_steerling",
            pat_str=base_enc._pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens={
                **base_enc._special_tokens,
                self.PAD_TOKEN: pad_token_id,
                self.BOS_TOKEN: bos_token_id,
                self.ENDOFCHUNK_TOKEN: endofchunk_token_id,
                self.MASK_TOKEN: mask_token_id,
            },
        )

        self._pad_token_id = pad_token_id
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id
        self._endofchunk_token_id = endofchunk_token_id
        self._mask_token_id = mask_token_id

        self._special_token_ids = {
            pad_token_id, bos_token_id, eos_token_id, endofchunk_token_id, mask_token_id
        }

        kwargs.pop("pad_token", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.pop("additional_special_tokens", None)

        super().__init__(
            pad_token=self.PAD_TOKEN,
            bos_token=self.BOS_TOKEN,
            eos_token=self.EOS_TOKEN,
            additional_special_tokens=[self.ENDOFCHUNK_TOKEN, self.MASK_TOKEN],
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.n_vocab

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    @property
    def endofchunk_token_id(self) -> int:
        return self._endofchunk_token_id

    def get_vocab(self) -> dict[str, int]:
        vocab = {}
        for token, idx in self._tokenizer._special_tokens.items():
            vocab[token] = idx
        return vocab

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        token_ids = self._tokenizer.encode(text, disallowed_special=())
        return [str(tid) for tid in token_ids]

    def _convert_token_to_id(self, token: str) -> int:
        special = self._tokenizer._special_tokens
        if token in special:
            return special[token]
        try:
            return int(token)
        except ValueError:
            ids = self._tokenizer.encode(token, disallowed_special=())
            return ids[0] if ids else self._pad_token_id

    def _convert_id_to_token(self, index: int) -> str:
        for name, idx in self._tokenizer._special_tokens.items():
            if idx == index:
                return name
        try:
            return self._tokenizer.decode([index])
        except Exception:
            return f"<|token_{index}|>"

    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        ids = []
        special_names = self._tokenizer._special_tokens
        for t in tokens:
            if t in special_names:
                continue
            try:
                tid = int(t)
                if tid not in self._special_token_ids:
                    ids.append(tid)
            except ValueError:
                re_encoded = self._tokenizer.encode(t, disallowed_special=())
                ids.extend(re_encoded)
        return self._tokenizer.decode(ids)

    def _decode(self, token_ids, skip_special_tokens: bool = False, **kwargs) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens:
            return self._tokenizer.decode(
                [t for t in token_ids if t not in self._special_token_ids]
            )
        parts: list[str] = []
        buffer: list[int] = []
        id_to_name = {idx: name for name, idx in self._tokenizer._special_tokens.items()}
        for tid in token_ids:
            if tid in self._special_token_ids:
                if buffer:
                    parts.append(self._tokenizer.decode(buffer))
                    buffer = []
                name = id_to_name.get(tid)
                if name:
                    parts.append(name)
            else:
                buffer.append(tid)
        if buffer:
            parts.append(self._tokenizer.decode(buffer))
        return "".join(parts)

    def build_inputs_with_special_tokens(self, token_ids_0: list[int], token_ids_1: list[int] | None = None) -> list[int]:
        """Return token IDs as-is — no BOS/EOS wrapping."""
        return token_ids_0

    def save_vocabulary(self, save_directory: str, filename_prefix: str | None = None) -> tuple[str, ...]:
        """No-op: tiktoken encodings are built-in."""
        return ()
'''

MODELING_TEMPLATE = '''
from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers import PreTrainedModel

from .configuration_steerling import SteerlingConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FlexAttention imports (CUDA-only, with fallback to SDPA)
# ---------------------------------------------------------------------------
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

# flex_attention requires Triton compilation which may fail on some GPUs
# (e.g. insufficient shared memory). Set STEERLING_USE_FLEX_ATTN=1 to enable.
_USE_FLEX_ATTN = os.environ.get("STEERLING_USE_FLEX_ATTN", "0") == "1"
if not _USE_FLEX_ATTN:
    _FLEX_ATTN_AVAILABLE = False

if torch.cuda.is_available() and _FLEX_ATTN_AVAILABLE:
    compiled_flex_attention = torch.compile(flex_attention, fullgraph=True)
else:
    compiled_flex_attention = flex_attention

# Threshold above which dense concept operations are forbidden
LARGE_CONCEPT_THRESHOLD = 50000


# ===========================================================================
# Output dataclasses
# ===========================================================================


@dataclass
class ConceptHeadOutput:
    """Output from ConceptHead forward pass."""

    features: Tensor
    gt_features: Tensor | None
    logits: Tensor | None
    predicted: Tensor
    weights: Tensor | None = None
    topk_indices: Tensor | None = None
    topk_logits: Tensor | None = None
    hidden: Tensor | None = None


@dataclass
class SteerlingOutput:
    """
    Output from Steerling forward pass with concept decomposition.

    Attributes:
        hidden: Raw transformer hidden states (B, T, D)
        known_features: Final known features (B, T, D)
        known_logits: Known concept logits (B, T, C_known) or None
        known_predicted: Predicted known features (B, T, D)
        known_weights: Known concept weights (B, T, C_known) or None
        known_topk_indices: Top-k known concept indices (B, T, k)
        known_topk_logits: Top-k known concept logits (B, T, k)
        unk: Residual unknown features (B, T, D)
        unk_hat: Predicted unknown features (B, T, D) or None
        unk_for_lm: Unknown features used for LM head (B, T, D)
        unknown_logits: Unknown concept logits or None
        unknown_weights: Unknown concept weights or None
        unknown_topk_indices: Top-k unknown concept indices or None
        unknown_topk_logits: Top-k unknown concept logits or None
        composed: Final composed features for LM head (B, T, D)
        epsilon: Epsilon correction term or None
    """

    hidden: Tensor
    known_features: Tensor
    known_logits: Tensor | None
    known_predicted: Tensor
    known_weights: Tensor | None
    known_topk_indices: Tensor | None = None
    known_topk_logits: Tensor | None = None
    unk: Tensor | None = None
    unk_hat: Tensor | None = None
    unk_for_lm: Tensor | None = None
    unknown_logits: Tensor | None = None
    unknown_weights: Tensor | None = None
    unknown_topk_indices: Tensor | None = None
    unknown_topk_logits: Tensor | None = None
    composed: Tensor | None = None
    epsilon: Tensor | None = None


# ===========================================================================
# Primitives: RMSNorm, RotaryEmbedding, MLP
# ===========================================================================


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, size: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(size))

    def forward(self, x: Tensor) -> Tensor:
        og = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(og)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (RoPE)."""

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 4096,
        base: float = 500000.0,
        full_precision: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_theta = base
        self.full_precision = full_precision
        self._cache: dict[str, Tensor] = {}
        self._build_cache(max_seq_len, torch.device("cpu"))

    def _build_cache(self, seq_len: int, device: torch.device) -> tuple[Tensor, Tensor]:
        pos_sin = self._cache.get("sin")
        pos_cos = self._cache.get("cos")
        if (
            pos_sin is not None
            and pos_cos is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_sin.device == device
        ):
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            inv_freq = 1.0 / (
                self.rope_theta
                ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float) / self.dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.outer(seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin = positions.sin()[None, None, :, :]
            pos_cos = positions.cos()[None, None, :, :]

        self._cache["sin"] = pos_sin
        self._cache["cos"] = pos_cos
        return pos_sin, pos_cos

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        if self.full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self._build_cache(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)

            q_ = (q_ * pos_cos[:, :, key_len - query_len : key_len, :]) + (
                self._rotate_half(q_) * pos_sin[:, :, key_len - query_len : key_len, :]
            )
            k_ = (k_ * pos_cos) + (self._rotate_half(k_) * pos_sin)

        return q_.type_as(q), k_.type_as(k)


class MLP(nn.Module):
    """MLP with SwiGLU or standard activation."""

    def __init__(self, config: SteerlingConfig):
        super().__init__()
        if config.intermediate_size is not None:
            intermediate_size = config.intermediate_size
        else:
            intermediate_size = config.mlp_ratio * config.n_embd

        self.mlp_type = config.mlp_type

        if config.mlp_type == "swiglu":
            self.c_fc = nn.Linear(config.n_embd, 2 * intermediate_size, bias=config.use_bias)
            self.c_proj = nn.Linear(intermediate_size, config.n_embd, bias=config.use_bias)
        else:
            self.c_fc = nn.Linear(config.n_embd, intermediate_size, bias=config.use_bias)
            self.c_proj = nn.Linear(intermediate_size, config.n_embd, bias=config.use_bias)
            act = config.activation
            if act == "gelu":
                self.activation = nn.GELU(approximate="tanh")
            elif act == "relu":
                self.activation = nn.ReLU()
            elif act == "silu":
                self.activation = nn.SiLU()
            else:
                raise ValueError(f"Unknown activation: {act}")

    def forward(self, x: Tensor) -> Tensor:
        if self.mlp_type == "swiglu":
            gate_up = self.c_fc(x)
            up, gate = gate_up.chunk(2, dim=-1)
            return self.c_proj(F.silu(gate) * up)
        else:
            return self.c_proj(self.activation(self.c_fc(x)))


# ===========================================================================
# Block-causal attention
# ===========================================================================


def block_causal_mask_mod(
    b: Any, h: Any, q_idx: Tensor, kv_idx: Tensor, *, block_size: int
) -> Tensor:
    """Block-causal mask: causal across blocks, bidirectional within blocks."""
    return q_idx // block_size >= kv_idx // block_size


def fast_create_block_causal_mask(
    attn_block_size: int,
    seq_length: int,
    mask_block_size: int,
    device: torch.device,
):
    """Analytically compute sparse block structure for flex_attention."""
    if not _FLEX_ATTN_AVAILABLE or _dense_to_ordered is None or BlockMask is None:
        raise RuntimeError("flex_attention not available")

    num_mask_blocks = -(-seq_length // mask_block_size)
    attn_blocks_per_mask_block, rem = divmod(mask_block_size, attn_block_size)
    if rem != 0:
        raise ValueError(
            f"mask_block_size ({mask_block_size}) must be divisible by "
            f"attn_block_size ({attn_block_size})"
        )

    num_attn_blocks = num_mask_blocks * attn_blocks_per_mask_block
    lowres = torch.tril(torch.ones(num_attn_blocks, num_attn_blocks, dtype=torch.bool, device=device))
    block_attn_count = (
        lowres.reshape(num_mask_blocks, attn_blocks_per_mask_block, num_mask_blocks, attn_blocks_per_mask_block)
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
    q: Tensor, k: Tensor, v: Tensor, diff_block_size: int, mask_cache: dict, enable_gqa: bool = False
) -> Tensor:
    """Fallback using SDPA with dense mask when flex_attention is unavailable."""
    B, H, T, D = q.shape
    device, dtype = q.device, q.dtype
    cache_key = f"sdpa_{T}_{device}_{dtype}"
    if cache_key not in mask_cache:
        q_idx = torch.arange(T, device=device).unsqueeze(1)
        kv_idx = torch.arange(T, device=device).unsqueeze(0)
        bool_mask = q_idx // diff_block_size >= kv_idx // diff_block_size
        attn_mask = torch.zeros(T, T, device=device, dtype=dtype)
        attn_mask.masked_fill_(~bool_mask, float("-inf"))
        mask_cache[cache_key] = attn_mask

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=mask_cache[cache_key], dropout_p=0.0, is_causal=False, enable_gqa=enable_gqa
    )


class BlockCausalSelfAttention(nn.Module):
    """Block-causal self-attention with FlexAttention and optional GQA."""

    FLEX_MASK_BLOCK_SIZE = 128

    def __init__(self, config: SteerlingConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.n_kv_heads = config.n_head if config.n_kv_heads is None else config.n_kv_heads

        assert self.n_head % self.n_kv_heads == 0
        kv_out = self.n_kv_heads * self.head_dim
        self.c_attn = nn.Linear(config.n_embd, config.n_embd + 2 * kv_out, bias=config.use_bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.use_bias)

        if config.use_qk_norm:
            if config.use_rms_norm:
                self.q_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
                self.k_norm = RMSNorm(self.head_dim, eps=config.norm_eps)
            else:
                self.q_norm = nn.LayerNorm(self.head_dim)
                self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None

        if config.use_rope:
            self.rope = RotaryEmbedding(
                dim=self.head_dim,
                max_seq_len=config.block_size,
                base=config.rope_base,
                full_precision=config.rope_full_precision,
            )
        else:
            self.rope = None

        self._mask_cache: dict = {}
        self._logged = False

    def _get_block_mask(self, T: int, device: torch.device):
        cache_key = f"flex_{T}_{device}"
        if cache_key not in self._mask_cache:
            diff_bs = self.config.diff_block_size
            mbs = self.FLEX_MASK_BLOCK_SIZE
            if mbs % diff_bs != 0:
                mbs = diff_bs * (mbs // diff_bs) or diff_bs
            self._mask_cache[cache_key] = fast_create_block_causal_mask(diff_bs, T, mbs, device)
        return self._mask_cache[cache_key]

    def forward(self, x: Tensor) -> Tensor:
        B, T, C = x.size()
        device = x.device
        use_flex = _FLEX_ATTN_AVAILABLE and x.is_cuda and flex_attention is not None

        qkv = self.c_attn(x)
        if self.config.clip_qkv is not None:
            qkv = qkv.clamp(min=-self.config.clip_qkv, max=self.config.clip_qkv)

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
                y = compiled_flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
            else:
                y = flex_attention(q, k, v, block_mask=block_mask, enable_gqa=True)
        else:
            y = sdpa_with_block_causal_mask(q, k, v, self.config.diff_block_size, self._mask_cache, enable_gqa=True)

        y = y.transpose(1, 2).reshape(B, T, C)
        return self.c_proj(y)


class SteerlingBlock(nn.Module):
    """Transformer block with block-causal attention."""

    def __init__(self, config: SteerlingConfig):
        super().__init__()
        if config.use_rms_norm:
            self.ln_1 = RMSNorm(config.n_embd, eps=config.norm_eps)
            self.ln_2 = RMSNorm(config.n_embd, eps=config.norm_eps)
        else:
            self.ln_1 = nn.LayerNorm(config.n_embd)
            self.ln_2 = nn.LayerNorm(config.n_embd)

        self.norm_order = config.norm_order
        self.attn = BlockCausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_order == "pre":
            x = x + self.attn(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        else:
            x = x + self.ln_1(self.attn(x))
            x = x + self.ln_2(self.mlp(x))
        return x


# ===========================================================================
# Backbone: transformer without concept heads
# ===========================================================================


class SteerlingBackbone(nn.Module):
    """Block-causal transformer backbone (no concept heads)."""

    def __init__(self, config: SteerlingConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.blocks = nn.ModuleList([SteerlingBlock(config) for _ in range(config.n_layers)])

        if config.use_rms_norm:
            self.ln_f = RMSNorm(config.n_embd, eps=config.norm_eps)
        else:
            self.ln_f = nn.LayerNorm(config.n_embd)

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        if config.weight_sharing:
            self.lm_head.weight = self.tok_emb.weight

    def forward(
        self,
        input_ids: Tensor,
        *,
        input_embeds: Tensor | None = None,
        return_hidden: bool = False,
    ) -> Tensor:
        if input_embeds is not None:
            x = input_embeds
        elif input_ids is not None:
            x = self.tok_emb(input_ids)
        else:
            raise ValueError("Either input_ids or input_embeds must be provided")

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        if return_hidden:
            return x

        return self.lm_head(x)


# ===========================================================================
# Concept head
# ===========================================================================


class ConceptHead(nn.Module):
    """
    Concept decomposition head with memory-efficient streaming operations.

    Supports both known and unknown concepts with streaming top-k selection.
    """

    def __init__(
        self,
        n_concepts: int,
        concept_dim: int,
        n_embd: int,
        is_unknown: bool = False,
        use_attention: bool = False,
        topk: int | None = 16,
        topk_features: int | None = None,
        block_size: int = 4096,
        pad_multiple: int = 16,
        store_unknown_weights: bool = False,
        apply_topk_to_unknown: bool = False,
        topk_on_logits: bool = False,
        factorize: bool = False,
        factorize_rank: int = 256,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.concept_dim = concept_dim
        self.n_embd = n_embd
        self.is_unknown = is_unknown
        self.use_attention = use_attention
        self.topk = topk
        self.topk_features = topk_features if topk_features is not None else topk
        self.block_size = block_size
        self.pad_multiple = pad_multiple
        self.store_unknown_weights = store_unknown_weights
        self.apply_topk_to_unknown = apply_topk_to_unknown
        self.topk_on_logits = topk_on_logits
        self.factorize = factorize
        self.factorize_rank = factorize_rank
        self._is_large = n_concepts > LARGE_CONCEPT_THRESHOLD

        self.n_concepts_padded = ((n_concepts + pad_multiple - 1) // pad_multiple) * pad_multiple

        # Embeddings and predictors
        if factorize:
            self.embedding_coef = nn.Embedding(self.n_concepts_padded, factorize_rank)
            self.embedding_basis = nn.Linear(factorize_rank, concept_dim, bias=False)
            self.concept_embedding = None

            if not use_attention:
                self.predictor_down = nn.Linear(n_embd, factorize_rank, bias=False)
                self.predictor_up = nn.Linear(factorize_rank, self.n_concepts_padded, bias=False)
                self.concept_predictor = None
            else:
                self.concept_query_projection = nn.Linear(n_embd, concept_dim, bias=False)
                self.predictor_down = None
                self.predictor_up = None
                self.concept_predictor = None
        else:
            self.concept_embedding = nn.Embedding(self.n_concepts_padded, concept_dim)
            self.embedding_coef = None
            self.embedding_basis = None

            if use_attention:
                self.concept_query_projection = nn.Linear(n_embd, concept_dim, bias=False)
                self.concept_predictor = None
            else:
                self.concept_predictor = nn.Linear(n_embd, self.n_concepts_padded, bias=False)

            self.predictor_down = None
            self.predictor_up = None

    # -- Embedding access --

    def _get_embedding_weight(self) -> Tensor:
        """Get full (C, D) embedding matrix."""
        if self.concept_embedding is not None:
            return self.concept_embedding.weight
        return self.embedding_basis(self.embedding_coef.weight)

    def _get_embedding(self, indices: Tensor) -> Tensor:
        """Get embeddings for specific indices."""
        if self.concept_embedding is not None:
            return self.concept_embedding(indices)
        coef = self.embedding_coef(indices)
        return self.embedding_basis(coef)

    def _get_predictor_weight(self) -> Tensor | None:
        """Get full (C, D) predictor weight matrix."""
        if self.concept_predictor is not None:
            return self.concept_predictor.weight
        if self.predictor_down is not None and self.predictor_up is not None:
            return self.predictor_up.weight @ self.predictor_down.weight
        return None

    @staticmethod
    def _safe_index(weight: Tensor, indices: Tensor) -> Tensor:
        """DTensor-safe indexing via F.embedding."""
        original_shape = indices.shape
        flat = indices.reshape(-1)
        result = F.embedding(flat, weight)
        return result.reshape(*original_shape, -1)

    @staticmethod
    def _merge_topk(topv: Tensor, topi: Tensor, v_blk: Tensor, i_blk: Tensor, k: int):
        cand_v = torch.cat([topv, v_blk], dim=1)
        cand_i = torch.cat([topi, i_blk], dim=1)
        new_v, sel = torch.topk(cand_v, k, dim=1)
        new_i = torch.gather(cand_i, 1, sel)
        return new_v, new_i

    # -- Feature computation: dense streaming --

    @staticmethod
    def linear_block_features(hidden: Tensor, predictor_weight: Tensor, embeddings: Tensor, block_size: int = 4096) -> Tensor:
        B, T, D = hidden.shape
        C = predictor_weight.size(0)
        output = torch.zeros(B, T, D, dtype=hidden.dtype, device=hidden.device)
        flat_h = hidden.reshape(-1, D)
        W_t = predictor_weight.t().contiguous()
        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            logits_block = (flat_h @ W_t[:, start:end]).to(torch.float32).clamp(-15, 15)
            weights_block = torch.sigmoid(logits_block)
            output.add_((weights_block @ embeddings[start:end].to(weights_block.dtype)).reshape(B, T, D))
        return output.to(hidden.dtype)

    @staticmethod
    def attention_block_features(query: Tensor, embeddings: Tensor, block_size: int = 4096) -> Tensor:
        B, T, D = query.shape
        C = embeddings.shape[0]
        scale = 1.0 / math.sqrt(D)
        flat_q = query.reshape(-1, D)
        emb_T = embeddings.t().contiguous()
        output = torch.zeros(B * T, D, dtype=query.dtype, device=query.device)
        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            scores = (flat_q @ emb_T[:, start:end]).to(torch.float32) * scale
            scores = scores.clamp(-15, 15)
            output.add_(torch.sigmoid(scores) @ embeddings[start:end].to(torch.float32))
        return output.reshape(B, T, D).to(query.dtype)

    # -- Feature computation: streaming top-k --

    @staticmethod
    def linear_features_topk_streaming(
        hidden: Tensor, predictor_weight: Tensor, embeddings: Tensor, k: int, block_size: int = 4096, topk_on_logits: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, T, D = hidden.shape
        C = predictor_weight.size(0)
        BT = B * T
        device = hidden.device
        k = min(k, C)
        flat_h = hidden.reshape(BT, D)
        W_t = predictor_weight.t().contiguous()
        topv = torch.full((BT, k), float("-inf"), device=device, dtype=hidden.dtype)
        topi = torch.zeros((BT, k), device=device, dtype=torch.long)

        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            logits_blk = (flat_h @ W_t[:, start:end]).to(torch.float32).clamp_(-15, 15)
            vals_blk = logits_blk if topk_on_logits else torch.sigmoid(logits_blk)
            blk_k = min(k, end - start)
            v_blk, idx_blk = torch.topk(vals_blk, blk_k, dim=1)
            i_blk = idx_blk + start
            if blk_k < k:
                pad_v = torch.full((BT, k - blk_k), float("-inf"), device=device, dtype=torch.float32)
                pad_i = torch.zeros((BT, k - blk_k), device=device, dtype=torch.long)
                v_blk = torch.cat([v_blk, pad_v], dim=1)
                i_blk = torch.cat([i_blk, pad_i], dim=1)
            topv, topi = ConceptHead._merge_topk(topv, topi, v_blk, i_blk, k)

        W_sel = ConceptHead._safe_index(predictor_weight, topi)
        logits_sel = torch.einsum("bd,bkd->bk", flat_h.to(torch.float32), W_sel.to(torch.float32)).clamp(-15, 15)
        del W_sel
        weights_sel = torch.sigmoid(logits_sel)
        E_sel = ConceptHead._safe_index(embeddings, topi)
        features = torch.einsum("bk,bkd->bd", weights_sel, E_sel.to(weights_sel.dtype))
        return features.reshape(B, T, D).to(hidden.dtype), topi.reshape(B, T, k), logits_sel.reshape(B, T, k)

    @staticmethod
    def attention_features_topk_streaming(
        query: Tensor, embeddings: Tensor, k: int, block_size: int = 4096, topk_on_logits: bool = False
    ) -> tuple[Tensor, Tensor, Tensor]:
        B, T, D = query.shape
        C = embeddings.shape[0]
        BT = B * T
        device = query.device
        scale = 1.0 / math.sqrt(D)
        k = min(k, C)
        flat_q = query.reshape(BT, D)
        emb_T = embeddings.t().contiguous()
        topv = torch.full((BT, k), float("-inf"), device=device, dtype=query.dtype)
        topi = torch.zeros((BT, k), device=device, dtype=torch.long)

        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            logits_blk = (flat_q @ emb_T[:, start:end]).to(torch.float32) * scale
            logits_blk = logits_blk.clamp(-15, 15)
            vals_blk = logits_blk if topk_on_logits else torch.sigmoid(logits_blk)
            blk_k = min(k, end - start)
            v_blk, idx_blk = torch.topk(vals_blk, blk_k, dim=1)
            i_blk = idx_blk + start
            if blk_k < k:
                pad_v = torch.full((BT, k - blk_k), float("-inf"), device=device, dtype=torch.float32)
                pad_i = torch.zeros((BT, k - blk_k), device=device, dtype=torch.long)
                v_blk = torch.cat([v_blk, pad_v], dim=1)
                i_blk = torch.cat([i_blk, pad_i], dim=1)
            topv, topi = ConceptHead._merge_topk(topv, topi, v_blk, i_blk, k)

        E_sel = ConceptHead._safe_index(embeddings, topi)
        logits_sel = torch.einsum("bd,bkd->bk", flat_q.to(torch.float32), E_sel.to(torch.float32)) * scale
        logits_sel = logits_sel.clamp(-15, 15)
        features = torch.einsum("bk,bkd->bd", torch.sigmoid(logits_sel), E_sel.to(torch.float32))
        return features.reshape(B, T, D).to(query.dtype), topi.reshape(B, T, k), logits_sel.reshape(B, T, k)

    # -- Feature computation: factorized --

    def attention_features_topk_factorized(self, query: Tensor, k: int, block_size: int = 4096) -> tuple[Tensor, Tensor, Tensor]:
        B, T, D = query.shape
        BT = B * T
        C = self.n_concepts
        device = query.device
        scale = 1.0 / math.sqrt(D)
        k = min(k, C)
        flat_q = query.reshape(BT, D)

        coef = self.embedding_coef.weight[:C]
        basis_weight = self.embedding_basis.weight
        q_compressed = flat_q @ basis_weight

        topv = torch.full((BT, k), float("-inf"), device=device, dtype=query.dtype)
        topi = torch.zeros((BT, k), device=device, dtype=torch.long)

        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            scores = (q_compressed.float() @ coef[start:end].T.float()) * scale
            scores = scores.clamp(-15, 15)
            blk_k = min(k, end - start)
            v_chunk, idx_chunk = torch.topk(scores, blk_k, dim=1)
            i_chunk = idx_chunk + start
            if blk_k < k:
                pad_v = torch.full((BT, k - blk_k), float("-inf"), device=device, dtype=torch.float32)
                pad_i = torch.zeros((BT, k - blk_k), device=device, dtype=torch.long)
                v_chunk = torch.cat([v_chunk, pad_v], dim=1)
                i_chunk = torch.cat([i_chunk, pad_i], dim=1)
            topv, topi = self._merge_topk(topv, topi, v_chunk, i_chunk, k)

        coef_sel = self.embedding_coef(topi)
        logits_sel = torch.einsum("br,bkr->bk", q_compressed.float(), coef_sel.float()) * scale
        logits_sel = logits_sel.clamp(-15, 15)
        weighted_coef = torch.einsum("bk,bkr->br", torch.sigmoid(logits_sel), coef_sel.float())
        features = weighted_coef @ basis_weight.T.float()
        return features.reshape(B, T, D).to(query.dtype), topi.reshape(B, T, k), logits_sel.reshape(B, T, k)

    def linear_features_topk_factorized(self, hidden: Tensor, k: int, block_size: int = 4096) -> tuple[Tensor, Tensor, Tensor]:
        B, T, D = hidden.shape
        BT = B * T
        C = self.n_concepts
        device = hidden.device
        k = min(k, C)
        flat_h = hidden.reshape(BT, D)

        down_weight = self.predictor_down.weight
        up_weight = self.predictor_up.weight[:C]
        basis_weight = self.embedding_basis.weight
        h_compressed = flat_h @ down_weight.T

        topv = torch.full((BT, k), float("-inf"), device=device, dtype=hidden.dtype)
        topi = torch.zeros((BT, k), device=device, dtype=torch.long)

        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            logits_chunk = h_compressed.float() @ up_weight[start:end].T.float()
            logits_chunk = logits_chunk.clamp(-15, 15)
            blk_k = min(k, end - start)
            v_chunk, idx_chunk = torch.topk(logits_chunk, blk_k, dim=1)
            i_chunk = idx_chunk + start
            if blk_k < k:
                pad_v = torch.full((BT, k - blk_k), float("-inf"), device=device, dtype=torch.float32)
                pad_i = torch.zeros((BT, k - blk_k), device=device, dtype=torch.long)
                v_chunk = torch.cat([v_chunk, pad_v], dim=1)
                i_chunk = torch.cat([i_chunk, pad_i], dim=1)
            topv, topi = self._merge_topk(topv, topi, v_chunk, i_chunk, k)

        coef_sel = self.embedding_coef(topi)
        up_sel = self._safe_index(self.predictor_up.weight[:C], topi)
        logits_sel = torch.einsum("br,bkr->bk", h_compressed.float(), up_sel.float()).clamp(-15, 15)
        weighted_coef = torch.einsum("bk,bkr->br", torch.sigmoid(logits_sel), coef_sel.float())
        features = weighted_coef @ basis_weight.T.float()
        return features.reshape(B, T, D).to(hidden.dtype), topi.reshape(B, T, k), logits_sel.reshape(B, T, k)

    def attention_block_features_factorized(self, query: Tensor, block_size: int = 4096) -> Tensor:
        B, T, D = query.shape
        BT = B * T
        C = self.n_concepts
        device = query.device
        scale = 1.0 / math.sqrt(D)
        flat_q = query.reshape(BT, D)
        coef = self.embedding_coef.weight[:C]
        basis_weight = self.embedding_basis.weight
        q_compressed = flat_q @ basis_weight
        output = torch.zeros(BT, D, dtype=query.dtype, device=device)
        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            scores = (q_compressed @ coef[start:end].T).float() * scale
            scores = scores.clamp(-15, 15)
            weights = torch.sigmoid(scores)
            weighted_coef = weights @ coef[start:end].float()
            output.add_(weighted_coef @ basis_weight.T.float())
        return output.reshape(B, T, D).to(query.dtype)

    def linear_block_features_factorized(self, hidden: Tensor, block_size: int = 4096) -> Tensor:
        B, T, D = hidden.shape
        BT = B * T
        C = self.n_concepts
        device = hidden.device
        flat_h = hidden.reshape(BT, D)
        coef = self.embedding_coef.weight[:C]
        basis_weight = self.embedding_basis.weight
        down_weight = self.predictor_down.weight
        up_weight = self.predictor_up.weight[:C]
        h_compressed = flat_h @ down_weight.T
        output = torch.zeros(BT, D, dtype=hidden.dtype, device=device)
        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            logits_chunk = h_compressed.float() @ up_weight[start:end].T.float()
            logits_chunk = logits_chunk.clamp(-15, 15)
            weights = torch.sigmoid(logits_chunk)
            weighted_coef = weights @ coef[start:end].float()
            output.add_(weighted_coef @ basis_weight.T.float())
        return output.reshape(B, T, D).to(hidden.dtype)

    # -- Sparse logit computation (for attribution) --

    def compute_logits_for_indices(self, hidden: Tensor, indices: Tensor) -> Tensor:
        """Compute logits for specific concept indices only."""
        if hidden.dim() == 2:
            flat_h, flat_idx, output_shape = hidden, indices, indices.shape
        else:
            B, T, D = hidden.shape
            flat_h = hidden.reshape(B * T, -1)
            flat_idx = indices.reshape(B * T, -1)
            output_shape = indices.shape

        n_valid = self.n_concepts
        indices_safe = flat_idx.clamp(0, n_valid - 1)

        if self.use_attention:
            query = self.concept_query_projection(flat_h.unsqueeze(0)).squeeze(0)
            scale = 1.0 / math.sqrt(self.concept_dim)
            E_sel = self._get_embedding(indices_safe)
            logits = torch.einsum("md,mkd->mk", query.float(), E_sel.float()) * scale
        else:
            if self.factorize:
                W = self._get_predictor_weight()[:n_valid]
            else:
                W = self.concept_predictor.weight[:n_valid]
            W_sel = self._safe_index(W, indices_safe)
            logits = torch.einsum("md,mkd->mk", flat_h.float(), W_sel.float())

        return logits.clamp(-15, 15).reshape(output_shape)

    def get_concept_weights(self, hidden: Tensor, concept_ids: Tensor) -> Tensor:
        """Get sigmoid weights for specific concepts (for attribution)."""
        if concept_ids.dim() == 1:
            if hidden.dim() == 2:
                concept_ids = concept_ids.unsqueeze(0).expand(hidden.size(0), -1)
            else:
                B, T, _ = hidden.shape
                concept_ids = concept_ids.unsqueeze(0).unsqueeze(0).expand(B, T, -1)
        logits = self.compute_logits_for_indices(hidden, concept_ids)
        return torch.sigmoid(logits)

    # -- Intervention support --

    def _apply_sparse_interventions(
        self, features: Tensor, hidden: Tensor, intervene_ids: Tensor, intervene_vals: Tensor
    ) -> Tensor:
        """Apply sparse interventions: features += (new_val - current_weight) * embedding[c]."""
        valid = intervene_ids != -1
        if not valid.any():
            return features
        ids_safe = intervene_ids.clamp(0, self.n_concepts - 1)
        current_logits = self.compute_logits_for_indices(hidden, ids_safe)
        current_weights = torch.sigmoid(current_logits)
        emb = self._get_embedding(ids_safe)
        delta = (intervene_vals - current_weights) * valid.float()
        correction = (delta.unsqueeze(-1) * emb).sum(dim=2)
        return features + correction

    # -- Forward pass --

    @torch.compiler.disable
    def forward(
        self,
        hidden: Tensor,
        intervene_ids: Tensor | None = None,
        intervene_vals: Tensor | None = None,
        return_logits: bool = False,
        store_hidden: bool = False,
    ) -> ConceptHeadOutput:
        """
        Forward pass for concept decomposition (inference only, no teacher forcing).

        Args:
            hidden: Transformer hidden states (B, T, n_embd)
            intervene_ids: Concept IDs to intervene on (B, T, K_int), -1 = skip
            intervene_vals: Intervention strength values (B, T, K_int)
            return_logits: If True, compute full (B, T, C) logits.
            store_hidden: If True, store hidden in output for attribution.

        Returns:
            ConceptHeadOutput with features, predicted, topk_indices, topk_logits
        """
        has_interventions = intervene_ids is not None and intervene_vals is not None
        n_valid = self.n_concepts

        topk_indices: Tensor | None = None
        topk_logits: Tensor | None = None

        apply_topk = self.topk is not None and (not self.is_unknown or self.apply_topk_to_unknown)
        k_features = self.topk_features if self.topk_features is not None else self.topk

        if self.factorize:
            if self.use_attention:
                query = self.concept_query_projection(hidden)
                if apply_topk:
                    predicted, topk_indices, topk_logits = self.attention_features_topk_factorized(
                        query, k=k_features, block_size=self.block_size
                    )
                else:
                    predicted = self.attention_block_features_factorized(query, block_size=self.block_size)
            else:
                if apply_topk:
                    predicted, topk_indices, topk_logits = self.linear_features_topk_factorized(
                        hidden, k=k_features, block_size=self.block_size
                    )
                else:
                    predicted = self.linear_block_features_factorized(hidden, block_size=self.block_size)
        elif apply_topk:
            E = self._get_embedding_weight()[:n_valid]
            if self.use_attention:
                query = self.concept_query_projection(hidden)
                predicted, topk_indices, topk_logits = self.attention_features_topk_streaming(
                    query, E, k=k_features, block_size=self.block_size, topk_on_logits=self.topk_on_logits
                )
            else:
                W = self.concept_predictor.weight[:n_valid]
                predicted, topk_indices, topk_logits = self.linear_features_topk_streaming(
                    hidden, W, E, k=k_features, block_size=self.block_size, topk_on_logits=self.topk_on_logits
                )
        else:
            E = self._get_embedding_weight()[:n_valid]
            if self.use_attention:
                query = self.concept_query_projection(hidden)
                predicted = self.attention_block_features(query, E, block_size=self.block_size)
            else:
                W = self.concept_predictor.weight[:n_valid]
                predicted = self.linear_block_features(hidden, W, E, block_size=self.block_size)

        # Slice top-k for loss from larger top-k for features
        if (
            topk_indices is not None
            and self.topk is not None
            and self.topk_features is not None
            and self.topk_features > self.topk
        ):
            _, rerank_idx = torch.topk(topk_logits, self.topk, dim=-1)
            topk_indices = torch.gather(topk_indices, -1, rerank_idx)
            topk_logits = torch.gather(topk_logits, -1, rerank_idx)

        # Apply sparse interventions
        if has_interventions:
            predicted = self._apply_sparse_interventions(predicted, hidden, intervene_ids, intervene_vals)

        return ConceptHeadOutput(
            features=predicted,
            gt_features=None,
            logits=None,
            predicted=predicted,
            weights=None,
            topk_indices=topk_indices,
            topk_logits=topk_logits,
            hidden=hidden.detach() if store_hidden else None,
        )


# ===========================================================================
# Main model: SteerlingForCausalLM
# ===========================================================================


class SteerlingForCausalLM(PreTrainedModel):
    """
    Steerling: Interpretable Causal Diffusion Language Model.

    Wraps a block-causal transformer with concept decomposition heads for
    interpretability and steering. Supports HuggingFace's from_pretrained().

    The model decomposes hidden states into:
        hidden -> known_features + unknown_features + epsilon = composed -> logits

    Usage:
        model = SteerlingForCausalLM.from_pretrained("guidelabs/steerling-8b", trust_remote_code=True)
        logits, outputs = model(input_ids)
    """

    config_class = SteerlingConfig
    supports_gradient_checkpointing = False

    def __init__(self, config: SteerlingConfig):
        super().__init__(config)

        # Transformer backbone
        self.transformer = SteerlingBackbone(config)

        # Known concept head
        self.known_head = ConceptHead(
            n_concepts=config.n_concepts,
            concept_dim=config.concept_dim,
            n_embd=config.n_embd,
            is_unknown=False,
            use_attention=config.use_attention_known,
            topk=config.topk_known,
            topk_features=config.topk_known_features,
            block_size=config.concept_block_size,
            pad_multiple=config.pad_multiple,
            store_unknown_weights=False,
            apply_topk_to_unknown=False,
            topk_on_logits=config.topk_on_logits,
            factorize=False,
        )

        # Unknown concept head (optional)
        if config.use_unknown:
            self.unknown_head = ConceptHead(
                n_concepts=config.n_unknown_concepts,
                concept_dim=config.concept_dim,
                n_embd=config.n_embd,
                is_unknown=True,
                use_attention=config.use_attention_unknown,
                topk=config.unknown_topk,
                block_size=config.concept_block_size,
                pad_multiple=config.pad_multiple,
                store_unknown_weights=config.store_unknown_weights,
                apply_topk_to_unknown=config.apply_topk_to_unknown,
                topk_on_logits=config.topk_on_logits,
                factorize=config.factorize_unknown,
                factorize_rank=config.factorize_rank,
            )
        else:
            self.unknown_head = None

        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights (called by PreTrainedModel.post_init)."""
        std = 0.02
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

    _tied_weights_keys = ["transformer.lm_head.weight"]

    def _tie_weights(self) -> None:
        """Tie LM head to token embedding weights."""
        if self.config.weight_sharing:
            self.transformer.lm_head.weight = self.transformer.tok_emb.weight

    def forward(
        self,
        input_ids: Tensor,
        *,
        input_embeds: Tensor | None = None,
        intervene_known_ids: Tensor | None = None,
        intervene_known_vals: Tensor | None = None,
        intervene_unknown_ids: Tensor | None = None,
        intervene_unknown_vals: Tensor | None = None,
        position_injection: Tensor | None = None,
        steering_inject_layer: int | None = None,
        steering_inject_alpha: float | None = None,
        minimal_output: bool = False,
        unknown_topk: int = 64,
    ) -> tuple[Tensor, SteerlingOutput]:
        """
        Forward pass with concept decomposition.

        Args:
            input_ids: Token IDs (B, T). May contain mask tokens.
            input_embeds: Pre-computed embeddings (B, T, D). Overrides input_ids.
            intervene_known_ids: Known concept IDs to override (B, T, K), -1 = skip.
            intervene_known_vals: Override values for known concepts (B, T, K).
            intervene_unknown_ids: Unknown concept IDs to override (B, T, K).
            intervene_unknown_vals: Override values for unknown concepts (B, T, K).
            position_injection: Per-position steering injection (B, T, D).
            steering_inject_layer: Inject at layers >= this (1-indexed).
            steering_inject_alpha: Injection strength. Defaults to config value.
            minimal_output: If True, skip computing unknown top-k (faster).
            unknown_topk: Number of unknown top-k to compute for output.

        Returns:
            logits: LM logits (B, T, vocab_size)
            outputs: SteerlingOutput with concept decomposition details
        """
        inject_alpha = steering_inject_alpha if steering_inject_alpha is not None else self.config.inject_alpha

        # Get hidden states from transformer
        if position_injection is not None and steering_inject_layer is not None:
            hidden = self._forward_with_injection(
                input_ids, input_embeds, position_injection, steering_inject_layer, inject_alpha
            )
        else:
            hidden = self.transformer(input_ids, input_embeds=input_embeds, return_hidden=True)

        # Known concept head (no teacher forcing at inference)
        known_out = self.known_head(
            hidden,
            intervene_ids=intervene_known_ids,
            intervene_vals=intervene_known_vals,
            return_logits=not minimal_output and not self.known_head._is_large,
        )
        known_features = known_out.features.to(hidden.dtype)

        # Residual unknown
        unk = hidden - known_features.detach()

        # Unknown concept head
        unk_for_lm = unk
        unknown_out: ConceptHeadOutput | None = None
        unk_hat: Tensor | None = None
        if self.unknown_head is not None:
            unknown_out = self.unknown_head(
                hidden.detach(),
                intervene_ids=intervene_unknown_ids,
                intervene_vals=intervene_unknown_vals,
                return_logits=not minimal_output and not self.unknown_head._is_large,
            )
            unk_hat = unknown_out.features.to(hidden.dtype)
            unk_for_lm = unk_hat.detach()

        # Epsilon correction
        epsilon = None
        if self.config.use_epsilon_correction and intervene_known_ids is None:
            epsilon = hidden - (unk_for_lm + known_features)
            unk_for_lm = unk_for_lm + epsilon

        # Compose and compute logits
        composed = unk_for_lm + known_features
        logits = self.transformer.lm_head(composed)

        # Compute unknown top-k if needed for output
        _unk_topk_indices = unknown_out.topk_indices if unknown_out else None
        _unk_topk_logits = unknown_out.topk_logits if unknown_out else None

        if (
            not minimal_output
            and self.unknown_head is not None
            and unknown_out is not None
            and _unk_topk_indices is None
            and unknown_topk > 0
        ):
            with torch.no_grad():
                if self.unknown_head.factorize:
                    if self.unknown_head.use_attention:
                        _query = self.unknown_head.concept_query_projection(hidden.detach())
                        _, _unk_topk_indices, _unk_topk_logits = (
                            self.unknown_head.attention_features_topk_factorized(
                                _query, k=unknown_topk, block_size=self.unknown_head.block_size
                            )
                        )
                    else:
                        _, _unk_topk_indices, _unk_topk_logits = (
                            self.unknown_head.linear_features_topk_factorized(
                                hidden.detach(), k=unknown_topk, block_size=self.unknown_head.block_size
                            )
                        )
                else:
                    _E = self.unknown_head._get_embedding_weight()[: self.unknown_head.n_concepts]
                    if self.unknown_head.use_attention:
                        _query = self.unknown_head.concept_query_projection(hidden.detach())
                        _, _unk_topk_indices, _unk_topk_logits = (
                            self.unknown_head.attention_features_topk_streaming(
                                _query, _E, k=unknown_topk, block_size=self.unknown_head.block_size
                            )
                        )
                    else:
                        _W = self.unknown_head.concept_predictor.weight[: self.unknown_head.n_concepts]
                        _, _unk_topk_indices, _unk_topk_logits = (
                            self.unknown_head.linear_features_topk_streaming(
                                hidden.detach(), _W, _E, k=unknown_topk, block_size=self.unknown_head.block_size
                            )
                        )

        outputs = SteerlingOutput(
            hidden=hidden,
            known_features=known_features,
            known_logits=known_out.logits,
            known_predicted=known_out.predicted,
            known_weights=known_out.weights,
            known_topk_indices=known_out.topk_indices,
            known_topk_logits=known_out.topk_logits,
            unk=unk,
            unk_hat=unk_hat,
            unk_for_lm=unk_for_lm,
            unknown_logits=unknown_out.logits if unknown_out else None,
            unknown_weights=unknown_out.weights if unknown_out else None,
            unknown_topk_indices=_unk_topk_indices,
            unknown_topk_logits=_unk_topk_logits,
            composed=composed,
            epsilon=epsilon,
        )

        return logits, outputs

    def _forward_with_injection(
        self,
        input_ids: Tensor,
        input_embeds: Tensor | None,
        position_injection: Tensor,
        inject_layer: int,
        inject_alpha: float,
    ) -> Tensor:
        """Forward through transformer blocks with steering injection."""
        if input_embeds is not None:
            x = input_embeds
        else:
            x = self.transformer.tok_emb(input_ids)

        for i, block in enumerate(self.transformer.blocks):
            x = block(x)
            if (i + 1) >= inject_layer:
                x = x + inject_alpha * position_injection

        return self.transformer.ln_f(x)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.tok_emb.weight.numel()
        return n_params
'''


# ===================================================================
# File generation
# ===================================================================

def compact_json(obj: dict) -> str:
    """JSON with indent=2 but short arrays kept on one line."""
    raw = json.dumps(obj, indent=2)
    # Collapse multi-line arrays where each element is on its own indented line
    def _collapse(m: re.Match) -> str:
        inner = m.group(1)
        # Parse individual elements (strip trailing commas and whitespace)
        elements = [line.strip().rstrip(",") for line in inner.strip().splitlines()]
        return "[" + ", ".join(elements) + "]"
    return re.sub(
        r'\[\s*\n((?:\s+(?:"[^"]*"|null),?\s*\n)+)\s*\]',
        _collapse,
        raw,
    ) + "\n"


FILES = {
    "config.json": lambda: compact_json(CONFIG_JSON),
    "tokenizer_config.json": lambda: compact_json(TOKENIZER_CONFIG_JSON),
    "configuration_steerling.py": lambda: CONFIGURATION_TEMPLATE,
    "tokenization_steerling.py": lambda: TOKENIZATION_TEMPLATE,
    "modeling_steerling.py": lambda: MODELING_TEMPLATE,
}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Generating HF files in {OUTPUT_DIR}/")
    for filename, content_fn in FILES.items():
        path = OUTPUT_DIR / filename
        content = content_fn()
        path.write_text(content)
        print(f"  {filename} ({len(content.splitlines())} lines)")

    print("Done.")


if __name__ == "__main__":
    main()

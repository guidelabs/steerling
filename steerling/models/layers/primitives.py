"""
Transformer building blocks: RMSNorm, RotaryEmbedding, MLP.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.
    """

    def __init__(self, config, size: int | None = None):
        super().__init__()
        self.eps = getattr(config, "norm_eps", 1e-5)
        norm_size = size if size is not None else config.n_embd
        self.weight = nn.Parameter(torch.ones(norm_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        og = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(og)


class BufferCache:
    """Simple cache for storing tensors (used by RotaryEmbedding)."""

    def __init__(self):
        self._cache: dict[str, torch.Tensor] = {}

    def get(self, key: str) -> torch.Tensor | None:
        return self._cache.get(key)

    def __setitem__(self, key: str, value: torch.Tensor):
        self._cache[key] = value

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._cache[key]


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embeddings (RoPE).

    Applies rotary embeddings to queries and keys for position information.

    Args:
        dim: Dimension of the rotary embeddings (typically head_dim)
        max_seq_len: Maximum sequence length to cache
        base: Base for inverse frequency computation (theta)
        rope_full_precision: Whether to compute RoPE in full precision
    """

    def __init__(
        self,
        dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        rope_full_precision: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.rope_theta = base
        self.rope_full_precision = rope_full_precision
        self.__cache = BufferCache()

        # Warm up cache on CPU
        self.get_rotary_embedding(max_seq_len, torch.device("cpu"))

    def get_rotary_embedding(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """Get or compute rotary embeddings for given sequence length."""

        pos_sin = self.__cache.get("rope_pos_sin")
        pos_cos = self.__cache.get("rope_pos_cos")

        if (
            pos_sin is not None
            and pos_cos is not None
            and pos_sin.shape[-2] >= seq_len
            and pos_cos.shape[-2] >= seq_len
        ):
            if pos_sin.device != device:
                pos_sin = pos_sin.to(device)
                self.__cache["rope_pos_sin"] = pos_sin
            if pos_cos.device != device:
                pos_cos = pos_cos.to(device)
                self.__cache["rope_pos_cos"] = pos_cos
            return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]

        with torch.autocast(device.type, enabled=False):
            inv_freq = 1.0 / (
                self.rope_theta ** (torch.arange(0, self.dim, 2, device=device, dtype=torch.float) / self.dim)
            )
            seq = torch.arange(seq_len, device=device, dtype=torch.float)
            freqs = torch.outer(seq, inv_freq)
            positions = torch.cat((freqs, freqs), dim=-1)
            pos_sin = positions.sin()[None, None, :, :]
            pos_cos = positions.cos()[None, None, :, :]

        self.__cache["rope_pos_sin"] = pos_sin
        self.__cache["rope_pos_cos"] = pos_cos
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half the hidden dims of the input."""

        B, nh, T, hs = x.size()
        x = x.view(B, nh, T, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(
        self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Apply rotary position embeddings to input tensor."""

        return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary embeddings to queries and keys."""

        if self.rope_full_precision:
            q_, k_ = q.float(), k.float()
        else:
            q_, k_ = q, k

        with torch.autocast(q.device.type, enabled=False):
            query_len, key_len = q_.shape[-2], k_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(key_len, q_.device)
            pos_sin = pos_sin.type_as(q_)
            pos_cos = pos_cos.type_as(q_)

            q_ = self.apply_rotary_pos_emb(
                pos_sin[:, :, key_len - query_len : key_len, :],
                pos_cos[:, :, key_len - query_len : key_len, :],
                q_,
            )
            k_ = self.apply_rotary_pos_emb(pos_sin, pos_cos, k_)

        return q_.type_as(q), k_.type_as(k)


class MLP(nn.Module):
    """
    Multi-Layer Perceptron with SwiGLU or standard activation.

    Args:
        config: Model config with n_embd, mlp_ratio, use_bias, mlp_type, activation
    """

    def __init__(self, config):
        super().__init__()

        if hasattr(config, "intermediate_size") and config.intermediate_size is not None:
            intermediate_size = config.intermediate_size
        else:
            intermediate_size = getattr(config, "mlp_ratio", 4) * config.n_embd

        use_bias = config.use_bias
        mlp_type = config.mlp_type

        if mlp_type == "swiglu":
            self.c_fc = nn.Linear(config.n_embd, 2 * intermediate_size, bias=use_bias)
            self.c_proj = nn.Linear(intermediate_size, config.n_embd, bias=use_bias)
            self.activation = None
        else:
            self.c_fc = nn.Linear(config.n_embd, intermediate_size, bias=use_bias)
            self.c_proj = nn.Linear(intermediate_size, config.n_embd, bias=use_bias)
            act_map = {
                "gelu": nn.GELU(approximate="tanh"),
                "relu": nn.ReLU(),
                "silu": nn.SiLU(),
            }
            self.activation = act_map[config.activation]

        self.c_proj.SCALE_INIT = 1  # type: ignore
        self.config = config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mlp_type = getattr(self.config, "mlp_type", "swiglu")

        if mlp_type == "swiglu":
            gate_up = self.c_fc(x)
            up, gate = gate_up.chunk(2, dim=-1)
            intermediate = F.silu(gate) * up
        else:
            intermediate = self.c_fc(x)
            intermediate = self.activation(intermediate)  # type: ignore

        return self.c_proj(intermediate)

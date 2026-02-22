"""CausalDiffusionLM model configuration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, model_validator


class CausalDiffusionConfig(BaseModel):
    """
    Configuration for CausalDiffusionLM (This is a block-causal discrete diffusion language model.)
    """

    model_type: Literal["causal_diffusion"] = "causal_diffusion"
    interpretable: bool = False

    # Architecture
    n_layers: int = 32
    n_head: int = 32
    n_embd: int = 4096
    block_size: int = 4096
    n_kv_heads: int | None = 4
    diff_block_size: int = 64

    # Normalization
    use_rms_norm: bool = True
    norm_eps: float = 1e-5
    norm_order: Literal["pre", "post"] = "post"
    use_qk_norm: bool = True

    # Position encoding
    use_rope: bool = True
    rope_base: float = 500000.0
    rope_full_precision: bool = True

    # MLP
    mlp_type: Literal["swiglu", "standard"] = "swiglu"
    activation: Literal["gelu", "relu", "silu"] = "gelu"
    mlp_ratio: int = 4
    intermediate_size: int | None = None

    # Other
    use_bias: bool = False
    clip_qkv: float | None = 10.0
    weight_sharing: bool = True

    @model_validator(mode="after")
    def validate_model(self) -> CausalDiffusionConfig:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.n_kv_heads is not None:
            if self.n_kv_heads <= 0:
                raise ValueError(f"n_kv_heads ({self.n_kv_heads}) must be >= 1")
            if self.n_head % self.n_kv_heads != 0:
                raise ValueError(
                    f"n_head ({self.n_head}) must be divisible by n_kv_heads ({self.n_kv_heads})"
                )
        return self

    @classmethod
    def from_json(cls, path: str | Path) -> CausalDiffusionConfig:
        """Load config from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)

    def to_json(self, path: str | Path) -> None:
        """Save config to a JSON file."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.model_dump(), f, indent=2)

    model_config = {"extra": "forbid", "validate_assignment": True}

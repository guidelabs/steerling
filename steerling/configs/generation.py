"""Generation configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """Configuration for text generation with SteerlingGenerator."""

    max_new_tokens: int = Field(default=256, gt=0)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    top_k: int | None = Field(default=None, gt=0)
    seed: int | None = None
    tokens_per_step: int = Field(default=1, gt=0)
    use_entropy_sampling: bool = True
    repetition_penalty: float = Field(default=1.2, ge=1.0)

    # Stop conditions
    include_eos_in_stop: bool = True

    # Concept steering
    steer_known: dict[int, float] | None = None
    steer_unknown: dict[int, float] | None = None

    model_config = {"extra": "forbid"}

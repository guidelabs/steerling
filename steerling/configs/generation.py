"""Generation configuration."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """Configuration for text generation with SteerlingGenerator."""

    max_new_tokens: int = Field(default=256, gt=0)
    steps: int = Field(default=256, gt=0)
    temperature: float = Field(default=0.0, ge=0.0)
    cfg_scale: float = Field(default=0.0, ge=0.0)
    seed: int | None = None

    # Stop conditions
    stop_tokens: list[int] | None = None

    model_config = {"extra": "forbid"}

"""Configuration for attribution baselines."""

from enum import StrEnum

from pydantic import BaseModel


class BaselineMode(StrEnum):
    MASK = "mask"
    PAD = "pad"
    ZERO = "zero"


class BaselineConfig(BaseModel):
    """
    Configuration for the integrated-gradients baseline.

    The baseline is the "absence of input" reference the attribution path starts from:
    - MASK: the model's mask-token embedding (default, recommended for Steerling, since
      mask is the trained absence-of-information state).
    - PAD: the model's pad-token embedding.
    - ZERO: a zero vector (the standard IG baseline).

    token_id overrides the mode-inferred token. For MASK/PAD the id is supplied by the
    caller (the faithful attributor threads it from the generator); ZERO ignores it.
    """

    mode: BaselineMode = BaselineMode.MASK
    token_id: int | None = None

    model_config = {"frozen": True}

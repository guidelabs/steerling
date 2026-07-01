"""Steering configuration (inference-only).

Steering injects a concept direction into the residual stream at layers
>= ``inject_layer`` and only at currently-masked positions (mask-aligned
injection). A positive ``mai_lm_target`` amplifies the concept; a negative
value suppresses it, which is the basis for concept unlearning.

``relu_logit_mask`` is an independent mechanism: it subtracts
``strength * relu(concept-vocab alignment)`` from the logits, asymmetrically
suppressing only the tokens a concept POSITIVELY promotes.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class SteeringConfig(BaseModel):
    """Configuration for concept steering during generation.

    The two supported recipes are the ``injection`` and ``injection_relu``
    presets. Build configs through them rather than setting fields directly:

    - ``SteeringConfig.injection(...)`` amplifies a concept (positive injection).
    - ``SteeringConfig.injection_relu(...)`` unlearns a concept (negative
      injection plus a focal-concept logit mask).
    """

    concept_ids: list[int] = Field(
        default_factory=list,
        description="Known concept IDs to steer. Embeddings are summed into one direction.",
    )
    mai_lm_target: float = Field(
        default=0.0,
        description="Injection strength on the top-aligned token (signed). Negative  to suppress.",
    )
    normalize_mai_lm_target: bool = Field(
        default=True,
        description=(
            "If True, divide mai_lm_target by the direction's peak LM-head alignment so the "
            "value reads in logit units. If False, mai_lm_target is the raw injection alpha. "
            "Unlearning uses False."
        ),
    )
    inject_layer: int | None = Field(
        default=None,
        ge=0,
        description="Inject at transformer layers with index >= this. Defaults to n_layers // 2.",
    )
    inject_alpha_schedule: Literal["fixed", "hard_cutoff"] = Field(
        default="fixed",
        description=(
            "Alpha schedule across generation. 'fixed' holds alpha constant. 'hard_cutoff' applies "
            "full alpha for the first cutoff_tokens committed tokens, then zero, which plants the "
            "concept early and lets the rest generate unsteered to preserve quality."
        ),
    )
    cutoff_tokens: int = Field(
        default=32,
        ge=1,
        description="hard_cutoff only: tokens generated at full alpha before injection stops.",
    )
    relu_logit_mask: dict[int, float] | None = Field(
        default=None,
        description=(
            "Known concept ID -> suppression strength on positively aligned vocab. "
            "Subtracts strength * relu(concept-vocab alignment) from logits."
        ),
    )

    model_config = {"extra": "forbid"}

    @classmethod
    def injection(
        cls,
        concept_ids: int | list[int],
        mai_lm_target: float = 8.0,
        inject_layer: int = 8,
        cutoff_tokens: int = 32,
    ) -> SteeringConfig:
        """
        Amplify a concept by positive injection (mai_lm_target reads in logit units).

        Uses the hard_cutoff schedule: full alpha for the first cutoff_tokens
        tokens, then unsteered, which preserves generation quality.
        """
        ids = [concept_ids] if isinstance(concept_ids, int) else list(concept_ids)
        return cls(
            concept_ids=ids,
            mai_lm_target=mai_lm_target,
            normalize_mai_lm_target=True,
            inject_layer=inject_layer,
            inject_alpha_schedule="hard_cutoff",
            cutoff_tokens=cutoff_tokens,
        )

    @classmethod
    def injection_relu(
        cls,
        concept_id: int,
        group_concept_ids: list[int] | None = None,
        mai_lm_target: float = -7.0,
        inject_layer: int | None = None,
        relu_strength: float = 20.0,
    ) -> SteeringConfig:
        """
        Unlearn a concept by negative injection plus a logit mask.

        The injection direction is built from the focal concept plus any
        group_concept_ids (summed), matching scalex's group injection. The logit
        mask targets the focal concept_id only. mai_lm_target is the raw injection
        alpha (negative suppresses). Uses the fixed schedule (constant alpha).
        """
        ids = [concept_id] + [c for c in (group_concept_ids or []) if c != concept_id]
        return cls(
            concept_ids=ids,
            mai_lm_target=mai_lm_target,
            normalize_mai_lm_target=False,
            inject_layer=inject_layer,
            inject_alpha_schedule="fixed",
            relu_logit_mask={concept_id: relu_strength},
        )

    @property
    def has_injection(self) -> bool:
        return bool(self.concept_ids)

    @property
    def has_logit_mask(self) -> bool:
        return bool(self.relu_logit_mask)

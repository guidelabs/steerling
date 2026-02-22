"""Concept decomposition configuration (inference-only)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ConceptConfig(BaseModel):
    """
    Configuration for interpretable concept decomposition.
    """

    # Concept counts
    n_concepts: int = Field(33732, description="Number of known concepts")
    n_unknown_concepts: int | None = Field(101196, description="Number of unknown concepts")

    # Embedding configuration
    max_concepts: int = Field(16, description="Max concepts per token in batch (K dimension)")
    concept_dim: int = Field(4096, description="Concept embedding dimension. Must equal model.n_embd")
    use_attention_known: bool = Field(False, description="Use attention for known concepts (vs linear)")
    use_attention_unknown: bool = Field(False, description="Use attention for unknown concepts")
    topk_known: int | None = Field(16, description="Top-k sparsity for known concepts")
    topk_known_features: int | None = Field(
        32,
        description="Top-k for known concept features going into LM head. If None, defaults to topk_known.",
    )

    use_unknown: bool = Field(True, description="Use unknown concept decomposition")

    # Unknown factorization
    # we factorize the unknown head, so these are the factorization parameters
    factorize_unknown: bool = Field(True, description="Use low-rank factorized embeddings for unknown head")
    factorize_rank: int = Field(256, ge=16, le=1024, description="Rank for factorized embeddings")
    unknown_topk: int | None = Field(128, description="Top-k for unknown head")

    use_epsilon_correction: bool = Field(True, description="Add correction: unk += (hidden - (unk + known))")

    # ConceptHead parameters
    block_size: int = Field(4096, description="Block size for memory-efficient operations")
    pad_multiple: int = Field(16, description="Pad n_concepts to multiple of this for GPU efficiency")
    topk_on_logits: bool = Field(False, description="Apply top-k on logits vs weights")
    store_unknown_weights: bool = Field(False, description="Store unknown logits/weights")
    apply_topk_to_unknown: bool = Field(True, description="Apply top-k to unknown concepts")

    # Steering
    inject_layer: int = Field(16, description="Inject steering at layers >= this")
    inject_alpha: float = Field(1.0, description="Steering injection strength")

    model_config = {"extra": "forbid", "validate_assignment": True}

"""Output types for interpretable Steerling models."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class InterpretableOutput:
    """
    Full output from InterpretableCausalDiffusionLM; it contains all decomposition components for attribution and analysis.
    """

    hidden: Tensor  # Transformer hidden states (B, T, D)
    known_features: Tensor  # Weighted known concept features (B, T, D)
    known_logits: Tensor | None  # Full concept logits (B, T, C) or None
    known_gt_features: Tensor | None  # GT pooled features (B, T, D) or None
    known_predicted: Tensor  # Predicted known features before TF (B, T, D)
    known_weights: Tensor | None  # Full concept weights (B, T, C) or None
    known_topk_indices: Tensor | None  # Top-k concept indices (B, T, k)
    known_topk_logits: Tensor | None  # Logits for top-k concepts (B, T, k)
    unk: Tensor  # True unknown residual: hidden - known_features.detach()
    unk_hat: Tensor | None  # Predicted unknown features (B, T, D)
    unk_for_lm: Tensor  # Unknown features used in final composition
    unknown_logits: Tensor | None  # Unknown concept logits
    unknown_weights: Tensor | None  # Unknown concept weights
    unknown_topk_indices: Tensor | None  # Unknown top-k indices
    unknown_topk_logits: Tensor | None  # Unknown top-k logits
    composed: Tensor  # Final composed features (B, T, D)
    epsilon: Tensor | None  # Epsilon correction term
    epsilon_true: Tensor | None  # True epsilon: hidden - (known_predicted + unk_hat)

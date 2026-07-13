"""Tests for interpretable model generation and top-k concept selection."""

import torch

from steerling.models.interpretable.interpretable_causal_diffusion import (
    InterpretableCausalDiffusionLM,
)


class TestTopKConcepts:
    def test_known_topk_shape_matches_config(self, tiny_config, tiny_concept_config, tokenizer, device):
        """When apply_topk is True, known topk_indices last dim equals k_features."""
        model = InterpretableCausalDiffusionLM(
            tiny_config,
            tiny_concept_config,
            vocab_size=tokenizer.vocab_size,
        ).to(device)
        model.eval()

        B, T = 1, 16
        input_ids = torch.randint(0, tokenizer.vocab_size, (B, T), device=device)

        with torch.no_grad():
            logits, outputs = model(input_ids, minimal_output=False)

        k_known = tiny_concept_config.topk_known_features or tiny_concept_config.topk_known
        assert outputs.known_topk_indices is not None
        assert outputs.known_topk_indices.shape == (B, T, k_known)
        assert outputs.known_topk_logits.shape == (B, T, k_known)

        # Number of unique concepts per position should be <= k_features
        for t in range(T):
            n_unique = outputs.known_topk_indices[0, t].unique().numel()
            assert n_unique <= k_known

    def test_known_topk_indices_in_range(self, tiny_config, tiny_concept_config, tokenizer, device):
        """Known top-k indices must be valid concept IDs (< n_concepts)."""
        model = InterpretableCausalDiffusionLM(
            tiny_config,
            tiny_concept_config,
            vocab_size=tokenizer.vocab_size,
        ).to(device)
        model.eval()

        B, T = 1, 16
        input_ids = torch.randint(0, tokenizer.vocab_size, (B, T), device=device)

        with torch.no_grad():
            logits, outputs = model(input_ids, minimal_output=False)

        assert outputs.known_topk_indices.max() < tiny_concept_config.n_concepts
        assert outputs.known_topk_indices.min() >= 0

    def test_unknown_topk_shape(self, tiny_config, tiny_concept_config, tokenizer, device):
        """When apply_topk_to_unknown is True, unknown topk_indices has correct shape."""
        assert tiny_concept_config.apply_topk_to_unknown is True
        model = InterpretableCausalDiffusionLM(
            tiny_config,
            tiny_concept_config,
            vocab_size=tokenizer.vocab_size,
        ).to(device)
        model.eval()

        B, T = 1, 16
        input_ids = torch.randint(0, tokenizer.vocab_size, (B, T), device=device)

        with torch.no_grad():
            logits, outputs = model(input_ids, minimal_output=False)

        k_unk = tiny_concept_config.unknown_topk
        if outputs.unknown_topk_indices is not None:
            assert outputs.unknown_topk_indices.shape == (B, T, k_unk)
            assert outputs.unknown_topk_logits.shape == (B, T, k_unk)

    def test_unknown_topk_indices_in_range(self, tiny_config, tiny_concept_config, tokenizer, device):
        """Unknown top-k indices must be valid concept IDs (< n_unknown_concepts)."""
        model = InterpretableCausalDiffusionLM(
            tiny_config,
            tiny_concept_config,
            vocab_size=tokenizer.vocab_size,
        ).to(device)
        model.eval()

        B, T = 1, 16
        input_ids = torch.randint(0, tokenizer.vocab_size, (B, T), device=device)

        with torch.no_grad():
            logits, outputs = model(input_ids, minimal_output=False)

        if outputs.unknown_topk_indices is not None:
            assert outputs.unknown_topk_indices.max() < tiny_concept_config.n_unknown_concepts
            assert outputs.unknown_topk_indices.min() >= 0

    def test_topk_features_k_concepts_go_into_lm_head(
        self, tiny_config, tiny_concept_config, tokenizer, device
    ):
        """Out of n_concepts, only topk_known_features have non-zero weight in features."""
        model = InterpretableCausalDiffusionLM(
            tiny_config,
            tiny_concept_config,
            vocab_size=tokenizer.vocab_size,
        ).to(device)
        model.eval()

        k_features = tiny_concept_config.topk_known_features or tiny_concept_config.topk_known
        n_concepts = tiny_concept_config.n_concepts

        B, T = 1, 16
        input_ids = torch.randint(0, tokenizer.vocab_size, (B, T), device=device)

        with torch.no_grad():
            hidden = model.transformer(input_ids, return_hidden=True)

        # Compute full dense weights (all n_concepts) via _compute_weights
        head = model.known_head
        if head.factorize:
            W = head._get_predictor_weight()[:n_concepts]
            raw_logits = hidden @ W.T
        else:
            raw_logits = head.concept_predictor(hidden)[..., :n_concepts]
        concept_logits = raw_logits.float().clamp(-15, 15)
        weights_all = torch.sigmoid(concept_logits)

        # Apply topk_with_cutoff — should zero out all but k_features per position
        weights_sparse = head.topk_with_cutoff(weights_all)
        nonzero_per_pos = (weights_sparse > 0.0).sum(-1).float()

        assert nonzero_per_pos.max().item() <= k_features, (
            f"Expected at most {k_features} non-zero weights, got {int(nonzero_per_pos.max())}"
        )
        assert nonzero_per_pos.min().item() > 0, "Should have at least 1 non-zero weight"

"""
Tests for faithful concept attribution.

Tests the core attribution math against the platform's implementation,
without requiring a GPU or model weights.

Run: pytest tests/test_attribution.py -v
"""

import torch
import pytest

from steerling.attribution.concept_attribution import (
    AttributionAccumulator,
    CommitEvent,
    ConceptLabels,
    OutputToConceptAttribution,
    chunk_attribution,
    find_chunk_boundaries,
)


# ---------------------------------------------------------------------------
# Helpers: fake concept heads for testing
# ---------------------------------------------------------------------------


class FakeConceptHead:
    """Minimal concept head that returns deterministic embeddings."""

    def __init__(self, n_concepts: int, dim: int, device: torch.device):
        self.n_concepts = n_concepts
        self.topk = 4
        self.block_size = 4096
        self.factorize = True
        # Fixed embeddings: concept i -> vector of all (i+1)/n_concepts
        self._embeddings = torch.randn(n_concepts, dim, device=device)

    def _get_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """[..., K] -> [..., K, D]"""
        return self._embeddings[indices]

    def linear_features_topk_factorized(self, hidden, k, block_size=4096):
        """Fake unknown topk: return first k concepts with random logits."""
        B, P, D = hidden.shape
        indices = torch.arange(k, device=hidden.device).unsqueeze(0).unsqueeze(0).expand(B, P, k)
        logits = torch.randn(B, P, k, device=hidden.device)
        features = torch.zeros(B, P, D, device=hidden.device)
        return features, indices, logits


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def setup(device):
    """Create fake heads and LM head weight for testing."""
    dim = 32
    vocab_size = 100
    n_known = 50
    n_unknown = 80

    known_head = FakeConceptHead(n_known, dim, device)
    unknown_head = FakeConceptHead(n_unknown, dim, device)
    lm_head_weight = torch.randn(vocab_size, dim, device=device)

    return known_head, unknown_head, lm_head_weight, dim


# ---------------------------------------------------------------------------
# Test: attribution math matches platform's head_contrib
# ---------------------------------------------------------------------------


class TestAttributionMath:
    """Verify C(c_i, y_t) = sigmoid(logit_i) * (embedding_i . W_{y_t})."""

    def test_known_contribution_formula(self, setup, device):
        """Known contribution matches: sigmoid(logit) * dot(embedding, W_yt)."""
        known_head, _, lm_head_weight, dim = setup

        # Single position, k=4 concepts
        token_id = 5
        k = 4
        indices = torch.tensor([0, 1, 2, 3], device=device)
        logits = torch.tensor([1.0, -0.5, 2.0, 0.1], device=device)

        W_yt = lm_head_weight[token_id].float()
        emb = known_head._get_embedding(indices).float()  # [K, D]

        # Platform formula
        dots = torch.einsum("kd,d->k", emb, W_yt)
        w = torch.sigmoid(logits.float())
        expected = w * dots

        # Our accumulator does the same with batch dims
        accumulator = AttributionAccumulator(
            seq_len=10, k_known=k, k_unk=0, device=device,
            lm_head_weight=lm_head_weight, known_head=known_head,
        )
        accumulator.commit(CommitEvent(
            step=0,
            positions=torch.tensor([0], device=device),
            token_ids=torch.tensor([token_id], device=device),
            target_logits=torch.tensor([0.0], device=device),  # doesn't affect contrib
            known_indices=indices.unsqueeze(0),  # [1, K]
            known_logits=logits.unsqueeze(0),  # [1, K]
            unk_indices=None,
            unk_logits=None,
        ))

        actual = accumulator.known_contributions[0, 0]  # [K]
        assert torch.allclose(actual, expected, atol=1e-6), \
            f"Known contributions don't match:\n  expected={expected}\n  actual={actual}"

    def test_unknown_contribution_formula(self, setup, device):
        """Unknown contribution uses same formula as known."""
        _, unknown_head, lm_head_weight, dim = setup
        known_head = FakeConceptHead(10, dim, device)

        token_id = 3
        k_unk = 4
        u_indices = torch.tensor([0, 1, 2, 3], device=device)
        u_logits = torch.tensor([0.5, -1.0, 0.3, 1.5], device=device)

        W_yt = lm_head_weight[token_id].float()
        u_emb = unknown_head._get_embedding(u_indices).float()
        u_dots = torch.einsum("kd,d->k", u_emb, W_yt)
        u_w = torch.sigmoid(u_logits.float())
        expected = u_w * u_dots

        accumulator = AttributionAccumulator(
            seq_len=10, k_known=2, k_unk=k_unk, device=device,
            lm_head_weight=lm_head_weight, known_head=known_head,
            unknown_head=unknown_head,
        )
        accumulator.commit(CommitEvent(
            step=0,
            positions=torch.tensor([0], device=device),
            token_ids=torch.tensor([token_id], device=device),
            target_logits=torch.tensor([0.0], device=device),
            known_indices=torch.zeros(1, 2, dtype=torch.long, device=device),
            known_logits=torch.zeros(1, 2, device=device),
            unk_indices=u_indices.unsqueeze(0),
            unk_logits=u_logits.unsqueeze(0),
        ))

        actual = accumulator.unk_contributions[0, 0]  # [K_unk]
        assert torch.allclose(actual, expected, atol=1e-6)

    def test_residual_exact_by_linearity(self, setup, device):
        """Residual = target_logit - sum(known) - sum(unknown)."""
        known_head, unknown_head, lm_head_weight, dim = setup

        token_id = 7
        k_known, k_unk = 4, 4
        k_indices = torch.arange(k_known, device=device)
        k_logits = torch.randn(k_known, device=device)
        u_indices = torch.arange(k_unk, device=device)
        u_logits = torch.randn(k_unk, device=device)

        # Compute expected target logit from composed representation
        # (for this test, just set an arbitrary target logit)
        target_logit = torch.tensor([42.0], device=device)

        accumulator = AttributionAccumulator(
            seq_len=10, k_known=k_known, k_unk=k_unk, device=device,
            lm_head_weight=lm_head_weight, known_head=known_head,
            unknown_head=unknown_head,
        )
        accumulator.commit(CommitEvent(
            step=0,
            positions=torch.tensor([0], device=device),
            token_ids=torch.tensor([token_id], device=device),
            target_logits=target_logit,
            known_indices=k_indices.unsqueeze(0),
            known_logits=k_logits.unsqueeze(0),
            unk_indices=u_indices.unsqueeze(0),
            unk_logits=u_logits.unsqueeze(0),
        ))

        k_sum = accumulator.known_contributions[0, 0].sum()
        u_sum = accumulator.unk_contributions[0, 0].sum()
        residual = accumulator.residual_contribution[0, 0]

        reconstructed = k_sum + u_sum + residual
        assert torch.allclose(reconstructed, target_logit, atol=1e-5), \
            f"Residual check failed: {k_sum} + {u_sum} + {residual} = {reconstructed} != {target_logit}"

    def test_float32_precision(self, setup, device):
        """Attribution math is done in float32 even with bfloat16 inputs."""
        known_head, _, lm_head_weight, _ = setup

        # Simulate bfloat16 logits
        bf16_logits = torch.tensor([1.0, -0.5], device=device).bfloat16()

        accumulator = AttributionAccumulator(
            seq_len=10, k_known=2, k_unk=0, device=device,
            lm_head_weight=lm_head_weight, known_head=known_head,
        )
        accumulator.commit(CommitEvent(
            step=0,
            positions=torch.tensor([0], device=device),
            token_ids=torch.tensor([1], device=device),
            target_logits=torch.tensor([1.0], device=device),
            known_indices=torch.tensor([[0, 1]], device=device),
            known_logits=bf16_logits.unsqueeze(0),
            unk_indices=None,
            unk_logits=None,
        ))

        # Contributions should be float32
        assert accumulator.known_contributions.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test: accumulator scatter-write
# ---------------------------------------------------------------------------


class TestAccumulator:
    """Test that the accumulator correctly writes to the right positions."""

    def test_multiple_commits(self, setup, device):
        """Multiple commits write to different positions without interference."""
        known_head, _, lm_head_weight, _ = setup
        k = 4

        accumulator = AttributionAccumulator(
            seq_len=10, k_known=k, k_unk=0, device=device,
            lm_head_weight=lm_head_weight, known_head=known_head,
        )

        # Commit positions 0, 1
        accumulator.commit(CommitEvent(
            step=0,
            positions=torch.tensor([0, 1], device=device),
            token_ids=torch.tensor([5, 6], device=device),
            target_logits=torch.tensor([1.0, 2.0], device=device),
            known_indices=torch.arange(k, device=device).unsqueeze(0).expand(2, k),
            known_logits=torch.randn(2, k, device=device),
            unk_indices=None,
            unk_logits=None,
        ))

        # Commit position 5
        accumulator.commit(CommitEvent(
            step=1,
            positions=torch.tensor([5], device=device),
            token_ids=torch.tensor([7], device=device),
            target_logits=torch.tensor([3.0], device=device),
            known_indices=torch.arange(k, device=device).unsqueeze(0),
            known_logits=torch.randn(1, k, device=device),
            unk_indices=None,
            unk_logits=None,
        ))

        # Positions 0, 1, 5 should have non-zero contributions
        assert accumulator.known_contributions[0, 0].abs().sum() > 0
        assert accumulator.known_contributions[0, 1].abs().sum() > 0
        assert accumulator.known_contributions[0, 5].abs().sum() > 0

        # Position 3 should still be zero
        assert accumulator.known_contributions[0, 3].abs().sum() == 0

    def test_result_fields(self, setup, device):
        """result() returns OutputToConceptAttribution with expected fields."""
        known_head, _, lm_head_weight, _ = setup

        accumulator = AttributionAccumulator(
            seq_len=5, k_known=2, k_unk=0, device=device,
            lm_head_weight=lm_head_weight, known_head=known_head,
        )

        result = accumulator.result()
        assert isinstance(result, OutputToConceptAttribution)
        assert result.known_indices.shape == (1, 5, 2)
        assert result.known_weights.shape == (1, 5, 2)
        assert result.known_contributions.shape == (1, 5, 2)
        assert result.target_token_ids.shape == (1, 5)
        assert result.target_logits.shape == (1, 5)
        assert result.epsilon_contribution.shape == (1, 5)

    def test_committed_tracking(self, setup, device):
        """Accumulator tracks which positions have been committed."""
        known_head, _, lm_head_weight, _ = setup
        k = 2

        accumulator = AttributionAccumulator(
            seq_len=5, k_known=k, k_unk=0, device=device,
            lm_head_weight=lm_head_weight, known_head=known_head,
        )

        assert accumulator.coverage == 0.0

        accumulator.commit(CommitEvent(
            step=0,
            positions=torch.tensor([1, 3], device=device),
            token_ids=torch.tensor([5, 6], device=device),
            target_logits=torch.tensor([1.0, 2.0], device=device),
            known_indices=torch.arange(k, device=device).unsqueeze(0).expand(2, k),
            known_logits=torch.randn(2, k, device=device),
            unk_indices=None, unk_logits=None,
        ))

        assert accumulator.coverage == pytest.approx(2 / 5)
        assert accumulator.committed[1] and accumulator.committed[3]
        assert not accumulator.committed[0]
        assert accumulator.commit_step[1] == 0
        assert accumulator.commit_step[3] == 0
        assert accumulator.commit_step[0] == -2

    def test_empty_commit(self, setup, device):
        """Committing zero positions is a no-op."""
        known_head, _, lm_head_weight, _ = setup

        accumulator = AttributionAccumulator(
            seq_len=5, k_known=2, k_unk=0, device=device,
            lm_head_weight=lm_head_weight, known_head=known_head,
        )
        accumulator.commit(CommitEvent(
            step=0,
            positions=torch.tensor([], dtype=torch.long, device=device),
            token_ids=torch.tensor([], dtype=torch.long, device=device),
            target_logits=torch.tensor([], device=device),
            known_indices=torch.zeros(0, 2, dtype=torch.long, device=device),
            known_logits=torch.zeros(0, 2, device=device),
            unk_indices=None,
            unk_logits=None,
        ))

        assert accumulator.known_contributions.abs().sum() == 0


# ---------------------------------------------------------------------------
# Test: chunk attribution
# ---------------------------------------------------------------------------


class TestChunkAttribution:
    """Test chunk-level normalization and aggregation."""

    def test_normalization_scale_invariant(self, device):
        """Chunk attribution is scale-invariant across positions."""
        # Two positions with very different scales
        result = OutputToConceptAttribution(
            target_token_ids=torch.zeros(1, 2, dtype=torch.long, device=device),
            target_logits=torch.tensor([[15.0, 0.15]], device=device),
            known_indices=torch.tensor([[[0, 1], [0, 1]]], device=device),
            known_weights=torch.ones(1, 2, 2, device=device),
            known_contributions=torch.tensor([[[10.0, 5.0], [0.1, 0.05]]], device=device),
            unk_indices=torch.zeros(1, 2, 0, dtype=torch.long, device=device),
            unk_weights=torch.zeros(1, 2, 0, device=device),
            unk_contributions=torch.zeros(1, 2, 0, device=device),
            epsilon_contribution=torch.tensor([[0.0, 0.0]], device=device),
        )

        entries, eps_pct = chunk_attribution(result, 0, 2, batch=0)

        # Both positions have same relative distribution (2:1),
        # so chunk result should reflect that regardless of scale
        concept_0 = [e for e in entries if "0" in e["label"]][0]
        concept_1 = [e for e in entries if "1" in e["label"]][0]
        ratio = concept_0["contribution"] / concept_1["contribution"]
        assert abs(ratio - 2.0) < 0.01, f"Expected ratio ~2.0, got {ratio}"

    def test_epsilon_percentage(self, device):
        """Epsilon percentage is computed correctly."""
        result = OutputToConceptAttribution(
            target_token_ids=torch.zeros(1, 1, dtype=torch.long, device=device),
            target_logits=torch.tensor([[10.0]], device=device),
            known_indices=torch.tensor([[[0]]], device=device),
            known_weights=torch.ones(1, 1, 1, device=device),
            known_contributions=torch.tensor([[[9.0]]], device=device),
            unk_indices=torch.zeros(1, 1, 0, dtype=torch.long, device=device),
            unk_weights=torch.zeros(1, 1, 0, device=device),
            unk_contributions=torch.zeros(1, 1, 0, device=device),
            epsilon_contribution=torch.tensor([[1.0]], device=device),
        )

        _, eps_pct = chunk_attribution(result, 0, 1, batch=0)
        # eps / (|known| + |eps|) = 1 / (9 + 1) = 10%
        assert abs(eps_pct - 10.0) < 0.1, f"Expected ~10%, got {eps_pct}%"

    def test_unknown_offset(self, device):
        """Unknown concept IDs are offset by num_known_concepts."""
        labels = ConceptLabels()  # no CSV, will use fallback labels

        result = OutputToConceptAttribution(
            target_token_ids=torch.zeros(1, 1, dtype=torch.long, device=device),
            target_logits=torch.tensor([[1.0]], device=device),
            known_indices=torch.zeros(1, 1, 1, dtype=torch.long, device=device),
            known_weights=torch.ones(1, 1, 1, device=device),
            known_contributions=torch.zeros(1, 1, 1, device=device),
            unk_indices=torch.tensor([[[5]]], device=device),
            unk_weights=torch.ones(1, 1, 1, device=device),
            unk_contributions=torch.tensor([[[1.0]]], device=device),
            epsilon_contribution=torch.tensor([[0.0]], device=device),
        )

        entries, _ = chunk_attribution(
            result, 0, 1, batch=0,
            concept_labels=labels,
            num_known_concepts=1000,
        )

        disc_entries = [e for e in entries if e["type"] == "discovered"]
        assert len(disc_entries) == 1
        # Label should use offset: 5 + 1000 = 1005
        assert "1005" in disc_entries[0]["label"]


# ---------------------------------------------------------------------------
# Test: find_chunks
# ---------------------------------------------------------------------------


class TestFindChunkBoundaries:
    """Test chunk boundary detection (matches scalex find_chunk_boundaries)."""

    def test_basic_eoc_split(self):
        """Splits at EOC tokens."""
        ids = [1, 2, 99, 3, 4, 99, 5]
        chunks = find_chunk_boundaries(ids, eoc_id=99, include_final_chunk=True)
        assert chunks == [(0, 2), (3, 5), (6, 7)]

    def test_no_eoc(self):
        """No EOC tokens, include_final_chunk=True returns whole sequence."""
        ids = [1, 2, 3, 4, 5]
        chunks = find_chunk_boundaries(ids, eoc_id=-1, include_final_chunk=True)
        assert chunks == [(0, 5)]

    def test_no_eoc_no_final(self):
        """No EOC tokens, include_final_chunk=False returns empty."""
        ids = [1, 2, 3, 4, 5]
        chunks = find_chunk_boundaries(ids, eoc_id=-1, include_final_chunk=False)
        assert chunks == []

    def test_start_index_skips_prompt(self):
        """start_index skips prompt tokens."""
        ids = [10, 11, 12, 1, 2, 99, 3, 4]
        chunks = find_chunk_boundaries(ids, eoc_id=99, start_index=3, include_final_chunk=True)
        assert chunks == [(3, 5), (6, 8)]

    def test_stop_ids_terminates(self):
        """stop_ids terminates chunking at the stop token."""
        ids = [1, 2, 99, 3, 4, 50, 5, 6]
        chunks = find_chunk_boundaries(ids, eoc_id=99, stop_ids=[50])
        assert chunks == [(0, 2), (3, 5)]

    def test_consecutive_eoc(self):
        """Consecutive EOC tokens produce empty chunks (start == end)."""
        ids = [1, 99, 99, 2]
        chunks = find_chunk_boundaries(ids, eoc_id=99, include_final_chunk=True)
        # The empty chunk (2,2) between consecutive EOCs is expected
        assert chunks == [(0, 1), (2, 2), (3, 4)]


# ---------------------------------------------------------------------------
# Test: concept labels
# ---------------------------------------------------------------------------


class TestConceptLabels:
    def test_fallback_known(self):
        labels = ConceptLabels()
        assert labels.label(42, "known") == "Known: #42"

    def test_fallback_discovered(self):
        labels = ConceptLabels()
        assert labels.label(7, "discovered") == "Discovered: #7"

    def test_missing_file(self, tmp_path):
        labels = ConceptLabels(tmp_path / "nonexistent.csv")
        assert labels.label(0, "known") == "Known: #0"


class TestOutputToConceptAttribution:
    """Test methods on the OutputToConceptAttribution dataclass."""

    def _make_attr(self, device):
        return OutputToConceptAttribution(
            target_token_ids=torch.tensor([[1, 2, 3]], device=device),
            target_logits=torch.tensor([[10.0, 20.0, 30.0]], device=device),
            known_indices=torch.tensor([[[0, 1, 2], [3, 4, 5], [6, 7, 8]]], device=device),
            known_weights=torch.tensor([[[0.9, 0.5, 0.1], [0.8, 0.4, 0.2], [0.7, 0.3, 0.6]]], device=device),
            known_contributions=torch.tensor([[[5.0, 3.0, 1.0], [8.0, 6.0, 2.0], [10.0, 4.0, 7.0]]], device=device),
            unk_indices=torch.tensor([[[0, 1], [2, 3], [4, 5]]], device=device),
            unk_weights=torch.tensor([[[0.6, 0.3], [0.5, 0.2], [0.4, 0.1]]], device=device),
            unk_contributions=torch.tensor([[[0.5, 0.3], [1.5, 1.0], [3.0, 2.0]]], device=device),
            epsilon_contribution=torch.tensor([[0.2, 1.5, 4.0]], device=device),
        )

    def test_verify_passes(self, device):
        """Verify passes when contributions sum to target logits."""
        attr = self._make_attr(device)
        result = attr.verify()
        assert result.passed
        assert result.max_abs_error < 1e-4

    def test_verify_fails_on_mismatch(self, device):
        """Verify fails when target logits don't match contributions."""
        attr = OutputToConceptAttribution(
            target_token_ids=torch.tensor([[1]], device=device),
            target_logits=torch.tensor([[999.0]], device=device),
            known_indices=torch.tensor([[[0]]], device=device),
            known_weights=torch.tensor([[[0.5]]], device=device),
            known_contributions=torch.tensor([[[1.0]]], device=device),
            unk_indices=torch.zeros(1, 1, 0, dtype=torch.long, device=device),
            unk_weights=torch.zeros(1, 1, 0, device=device),
            unk_contributions=torch.zeros(1, 1, 0, device=device),
            epsilon_contribution=torch.tensor([[0.0]], device=device),
        )
        result = attr.verify()
        assert not result.passed

    def test_top_k(self, device):
        """top_k selects concepts with largest absolute contribution."""
        attr = self._make_attr(device)
        top2 = attr.top_k(2)

        assert top2.known_contributions.shape == (1, 3, 2)
        assert top2.known_indices.shape == (1, 3, 2)
        assert top2.known_weights.shape == (1, 3, 2)
        assert top2.unk_contributions.shape == (1, 3, 2)

        # At position 0: contributions [5.0, 3.0, 1.0] → top-2 are 5.0, 3.0
        top2_vals = top2.known_contributions[0, 0].sort(descending=True).values
        assert torch.allclose(top2_vals, torch.tensor([5.0, 3.0], device=device))

    def test_to_dataframe(self, device):
        """to_dataframe returns a DataFrame with expected columns and rows."""
        attr = self._make_attr(device)
        df = attr.to_dataframe()

        expected_cols = {"batch", "position", "target_token_id", "target_logit",
                         "concept_type", "concept_id", "weight", "contribution", "epsilon"}
        assert expected_cols == set(df.columns)

        # 3 positions × (3 known + 2 unknown) = 15 rows
        assert len(df) == 15
        assert set(df["concept_type"].unique()) == {"known", "discovered"}

    def test_chunk_aggregation_end_to_end(self, device):
        """Full pipeline: accumulator → result → chunk_attribution produces correct chunks."""
        dim = 16
        vocab = 100
        k_known = 3
        k_unk = 2
        seq_len = 10
        eoc_id = 99

        known_head = FakeConceptHead(20, dim, device)
        unknown_head = FakeConceptHead(30, dim, device)
        lm_head_weight = torch.randn(vocab, dim, device=device)

        accumulator = AttributionAccumulator(
            seq_len=seq_len, k_known=k_known, k_unk=k_unk, device=device,
            lm_head_weight=lm_head_weight, known_head=known_head,
            unknown_head=unknown_head,
        )

        # Simulate generation: commit all positions
        # Token sequence: [prompt, prompt, 1, 2, EOC, 3, 4, 5, EOC, 6]
        token_ids = [10, 11, 1, 2, eoc_id, 3, 4, 5, eoc_id, 6]
        prompt_len = 2

        for pos in range(seq_len):
            accumulator.commit(CommitEvent(
                step=pos,
                positions=torch.tensor([pos], device=device),
                token_ids=torch.tensor([token_ids[pos]], device=device),
                target_logits=torch.tensor([float(pos)], device=device),
                known_indices=torch.arange(k_known, device=device).unsqueeze(0),
                known_logits=torch.randn(1, k_known, device=device),
                unk_indices=torch.arange(k_unk, device=device).unsqueeze(0),
                unk_logits=torch.randn(1, k_unk, device=device),
            ))

        assert accumulator.coverage == 1.0

        attr = accumulator.result()
        assert attr.verify().passed

        # Find chunks (skip prompt)
        chunks = find_chunk_boundaries(
            token_ids, eoc_id=eoc_id,
            start_index=prompt_len, include_final_chunk=True,
        )

        # Expected: [2,4), [5,8), [9,10)
        assert len(chunks) == 3
        assert chunks[0] == (2, 4)
        assert chunks[1] == (5, 8)
        assert chunks[2] == (9, 10)

        # chunk_attribution should return entries for each chunk
        for start, end in chunks:
            entries, eps_pct = chunk_attribution(attr, start, end, batch=0)
            assert len(entries) > 0
            assert isinstance(eps_pct, float)
            # Every entry has required fields
            for e in entries:
                assert "label" in e
                assert "type" in e
                assert "contribution" in e
            # Contributions + epsilon should roughly account for all mass
            total = sum(abs(e["contribution"]) for e in entries) + abs(eps_pct / 100)
            assert total > 0

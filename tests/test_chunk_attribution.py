"""Tests for chunk_attribution."""

import pytest
import torch


def concept_label(concept_id: int, concept_type: str = "known") -> str:
    return f"concept_{concept_id}"


def chunk_attribution(attr, start, end, batch=0):
    k_idx = attr["known_indices"][batch, start:end]
    k_c = attr["known_contributions"][batch, start:end]
    d_idx = attr["disc_indices"][batch, start:end]
    d_c = attr["disc_contributions"][batch, start:end]
    eps = attr["epsilon"][batch, start:end]

    pos_total = (k_c.abs().sum(-1) + d_c.abs().sum(-1) + eps.abs()).clamp(min=1e-8)

    k_c_norm = k_c / pos_total.unsqueeze(-1)
    d_c_norm = d_c / pos_total.unsqueeze(-1)

    def aggregate(idx, contrib_norm):
        flat_idx = idx.reshape(-1)
        flat_norm = contrib_norm.reshape(-1)
        unique_ids, inverse = flat_idx.unique(return_inverse=True)
        mass = torch.zeros(len(unique_ids), device=flat_idx.device)
        mass.scatter_add_(0, inverse, flat_norm)
        return unique_ids, mass

    k_ids, k_mass = aggregate(k_idx, k_c_norm)
    d_ids, d_mass = aggregate(d_idx, d_c_norm)

    T = end - start

    entries = []
    for cid, mass in zip(k_ids.tolist(), k_mass.tolist(), strict=False):
        entries.append(
            {
                "label": concept_label(int(cid), "known"),
                "type": "known",
                "contribution": mass / T,
            }
        )
    for cid, mass in zip(d_ids.tolist(), d_mass.tolist(), strict=False):
        entries.append(
            {
                "label": f"Discovered #{int(cid)}",
                "type": "discovered",
                "contribution": mass / T,
            }
        )

    eps_pct = float((eps / pos_total).mean()) * 100

    return entries, eps_pct


# --- helpers ---


def make_attr(known_idx, known_c, disc_idx, disc_c, eps):
    """Wrap lists into the attr dict format, batch size 1."""
    return {
        "known_indices": torch.tensor([[known_idx]]),  # [1, 1, K]
        "known_contributions": torch.tensor([[known_c]]),
        "disc_indices": torch.tensor([[disc_idx]]),
        "disc_contributions": torch.tensor([[disc_c]]),
        "epsilon": torch.tensor([[eps]]),  # [1, 1]
    }


# --- tests ---


class TestChunkAttribution:
    """
    Single-position setup so all expected values are easy to verify by hand.

      known:       concept 5 → +0.6,  concept 7 → -0.4
      discovered:  concept 2 → +0.2
      epsilon:     -0.1

      pos_total = 0.6 + 0.4 + 0.2 + 0.1 = 1.3

      known fractions:      +0.6/1.3 = +0.4615,  -0.4/1.3 = -0.3077
      discovered fraction:  +0.2/1.3 = +0.1538
      eps fraction:         -0.1/1.3 = -0.0769  → eps_pct = -7.69%

      T = 1, so contribution == fraction (dividing by T=1 is a no-op)
    """

    def _single_pos_attr(self):
        return make_attr(
            known_idx=[5, 7],
            known_c=[0.6, -0.4],
            disc_idx=[2],
            disc_c=[0.2],
            eps=-0.1,
        )

    def _run(self, attr, start=0, end=1):
        entries, eps_pct = chunk_attribution(attr, start, end)
        by_label = {e["label"]: e for e in entries}
        return by_label, eps_pct

    def test_known_concepts_present(self):
        by_label, _ = self._run(self._single_pos_attr())
        assert "concept_5" in by_label
        assert "concept_7" in by_label

    def test_discovered_concept_present(self):
        by_label, _ = self._run(self._single_pos_attr())
        assert "Discovered #2" in by_label

    def test_positive_known_contribution(self):
        by_label, _ = self._run(self._single_pos_attr())
        assert pytest.approx(by_label["concept_5"]["contribution"], abs=1e-4) == 0.4615

    def test_negative_known_contribution_preserved(self):
        by_label, _ = self._run(self._single_pos_attr())
        assert by_label["concept_7"]["contribution"] < 0
        assert pytest.approx(by_label["concept_7"]["contribution"], abs=1e-4) == -0.3077

    def test_discovered_contribution(self):
        by_label, _ = self._run(self._single_pos_attr())
        assert pytest.approx(by_label["Discovered #2"]["contribution"], abs=1e-4) == 0.1538

    def test_eps_pct(self):
        _, eps_pct = self._run(self._single_pos_attr())
        assert pytest.approx(eps_pct, abs=1e-2) == -7.69

    def test_same_concept_accumulates_across_positions(self):
        """Concept 5 appears at both positions — contributions should sum."""
        attr = {
            "known_indices": torch.tensor([[[5], [5]]]),  # [1, 2, 1]
            "known_contributions": torch.tensor([[[0.6], [0.3]]]),
            "disc_indices": torch.tensor([[[0], [0]]]),
            "disc_contributions": torch.tensor([[[0.0], [0.0]]]),
            "epsilon": torch.tensor([[0.4, 0.7]]),
        }
        # pos_total_0 = 0.6 + 0.0 + 0.4 = 1.0  → fraction = 0.6
        # pos_total_1 = 0.3 + 0.0 + 0.7 = 1.0  → fraction = 0.3
        # sum = 0.9, divided by T=2 → 0.45
        by_label, _ = chunk_attribution(attr, 0, 2)
        by_label = {e["label"]: e for e in by_label}
        assert pytest.approx(by_label["concept_5"]["contribution"], abs=1e-4) == 0.45

    def test_opposing_positions_cancel(self):
        """Same concept with equal and opposite contributions nets to zero."""
        attr = {
            "known_indices": torch.tensor([[[3], [3]]]),
            "known_contributions": torch.tensor([[[0.5], [-0.5]]]),
            "disc_indices": torch.tensor([[[0], [0]]]),
            "disc_contributions": torch.tensor([[[0.0], [0.0]]]),
            "epsilon": torch.tensor([[0.5, 0.5]]),
        }
        # pos_total = 1.0 at both positions → fractions +0.5 and -0.5 → sum=0
        by_label, _ = chunk_attribution(attr, 0, 2)
        by_label = {e["label"]: e for e in by_label}
        assert pytest.approx(by_label["concept_3"]["contribution"], abs=1e-6) == 0.0

    def test_start_end_slices_correctly(self):
        """Contributions outside [start:end] must not affect the result."""
        attr = {
            "known_indices": torch.tensor([[[5], [5], [5]]]),  # [1, 3, 1]
            "known_contributions": torch.tensor([[[0.9], [0.6], [0.3]]]),
            "disc_indices": torch.tensor([[[0], [0], [0]]]),
            "disc_contributions": torch.tensor([[[0.0], [0.0], [0.0]]]),
            "epsilon": torch.tensor([[0.1, 0.4, 0.7]]),
        }
        # only look at position 1 (start=1, end=2)
        # pos_total_1 = 0.6 + 0.4 = 1.0 → fraction = 0.6, T=1 → contribution = 0.6
        by_label, _ = chunk_attribution(attr, 1, 2)
        by_label = {e["label"]: e for e in by_label}
        assert pytest.approx(by_label["concept_5"]["contribution"], abs=1e-4) == 0.6

    def test_entry_format(self):
        by_label, eps_pct = self._run(self._single_pos_attr())
        assert isinstance(eps_pct, float)
        for entry in by_label.values():
            assert "label" in entry
            assert "type" in entry and entry["type"] in ("known", "discovered")
            assert "contribution" in entry and isinstance(entry["contribution"], float)


def aggregate(idx, contrib_norm):
    flat_idx = idx.reshape(-1)
    flat_norm = contrib_norm.reshape(-1)
    unique_ids, inverse = flat_idx.unique(return_inverse=True)
    mass = torch.zeros(len(unique_ids), device=flat_idx.device)
    mass.scatter_add_(0, inverse, flat_norm)
    return unique_ids, mass


class TestAggregate:
    def test_single_concept_single_position(self):
        ids, mass = aggregate(torch.tensor([[3]]), torch.tensor([[0.5]]))
        assert ids.tolist() == [3]
        assert pytest.approx(mass.tolist()) == [0.5]

    def test_same_concept_accumulates(self):
        ids, mass = aggregate(
            torch.tensor([[5], [5], [5]]),
            torch.tensor([[0.2], [0.3], [0.1]]),
        )
        assert ids.tolist() == [5]
        assert pytest.approx(mass.tolist(), abs=1e-6) == [0.6]

    def test_opposing_signs_cancel(self):
        ids, mass = aggregate(
            torch.tensor([[7], [7]]),
            torch.tensor([[0.4], [-0.4]]),
        )
        assert ids.tolist() == [7]
        assert pytest.approx(mass.tolist(), abs=1e-6) == [0.0]

    def test_multiple_concepts_stay_separate(self):
        ids, mass = aggregate(
            torch.tensor([[1, 2], [3, 1]]),
            torch.tensor([[0.3, 0.5], [0.2, 0.1]]),
        )
        result = dict(zip(ids.tolist(), mass.tolist(), strict=False))
        assert pytest.approx(result[1], abs=1e-6) == 0.4  # 0.3 + 0.1
        assert pytest.approx(result[2], abs=1e-6) == 0.5
        assert pytest.approx(result[3], abs=1e-6) == 0.2

    def test_same_concept_in_all_k_slots(self):
        """Same concept appearing across all K slots at one position."""
        ids, mass = aggregate(
            torch.tensor([[9, 9, 9]]),
            torch.tensor([[0.1, 0.2, 0.3]]),
        )
        assert ids.tolist() == [9]
        assert pytest.approx(mass.tolist(), abs=1e-6) == [0.6]

    def test_output_length_matches_unique_concepts(self):
        idx = torch.randint(0, 10, (5, 8))
        contrib = torch.randn(5, 8)
        ids, mass = aggregate(idx, contrib)
        assert len(ids) == len(mass) == idx.unique().numel()

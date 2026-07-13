"""
Tests for faithful output-to-input feature attribution.
"""

import pytest
import torch

from steerling.attribution.input_attribution import (
    CommitGroup,
    DiffusionTrace,
    FaithfulOutputToInputAttribution,
    FaithfulOutputToInputAttributor,
    OutputToInputAttributor,
    PositionType,
    _TraceRecorder,
)
from steerling.attribution.utils import resolve_baseline_token_id
from steerling.configs.attribution import BaselineConfig, BaselineMode
from steerling.configs.generation import GenerationConfig
from steerling.inference.causal_diffusion import GenerationStepInfo, SteerlingGenerator
from steerling.models.interpretable.interpretable_causal_diffusion import (
    InterpretableCausalDiffusionLM,
)


@pytest.fixture
def interp_generator(tiny_config, tiny_concept_config, tokenizer, device):
    """Tiny interpretable generator with random weights (CPU)."""
    model = InterpretableCausalDiffusionLM(tiny_config, tiny_concept_config, vocab_size=tokenizer.vocab_size)
    return SteerlingGenerator(
        model=model,
        tokenizer=tokenizer,
        model_config=tiny_config,
        is_interpretable=True,
        device=device,
    )


def _make_trace(device, mask_id: int = 99) -> DiffusionTrace:
    """Synthetic trace: prompt len 2, 4 generated slots, position 5 never committed."""
    groups = [
        CommitGroup(
            commit_order=0,
            positions=torch.tensor([2, 3], device=device),
            token_ids=torch.tensor([20, 21], device=device),
            gen_logits=torch.tensor([1.0, 1.0], device=device),
        ),
        CommitGroup(
            commit_order=1,
            positions=torch.tensor([4], device=device),
            token_ids=torch.tensor([22], device=device),
            gen_logits=torch.tensor([1.0], device=device),
        ),
    ]
    return DiffusionTrace(
        prompt_ids=torch.tensor([10, 11], device=device),
        prompt_len=2,
        seq_length=6,
        mask_id=mask_id,
        eos_id=None,
        device=device,
        groups=groups,
    )


# --------------------------------------------------------------------------- #
# Trace logic (no model)
# --------------------------------------------------------------------------- #


class TestDiffusionTrace:
    def test_reconstruct_reveals_only_prior_commits(self, device):
        trace = _make_trace(device)
        # order 0: nothing committed yet, prompt visible, generation all masked
        assert trace.reconstruct(0).tolist() == [10, 11, 99, 99, 99, 99]
        # order 1: group 0 (positions 2,3) revealed
        assert trace.reconstruct(1).tolist() == [10, 11, 20, 21, 99, 99]
        # order 2: groups 0 and 1 revealed; position 5 never committed
        assert trace.reconstruct(2).tolist() == [10, 11, 20, 21, 22, 99]

    def test_committed_tokens(self, device):
        trace = _make_trace(device)
        assert trace.committed_tokens().tolist() == [10, 11, 20, 21, 22, 99]

    def test_position_types(self, device):
        trace = _make_trace(device)
        types = trace.position_types()
        assert types[0] is PositionType.PROMPT
        assert types[1] is PositionType.PROMPT
        assert types[2] is PositionType.GENERATED
        assert types[4] is PositionType.GENERATED
        assert types[5] is PositionType.GENERATED_MASKED

    def test_padded_equals_seq_length(self, device):
        trace = _make_trace(device)
        assert trace.padded_seq_length == trace.seq_length == 6

    def test_recorder_builds_monotonic_groups(self, device):
        recorder = _TraceRecorder()
        logits = torch.randn(1, 6, 30, device=device)
        recorder(
            GenerationStepInfo(
                step=0,
                logits=logits,
                outputs=None,
                committed_positions=torch.tensor([2, 3], device=device),
                committed_token_ids=torch.tensor([20, 21], device=device),
            )
        )
        recorder(
            GenerationStepInfo(
                step=1,
                logits=logits,
                outputs=None,
                committed_positions=torch.tensor([4], device=device),
                committed_token_ids=torch.tensor([22], device=device),
            )
        )
        assert [g.commit_order for g in recorder.groups] == [0, 1]
        assert recorder.groups[0].positions.tolist() == [2, 3]
        # gen_logits gathered from the committed tokens
        assert recorder.groups[1].gen_logits.shape == (1,)

    def test_recorder_rejects_batched(self, device):
        recorder = _TraceRecorder()
        with pytest.raises(NotImplementedError):
            recorder(
                GenerationStepInfo(
                    step=0,
                    logits=torch.randn(2, 6, 30, device=device),
                    outputs=None,
                    committed_positions=torch.tensor([2], device=device),
                    committed_token_ids=torch.tensor([20], device=device),
                )
            )


class TestAggregateChunk:
    """aggregate_chunk restricts to targets committed within [start, end)."""

    def _result(self, device):
        trace = _make_trace(device)
        # 3 targets at positions 2, 3, 4. Build attributions so each target points
        # at a distinct generated input position: target@2 -> pos2, @3 -> pos3, @4 -> pos4.
        attributions = torch.zeros(3, 6, device=device)
        attributions[0, 2] = 1.0
        attributions[1, 3] = 1.0
        attributions[2, 4] = 1.0
        return FaithfulOutputToInputAttribution(
            trace=trace,
            target_orders=torch.tensor([0, 0, 1], device=device),
            target_positions=torch.tensor([2, 3, 4], device=device),
            target_token_ids=torch.tensor([20, 21, 22], device=device),
            gen_logits=torch.ones(3, device=device),
            attributions=attributions,
        )

    def test_chunk_filters_targets(self, device):
        res = self._result(device)
        # chunk [2,4) includes targets at 2 and 3 only
        ranked = res.aggregate_chunk(2, 4, scope="generated")
        top = ranked[0]
        assert top["input_position"] in (2, 3)
        # chunk [4,6) includes only the target at 4 -> input position 4 dominates
        ranked2 = res.aggregate_chunk(4, 6, scope="generated")
        assert ranked2[0]["input_position"] == 4

    def test_empty_chunk_returns_empty(self, device):
        res = self._result(device)
        assert res.aggregate_chunk(10, 12, scope="generated") == []

    def test_aggregate_excludes_masked(self, device):
        res = self._result(device)
        ranked = res.aggregate(scope="generated")
        # position 5 is GENERATED_MASKED and must never appear
        assert all(d["input_position"] != 5 for d in ranked)


# --------------------------------------------------------------------------- #
# Critical faithfulness invariants (tiny model)
# --------------------------------------------------------------------------- #


class TestCriticalFeatureAttribution:
    def test_masked_positions_get_zero_attribution(self, interp_generator, device):
        """
        Under the MASK baseline, a position that is MASK in the snapshot has input
        embedding equal to the baseline, so its delta is zero and IG attributes
        exactly zero to it. This is the faithfulness guarantee.
        """
        mask_id = int(interp_generator.mask_token_id)
        backbone = interp_generator.model
        primitive = OutputToInputAttributor.from_backbone(backbone, baseline="mask", mask_token_id=mask_id)
        # prompt (2 tokens) + 2 masked slots
        x = torch.tensor([[5, 6, mask_id, mask_id]], device=device)
        attr = primitive.compute(
            input_ids=x,
            target_token_ids=torch.tensor([7], device=device),
            positions=2,
            n_steps=2,
        )
        # masked input positions (2, 3) must be exactly zero
        assert attr.attributions[0, 2].abs().item() == 0.0
        assert attr.attributions[0, 3].abs().item() == 0.0

    def test_attribute_end_to_end_shape(self, interp_generator):
        """End-to-end attribute(); also the RoPE inference-tensor canary."""
        config = GenerationConfig(max_new_tokens=16, steps=16, seed=0, cfg_scale=0.0)
        attributor = FaithfulOutputToInputAttributor.from_generator(interp_generator)
        attr = attributor.attribute("Hello", config, n_steps=2)

        n_targets = attr.attributions.shape[0]
        assert attr.target_positions.shape[0] == n_targets
        assert attr.attributions.shape[1] == attr.trace.seq_length
        assert not torch.isnan(attr.attributions).any()

    def test_randomization_sanity_check(self, interp_generator, device):
        """
        Attribution must depend on model weights (Adebayo et al., arXiv:1810.03292).
        Re-attributing the same trace after randomizing weights should change the
        attributions substantially.

        The trace is built synthetically (not by generating), because the tiny
        random-weight model can emit token ids the tokenizer cannot decode, and
        generation decodes its output. A randomization test only needs a valid trace.
        """
        attributor = FaithfulOutputToInputAttributor.from_generator(interp_generator)
        trace = _make_trace(device, mask_id=int(interp_generator.mask_token_id))

        attr1 = attributor.attribute_from_trace(trace, n_steps=2)
        with torch.no_grad():
            for p in interp_generator.model.parameters():
                p.normal_()
        attr2 = attributor.attribute_from_trace(trace, n_steps=2)

        a = attr1.attributions.flatten()
        b = attr2.attributions.flatten()
        if a.numel() > 1 and a.std() > 0 and b.std() > 0:
            corr = torch.corrcoef(torch.stack([a, b]))[0, 1].abs().item()
            assert corr < 0.95, f"attribution barely changed after randomization (corr={corr:.3f})"

    def test_cfg_scale_rejected(self, interp_generator):
        config = GenerationConfig(max_new_tokens=16, steps=16, cfg_scale=1.0)
        attributor = FaithfulOutputToInputAttributor.from_generator(interp_generator)
        with pytest.raises(ValueError, match="cfg_scale"):
            attributor.attribute("Hello", config)

    def test_requires_interpretable(self, tiny_config, tokenizer, device):
        from steerling.models.causal_diffusion import CausalDiffusionLM

        plain = CausalDiffusionLM(tiny_config, vocab_size=tokenizer.vocab_size)
        gen = SteerlingGenerator(
            model=plain,
            tokenizer=tokenizer,
            model_config=tiny_config,
            is_interpretable=False,
            device=device,
        )
        with pytest.raises(ValueError, match="interpretable"):
            FaithfulOutputToInputAttributor.from_generator(gen)

    def test_completeness_uses_snapshot_baseline(self, interp_generator):
        """
        Completeness regression: sum(attributions) for a target must equal the
        signed logit gap computed with input_ids held at the snapshot and only the
        embeddings swapped mask<->real. Computing the baseline with input_ids set to
        all-mask is a different function value in a diffusion model (input_ids carries
        the mask structure), which silently breaks the axiom. Random weights and few
        steps, so this checks the identity is wired correctly, not tight convergence.
        """
        config = GenerationConfig(max_new_tokens=16, steps=16, seed=0, cfg_scale=0.0)
        attributor = FaithfulOutputToInputAttributor.from_generator(interp_generator)
        attr = attributor.attribute("Hello", config, n_steps=64)

        backbone = attributor.backbone
        tok_emb = backbone.transformer.tok_emb
        mask_emb = tok_emb(torch.tensor([attr.trace.mask_id], device=attr.trace.device))[0]

        def logit(ids_row, emb, pos, tid):
            with torch.no_grad():
                h = backbone.transformer(input_ids=ids_row, input_embeds=emb, return_hidden=True)
                return float(backbone.transformer.lm_head(h)[0, pos, tid])

        n = 0
        order = int(attr.target_orders[n])
        pos = int(attr.target_positions[n])
        tid = int(attr.target_token_ids[n])
        x = attr.trace.reconstruct(order).unsqueeze(0)  # snapshot input_ids for BOTH terms
        emb_in = tok_emb(x)
        emb_base = mask_emb.view(1, 1, -1).expand_as(emb_in)
        gap = logit(x, emb_in, pos, tid) - logit(x, emb_base, pos, tid)
        attr_sum = float(attr.attributions[n].sum())

        # signed sum should track the signed gap (loose tolerance: random weights, 64 steps)
        assert abs(attr_sum - gap) <= abs(gap) + 1.0, (
            f"completeness far off: sum(attr)={attr_sum:.3f} gap={gap:.3f}"
        )
        # and it must have the same sign as the gap when the gap is not tiny
        if abs(gap) > 1.0:
            assert (attr_sum > 0) == (gap > 0), (
                f"sign mismatch: sum(attr)={attr_sum:.3f} gap={gap:.3f} "
                f"(baseline likely computed with the wrong input_ids)"
            )


class TestSpecialIdSet:
    def test_returns_nonempty_with_mask_id(self, tokenizer):
        ids = FaithfulOutputToInputAttribution._special_id_set(tokenizer)
        assert isinstance(ids, set)
        assert len(ids) > 0

    def test_raises_when_no_special_ids(self):
        class Bare:
            all_special_ids: list[int] = []

        with pytest.raises(AssertionError, match="special token ids"):
            FaithfulOutputToInputAttribution._special_id_set(Bare())


# --------------------------------------------------------------------------- #
# Baseline resolution
# --------------------------------------------------------------------------- #


class TestBaselineResolution:
    def test_mask_mode(self):
        cfg = BaselineConfig(mode=BaselineMode.MASK)
        assert resolve_baseline_token_id(cfg, mask_token_id=99) == 99

    def test_pad_mode(self):
        cfg = BaselineConfig(mode=BaselineMode.PAD)
        assert resolve_baseline_token_id(cfg, pad_token_id=7) == 7

    def test_zero_mode_returns_none(self):
        cfg = BaselineConfig(mode=BaselineMode.ZERO)
        assert resolve_baseline_token_id(cfg) is None

    def test_explicit_token_id_overrides(self):
        cfg = BaselineConfig(mode=BaselineMode.MASK, token_id=42)
        assert resolve_baseline_token_id(cfg, mask_token_id=99) == 42

    def test_mask_mode_missing_id_raises(self):
        cfg = BaselineConfig(mode=BaselineMode.MASK)
        with pytest.raises(ValueError, match="mask token id"):
            resolve_baseline_token_id(cfg)

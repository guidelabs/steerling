"""
Output -> input feature attribution via integrated gradients.

Answers: "which input tokens caused this generated token?"

Faithfulness: in a masked diffusion model the explained token sits at a masked
slot inside a partially-masked sequence, and tokens commit in a confidence order.
Attributing against the prompt alone or the finished text is unfaithful. The
(our) faithful attributor replays the exact snapshot the model saw when each token was
committed. IG uses a MASK baseline (the trained absence-of-information state), so
positions still masked in a snapshot get zero attribution by-construction.

Single-step (Input x Gradient) eq:
    C(x_t, y_j) = (E[x_t] - E[mask])^T @ grad_{E[x_t]} logit_{y_j}

Multi-step IG averages over right-Riemann interpolation points between baseline and
input; n_steps=32 is recommended for faithful magnitudes, n_steps=1 for token
selection only.

Interpretable models only: the generation callback that records snapshots fires on
the interpretable forward path. Steered generation is not yet supported. The
replay would have to re-apply the same injection.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from steerling.attribution.utils import (
    get_baseline_embedding,
    integrated_gradients,
    normalize_attributions,
)
from steerling.configs.attribution import BaselineConfig, BaselineMode

if TYPE_CHECKING:
    from steerling.configs.generation import GenerationConfig
    from steerling.inference.causal_diffusion import GenerationStepInfo, SteerlingGenerator


# --------------------------------------------------------------------------- #
# Single-target token attribution primitive.
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class OutputToInputAttribution:
    """
    Result class.
    Attribution of a target token to positions of one input sequence.

    ``attributions[b, t]`` is how much input position ``t`` contributed to the
    logit of ``target_token_ids[b]`` at ``positions[b]``. ``input_ids`` is the full
    sequence the model conditioned on (for diffusion, a partial-mask snapshot).
    """

    input_ids: Tensor  # [B, T]
    target_token_ids: Tensor  # [B]
    positions: int | Tensor
    attributions: Tensor  # [B, T]

    def normalize(self, eps: float = 1e-6) -> OutputToInputAttribution:
        return OutputToInputAttribution(
            input_ids=self.input_ids,
            target_token_ids=self.target_token_ids,
            positions=self.positions,
            attributions=normalize_attributions(self.attributions, dim=-1, eps=eps),
        )

    def top_tokens(self, k: int, batch_idx: int = 0, signed: bool = True) -> list[tuple[int, float]]:
        """Top-k input positions for one target as (position, signed_score)."""
        scores = self.attributions[batch_idx]
        kk = min(k, scores.shape[0])
        keys = scores if signed else scores.abs()
        _, topk_idx = torch.topk(keys, kk)
        return [(int(topk_idx[i]), float(scores[topk_idx[i]])) for i in range(kk)]

    def to_dataframe(self):
        import pandas as pd

        rows = []
        batch_size, seq_len = self.input_ids.shape
        for b in range(batch_size):
            target_id = int(self.target_token_ids[b])
            pos = int(self.positions) if isinstance(self.positions, int) else int(self.positions[b])
            for t in range(seq_len):
                rows.append(
                    {
                        "batch": b,
                        "target_token_id": target_id,
                        "output_position": pos,
                        "input_position": t,
                        "input_token_id": int(self.input_ids[b, t]),
                        "attribution": float(self.attributions[b, t]),
                    }
                )
        return pd.DataFrame(rows)


class OutputToInputAttributor:
    """
    Integrated-gradients primitive.
    """

    def __init__(
        self,
        backbone: Any,
        baseline_config: BaselineConfig | None = None,
        *,
        mask_token_id: int | None = None,
        pad_token_id: int | None = None,
    ):
        self.backbone = backbone
        self.baseline_config = baseline_config or BaselineConfig()
        self._mask_token_id = mask_token_id
        self._pad_token_id = pad_token_id
        self._baseline_embedding: Tensor | None = None

    @classmethod
    def from_backbone(
        cls,
        backbone: Any,
        baseline: BaselineConfig | str = "mask",
        *,
        mask_token_id: int | None = None,
        pad_token_id: int | None = None,
    ) -> OutputToInputAttributor:
        if isinstance(baseline, str):
            baseline = BaselineConfig(mode=BaselineMode(baseline))
        return cls(backbone, baseline, mask_token_id=mask_token_id, pad_token_id=pad_token_id)

    def _get_baseline_embedding(self) -> Tensor:
        if self._baseline_embedding is None:
            self._baseline_embedding = get_baseline_embedding(
                self.backbone,
                self.baseline_config,
                mask_token_id=self._mask_token_id,
                pad_token_id=self._pad_token_id,
            )
        return self._baseline_embedding

    @torch.compiler.disable
    def compute(
        self,
        input_ids: Tensor,
        target_token_ids: Tensor,
        positions: int | Tensor,
        n_steps: int = 1,
        baseline_embedding: Tensor | None = None,
    ) -> OutputToInputAttribution:
        """
        Attribute target tokens at given positions within ``input_ids``.

        ``input_ids`` must be the exact sequence the model conditioned on; for a
        diffusion model this is a partial-mask snapshot and the explained positions
        are masked slots within it. ``baseline_embedding`` ([D] or [T, D]) overrides
        the configured baseline; the faithful attributor passes a per-position one.
        """
        baseline_emb = (
            baseline_embedding if baseline_embedding is not None else self._get_baseline_embedding()
        )
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if isinstance(positions, int):
            pos_tensor = torch.full((batch_size,), positions, device=device, dtype=torch.long)
        else:
            pos_tensor = positions.to(device)

        assert int(pos_tensor.min()) >= 0 and int(pos_tensor.max()) < seq_len, (
            f"positions must index into input_ids (length {seq_len}); got "
            f"[{int(pos_tensor.min())}, {int(pos_tensor.max())}]. Pass the full sequence "
            f"the model conditioned on (the partial-mask snapshot)."
        )

        def target_fn(logits: Tensor, outputs: object, _tids=target_token_ids, _pos=pos_tensor) -> Tensor:
            batch_idx = torch.arange(batch_size, device=device)
            pos_logits = logits[batch_idx, _pos]  # [B, V]
            return pos_logits[batch_idx, _tids]  # [B]

        attributions = integrated_gradients(
            backbone=self.backbone,
            input_ids=input_ids,
            baseline_embedding=baseline_emb,
            target_fn=target_fn,
            n_steps=n_steps,
        )

        return OutputToInputAttribution(
            input_ids=input_ids.detach(),
            target_token_ids=target_token_ids.detach(),
            positions=positions,
            attributions=attributions.detach(),
        )


# --------------------------------------------------------------------------- #
# Diffusion replay: trace + faithful generation-level attributor.
# --------------------------------------------------------------------------- #

_NEVER_COMMITTED = torch.iinfo(torch.long).max


class PositionType(StrEnum):
    """Role of an input position in a reconstructed snapshot."""

    PROMPT = "prompt"
    GENERATED = "generated"  # committed during generation
    GENERATED_MASKED = "generated_masked"  # generation region, never committed


@dataclass
class CommitGroup:
    """Tokens committed together in one step_callback invocation."""

    order: int
    positions: Tensor  # [P]
    token_ids: Tensor  # [P]
    gen_logits: Tensor  # [P] generation-time logit of each committed token


@dataclass
class DiffusionTrace:
    """
    Replayable record of one generation: the commit schedule needed to
    reconstruct the input the model saw at every step.

    The open generator builds ``x`` as prompt + all-masked generation region with
    no padding and no EOS tail, so ``padded_seq_length == seq_length`` and there is
    no EOS-pad region.
    """

    prompt_ids: Tensor
    prompt_len: int
    seq_length: int
    mask_id: int
    eos_id: int | None
    device: torch.device
    groups: list[CommitGroup]

    def __post_init__(self) -> None:
        self.padded_seq_length = self.seq_length
        self._base = self._build_base()
        self._order_at, self._token_at = self._build_caches()

    def _build_base(self) -> Tensor:
        """prompt | masks, the sequence generate_full starts from."""
        x = torch.full((self.seq_length,), self.mask_id, dtype=torch.long, device=self.device)
        x[: self.prompt_len] = self.prompt_ids
        return x

    def _build_caches(self) -> tuple[Tensor, Tensor]:
        order_at = torch.full((self.seq_length,), _NEVER_COMMITTED, dtype=torch.long, device=self.device)
        order_at[: self.prompt_len] = -1
        token_at = self._base.clone()
        for g in self.groups:
            order_at[g.positions] = g.order
            token_at[g.positions] = g.token_ids
        return order_at, token_at

    def committed_tokens(self) -> Tensor:
        return self._token_at

    def reconstruct(self, order: int) -> Tensor:
        """The input the model saw when producing commit group ``order``."""
        revealed = self._order_at < order
        return torch.where(revealed, self._token_at, self._base)

    def position_types(self) -> list[PositionType]:
        order = self._order_at
        types = [PositionType.GENERATED] * self.seq_length
        for p in range(self.prompt_len):
            types[p] = PositionType.PROMPT
        for p in range(self.prompt_len, self.seq_length):
            if int(order[p]) == _NEVER_COMMITTED:
                types[p] = PositionType.GENERATED_MASKED
        return types


class _TraceRecorder:
    """step_callback that builds a DiffusionTrace live during generation."""

    def __init__(self) -> None:
        self.groups: list[CommitGroup] = []
        self._order = 0

    def __call__(self, info: GenerationStepInfo) -> None:
        if info.logits.shape[0] != 1:
            raise NotImplementedError(
                "Faithful attribution records one generation at a time (B=1). Call per prompt for now."
            )
        # Generation runs under @torch.inference_mode(), so tensors created here are
        # inference tensors, and indexing logits with them later poisons autograd
        # ("inference tensors cannot be saved for backward"). .tolist() alone does not
        # help, because the following torch.tensor(...) still runs inside inference
        # mode. Build the stored tensors in an explicit normal-mode block.
        device = info.logits.device
        pos_list = info.committed_positions.tolist()
        tok_list = info.committed_token_ids.tolist()
        gen = info.logits[0, info.committed_positions].gather(-1, info.committed_token_ids.unsqueeze(-1))
        gen_list = gen.squeeze(-1).tolist()
        with torch.inference_mode(False):
            pos = torch.tensor(pos_list, dtype=torch.long, device=device)
            tok = torch.tensor(tok_list, dtype=torch.long, device=device)
            gen_logits = torch.tensor(gen_list, device=device)
        self.groups.append(
            CommitGroup(order=self._order, positions=pos, token_ids=tok, gen_logits=gen_logits)
        )
        self._order += 1


@dataclass
class FaithfulOutputToInputAttribution:
    """
    Per-token output -> input attribution for a whole generation.

    ``attributions[n]`` is the [T] attribution of input positions to the n-th
    committed target, from the snapshot the model saw when it was committed.
    """

    trace: DiffusionTrace
    target_orders: Tensor  # [N]
    target_positions: Tensor  # [N]
    target_token_ids: Tensor  # [N]
    gen_logits: Tensor  # [N]
    attributions: Tensor  # [N, T]

    def _scope_positions(self, scope: str) -> Tensor:
        types = self.trace.position_types()
        keep = torch.zeros(self.trace.seq_length, dtype=torch.bool)
        for p, t in enumerate(types):
            if (
                (scope == "prompt" and t is PositionType.PROMPT)
                or (scope == "generated" and t is PositionType.GENERATED)
                or (scope == "all" and t in (PositionType.PROMPT, PositionType.GENERATED))
            ):
                keep[p] = True
        return keep.nonzero(as_tuple=False).squeeze(-1)

    def explanation(self, output_position: int) -> Tensor:
        """[T] attribution vector for the target committed at ``output_position``."""
        idx = (self.target_positions == output_position).nonzero(as_tuple=False)
        if idx.numel() == 0:
            raise KeyError(f"No target was committed at position {output_position}")
        return self.attributions[int(idx[0])]

    def _aggregate_targets(self, target_idx: Tensor, scope: str, reduce: str, signed: bool) -> list[dict]:
        """Aggregate a chosen subset of targets over a scope. Shared by aggregate/aggregate_chunk."""
        positions = self._scope_positions(scope).to(self.attributions.device)
        if target_idx.numel() == 0 or positions.numel() == 0:
            return []
        scoped = self.attributions.index_select(0, target_idx).index_select(1, positions)  # [n, S]

        denom = scoped.abs().sum(dim=-1, keepdim=True).clamp(min=1e-8)
        vals = scoped / denom  # equal mass per target
        if not signed:
            vals = vals.abs()
        agg = vals.mean(dim=0) if reduce == "mean" else vals.sum(dim=0)  # [S]

        tokens = self.trace.committed_tokens().index_select(0, positions)
        sort_key = agg if signed else agg.abs()
        ranked = torch.argsort(sort_key, descending=True).tolist()
        return [
            {
                "input_position": int(positions[i]),
                "input_token_id": int(tokens[i]),
                "score": float(agg[i]),
            }
            for i in ranked
        ]

    def aggregate(self, scope: str = "prompt", reduce: str = "mean", signed: bool = True) -> list[dict]:
        """
        Rank input positions by attribution aggregated across all targets.

        Each target is normalized over the scope by its absolute mass (equal vote per
        generated token), then reduced. signed=True ranks by net effect, signed=False
        by magnitude. generated_masked positions are always excluded.
        """
        all_targets = torch.arange(self.attributions.shape[0], device=self.attributions.device)
        return self._aggregate_targets(all_targets, scope, reduce, signed)

    def aggregate_chunk(
        self, start: int, end: int, scope: str = "generated", reduce: str = "mean", signed: bool = True
    ) -> list[dict]:
        """
        Like ``aggregate`` but only over targets committed within ``[start, end)``.

        Use with ``find_chunks`` (from concept_attribution) to attribute each output
        chunk separately. scope defaults to 'generated'; pass 'prompt' to rank which
        prompt tokens drove a given output chunk.
        """
        pos = self.target_positions
        mask = (pos >= start) & (pos < end)
        target_idx = mask.nonzero(as_tuple=False).squeeze(-1)
        return self._aggregate_targets(target_idx, scope, reduce, signed)

    def top_input_tokens(self, k: int, scope: str = "prompt") -> list[dict]:
        """Top-k input positions by aggregated importance within ``scope``."""
        return self.aggregate(scope=scope)[:k]

    def to_dataframe(self):
        import pandas as pd

        types = self.trace.position_types()
        tokens = self.trace.committed_tokens()
        n_targets, seq_len = self.attributions.shape
        rows = []
        for n in range(n_targets):
            target_pos = int(self.target_positions[n])
            target_tok = int(self.target_token_ids[n])
            order = int(self.target_orders[n])
            for p in range(seq_len):
                rows.append(
                    {
                        "commit_order": order,
                        "target_position": target_pos,
                        "target_token_id": target_tok,
                        "input_position": p,
                        "input_token_id": int(tokens[p]),
                        "input_type": types[p].value,
                        "attribution": float(self.attributions[n, p]),
                    }
                )
        return pd.DataFrame(rows)

    @staticmethod
    def _special_id_set(tokenizer: Any) -> set[int]:
        """Special token ids to exclude from heatmaps."""
        ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        # chunk/turn separators that may not be registered as 'special'
        for name in ("eot_id", "endofchunk_token_id"):
            tok_id = getattr(tokenizer, name, None)
            if tok_id is not None:
                ids.add(int(tok_id))
        assert ids, (
            f"No special token ids found on {type(tokenizer).__name__}. "
            f"Expected all_special_ids to be populated."
        )
        return ids

    def to_visualization(
        self,
        tokenizer: Any,
        output_position: int | None = None,
        scope: str = "prompt",
        drop_special: bool = True,
        drop_blank: bool = True,
    ) -> str:
        """Text heatmap over input tokens within ``scope`` (aggregate, or one target)."""
        blocks = " ░▒▓█"
        positions = self._scope_positions(scope)
        tokens = self.trace.committed_tokens()

        special = set()
        if drop_special or drop_blank:
            special = self._special_id_set(tokenizer)

            def keep(p: int) -> bool:
                tok_id = int(tokens[p])
                decoded = tokenizer.decode([tok_id]).strip()
                return (
                    tok_id not in special
                    and decoded != ""
                    and decoded.lower() not in {"user", "assistant", "system"}
                )

            positions = positions[torch.tensor([keep(int(p)) for p in positions], dtype=torch.bool)]

        if output_position is None:
            score_by_pos = {d["input_position"]: abs(d["score"]) for d in self.aggregate(scope=scope)}
            vals = torch.tensor([score_by_pos[int(p)] for p in positions])
        else:
            vec = self.explanation(output_position).abs().cpu()
            vals = vec.index_select(0, positions)

        max_val = vals.max().clamp(min=1e-8)
        normed = (vals / max_val).clamp(0, 1)
        parts = []
        for i, p in enumerate(positions.tolist()):
            token_str = tokenizer.decode([int(tokens[p])])
            level = int(normed[i].item() * (len(blocks) - 1))
            parts.append(f"{blocks[level]}{token_str}")
        return "".join(parts)


def _clear_inference_caches(model: Any) -> None:
    """
    Drop tensors cached during inference-mode generation before running IG.

    ``generate_full`` runs under ``@torch.inference_mode()``, so caches it fills
    (RoPE sin/cos tables, block-causal attention masks) hold inference tensors. IG
    then runs forwards with grad enabled, and autograd rejects inference tensors.
    Clearing forces these to rebuild as normal tensors on the first IG forward.
    No-op for any cache attribute that is absent.
    """
    for module in model.modules():
        rope_cache = getattr(module, "_RotaryEmbedding__cache", None)
        if rope_cache is not None and hasattr(rope_cache, "_cache"):
            rope_cache._cache.clear()
        for attr in ("_mask_cache", "_sdpa_mask_cache"):
            cache = getattr(module, attr, None)
            if isinstance(cache, dict):
                cache.clear()


class FaithfulOutputToInputAttributor:
    """
    Record a generation trace, reconstruct each snapshot, and run the IG primitive
    per commit group. Interpretable models only; steered generation is not supported.

    Usage:
        attributor = FaithfulOutputToInputAttributor.from_generator(generator)
        attr = attributor.attribute("Tell me a story", config, n_steps=32)
        print(attr.top_input_tokens(k=5, scope="prompt"))
    """

    def __init__(self, generator: SteerlingGenerator) -> None:
        if not generator.is_interpretable:
            raise ValueError("Faithful feature attribution requires an interpretable model.")
        self.generator = generator
        self.backbone = generator.model  # InterpretableCausalDiffusionLM
        self.device = generator.device
        self._primitive = OutputToInputAttributor.from_backbone(
            self.backbone,
            baseline="mask",
            mask_token_id=int(generator.mask_token_id),
        )

    @classmethod
    def from_generator(cls, generator: SteerlingGenerator) -> FaithfulOutputToInputAttributor:
        return cls(generator=generator)

    def attribute(
        self,
        prompt: str | list[int],
        config: GenerationConfig | None = None,
        n_steps: int = 32,
    ) -> FaithfulOutputToInputAttribution:
        """Generate, record the trace, and attribute every committed token."""
        if config is None:
            from steerling.configs.generation import GenerationConfig as GenCfg

            config = GenCfg(max_new_tokens=128)
        if config.cfg_scale > 0.0:
            raise ValueError(
                "Faithful attribution needs cfg_scale=0: the recording callback only "
                "fires on the non-CFG interpretable forward path."
            )

        recorder = _TraceRecorder()
        prompt_tensor = self._encode_prompt(prompt)
        self.generator.generate_full(prompt_tensor, config, step_callback=recorder)
        trace = self._build_trace(prompt_tensor, config, recorder.groups)
        return self.attribute_from_trace(trace, n_steps=n_steps)

    @torch.compiler.disable
    def attribute_from_trace(
        self, trace: DiffusionTrace, n_steps: int = 32
    ) -> FaithfulOutputToInputAttribution:
        """Attribute every committed token in a pre-recorded trace."""
        _clear_inference_caches(self.backbone)  # rebuild RoPE/mask caches as normal tensors
        baseline = self._baseline_embedding(trace)  # [T, D]

        attr_chunks: list[Tensor] = []
        orders: list[Tensor] = []
        positions: list[Tensor] = []
        tokens: list[Tensor] = []
        gen_logits: list[Tensor] = []

        for group in trace.groups:
            x = trace.reconstruct(group.order)  # [T]
            chunk = self._ig_group(x, group.positions, group.token_ids, baseline, n_steps)  # [P, T]
            attr_chunks.append(chunk)
            orders.append(torch.full_like(group.positions, group.order))
            positions.append(group.positions)
            tokens.append(group.token_ids)
            gen_logits.append(group.gen_logits)

        if not attr_chunks:
            warnings.warn("Trace contains no commits; nothing to attribute.", stacklevel=2)

        empty_long = torch.empty(0, dtype=torch.long, device=self.device)
        return FaithfulOutputToInputAttribution(
            trace=trace,
            target_orders=torch.cat(orders) if orders else empty_long,
            target_positions=torch.cat(positions) if positions else empty_long,
            target_token_ids=torch.cat(tokens) if tokens else empty_long,
            gen_logits=torch.cat(gen_logits) if gen_logits else torch.empty(0, device=self.device),
            attributions=(
                torch.cat(attr_chunks).detach()
                if attr_chunks
                else torch.empty(0, trace.seq_length, device=self.device)
            ),
        )

    def _ig_group(
        self, x: Tensor, positions: Tensor, token_ids: Tensor, baseline: Tensor, n_steps: int
    ) -> Tensor:
        """IG for the P targets committed in one group. Snapshot tiled into the batch."""
        n_targets = positions.shape[0]
        x_batch = x.unsqueeze(0).expand(n_targets, -1).contiguous()  # [P, T]
        attr = self._primitive.compute(
            input_ids=x_batch,
            target_token_ids=token_ids,
            positions=positions,
            n_steps=n_steps,
            baseline_embedding=baseline,  # [T, D]
        )
        return attr.attributions  # [P, T]

    def _baseline_embedding(self, trace: DiffusionTrace) -> Tensor:
        """Per-position baseline: MASK everywhere (no EOS tail in the open layout)."""
        ids = torch.full((trace.seq_length,), trace.mask_id, dtype=torch.long, device=self.device)
        with torch.no_grad():
            return self.backbone.transformer.tok_emb(ids)  # [T, D]

    def _encode_prompt(self, prompt: str | list[int]) -> Tensor:
        """Match generate_full's prompt handling: str -> encode, list[int] -> tensor."""
        if isinstance(prompt, str):
            ids = self.generator.tokenizer.encode(prompt, add_special_tokens=False)
        else:
            ids = list(prompt)
        return torch.tensor([ids], dtype=torch.long, device=self.device)

    def _build_trace(
        self, prompt_tensor: Tensor, config: GenerationConfig, groups: list[CommitGroup]
    ) -> DiffusionTrace:
        """Recompute the sequence layout generate_full used (no padding, no EOS tail)."""
        prompt_ids = prompt_tensor[0]
        prompt_len = int(prompt_ids.shape[0])
        seq_length = prompt_len + int(config.max_new_tokens)
        eos_id = int(self.generator.eos_token_id) if self.generator.eos_token_id is not None else None
        return DiffusionTrace(
            prompt_ids=prompt_ids.to(self.device),
            prompt_len=prompt_len,
            seq_length=seq_length,
            mask_id=int(self.generator.mask_token_id),
            eos_id=eos_id,
            device=self.device,
            groups=groups,
        )

"""Faithful concept attribution captured during generation."""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import torch
from torch import Tensor

from steerling import GenerationConfig, SteerlingGenerator
from steerling.attribution.trace import CommitRecord, DiffusionTrace
from steerling.attribution.utils import find_chunk_boundaries
from steerling.inference.causal_diffusion import GenerationOutput, GenerationStepInfo


@dataclass(frozen=True)
class VerificationResult:
    """Result of attribution decomposition verification."""

    passed: bool
    max_abs_error: float
    error_percentiles: dict[str, float]
    details: str

    def __repr__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"VerificationResult({status}, {self.details})"


@torch.no_grad()
def _compute_unknown_topk(
    unknown_head,
    hidden: Tensor,
    k: int = 64,
) -> tuple[Tensor | None, Tensor | None]:
    """
    Compute unknown concept top-k for a slice of positions.

    With ``minimal_output=True``, unknown top-k is not included in
    the model outputs. This helper computes it from hidden states
    for only the committed positions (typically tokens_per_step),
    keeping memory O(P * k).

    Args:
        unknown_head: The unknown ConceptHead, or None.
        hidden: [1, P, D] hidden states for positions of interest.
        k: Number of top concepts to return per position.

    Returns:
        (indices, logits) each [1, P, k], or (None, None).
    """
    if unknown_head is None or k <= 0:
        return None, None

    _, indices, logits = unknown_head.linear_features_topk_factorized(
        hidden,
        k=k,
        block_size=unknown_head.block_size,
    )

    return indices, logits  # [1, P, k], [1, P, k]


@dataclass(frozen=True)
class OutputToConceptAttribution:
    """
    Token-level attribution of output logits to concepts.

    All tensors use batch dimension first. For single examples, B=1.

    Attributes:
        target_token_ids: [B, T] tokens being explained
        target_logits: [B, T] logit values for those tokens
        known_indices: [B, T, k_known] active known concept indices
        known_weights: [B, T, k_known] sigmoid weights in [0, 1]
        known_contributions: [B, T, k_known] w_i * (v_i^T @ W_{y_t})
        unk_indices: [B, T, k_unk] active unknown concept indices
        unk_weights: [B, T, k_unk] sigmoid weights in [0, 1]
        unk_contributions: [B, T, k_unk] w_j * (v_j^T @ W_{y_t})
        epsilon_contribution: [B, T] residual contribution per position
        committed: [B, T] bool mask, True for positions that were committed
    """

    target_token_ids: Tensor
    target_logits: Tensor
    known_indices: Tensor
    known_weights: Tensor
    known_contributions: Tensor
    unk_indices: Tensor
    unk_weights: Tensor
    unk_contributions: Tensor
    epsilon_contribution: Tensor
    committed: Tensor

    def verify(self, atol: float = 1e-4) -> VerificationResult:
        """Verify contributions sum to target logits."""
        reconstructed = (
            self.known_contributions.sum(dim=-1)
            + self.unk_contributions.sum(dim=-1)
            + self.epsilon_contribution
        )
        errors = (reconstructed - self.target_logits).abs()

        return VerificationResult(
            passed=bool((errors < atol).all()),
            max_abs_error=float(errors.max()),
            error_percentiles={
                "p50": float(errors.median()),
                "p95": float(errors.quantile(0.95)),
                "p99": float(errors.quantile(0.99)),
            },
            details=f"Max error: {errors.max():.6f}, Mean: {errors.mean():.6f}",
        )

    def top_k(self, k: int) -> OutputToConceptAttribution:
        """Return attribution with only top-k concepts by absolute contribution."""
        k_known = min(k, self.known_contributions.shape[-1])
        k_unk = min(k, self.unk_contributions.shape[-1])

        if k_known == 0 and k_unk == 0:
            return self

        if k_known > 0:
            _, known_sel = torch.topk(self.known_contributions.abs(), k_known, dim=-1)
            known_indices = torch.gather(self.known_indices, -1, known_sel)
            known_weights = torch.gather(self.known_weights, -1, known_sel)
            known_contributions = torch.gather(self.known_contributions, -1, known_sel)
        else:
            known_indices = self.known_indices
            known_weights = self.known_weights
            known_contributions = self.known_contributions

        if k_unk > 0:
            _, unk_sel = torch.topk(self.unk_contributions.abs(), k_unk, dim=-1)
            unk_indices = torch.gather(self.unk_indices, -1, unk_sel)
            unk_weights = torch.gather(self.unk_weights, -1, unk_sel)
            unk_contributions = torch.gather(self.unk_contributions, -1, unk_sel)
        else:
            unk_indices = self.unk_indices
            unk_weights = self.unk_weights
            unk_contributions = self.unk_contributions

        return OutputToConceptAttribution(
            target_token_ids=self.target_token_ids,
            target_logits=self.target_logits,
            known_indices=known_indices,
            known_weights=known_weights,
            known_contributions=known_contributions,
            unk_indices=unk_indices,
            unk_weights=unk_weights,
            unk_contributions=unk_contributions,
            epsilon_contribution=self.epsilon_contribution,
            committed=self.committed,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame for analysis."""
        rows = []
        B, T = self.target_token_ids.shape
        k_known = self.known_indices.shape[-1]
        k_unk = self.unk_indices.shape[-1]

        for b in range(B):
            for t in range(T):
                if not self.committed[b, t]:
                    continue
                target_id = int(self.target_token_ids[b, t])
                target_logit = float(self.target_logits[b, t])
                eps = float(self.epsilon_contribution[b, t])

                for ki in range(k_known):
                    rows.append(
                        {
                            "batch": b,
                            "position": t,
                            "target_token_id": target_id,
                            "target_logit": target_logit,
                            "concept_type": "known",
                            "concept_id": int(self.known_indices[b, t, ki]),
                            "weight": float(self.known_weights[b, t, ki]),
                            "contribution": float(self.known_contributions[b, t, ki]),
                            "epsilon": eps,
                        }
                    )

                for ui in range(k_unk):
                    rows.append(
                        {
                            "batch": b,
                            "position": t,
                            "target_token_id": target_id,
                            "target_logit": target_logit,
                            "concept_type": "discovered",
                            "concept_id": int(self.unk_indices[b, t, ui]),
                            "weight": float(self.unk_weights[b, t, ui]),
                            "contribution": float(self.unk_contributions[b, t, ui]),
                            "epsilon": eps,
                        }
                    )

        return pd.DataFrame(rows)

    def save(self, path: str) -> None:
        """Save to parquet."""
        self.to_dataframe().to_parquet(path, index=False)


class AttributionAccumulator:
    """
    Accumulates concept attribution as tokens are committed during generation.

    Pre-allocates [1, T, K] buffers. Each commit() computes per-concept
    contributions (w_i * (v_i^T @ W_{y_t})) and scatter-writes them
    into the buffers.

    All computation in float32 for numerical precision.

    Args:
        seq_len: Total sequence length (prompt + generation).
        k_known: Number of known top-k concepts per position.
        k_unk: Number of unknown top-k concepts per position.
        device: Torch device.
        lm_head_weight: [V, D] LM head weight matrix.
        known_head: Known ConceptHead.
        unknown_head: Unknown ConceptHead, or None.
    """

    def __init__(
        self,
        seq_len: int,
        k_known: int,
        k_unk: int,
        device: torch.device,
        lm_head_weight: Tensor,
        known_head,
        unknown_head=None,
    ) -> None:
        self.seq_len = seq_len
        self.k_known = k_known
        self.k_unk = k_unk
        self.device = device
        self.lm_head_weight = lm_head_weight
        self.known_head = known_head
        self.unknown_head = unknown_head

        # Pre-allocate output buffers
        self.token_ids = torch.zeros(1, seq_len, dtype=torch.long, device=device)
        self.target_logits = torch.zeros(1, seq_len, device=device)

        self.known_indices = torch.zeros(1, seq_len, k_known, dtype=torch.long, device=device)
        self.known_weights = torch.zeros(1, seq_len, k_known, device=device)
        self.known_contributions = torch.zeros(1, seq_len, k_known, device=device)

        self.unk_indices = torch.zeros(1, seq_len, k_unk, dtype=torch.long, device=device)
        self.unk_weights = torch.zeros(1, seq_len, k_unk, device=device)
        self.unk_contributions = torch.zeros(1, seq_len, k_unk, device=device)

        self.residual_contribution = torch.zeros(1, seq_len, device=device)

        # Tracking
        self.committed = torch.zeros(seq_len, dtype=torch.bool, device=device)
        self.commit_denoising_step = torch.full((seq_len,), -2, dtype=torch.long, device=device)

    @torch.no_grad()
    def commit(self, event: CommitRecord) -> None:
        """
        Record attribution for newly committed positions.

        Computes C(c_i, y_t) = w_i * (v_i^T @ W_{y_t}) using the
        concept state from the current forward pass.

        Args:
            event: CommitRecord with positions, tokens, and concept
                head outputs from this denoising step.
        """
        pos = event.positions  # [P]
        P = pos.shape[0]
        if P == 0:
            return

        # All computation in float32
        target_logits_f = event.target_logits.float()

        # LM head weights for committed tokens: [P, D]
        W_yt = self.lm_head_weight[event.token_ids].float()

        # Known contributions
        k_emb = self.known_head._get_embedding(event.known_indices).float()  # [P, K, D]
        k_dots = torch.einsum("pkd,pd->pk", k_emb, W_yt)  # [P, K]
        k_w = torch.sigmoid(event.known_logits.float())  # [P, K]
        k_contrib = k_w * k_dots  # [P, K]

        # Unknown contributions
        if self.unknown_head is not None and event.unk_indices is not None and event.unk_logits is not None:
            u_emb = self.unknown_head._get_embedding(event.unk_indices).float()  # [P, K_unk, D]
            u_dots = torch.einsum("pkd,pd->pk", u_emb, W_yt)  # [P, K_unk]
            u_w = torch.sigmoid(event.unk_logits.float())  # [P, K_unk]
            u_contrib = u_w * u_dots  # [P, K_unk]
        else:
            u_w = torch.zeros(P, self.k_unk, device=self.device)
            u_contrib = torch.zeros(P, self.k_unk, device=self.device)

        # Residual (exact by linearity)
        residual = target_logits_f - k_contrib.sum(-1) - u_contrib.sum(-1)  # [P]

        # Scatter-write into pre-allocated buffers
        self.token_ids[0, pos] = event.token_ids
        self.target_logits[0, pos] = target_logits_f

        self.known_indices[0, pos] = event.known_indices
        self.known_weights[0, pos] = k_w
        self.known_contributions[0, pos] = k_contrib

        if event.unk_indices is not None:
            self.unk_indices[0, pos] = event.unk_indices
        self.unk_weights[0, pos] = u_w
        self.unk_contributions[0, pos] = u_contrib

        self.residual_contribution[0, pos] = residual

        self.committed[pos] = True
        self.commit_denoising_step[pos] = event.denoising_step

    def result(self) -> OutputToConceptAttribution:
        """Build final attribution from accumulated commits."""
        n_uncommitted = (~self.committed).sum().item()
        if n_uncommitted > 0:
            warnings.warn(
                f"{n_uncommitted} of {self.seq_len} positions were never "
                f"committed. Their attributions will be zero.",
                stacklevel=2,
            )

        return OutputToConceptAttribution(
            target_token_ids=self.token_ids,
            target_logits=self.target_logits,
            known_indices=self.known_indices,
            known_weights=self.known_weights,
            known_contributions=self.known_contributions,
            unk_indices=self.unk_indices,
            unk_weights=self.unk_weights,
            unk_contributions=self.unk_contributions,
            epsilon_contribution=self.residual_contribution,
            committed=self.committed.unsqueeze(0),
        )

    @property
    def coverage(self) -> float:
        """Fraction of positions that have been committed."""
        return float(self.committed.float().mean().item())

    @property
    def commit_order(self) -> Tensor:
        """[T] tensor of denoising steps for analyzing generation order."""
        return self.commit_denoising_step


AttributionResult = OutputToConceptAttribution


class AttributionEntry(TypedDict):
    label: str
    type: str  # "known" | "discovered"
    contribution: float


class ConceptLabels:
    """Maps concept IDs to human-readable labels.

    Supports parquet format (concept_labels.parquet with known + unknown)
    and legacy CSV format.
    """

    def __init__(self, concepts_path: Path | str | None = None):
        if concepts_path is None:
            env = os.environ.get("STEERLING_CONCEPTS_PATH")
            if env:
                concepts_path = env
            else:
                try:
                    from huggingface_hub import hf_hub_download

                    concepts_path = hf_hub_download("guidelabs/steerling", "concept_labels.parquet")
                except Exception:
                    pass  # fall through to empty DataFrame

        if concepts_path is None:
            self._df = pd.DataFrame(columns=pd.Index(["concept_id", "head", "concept_name"]))
        elif (p := Path(concepts_path)).exists():
            if p.suffix == ".parquet":
                self._df = pd.read_parquet(p)
            else:
                self._df = pd.read_csv(p)
                # Legacy CSV uses concept_idx; normalize to concept_id
                if "concept_idx" in self._df.columns and "concept_id" not in self._df.columns:
                    self._df = self._df.rename(columns={"concept_idx": "concept_id"})
            print(f"Loaded {len(self._df)} concept labels from {p}")
        else:
            print(f"Concept file not found at {p} — using fallback labels")
            self._df = pd.DataFrame(columns=pd.Index(["concept_id", "head", "concept_name"]))

        # Build lookup dicts for fast access
        self._labels: dict[int, str] = {}
        self._heads: dict[int, str] = {}
        for _, row in self._df.iterrows():
            cid = int(row["concept_id"])
            self._labels[cid] = str(row["concept_name"])
            if "head" in row.index:
                self._heads[cid] = str(row["head"])

    def label(self, concept_id: int, concept_type: str = "known") -> str:
        name = self._labels.get(concept_id)
        if name is None:
            if concept_type == "discovered":
                return f"Discovered: #{concept_id}"
            return f"Known: #{concept_id}"
        head = self._heads.get(concept_id, concept_type)
        prefix = "Discovered" if head == "unknown" else "Known"
        return f"{prefix}: {name}"


def chunk_attribution(
    attr: OutputToConceptAttribution,
    start: int,
    end: int,
    batch: int = 0,
    concept_labels: ConceptLabels | None = None,
    num_known_concepts: int = 0,
) -> tuple[list[AttributionEntry], float]:
    """
    Compute concept attribution over a chunk of tokens [start, end).

    Per position, each concept's contribution is normalized by the total absolute
    logit mass at that position (known + unknown + epsilon). This removes
    scale differences across positions so every token votes equally to the chunk.

    Args:
        attr: OutputToConceptAttribution from the accumulator.
        start: Start token index (inclusive).
        end: End token index (exclusive).
        batch: Batch index to use (default 0).
        concept_labels: Optional ConceptLabels for human-readable names.
        num_known_concepts: Offset for unknown concept IDs to avoid collision.

    Returns:
        entries: List of AttributionEntry dicts sorted by abs contribution.
        eps_pct: Epsilon as a percentage of total logit mass (mean over chunk).
    """
    labels = concept_labels or ConceptLabels()

    mask = attr.committed[batch, start:end]  # (T,)
    k_idx = attr.known_indices[batch, start:end][mask]  # (T', K_known)
    k_c = attr.known_contributions[batch, start:end][mask]  # (T', K_known)
    u_idx = attr.unk_indices[batch, start:end][mask]  # (T', K_unk)
    u_c = attr.unk_contributions[batch, start:end][mask]  # (T', K_unk)
    eps = attr.epsilon_contribution[batch, start:end][mask]  # (T',)

    pos_total = (k_c.abs().sum(-1) + u_c.abs().sum(-1) + eps.abs()).clamp(min=1e-8)  # (T,)

    k_c_norm = k_c / pos_total.unsqueeze(-1)  # (T, K_known)
    u_c_norm = u_c / pos_total.unsqueeze(-1)  # (T, K_unk)

    def aggregate(idx: Tensor, contrib_norm: Tensor) -> tuple[Tensor, Tensor]:
        flat_idx = idx.reshape(-1)
        flat_norm = contrib_norm.reshape(-1)
        unique_ids, inverse = flat_idx.unique(return_inverse=True)
        mass = torch.zeros(len(unique_ids), device=flat_idx.device)
        mass.scatter_add_(0, inverse, flat_norm)
        return unique_ids, mass

    k_ids, k_mass = aggregate(k_idx, k_c_norm)
    u_ids, u_mass = aggregate(u_idx, u_c_norm)

    T = end - start
    entries: list[AttributionEntry] = []

    for cid, mass in zip(k_ids.tolist(), k_mass.tolist(), strict=False):
        entries.append(
            AttributionEntry(
                label=labels.label(int(cid), "known"),
                type="known",
                contribution=float(mass) / T,
            )
        )
    for cid, mass in zip(u_ids.tolist(), u_mass.tolist(), strict=False):
        entries.append(
            AttributionEntry(
                label=labels.label(int(cid + num_known_concepts), "discovered"),
                type="discovered",
                contribution=float(mass) / T,
            )
        )

    entries.sort(key=lambda e: abs(e["contribution"]), reverse=True)
    eps_pct = float((eps / pos_total).mean()) * 100

    return entries, eps_pct


class FaithfulConceptAttributor:
    """
    Faithful concept attribution captured during generation.

    Each token's attribution comes from the forward pass where it was
    committed — the actual decision context with partial masking — rather
    than a post-hoc pass over the fully unmasked sequence.

    After calling ``attribute()``, the recorded ``DiffusionTrace`` is
    available as ``self.last_trace``. This can be passed directly to
    ``FaithfulOutputToInputAttributor.attribute_from_trace()`` to get
    input attribution without re-generating.

    Args:
        generator: A loaded SteerlingGenerator instance.
        concepts_path: Optional path to concept_labels.parquet for label lookup.
        unknown_topk: Number of unknown top-k concepts to compute
            per committed position.
    """

    def __init__(
        self,
        generator: SteerlingGenerator,
        concepts_path: Path | str | None = None,
        unknown_topk: int | None = None,
    ) -> None:
        self.generator = generator
        # Resolve backbone — may be wrapped
        if hasattr(generator.model, "model"):
            self.backbone: Any = generator.model.model
        else:
            self.backbone = generator.model
        self.device = generator.device
        self.labels = ConceptLabels(concepts_path)
        if unknown_topk is None:
            unk_head = getattr(self.backbone, "unknown_head", None)
            unknown_topk = getattr(unk_head, "topk", 64) or 64
        self.unknown_topk = unknown_topk
        self._num_known_concepts = getattr(getattr(self.backbone, "known_head", None), "n_concepts", 0)
        self.last_attribution: OutputToConceptAttribution | None = None
        self.last_trace: DiffusionTrace | None = None

    @torch.no_grad()
    def attribute(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        batch: int = 0,
    ) -> tuple[GenerationOutput, list[tuple[list[AttributionEntry], float]]]:
        """
        Generate text and capture faithful concept attribution.

        After this call, ``self.last_trace`` holds the ``DiffusionTrace``
        which can be passed to
        ``FaithfulOutputToInputAttributor.attribute_from_trace()``
        for input attribution without re-generating.

        Args:
            prompt: Input text prompt.
            config: Generation configuration. If None, uses defaults.
            batch: Batch index (default 0).

        Returns:
            (gen_output, chunks) where chunks is a list of
            (entries, eps_pct) per chunk.
        """
        if config is None:
            config = GenerationConfig(max_new_tokens=128, steps=128)

        if not self.generator.is_interpretable:
            raise RuntimeError("Attribution requires an interpretable model.")

        # Determine k from model
        k_known = getattr(self.backbone.known_head, "topk", 32)
        unknown_head = getattr(self.backbone, "unknown_head", None)
        k_unk = self.unknown_topk

        # Estimate sequence length for pre-allocation
        prompt_ids = self.generator.tokenizer.encode(prompt, add_special_tokens=False)
        seq_len = len(prompt_ids) + config.max_new_tokens

        # Create accumulator
        lm_head_weight = self.backbone.transformer.lm_head.weight
        accumulator = AttributionAccumulator(
            seq_len=seq_len,
            k_known=k_known,
            k_unk=k_unk,
            device=self.device,
            lm_head_weight=lm_head_weight,
            known_head=self.backbone.known_head,
            unknown_head=unknown_head,
        )

        # Step callback — fires at each commit, computes attribution inline
        # and also collects CommitRecords for building a DiffusionTrace
        unknown_head_ref = unknown_head
        unk_topk = self.unknown_topk
        commit_counter = [0]
        commit_records: list[CommitRecord] = []

        def on_commit(info: GenerationStepInfo) -> None:
            pos = info.committed_positions  # [P]
            tids = info.committed_token_ids  # [P]

            # Target logits from this forward pass
            target_lgt = info.logits[0, pos].gather(-1, tids.unsqueeze(-1)).squeeze(-1)

            # Known: extract top-k from outputs (sparse path)
            k_idx = info.outputs.known_topk_indices[0, pos]  # [P, K]
            k_lgt = info.outputs.known_topk_logits[0, pos]  # [P, K]

            # Unknown: compute from hidden for committed positions only
            hidden_p = info.outputs.hidden[:, pos, :]  # [1, P, D]
            u_idx, u_lgt = _compute_unknown_topk(
                unknown_head_ref,
                hidden_p,
                k=unk_topk,
            )

            record = CommitRecord(
                commit_order=commit_counter[0],
                denoising_step=info.step,
                positions=pos,
                token_ids=tids,
                target_logits=target_lgt,
                known_indices=k_idx,
                known_logits=k_lgt,
                unk_indices=u_idx[0] if u_idx is not None else None,
                unk_logits=u_lgt[0] if u_lgt is not None else None,
            )
            accumulator.commit(record)
            commit_records.append(record)
            commit_counter[0] += 1

        # Generate with callback
        gen_output = self.generator.generate_full(prompt, config, step_callback=on_commit)

        # Build chunk-level results
        attr = accumulator.result()
        self.last_attribution = attr  # expose for custom chunk ranges

        # Build DiffusionTrace for reuse by input attribution
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)
        eos_id = int(self.generator.eos_token_id) if self.generator.eos_token_id is not None else None
        self.last_trace = DiffusionTrace(
            prompt_ids=prompt_tensor,
            prompt_len=len(prompt_ids),
            seq_length=seq_len,
            mask_id=int(self.generator.mask_token_id),
            eos_id=eos_id,
            device=self.device,
            groups=commit_records,
        )

        eoc_id = getattr(self.generator.tokenizer, "endofchunk_token_id", None)
        eot_id = getattr(self.generator.tokenizer, "eot_id", None)
        stop = [eot_id] if eot_id is not None else None
        chunks = find_chunk_boundaries(
            gen_output.tokens.tolist(),
            eoc_id=eoc_id if eoc_id is not None else -1,
            start_index=gen_output.prompt_tokens,
            stop_ids=stop,
            include_final_chunk=True,
        )

        results = []
        for start, end in chunks:
            entries, eps_pct = chunk_attribution(
                attr,
                start,
                end,
                batch=batch,
                concept_labels=self.labels,
                num_known_concepts=self._num_known_concepts,
            )
            results.append((entries, eps_pct))

        return gen_output, results

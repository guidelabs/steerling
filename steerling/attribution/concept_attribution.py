"""
Concept attribution for Steerling models.

Provides per-position and per-chunk concept attribution using sparse top-k indices
from the ConceptHead forward pass.

Usage:
    from steerling import SteerlingGenerator, GenerationConfig
    from steerling.attribution import ConceptAttributor

    generator = SteerlingGenerator.from_pretrained("guidelabs/steerling-8b", device="cuda")
    attributor = ConceptAttributor(generator)

    entries, eps_pct = attributor.attribute(prompt="AI technology will")
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import pandas as pd
import torch
from torch import Tensor

from steerling import GenerationConfig, SteerlingGenerator


class AttributionEntry(TypedDict):
    label: str
    type: str  # "known" | "discovered"
    contribution: float


class AttributionResult(TypedDict):
    known_indices: Tensor  # (B, T, K_known)
    known_contributions: Tensor  # (B, T, K_known)
    disc_indices: Tensor  # (B, T, K_disc)
    disc_contributions: Tensor  # (B, T, K_disc)
    epsilon: Tensor  # (B, T)


class ConceptLabels:
    """Maps concept IDs to human-readable labels from a CSV file."""

    def __init__(self, concepts_path: Path | str | None = None):
        if concepts_path is not None:
            p = Path(concepts_path)
            if p.exists():
                self._df = pd.read_csv(p)
                print(f"Loaded {len(self._df)} concept labels from {p}")
            else:
                print(f"Concept file not found at {p} — using fallback labels")
                self._df = pd.DataFrame(columns=pd.Index(["concept_idx", "concept_name"]))
        else:
            self._df = pd.DataFrame(columns=pd.Index(["concept_idx", "concept_name"]))

    def label(self, concept_id: int, concept_type: str = "known") -> str:
        if concept_type == "discovered":
            return f"Discovered #{concept_id}"
        row = self._df[self._df["concept_idx"] == concept_id]
        if len(row) == 0:
            return f"Known #{concept_id}"
        return str(row.iloc[0]["concept_name"])


@torch.no_grad()
def compute_concept_attribution(
    outputs,
    logits: Tensor,
    backbone,
) -> AttributionResult:
    """
    Compute per-position concept contributions using sparse top-k indices.

    For each position, decomposes the predicted token logit into:
        - known concept contributions
        - discovered (unknown) concept contributions
        - epsilon residual

    Args:
        outputs: Backbone output object with known_topk_indices, known_topk_logits,
                 unknown_topk_indices, unknown_topk_logits attributes.
        logits: (B, T, V) token logits from the backbone.
        backbone: The backbone model with known_head, unknown_head, and lm_head.

    Returns:
        AttributionResult dict with indices, contributions, and epsilon tensors.
    """
    pred_ids = logits.argmax(dim=-1)  # (B, T)
    pred_logits = torch.gather(logits, -1, pred_ids.unsqueeze(-1)).squeeze(-1)  # (B, T)
    W_y = backbone.transformer.lm_head.weight[pred_ids]  # (B, T, D)

    def _head_contrib(
        head,
        topk_idx: Tensor | None,
        topk_logits: Tensor | None,
    ) -> tuple[Tensor, Tensor] | None:
        """Compute contribution from a single concept head's sparse top-k."""
        if topk_idx is None:
            return None
        emb = head._get_embedding(topk_idx)  # (B, T, k, D)
        dots = torch.einsum("btkd,btd->btk", emb, W_y)  # (B, T, k)
        assert topk_logits is not None  # narrowing for type
        w = torch.sigmoid(topk_logits.float())  # (B, T, k)
        return topk_idx, w * dots

    known = _head_contrib(
        backbone.known_head,
        topk_idx=outputs.known_topk_indices,
        topk_logits=outputs.known_topk_logits,
    )

    disc = None
    if getattr(backbone, "unknown_head", None) is not None:
        disc = _head_contrib(
            backbone.unknown_head,
            topk_idx=outputs.unknown_topk_indices,
            topk_logits=outputs.unknown_topk_logits,
        )

    bsz, tlen = pred_ids.shape
    device = pred_ids.device

    def _zero(k: int = 0) -> tuple[Tensor, Tensor]:
        return (
            torch.zeros(bsz, tlen, k, dtype=torch.long, device=device),
            torch.zeros(bsz, tlen, k, device=device),
        )

    k_idx, k_c = known if known is not None else _zero()
    d_idx, d_c = disc if disc is not None else _zero()

    eps = pred_logits - k_c.sum(-1) - d_c.sum(-1)

    return AttributionResult(
        known_indices=k_idx,
        known_contributions=k_c,
        disc_indices=d_idx,
        disc_contributions=d_c,
        epsilon=eps,
    )


def find_chunks(token_ids: Tensor, tokenizer) -> list[tuple[int, int]]:
    """
    Split token IDs at <|endofchunk|> boundaries.

    Args:
        token_ids: (L,) or (1, L) token tensor.
        tokenizer: Tokenizer with optional endofchunk_token_id attribute.

    Returns:
        List of (start, end) index pairs, one per chunk.
    """
    ids = token_ids.squeeze()
    eoc_id = getattr(tokenizer, "endofchunk_token_id", None)

    if eoc_id is None:
        return [(0, int(ids.shape[0]))]

    positions = (ids == eoc_id).nonzero(as_tuple=True)[0].tolist()
    chunks: list[tuple[int, int]] = []
    prev = 0
    for p in positions:
        if p > prev:
            chunks.append((prev, p))
        prev = p + 1
    if prev < len(ids):
        chunks.append((prev, len(ids)))
    return chunks


def chunk_attribution(
    attr: AttributionResult,
    start: int,
    end: int,
    batch: int = 0,
    concept_labels: ConceptLabels | None = None,
) -> tuple[list[AttributionEntry], float]:
    """
    Compute concept attribution over a chunk of tokens [start, end).

    Per position, each concept's contribution is normalized by the total absolute
    logit mass at that position (known + discovered + epsilon). This removes
    scale differences across positions so every token votes equally to the chunk.

    Args:
        attr: AttributionResult from compute_concept_attribution.
        start: Start token index (inclusive).
        end: End token index (exclusive).
        batch: Batch index to use (default 0).
        concept_labels: Optional ConceptLabels for human-readable names.

    Returns:
        entries: List of AttributionEntry dicts sorted by abs contribution.
        eps_pct: Epsilon as a percentage of total logit mass (mean over chunk).
    """
    labels = concept_labels or ConceptLabels()

    k_idx = attr["known_indices"][batch, start:end]  # (T, K_known)
    k_c = attr["known_contributions"][batch, start:end]  # (T, K_known)
    d_idx = attr["disc_indices"][batch, start:end]  # (T, K_disc)
    d_c = attr["disc_contributions"][batch, start:end]  # (T, K_disc)
    eps = attr["epsilon"][batch, start:end]  # (T,)

    pos_total = (k_c.abs().sum(-1) + d_c.abs().sum(-1) + eps.abs()).clamp(min=1e-8)  # (T,)

    k_c_norm = k_c / pos_total.unsqueeze(-1)  # (T, K_known)
    d_c_norm = d_c / pos_total.unsqueeze(-1)  # (T, K_disc)

    def aggregate(idx: Tensor, contrib_norm: Tensor) -> tuple[Tensor, Tensor]:
        flat_idx = idx.reshape(-1)
        flat_norm = contrib_norm.reshape(-1)
        unique_ids, inverse = flat_idx.unique(return_inverse=True)
        mass = torch.zeros(len(unique_ids), device=flat_idx.device)
        mass.scatter_add_(0, inverse, flat_norm)
        return unique_ids, mass

    k_ids, k_mass = aggregate(k_idx, k_c_norm)
    d_ids, d_mass = aggregate(d_idx, d_c_norm)

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
    for cid, mass in zip(d_ids.tolist(), d_mass.tolist(), strict=False):
        entries.append(
            AttributionEntry(
                label=labels.label(int(cid), "discovered"),
                type="discovered",
                contribution=float(mass) / T,
            )
        )

    entries.sort(key=lambda e: abs(e["contribution"]), reverse=True)
    eps_pct = float((eps / pos_total).mean()) * 100

    return entries, eps_pct


class ConceptAttributor:
    """
    High-level interface for concept attribution on Steerling models.

    Args:
        generator: A loaded SteerlingGenerator instance.
        concepts_path: Optional path to known_concepts.csv for label lookup.

    Example:
        attributor = ConceptAttributor(generator, concepts_path="assets/concepts/known_concepts.csv")
        results = attributor.attribute("AI technology will")
        for chunk_entries, eps_pct in results:
            for entry in chunk_entries[:5]:
                print(entry)
    """

    def __init__(
        self,
        generator: SteerlingGenerator,
        concepts_path: Path | str | None = None,
    ):
        self.generator = generator
        self.backbone: Any = generator.model.model if hasattr(generator.model, "model") else generator.model
        self.device = generator.device
        self.labels = ConceptLabels(concepts_path)

    @torch.no_grad()
    def attribute(
        self,
        prompt: str,
        config: GenerationConfig | None = None,
        batch: int = 0,
    ) -> list[tuple[list[AttributionEntry], float]]:
        """
        Generate text from prompt and compute chunk-level concept attribution.

        Args:
            prompt: Input text prompt.
            config: GenerationConfig (defaults to max_new_tokens=128).
            batch: Batch index (default 0).

        Returns:
            List of (entries, eps_pct) tuples, one per chunk.
        """
        if config is None:
            config = GenerationConfig(max_new_tokens=128, steps=128, temperature=0.4)

        gen_output = self.generator.generate_full(prompt, config)
        tokens = gen_output.tokens.unsqueeze(0)  # (1, L)

        logits, outputs = self.backbone(tokens, minimal_output=True)

        attr = compute_concept_attribution(outputs, logits, self.backbone)
        chunks = find_chunks(tokens, self.generator.tokenizer)

        results = []
        for start, end in chunks:
            entries, eps_pct = chunk_attribution(attr, start, end, batch=batch, concept_labels=self.labels)
            results.append((entries, eps_pct))

        return results

    def attribute_tokens(
        self,
        tokens: Tensor,
        batch: int = 0,
    ) -> tuple[AttributionResult, list[tuple[int, int]]]:
        """
        Run attribution on a pre-tokenized tensor. Returns raw attribution
        result and chunk boundaries for custom downstream processing.

        Args:
            tokens: (1, L) token tensor already on the correct device.
            batch: Batch index (default 0).

        Returns:
            attr: Raw AttributionResult dict.
            chunks: List of (start, end) chunk boundaries.
        """
        logits, outputs = self.backbone(tokens, minimal_output=True)
        attr = compute_concept_attribution(outputs, logits, self.backbone)
        chunks = find_chunks(tokens, self.generator.tokenizer)
        return attr, chunks

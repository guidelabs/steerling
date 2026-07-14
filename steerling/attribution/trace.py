"""Shared commit record and trace types for attribution.

CommitRecord is the unified type for tokens committed during generation,
used by both concept attribution and input feature attribution.
DiffusionTrace replays the commit schedule for faithful attribution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

import torch
from torch import Tensor

_NEVER_COMMITTED = torch.iinfo(torch.long).max


@dataclass(frozen=True)
class CommitRecord:
    """Tokens committed together in one denoising step.

    Shared by concept attribution and input feature attribution.
    Concept fields (known_indices, known_logits, unk_indices, unk_logits)
    are only populated by the concept attributor; input attribution leaves
    them as None.

    Attributes:
        commit_order: Contiguous replay index (0, 1, 2, ...).
        denoising_step: Denoising iteration index from the generator.
        positions: [P] sequence positions committed this step.
        token_ids: [P] chosen token IDs.
        target_logits: [P] model's logit for the chosen token.
        known_indices: [P, K_known] top-k known concept indices, or None.
        known_logits: [P, K_known] pre-sigmoid logits for known concepts, or None.
        unk_indices: [P, K_unk] top-k unknown concept indices, or None.
        unk_logits: [P, K_unk] pre-sigmoid logits for unknown concepts, or None.
    """

    commit_order: int
    denoising_step: int
    positions: Tensor
    token_ids: Tensor
    target_logits: Tensor
    known_indices: Tensor | None = field(default=None)
    known_logits: Tensor | None = field(default=None)
    unk_indices: Tensor | None = field(default=None)
    unk_logits: Tensor | None = field(default=None)


class PositionType(StrEnum):
    """Role of an input position in a reconstructed snapshot."""

    PROMPT = "prompt"
    GENERATED = "generated"  # committed during generation
    GENERATED_MASKED = "generated_masked"  # generation region, never committed


@dataclass(frozen=True)
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
    groups: list[CommitRecord]

    def __post_init__(self) -> None:
        # Validate: orders must be contiguous from 0
        orders = sorted(g.commit_order for g in self.groups)
        if orders and orders != list(range(len(orders))):
            raise ValueError(f"CommitRecord orders must be contiguous from 0, got {orders}")

        # Validate: no position committed twice
        all_positions: list[int] = []
        for g in self.groups:
            all_positions.extend(g.positions.tolist())
        if len(all_positions) != len(set(all_positions)):
            raise ValueError("A position was committed more than once")

        object.__setattr__(self, "padded_seq_length", self.seq_length)
        object.__setattr__(self, "_base", self._build_base())
        order_at, token_at = self._build_caches()
        object.__setattr__(self, "_order_at", order_at)
        object.__setattr__(self, "_token_at", token_at)

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
            order_at[g.positions] = g.commit_order
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

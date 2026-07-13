"""Browse and search the concepts available for steering.

The concept catalog wraps ``assets/concepts/concept_labels.parquet`` so users can
find a concept ID before steering or unlearning. Names alone are noisy (a search
for "cat" matches "category" and "communication"), so search covers both the name
and the description, and ``SteerlingGenerator.concept_top_tokens`` is the reliable
cross-check for what a concept actually promotes in the loaded weights.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

_ENV_VAR = "STEERLING_CONCEPTS_PATH"
_HF_REPO = "guidelabs/steerling"
_HF_FILENAME = "concept_labels.parquet"


@dataclass(frozen=True)
class Concept:
    """A single concept the model can represent."""

    concept_id: int
    name: str
    description: str
    head: str
    group_id: int | None
    group_name: str | None
    is_steerable: bool
    is_tone: bool
    is_alignment: bool
    is_demographic: bool

    def __repr__(self) -> str:
        return f"Concept(id={self.concept_id}, name={self.name!r}, steerable={self.is_steerable})"


class ConceptCatalog:
    """Lookup and search over the concept label table."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.set_index("concept_id", drop=False)

    @classmethod
    def load(cls, path: str | Path | None = None) -> ConceptCatalog:
        """
        Load the catalog from a parquet file.

        Resolution order: explicit ``path``, then the ``STEERLING_CONCEPTS_PATH``
        environment variable, then the bundled ``assets/concepts`` file.
        """
        if path is not None:
            resolved = Path(path)
        elif os.environ.get(_ENV_VAR):
            resolved = Path(os.environ[_ENV_VAR])
        else:
            resolved = Path(hf_hub_download(_HF_REPO, _HF_FILENAME))
        if not resolved.is_file():
            raise FileNotFoundError(
                f"Concept labels not found at {resolved}. Pass path=... or set {_ENV_VAR}."
            )
        return cls(pd.read_parquet(resolved))

    def get(self, concept_id: int) -> Concept:
        """Return the concept with this ID."""
        if concept_id not in self._df.index:
            raise KeyError(f"No concept with id {concept_id}")
        return self._row_to_concept(self._df.loc[concept_id])

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        steerable_only: bool = True,
        head: str | None = "known",
    ) -> list[Concept]:
        """
        Rank concepts by how well their name and description match the query.

        Name matches are weighted more heavily than description matches. By default
        the search is limited to steerable known concepts, which is the set that
        can be steered.
        """
        df = self._df
        if head is not None:
            df = df[df["head"] == head]
        if steerable_only:
            df = df[df["is_steerable"]]

        name = df["concept_name"].str.lower().fillna("")
        desc = df["concept_description"].str.lower().fillna("")
        score = pd.Series(0, index=df.index)
        for term in (t for t in query.lower().split() if t):
            pattern = rf"\b{re.escape(term)}\b"
            score = score + name.str.count(pattern) * 3 + desc.str.count(pattern)

        hits = df.assign(_score=score)
        hits = hits[hits["_score"] > 0].sort_values("_score", ascending=False).head(limit)
        return [self._row_to_concept(row) for _, row in hits.iterrows()]

    def group(self, concept_id: int) -> list[int]:
        """
        Concept IDs sharing this concept's group (the focal ID first).

        Pass the result to ``SteeringConfig.injection`` / ``injection_relu`` for
        group steering, which sums the members into one direction.
        """
        focal = self.get(concept_id)
        if focal.group_id is None:
            return [concept_id]
        members = self._df[self._df["public_group_id"] == focal.group_id]["concept_id"].tolist()
        rest = [int(m) for m in members if int(m) != concept_id]
        return [concept_id] + rest

    def to_df(self) -> pd.DataFrame:
        """The underlying table, for custom filtering."""
        return self._df

    @staticmethod
    def _row_to_concept(row: pd.Series) -> Concept:
        gid = row["public_group_id"]
        return Concept(
            concept_id=int(row["concept_id"]),
            name=str(row["concept_name"]),
            description=str(row["concept_description"]),
            head=str(row["head"]),
            group_id=None if pd.isna(gid) else int(gid),
            group_name=None if pd.isna(row["group_name"]) else str(row["group_name"]),
            is_steerable=bool(row["is_steerable"]),
            is_tone=bool(row["is_tone"]),
            is_alignment=bool(row["is_alignment"]),
            is_demographic=bool(row["is_demographic"]),
        )

    def __len__(self) -> int:
        return len(self._df)

    def __contains__(self, concept_id: int) -> bool:
        return concept_id in self._df.index

    def __getitem__(self, concept_id: int) -> Concept:
        return self.get(concept_id)

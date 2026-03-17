"""
Concept decomposition head with memory-efficient streaming operations.

Supports both known and unknown concepts with:
- Streaming feature computation (no (B, T, C) allocation)
- Streaming top-k selection (memory O(B * T * k))
- Sparse logit computation for loss functions
- Teacher forcing and interventions (known concepts only)
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

logger = logging.getLogger(__name__)

# threshold above which dense operations are forbidden for safety
LARGE_CONCEPT_THRESHOLD = 50000


@dataclass
class ConceptHeadOutput:
    """Output from ConceptHead forward pass.

    Attributes:
        features: Final concept features after teacher forcing/intervention (B, T, D)
        gt_features: Ground truth pooled features. None for unknown heads. (B, T, D) or None
        logits: Full concept logits (B, T, C). Only set if return_logits=True. Usually None.
        predicted: Predicted features before teacher forcing mixing (B, T, D)
        weights: Full concept weights (B, T, C). Only set if return_logits=True. Usually None.
        topk_indices: Top-k concept indices (B, T, k). Set when using streaming top-k.
        topk_logits: Logits for top-k concepts (B, T, k). Set when using streaming top-k.
        hidden: Hidden states passed to this head (B, T, D). Stored for attribution.
    """

    features: Tensor
    gt_features: Tensor | None
    logits: Tensor | None
    predicted: Tensor
    weights: Tensor | None = None
    topk_indices: Tensor | None = None
    topk_logits: Tensor | None = None
    hidden: Tensor | None = None


class ConceptHead(nn.Module):
    """
    Concept decomposition head supporting both known and unknown concepts.
    Memory-efficient implementation that avoids (B, T, C) allocations by default.

    Modes:
    - Known (is_unknown=False): Supports GT, teacher forcing, top-k, interventions
    - Unknown (is_unknown=True): No GT, no teacher forcing

    Architectures:
    - use_attention=False: Linear predictor (n_embd -> n_concepts)
    - use_attention=True: Query projection + sigmoid attention over embeddings

    Factorization (for large unknown heads):
    - factorize=False: Dense embeddings (C, D) and predictor (D, C)
    - factorize=True: Factorized embeddings (C, r) @ (r, D) where r << D
                      Reduces memory by ~10-20x for large C

    Memory Safety:
    - Unknown heads with n_concepts > 50k cannot use dense operations
    - Interventions are only supported for known heads
    - return_logits=True is forbidden for large unknown heads
    - All tensor indexing uses F.embedding for DTensor safety

    Args:
        n_concepts: Number of concepts (C)
        concept_dim: Dimension of concept embeddings (should equal n_embd)
        n_embd: Model hidden dimension
        is_unknown: If True, skip GT pooling and teacher forcing
        use_attention: If True, use attention; else use linear predictor
        topk: Top-k sparsity for concept weights. None = no sparsity.
        block_size: Block size for memory-efficient operations
        pad_multiple: Pad n_concepts to a multiple of this for efficiency
        store_unknown_weights: If True and use_attention & is_unknown, store logits/weights
        apply_topk_to_unknown: If True, also apply top-k to unknown concepts
        topk_on_logits: If True, apply top-k on logits (then sigmoid). If False, on weights.
        teacher_force_alpha: If None, hard TF. If in [0,1], soft mixing.
        factorize: If True, use low-rank factorized embeddings
        factorize_rank: Rank for factorization (r). Lower = less memory, less expressivity.
    """

    # Concept Pooling (for known head GT features)
    class ConceptPooling(nn.Module):
        """Memory-efficient sum pooling using scatter-add."""

        def __init__(self, concept_dim: int):
            super().__init__()
            self.concept_dim = concept_dim

        def forward(
            self,
            concept_ids: Tensor,
            concept_mask: Tensor,
            concept_embeddings: nn.Embedding,
        ) -> Tensor:
            """
            Pool concept embeddings based on ground truth IDs.
            Uses scatter-add to avoid (B, T, K, D) allocation when K is sparse.

            Args:
                concept_ids: (B, T, K) concept indices, -1 for invalid
                concept_mask: (B, T, K) boolean mask for valid concepts
                concept_embeddings: Embedding layer to look up

            Returns:
                Pooled features (B, T, D)
            """
            B, T, K = concept_ids.shape
            D = concept_embeddings.embedding_dim
            device = concept_ids.device

            # Find valid positions
            valid_mask = concept_mask & (concept_ids != -1)

            # Output tensor
            pooled = torch.zeros(B, T, D, device=device, dtype=concept_embeddings.weight.dtype)

            if not valid_mask.any():
                return pooled

            # Get valid indices
            b_idx, t_idx, k_idx = torch.where(valid_mask)
            c_ids = concept_ids[b_idx, t_idx, k_idx].long()

            # Look up embeddings for valid concepts only
            emb = concept_embeddings(c_ids)  # (N_valid, D)

            # Scatter-add into output
            flat_idx = b_idx * T + t_idx  # (N_valid,)
            flat_idx = flat_idx.unsqueeze(-1).expand(-1, D)  # (N_valid, D)

            pooled_flat = pooled.view(B * T, D)
            pooled_flat.scatter_add_(0, flat_idx, emb)

            return pooled.view(B, T, D)

    def __init__(
        self,
        n_concepts: int,
        concept_dim: int,
        n_embd: int,
        is_unknown: bool = False,
        use_attention: bool = False,
        topk: int | None = 16,
        topk_features: int | None = None,
        block_size: int = 8192,
        *,
        pad_multiple: int = 16,
        store_unknown_weights: bool = False,
        apply_topk_to_unknown: bool = False,
        topk_on_logits: bool = False,
        # Factorization options
        factorize: bool = False,
        factorize_rank: int = 256,
    ):
        super().__init__()

        self.n_concepts = n_concepts
        self.concept_dim = concept_dim
        self.n_embd = n_embd
        self.is_unknown = is_unknown
        self.use_attention = use_attention
        self.topk = topk
        self.topk_features = topk_features if topk_features is not None else topk
        self.block_size = block_size

        # Flags
        self.pad_multiple = pad_multiple
        self.store_unknown_weights = store_unknown_weights
        self.apply_topk_to_unknown = apply_topk_to_unknown
        self.topk_on_logits = topk_on_logits

        # Factorization options
        self.factorize = factorize
        self.factorize_rank = factorize_rank

        # let's track if this is a "large" head where dense ops are forbidden
        self._is_large = n_concepts > LARGE_CONCEPT_THRESHOLD

        # Pad n_concepts to multiple of pad_multiple for efficiency
        self.n_concepts_padded = ((n_concepts + pad_multiple - 1) // pad_multiple) * pad_multiple

        # Weight initialization: Dense OR Factorized
        if factorize:
            # Factorized embeddings: E = coef @ basis
            # coef: (C, r), basis: (r, D) -> E: (C, D)
            self.embedding_coef = nn.Embedding(self.n_concepts_padded, factorize_rank)
            self.embedding_basis = nn.Linear(factorize_rank, concept_dim, bias=False)
            self.concept_embedding = None  # Don't allocate dense

            if not use_attention:
                # Factorized predictor: logits = hidden @ down @ up.T
                # down: (D, r), up: (r, C) -> effective W: (D, C)
                self.predictor_down = nn.Linear(n_embd, factorize_rank, bias=False)
                self.predictor_up = nn.Linear(factorize_rank, self.n_concepts_padded, bias=False)
                self.concept_predictor = None
            else:
                # Attention uses query projection (not factorized)
                self.concept_query_projection = nn.Linear(n_embd, concept_dim, bias=False)
                self.predictor_down = None
                self.predictor_up = None
                self.concept_predictor = None

            # Log memory savings
            dense_params = n_concepts * concept_dim * 2  # embedding + predictor
            factorized_params = (
                n_concepts * factorize_rank  # coef
                + factorize_rank * concept_dim  # basis
                + (n_embd * factorize_rank + factorize_rank * n_concepts if not use_attention else 0)
            )
            logger.info(f"[ConceptHead] Factorized mode: {n_concepts} concepts, rank={factorize_rank}")
            logger.info(
                f"[ConceptHead] Memory: {dense_params * 2 / 1e9:.2f} GB (dense) -> "
                f"{factorized_params * 2 / 1e9:.2f} GB (factorized) = "
                f"{(1 - factorized_params / dense_params) * 100:.1f}% reduction"
            )
        else:
            # Dense embeddings (what we had before)
            self.concept_embedding = nn.Embedding(self.n_concepts_padded, concept_dim)
            self.embedding_coef = None
            self.embedding_basis = None

            if use_attention:
                self.concept_query_projection = nn.Linear(n_embd, concept_dim, bias=False)
                self.concept_predictor = None
            else:
                self.concept_predictor = nn.Linear(n_embd, self.n_concepts_padded, bias=False)

            self.predictor_down = None
            self.predictor_up = None

        # Concept pooling for GT features (known head only)
        self.concept_pooling = self.ConceptPooling(concept_dim)

        if self.topk_features != self.topk:
            logger.info(
                f"[ConceptHead] {'Unknown' if is_unknown else 'Known'} head: "
                f"topk={self.topk} (loss), topk_features={self.topk_features} (features)"
            )

        if is_unknown and apply_topk_to_unknown:
            logger.info(f"[ConceptHead] Unknown head: apply_topk_to_unknown=True, topk={self.topk}")

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values."""
        if self.factorize:
            nn.init.normal_(self.embedding_coef.weight, mean=0.0, std=0.02)  # type: ignore
            nn.init.normal_(self.embedding_basis.weight, mean=0.0, std=0.02)  # type: ignore

            if self.predictor_down is not None:
                nn.init.normal_(self.predictor_down.weight, mean=0.0, std=0.02)
            if self.predictor_up is not None:
                nn.init.normal_(self.predictor_up.weight, mean=0.0, std=0.02)
        else:
            if self.concept_embedding is not None:
                nn.init.normal_(self.concept_embedding.weight, mean=0.0, std=0.02)

            if self.concept_predictor is not None:
                nn.init.normal_(self.concept_predictor.weight, mean=0.0, std=0.02)

        if hasattr(self, "concept_query_projection") and self.concept_query_projection is not None:
            nn.init.normal_(self.concept_query_projection.weight, mean=0.0, std=0.02)

    def _check_dense_allowed(self, operation: str) -> None:
        """Raise error if dense operations are requested for large unknown heads."""

        if self.is_unknown and self._is_large:
            raise ValueError(
                f"{operation} requested for unknown head with {self.n_concepts} concepts. "
                f"This would allocate multi-GB tensors. Use streaming mode instead. "
                f"(Threshold: {LARGE_CONCEPT_THRESHOLD})"
            )

    @staticmethod
    def _safe_index(weight: Tensor, indices: Tensor) -> Tensor:
        """
        DTensor-safe indexing using F.embedding.

        Replaces weight[indices] which crashes under FSDP2/DTensor.

        Args:
            weight: (N, D) weight matrix
            indices: (...) indices to select

        Returns:
            (..., D) selected embeddings
        """
        original_shape = indices.shape
        flat_indices = indices.reshape(-1)
        flat_result = F.embedding(flat_indices, weight)
        return flat_result.reshape(*original_shape, -1)

    def _get_embedding_weight(self) -> Tensor:
        """
        Get full embedding matrix.

        For dense: returns concept_embedding.weight
        For factorized: computes coef @ basis (materializes full matrix)

        Returns:
            (C, D) embedding matrix
        """
        if self.concept_embedding is not None:
            return self.concept_embedding.weight
        else:
            # Factorized: E = coef @ basis
            return self.embedding_basis(self.embedding_coef.weight)  # type: ignore

    def _get_embedding(self, indices: Tensor) -> Tensor:
        """
        Get embeddings for specific indices (DTensor-safe).

        For dense: uses F.embedding
        For factorized: looks up coef, then applies basis

        Args:
            indices: (...) concept indices

        Returns:
            (..., D) embeddings
        """

        if self.concept_embedding is not None:
            # Dense path - use nn.Embedding's forward (DTensor-safe)
            return self.concept_embedding(indices)
        else:
            # Factorized path: E[i] = basis(coef[i])
            coef = self.embedding_coef(indices)  # (..., r)  # type: ignore
            return self.embedding_basis(coef)  # (..., D)  # type: ignore

    def _get_predictor_weight(self) -> Tensor | None:
        """
        Get full predictor weight matrix (for linear path only).

        Returns:
            (C, D) predictor weight, or None if using attention
        """
        if self.concept_predictor is not None:
            return self.concept_predictor.weight
        elif self.predictor_down is not None and self.predictor_up is not None:
            # Factorized: W = up.weight @ down.weight
            # up: (C, r), down: (r, D) → W: (C, D)
            return self.predictor_up.weight @ self.predictor_down.weight
        else:
            return None

    @staticmethod
    def _merge_topk(
        topv: Tensor,
        topi: Tensor,
        v_blk: Tensor,
        i_blk: Tensor,
        k: int,
    ) -> tuple[Tensor, Tensor]:
        """Efficient merge of two top-k sets. Memory: O(BT × 2k)."""
        cand_v = torch.cat([topv, v_blk], dim=1)  # (BT, 2k)
        cand_i = torch.cat([topi, i_blk], dim=1)  # (BT, 2k)
        new_v, sel = torch.topk(cand_v, k, dim=1)  # (BT, k)
        new_i = torch.gather(cand_i, 1, sel)  # (BT, k)
        return new_v, new_i

    @staticmethod
    def linear_block_features(
        hidden: Tensor,
        predictor_weight: Tensor,
        embeddings: Tensor,
        block_size: int = 4096,
    ) -> Tensor:
        """
        Memory-efficient linear prediction without materializing (B, T, C).

        Args:
            hidden: (B, T, D)
            predictor_weight: (C, D)
            embeddings: (C, D)
            block_size: Concepts per block

        Returns:
            Features (B, T, D)
        """
        B, T, D = hidden.shape
        C = predictor_weight.size(0)

        output = torch.zeros(B, T, D, dtype=hidden.dtype, device=hidden.device)
        flat_h = hidden.reshape(-1, D)
        W_t = predictor_weight.t().contiguous()

        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            logits_block = (flat_h @ W_t[:, start:end]).to(torch.float32)
            logits_block = logits_block.clamp(-15, 15)
            weights_block = torch.sigmoid(logits_block)
            E_block = embeddings[start:end].to(weights_block.dtype)
            output.add_((weights_block @ E_block).reshape(B, T, D))

        return output.to(hidden.dtype)

    @staticmethod
    def attention_block_features(
        query: Tensor,
        embeddings: Tensor,
        block_size: int = 4096,
    ) -> Tensor:
        """Memory-efficient attention features without materializing (B, T, C)."""

        B, T, D = query.shape
        C = embeddings.shape[0]
        scale = 1.0 / math.sqrt(D)

        flat_q = query.reshape(-1, D)
        emb_T = embeddings.t().contiguous()
        output = torch.zeros(B * T, D, dtype=query.dtype, device=query.device)

        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            scores = (flat_q @ emb_T[:, start:end]).to(torch.float32) * scale
            scores = scores.clamp(-15, 15)
            weights = torch.sigmoid(scores)
            output.add_(weights @ embeddings[start:end].to(weights.dtype))

        return output.reshape(B, T, D).to(query.dtype)

    # Dense Streaming Top-K Methods
    @staticmethod
    def linear_features_topk_streaming(
        hidden: Tensor,
        predictor_weight: Tensor,
        embeddings: Tensor,
        k: int,
        block_size: int = 4096,
        topk_on_logits: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Memory-efficient linear prediction with streaming top-k.

        Uses merge-k-with-k to keep memory O(BT × k), not O(BT × block_size).

        Args:
            hidden: (B, T, D)
            predictor_weight: (C, D)
            embeddings: (C, D)
            k: Number of top concepts
            block_size: Concepts per block
            topk_on_logits: If True, select top-k by logits; else by sigmoid

        Returns:
            features: (B, T, D) weighted concept features
            topk_indices: (B, T, k) indices of top-k concepts
            topk_logits: (B, T, k) logits for top-k concepts
        """
        B, T, D = hidden.shape
        C = predictor_weight.size(0)
        BT = B * T
        device = hidden.device
        k = min(k, C)

        flat_h = hidden.reshape(BT, D)
        W_t = predictor_weight.t().contiguous()

        # Initialize top-k trackers
        topv = torch.full((BT, k), float("-inf"), device=device, dtype=hidden.dtype)
        topi = torch.zeros((BT, k), device=device, dtype=torch.long)

        # Pass 1: Find global top-k per token via streaming merge
        for start in range(0, C, block_size):
            end = min(start + block_size, C)

            # Compute logits for this block
            logits_blk = (flat_h @ W_t[:, start:end]).to(torch.float32).clamp_(-15, 15)

            # Values to rank by
            vals_blk = logits_blk if topk_on_logits else torch.sigmoid(logits_blk)

            # Get top-k within this block
            blk_k = min(k, end - start)
            v_blk, idx_blk = torch.topk(vals_blk, blk_k, dim=1)
            i_blk = idx_blk + start  # Offset to global indices

            # Pad if block smaller than k
            if blk_k < k:
                pad_v = torch.full((BT, k - blk_k), float("-inf"), device=device, dtype=torch.float32)
                pad_i = torch.zeros((BT, k - blk_k), device=device, dtype=torch.long)
                v_blk = torch.cat([v_blk, pad_v], dim=1)
                i_blk = torch.cat([i_blk, pad_i], dim=1)

            # Merge with running top-k
            topv, topi = ConceptHead._merge_topk(topv, topi, v_blk, i_blk, k)

        # Pass 2: Compute features from top-k concepts only
        # (Maybe)DTensor-safe: use _safe_index instead of direct indexing
        W_sel = ConceptHead._safe_index(predictor_weight, topi)  # (BT, k, D)

        # Recompute logits for selected concepts (for accurate gradients)
        logits_sel = torch.einsum("bd,bkd->bk", flat_h.to(torch.float32), W_sel.to(torch.float32))
        logits_sel = logits_sel.clamp(-15, 15)

        # Free W_sel before allocating E_sel to reduce peak memory
        del W_sel

        # Compute weights
        weights_sel = torch.sigmoid(logits_sel)

        # Compute features (DTensor-safe)
        E_sel = ConceptHead._safe_index(embeddings, topi)  # (BT, k, D)
        features = torch.einsum("bk,bkd->bd", weights_sel.to(E_sel.dtype), E_sel)

        return (
            features.reshape(B, T, D).to(hidden.dtype),
            topi.reshape(B, T, k),
            logits_sel.reshape(B, T, k),
        )

    @staticmethod
    def attention_features_topk_streaming(
        query: Tensor,
        embeddings: Tensor,
        k: int,
        block_size: int = 4096,
        topk_on_logits: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Memory-efficient attention with streaming top-k."""

        B, T, D = query.shape
        C = embeddings.shape[0]
        BT = B * T
        device = query.device
        scale = 1.0 / math.sqrt(D)
        k = min(k, C)

        flat_q = query.reshape(BT, D)
        emb_T = embeddings.t().contiguous()

        topv = torch.full((BT, k), float("-inf"), device=device, dtype=query.dtype)
        topi = torch.zeros((BT, k), device=device, dtype=torch.long)

        # Pass 1: Find top-k
        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            logits_blk = (flat_q @ emb_T[:, start:end]).to(torch.float32) * scale
            logits_blk = logits_blk.clamp(-15, 15)

            vals_blk = logits_blk if topk_on_logits else torch.sigmoid(logits_blk)

            blk_k = min(k, end - start)
            v_blk, idx_blk = torch.topk(vals_blk, blk_k, dim=1)
            i_blk = idx_blk + start

            if blk_k < k:
                pad_v = torch.full((BT, k - blk_k), float("-inf"), device=device, dtype=torch.float32)
                pad_i = torch.zeros((BT, k - blk_k), device=device, dtype=torch.long)
                v_blk = torch.cat([v_blk, pad_v], dim=1)
                i_blk = torch.cat([i_blk, pad_i], dim=1)

            topv, topi = ConceptHead._merge_topk(topv, topi, v_blk, i_blk, k)

        # Pass 2: Compute features (DTensor-safe)
        E_sel = ConceptHead._safe_index(embeddings, topi)  # (BT, k, D)
        logits_sel = torch.einsum("bd,bkd->bk", flat_q.to(torch.float32), E_sel.to(torch.float32)) * scale
        logits_sel = logits_sel.clamp(-15, 15)
        weights_sel = torch.sigmoid(logits_sel)
        features = torch.einsum("bk,bkd->bd", weights_sel.to(E_sel.dtype), E_sel)

        return (
            features.reshape(B, T, D).to(query.dtype),
            topi.reshape(B, T, k),
            logits_sel.reshape(B, T, k),
        )

    def attention_block_features_factorized(
        self,
        query: Tensor,
        block_size: int = 4096,
    ) -> Tensor:
        """
        Memory-efficient factorized attention over ALL concepts.

        Uses factorized scoring and feature computation:
        - Scoring: (query @ basis.T) @ coef.T instead of query @ E.T
        - Features: (weights @ coef) @ basis instead of weights @ E

        FLOPs: O(BT * r * (D + C)) instead of O(BT * D * C)

        Args:
            query: (B, T, D) query vectors from concept_query_projection
            block_size: Concepts per block for chunked processing

        Returns:
            (B, T, D) weighted concept features
        """
        assert self.factorize, "Only valid for factorized head"

        B, T, D = query.shape
        BT = B * T
        C = self.n_concepts
        _ = self.factorize_rank
        device = query.device
        scale = 1.0 / math.sqrt(D)

        flat_q = query.reshape(BT, D)

        # Get factorized components
        coef = self.embedding_coef.weight[:C]  # (C, r)  # type: ignore
        basis_weight = self.embedding_basis.weight  # (D, r)  # type: ignore

        # Compress query once: (BT, D) @ (D, r) → (BT, r)
        q_compressed = flat_q @ basis_weight

        # Output accumulator
        output = torch.zeros(BT, D, dtype=query.dtype, device=device)

        # Process concepts in blocks
        _ = (C + block_size - 1) // block_size
        for _block_idx, start in enumerate(range(0, C, block_size)):
            end = min(start + block_size, C)
            coef_chunk = coef[start:end]  # (chunk, r)

            # Factorized scores: (BT, r) @ (r, chunk) → (BT, chunk)
            scores_chunk = (q_compressed @ coef_chunk.T).float() * scale
            scores_chunk = scores_chunk.clamp(-15, 15)
            weights_chunk = torch.sigmoid(scores_chunk)

            # Factorized features:
            # weights @ E_chunk = weights @ (coef_chunk @ basis)
            #                   = (weights @ coef_chunk) @ basis
            weighted_coef = weights_chunk @ coef_chunk.float()  # (BT, r)
            features_chunk = weighted_coef @ basis_weight.T.to(weighted_coef.dtype)  # (BT, D)

            output.add_(features_chunk)

        return output.reshape(B, T, D).to(query.dtype)

    def attention_features_topk_factorized(
        self,
        query: Tensor,
        k: int,
        block_size: int = 4096,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Memory-efficient factorized attention with streaming top-k.

        Pass 1: Find top-k concepts using factorized scoring
        Pass 2: Compute features using only top-k embeddings

        Args:
            query: (B, T, D) query vectors
            k: Number of top concepts per token
            block_size: Concepts per block

        Returns:
            features: (B, T, D) weighted concept features
            topk_indices: (B, T, k) top-k concept indices
            topk_logits: (B, T, k) logits for top-k concepts
        """
        assert self.factorize, "Only valid for factorized head"

        B, T, D = query.shape
        BT = B * T
        C = self.n_concepts
        _ = self.factorize_rank
        device = query.device
        scale = 1.0 / math.sqrt(D)
        k = min(k, C)

        flat_q = query.reshape(BT, D)

        # Get factorized components
        coef = self.embedding_coef.weight[:C]  # (C, r)  # type: ignore
        basis_weight = self.embedding_basis.weight  # (D, r)  # type: ignore

        # Compress query: (BT, D) @ (D, r) → (BT, r)
        q_compressed = flat_q @ basis_weight

        # Initialize top-k trackers
        topv = torch.full((BT, k), float("-inf"), device=device, dtype=query.dtype)
        topi = torch.zeros((BT, k), device=device, dtype=torch.long)

        # Pass 1: Find global top-k via streaming merge
        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            coef_chunk = coef[start:end]  # (chunk, r)

            # Factorized scores for chunk
            scores_chunk = (q_compressed.float() @ coef_chunk.T.float()) * scale
            scores_chunk = scores_chunk.clamp(-15, 15)

            # Top-k within chunk
            blk_k = min(k, end - start)
            v_chunk, idx_chunk = torch.topk(scores_chunk, blk_k, dim=1)
            i_chunk = idx_chunk + start  # Global indices

            # Pad if block smaller than k
            if blk_k < k:
                pad_v = torch.full((BT, k - blk_k), float("-inf"), device=device, dtype=torch.float32)
                pad_i = torch.zeros((BT, k - blk_k), device=device, dtype=torch.long)
                v_chunk = torch.cat([v_chunk, pad_v], dim=1)
                i_chunk = torch.cat([i_chunk, pad_i], dim=1)

            # Merge with running top-k
            topv, topi = self._merge_topk(topv, topi, v_chunk, i_chunk, k)

        # Pass 2: Compute features from top-k (stay in factorized rank-r space)
        # Work with (BT, k, r) coefficients instead of (BT, k, D) to save ~16x memory
        coef_sel = self.embedding_coef(topi)  # (BT, k, r)  # type: ignore

        # Recompute logits in compressed space (for accurate gradients)
        logits_sel = torch.einsum("br,bkr->bk", q_compressed.float(), coef_sel.float()) * scale
        logits_sel = logits_sel.clamp(-15, 15)

        # Compute weighted features: (weights @ coef_sel) @ basis.T
        weights_sel = torch.sigmoid(logits_sel)
        weighted_coef = torch.einsum("bk,bkr->br", weights_sel.to(coef_sel.dtype), coef_sel)
        features = weighted_coef @ basis_weight.T.to(weighted_coef.dtype)  # (BT, D)

        return (
            features.reshape(B, T, D).to(query.dtype),
            topi.reshape(B, T, k),
            logits_sel.reshape(B, T, k),
        )

    def linear_block_features_factorized(
        self,
        hidden: Tensor,
        block_size: int = 4096,
    ) -> Tensor:
        """
        Memory-efficient factorized linear prediction over ALL concepts.

        Uses factorized predictor: logits = hidden @ down @ up.T
        Uses factorized embeddings: features = weights @ coef @ basis

        Args:
            hidden: (B, T, D) hidden states
            block_size: Concepts per block

        Returns:
            (B, T, D) weighted concept features
        """
        assert self.factorize, "Only valid for factorized head"
        assert self.predictor_down is not None, "Linear path requires predictor"

        B, T, D = hidden.shape
        BT = B * T
        C = self.n_concepts
        _ = self.factorize_rank
        device = hidden.device

        flat_h = hidden.reshape(BT, D)

        # Get factorized components
        coef = self.embedding_coef.weight[:C]  # (C, r)  # type: ignore
        basis_weight = self.embedding_basis.weight  # (D, r)  # type: ignore
        down_weight = self.predictor_down.weight  # (r, D)
        up_weight = self.predictor_up.weight[:C]  # (C, r)  # type: ignore

        # Compress hidden: (BT, D) @ (D, r) → (BT, r)
        h_compressed = flat_h @ down_weight.T

        # Output accumulator
        output = torch.zeros(BT, D, dtype=hidden.dtype, device=device)

        # Process concepts in blocks
        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            up_chunk = up_weight[start:end]  # (chunk, r)
            coef_chunk = coef[start:end]  # (chunk, r)

            # Factorized logits (cast to float32 for stability)
            logits_chunk = h_compressed.float() @ up_chunk.T.float()
            logits_chunk = logits_chunk.clamp(-15, 15)
            weights_chunk = torch.sigmoid(logits_chunk)

            # Factorized features (all in float32)
            weighted_coef = weights_chunk @ coef_chunk.float()  # (BT, r)
            features_chunk = weighted_coef @ basis_weight.T.to(weighted_coef.dtype)  # (BT, D)

            output.add_(features_chunk)

        return output.reshape(B, T, D).to(hidden.dtype)

    def linear_features_topk_factorized(
        self,
        hidden: Tensor,
        k: int,
        block_size: int = 4096,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Memory-efficient factorized linear with streaming top-k.

        Args:
            hidden: (B, T, D) hidden states
            k: Number of top concepts per token
            block_size: Concepts per block

        Returns:
            features: (B, T, D) weighted concept features
            topk_indices: (B, T, k) top-k concept indices
            topk_logits: (B, T, k) logits for top-k concepts
        """
        assert self.factorize, "Only valid for factorized head"
        assert self.predictor_down is not None, "Linear path requires predictor"

        B, T, D = hidden.shape
        BT = B * T
        C = self.n_concepts
        _ = self.factorize_rank
        device = hidden.device
        k = min(k, C)

        flat_h = hidden.reshape(BT, D)

        # Get factorized components
        down_weight = self.predictor_down.weight  # (r, D)
        up_weight = self.predictor_up.weight[:C]  # (C, r)  # type: ignore
        basis_weight = self.embedding_basis.weight  # (D, r)  # type: ignore

        # Compress hidden: (BT, D) @ (D, r) → (BT, r)
        h_compressed = flat_h @ down_weight.T

        # Initialize top-k trackers
        topv = torch.full((BT, k), float("-inf"), device=device, dtype=hidden.dtype)
        topi = torch.zeros((BT, k), device=device, dtype=torch.long)

        # Pass 1: Find global top-k via streaming merge
        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            up_chunk = up_weight[start:end]  # (chunk, r)

            # Factorized logits for chunk (cast to float32)
            logits_chunk = h_compressed.float() @ up_chunk.T.float()
            logits_chunk = logits_chunk.clamp(-15, 15)

            # Top-k within chunk
            blk_k = min(k, end - start)
            v_chunk, idx_chunk = torch.topk(logits_chunk, blk_k, dim=1)
            i_chunk = idx_chunk + start

            # Pad if needed
            if blk_k < k:
                pad_v = torch.full((BT, k - blk_k), float("-inf"), device=device, dtype=torch.float32)
                pad_i = torch.zeros((BT, k - blk_k), device=device, dtype=torch.long)
                v_chunk = torch.cat([v_chunk, pad_v], dim=1)
                i_chunk = torch.cat([i_chunk, pad_i], dim=1)

            # Merge
            topv, topi = self._merge_topk(topv, topi, v_chunk, i_chunk, k)

        # Pass 2: Compute features from top-k (stay in factorized rank-r space)
        # Work with (BT, k, r) coefficients instead of (BT, k, D) to save ~16x memory
        coef_sel = self.embedding_coef(topi)  # (BT, k, r)  # type: ignore
        up_sel = self._safe_index(self.predictor_up.weight[:C], topi)  # (BT, k, r)  # type: ignore

        # Recompute logits in compressed space (for accurate gradients)
        logits_sel = torch.einsum("br,bkr->bk", h_compressed.float(), up_sel.float())
        logits_sel = logits_sel.clamp(-15, 15)

        # Compute weighted features: (weights @ coef_sel) @ basis.T
        weights_sel = torch.sigmoid(logits_sel)
        weighted_coef = torch.einsum("bk,bkr->br", weights_sel.to(coef_sel.dtype), coef_sel)
        features = weighted_coef @ basis_weight.T.to(weighted_coef.dtype)  # (BT, D)

        return (
            features.reshape(B, T, D).to(hidden.dtype),
            topi.reshape(B, T, k),
            logits_sel.reshape(B, T, k),
        )

    # Sparse Logit Computation (for loss functions)
    def compute_logits_for_indices(
        self,
        hidden: Tensor,
        indices: Tensor,
    ) -> Tensor:
        """
        Compute logits for specific concept indices only (sparse).

        Supports both dense and factorized heads.

        IMPORTANT: This function materializes (M, K, D) where M is the number of
        tokens in hidden. Only call this with small M (e.g., masked tokens only).

        Args:
            hidden: (M, D) or (B, T, D) hidden states
            indices: (M, K) or (B, T, K) concept indices

        Returns:
            logits: Same shape as indices
        """
        # Handle both (M, D) and (B, T, D) inputs
        if hidden.dim() == 2:
            M, D = hidden.shape
            K = indices.size(-1)
            flat_h = hidden
            flat_idx = indices
            output_shape = indices.shape
        else:
            B, T, D = hidden.shape
            K = indices.size(-1)
            M = B * T
            flat_h = hidden.reshape(M, D)
            flat_idx = indices.reshape(M, K)
            output_shape = indices.shape

        # Safety check for large allocations
        estimated_bytes = M * K * D * 2  # fp16
        if estimated_bytes > 1e9:  # 1 GB threshold
            warnings.warn(  # noqa: B028
                f"compute_logits_for_indices will allocate ~{estimated_bytes / 1e9:.1f} GB. "
                f"Consider reducing M={M} (use masked tokens only) or K={K}."
            )

        n_valid = self.n_concepts
        indices_safe = flat_idx.clamp(0, n_valid - 1)

        if self.use_attention:
            query = self.concept_query_projection(flat_h.unsqueeze(0)).squeeze(0)  # (M, D)
            scale = 1.0 / math.sqrt(self.concept_dim)

            # Get embeddings for indices (DTensor-safe, works for both dense and factorized)
            E_sel = self._get_embedding(indices_safe)  # (M, K, D)
            logits = torch.einsum("md,mkd->mk", query.float(), E_sel.float()) * scale
        else:
            # Linear predictor
            if self.factorize:
                W = self._get_predictor_weight()[:n_valid]  # type: ignore
                W_sel = self._safe_index(W, indices_safe)  # (M, K, D)
            else:
                W = self.concept_predictor.weight[:n_valid]  # type: ignore
                W_sel = self._safe_index(W, indices_safe)  # (M, K, D)

            logits = torch.einsum("md,mkd->mk", flat_h.float(), W_sel.float())

        return logits.clamp(-15, 15).reshape(output_shape)

    def get_concept_weights(
        self,
        hidden: Tensor,
        concept_ids: Tensor,
    ) -> Tensor:
        """
        Get sigmoid weights for specific concepts (for attribution).

        Args:
            hidden: (B, T, D) or (M, D) hidden states
            concept_ids: (B, T, K) or (M, K) or (K,) concept indices

        Returns:
            weights: Same shape as concept_ids, values in [0, 1]
        """
        # Handle (K,) input by expanding
        if concept_ids.dim() == 1:
            if hidden.dim() == 2:
                M = hidden.size(0)
                concept_ids = concept_ids.unsqueeze(0).expand(M, -1)
            else:
                B, T, _ = hidden.shape
                concept_ids = concept_ids.unsqueeze(0).unsqueeze(0).expand(B, T, -1)

        logits = self.compute_logits_for_indices(hidden, concept_ids)
        return torch.sigmoid(logits)

    @staticmethod
    def blocked_logits(
        query: Tensor,
        embeddings: Tensor,
        block_size: int = 8192,
        out_device: torch.device | None = None,
        out_dtype: torch.dtype = torch.float32,
    ) -> Tensor:
        """
        Compute concept logits in column blocks for memory efficiency.

        logits = query @ embeddings.T / sqrt(D)
        """
        B, T, D = query.shape
        C = embeddings.size(0)
        scale = 1.0 / math.sqrt(D)

        dev = query.device if out_device is None else out_device
        logits = torch.empty(B, T, C, device=dev, dtype=out_dtype)

        q = query.reshape(-1, D).to(torch.float32)
        Et = embeddings.t().contiguous().to(torch.float32)

        for s in range(0, C, block_size):
            e = min(s + block_size, C)
            scores = (q @ Et[:, s:e]) * scale
            scores = scores.clamp(-15, 15)
            logits[:, :, s:e] = scores.reshape(B, T, e - s).to(out_dtype)

        return logits

    @staticmethod
    def blocked_mix(
        weights: Tensor,
        embeddings: Tensor,
        block_size: int = 8192,
    ) -> Tensor:
        """
        Compute weighted sum of embeddings in column blocks.

        output = weights @ embeddings
        """
        B, T, C = weights.shape
        D = embeddings.size(1)

        out = torch.zeros(B, T, D, device=weights.device, dtype=weights.dtype)

        for s in range(0, C, block_size):
            e = min(s + block_size, C)
            w_blk = weights[:, :, s:e].to(torch.float32)
            V_blk = embeddings[s:e].to(w_blk.dtype)
            out.add_(w_blk @ V_blk)

        return out.to(weights.dtype)

    @staticmethod
    def sigmoid_block_attention(
        query: Tensor,
        embeddings: Tensor,
        block_size: int = 8192,
        return_logits: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Memory-efficient sigmoid attention using block processing."""

        B, T, D = query.shape
        C = embeddings.shape[0]
        scale = 1.0 / math.sqrt(D)

        flat_q = query.reshape(-1, D)
        emb_T = embeddings.t().contiguous()
        output = torch.zeros(B * T, D, dtype=query.dtype, device=query.device)

        logits: Tensor | None = None
        if return_logits:
            logits = torch.empty(B, T, C, dtype=torch.float32, device=query.device)

        for start in range(0, C, block_size):
            end = min(start + block_size, C)
            scores = (flat_q @ emb_T[:, start:end]).to(torch.float32) * scale
            scores = scores.clamp(-15, 15)

            if logits is not None:
                logits[:, :, start:end] = scores.reshape(B, T, end - start)

            weights = torch.sigmoid(scores)
            output.add_(weights @ embeddings[start:end].to(weights.dtype))

        output = output.reshape(B, T, D).to(query.dtype)

        if return_logits:
            assert logits is not None
            return output, logits
        return output

    def _apply_sparse_interventions(
        self,
        features: Tensor,
        hidden: Tensor,
        intervene_ids: Tensor,
        intervene_vals: Tensor,
    ) -> Tensor:
        """
        Apply sparse interventions matching original dense behavior.

        Original dense behavior:
            weights = sigmoid(logits)  # (B, T, C)
            weights[..., c] = new_val  # Override
            features = weights @ embeddings

        Sparse equivalent:
            features += (new_val - current_weight) * embedding[c]
        """
        B, T, D = features.shape

        # the size of intervene_ids.size(-1) is K_int

        valid = intervene_ids != -1  # (B, T, K_int)

        if not valid.any():
            return features

        ids_safe = intervene_ids.clamp(0, self.n_concepts - 1)

        # Get current weights for intervened concepts
        current_logits = self.compute_logits_for_indices(hidden, ids_safe)  # (B, T, K_int)
        current_weights = torch.sigmoid(current_logits)  # (B, T, K_int)

        # Get embeddings for intervened concepts
        emb = self._get_embedding(ids_safe)  # (B, T, K_int, D)

        # Compute delta: new_weight - current_weight
        delta = (intervene_vals - current_weights) * valid.float()  # (B, T, K_int)

        # Apply correction: features += Σ delta_c * embedding_c
        correction = (delta.unsqueeze(-1) * emb).sum(dim=2)  # (B, T, D)

        return features + correction

    def _apply_dense_interventions(
        self,
        concept_weight: Tensor,
        intervene_ids: Tensor,
        intervene_vals: Tensor,
    ) -> Tensor:
        """Apply interventions by overriding concept weights (dense path)."""

        n_valid = min(self.n_concepts, concept_weight.size(-1))
        valid_edit = intervene_ids != -1
        ids = intervene_ids.clamp(0, n_valid - 1).long()
        vals = intervene_vals.to(concept_weight.dtype)

        updates = torch.zeros_like(concept_weight)
        updates.scatter_add_(2, ids, torch.where(valid_edit, vals, torch.zeros_like(vals)))

        set_mask = torch.zeros_like(concept_weight, dtype=torch.bool)
        set_mask.scatter_(2, ids, valid_edit)

        return torch.where(set_mask, updates, concept_weight)

    def topk_with_cutoff(self, tensor: Tensor, dim: int = -1) -> Tensor:
        """
        Apply top-k sparsity, zeroing out all but top-k values.

        Args:
            tensor: Input tensor, typically (B, T, C)
            dim: Dimension to apply top-k (default: last)

        Returns:
            Sparse tensor with only top-k values preserved
        """
        assert dim == -1 or dim == tensor.dim() - 1

        if self.topk is None:
            return tensor

        padded = tensor.size(dim)
        n_valid = min(self.n_concepts, padded)

        if n_valid <= 0:
            return torch.zeros_like(tensor)

        # Only look at valid concepts
        x = tensor.narrow(dim, 0, n_valid)
        kk = min(self.topk, n_valid)

        # Get top-k values and indices
        topv, topi = torch.topk(x, kk, dim=dim)

        # Create sparse output
        out = torch.zeros_like(x)
        out.scatter_(dim, topi, topv)

        # Pad back if needed
        if n_valid < padded:
            pad_shape = list(out.shape)
            pad_shape[dim] = padded - n_valid
            pad_zeros = out.new_zeros(pad_shape)
            out = torch.cat([out, pad_zeros], dim=dim)

        return out

    def _compute_weights(self, concept_logits: Tensor, E: Tensor) -> Tensor:
        """Compute concept weights from logits, with optional top-k sparsity."""

        apply_topk = self.topk is not None and ((not self.is_unknown) or self.apply_topk_to_unknown)

        if apply_topk and self.topk_on_logits:
            logits_for_weights = self.topk_with_cutoff(concept_logits)
            weights = torch.sigmoid(logits_for_weights).to(E.dtype)
            return weights

        weights = torch.sigmoid(concept_logits).to(E.dtype)

        if apply_topk and not self.topk_on_logits:
            weights = self.topk_with_cutoff(weights)

        return weights

    @torch.compiler.disable  # type: ignore[attr-defined]
    def forward(
        self,
        hidden: Tensor,
        intervene_ids: Tensor | None = None,
        intervene_vals: Tensor | None = None,
        return_logits: bool = False,
        store_hidden: bool = False,
    ) -> ConceptHeadOutput:
        """
        Forward pass for concept decomposition (inference only, no teacher forcing).

        Args:
            hidden: Transformer hidden states (B, T, n_embd)
            intervene_ids: Concept IDs to intervene on (B, T, K_int), -1 = skip
            intervene_vals: Intervention strength values (B, T, K_int)
            return_logits: If True, compute full (B, T, C) logits. Forbidden for large heads.
            store_hidden: If True, store hidden in output for later attribution.

        Returns:
            ConceptHeadOutput with features, predicted, topk_indices, topk_logits
        """
        B, T, _ = hidden.shape

        has_interventions = intervene_ids is not None and intervene_vals is not None

        if return_logits:
            self._check_dense_allowed("return_logits=True")

        # Get valid concept count
        n_valid = self.n_concepts

        # Initialize outputs
        concept_logits: Tensor | None = None
        concept_weight: Tensor | None = None
        predicted: Tensor
        topk_indices: Tensor | None = None
        topk_logits: Tensor | None = None

        # Determine if we should apply top-k
        apply_topk = self.topk is not None and (not self.is_unknown or self.apply_topk_to_unknown)
        k_features = self.topk_features if self.topk_features is not None else self.topk

        # Dense interventions need the full weight matrix to modify before computing features
        use_dense_intervention = has_interventions and not self._is_large

        # --- Step 1: Compute features (always via top-k when configured) ---
        if use_dense_intervention:
            # DENSE INTERVENTION PATH: must compute all weights to apply interventions
            E = self._get_embedding_weight()[:n_valid]

            if self.use_attention:
                query = self.concept_query_projection(hidden)
                concept_logits = self.blocked_logits(query, E, block_size=self.block_size)
            else:
                if self.factorize:
                    W = self._get_predictor_weight()[:n_valid]  # type: ignore
                    raw_logits = hidden @ W.T
                else:
                    raw_logits = self.concept_predictor(hidden)[..., :n_valid]  # type: ignore
                concept_logits = raw_logits.float().clamp(-15, 15)

            concept_weight = self._compute_weights(concept_logits, E)

            assert intervene_ids is not None and intervene_vals is not None
            concept_weight = self._apply_dense_interventions(concept_weight, intervene_ids, intervene_vals)

            predicted = self.blocked_mix(concept_weight, E, block_size=self.block_size)

        elif self.factorize:
            # FACTORIZED PATH (memory efficient)
            if self.use_attention:
                query = self.concept_query_projection(hidden)

                if apply_topk:
                    predicted, topk_indices, topk_logits = self.attention_features_topk_factorized(
                        query,
                        k=k_features,  # type: ignore
                        block_size=self.block_size,
                    )
                else:
                    predicted = self.attention_block_features_factorized(query, block_size=self.block_size)
            else:
                if apply_topk:
                    predicted, topk_indices, topk_logits = self.linear_features_topk_factorized(
                        hidden,
                        k=k_features,  # type: ignore
                        block_size=self.block_size,
                    )
                else:
                    predicted = self.linear_block_features_factorized(hidden, block_size=self.block_size)

        elif apply_topk:
            # DENSE STREAMING TOP-K PATH
            E = self._get_embedding_weight()[:n_valid]

            if self.use_attention:
                query = self.concept_query_projection(hidden)
                predicted, topk_indices, topk_logits = self.attention_features_topk_streaming(
                    query,
                    E,
                    k=k_features,  # type: ignore
                    block_size=self.block_size,
                    topk_on_logits=self.topk_on_logits,
                )
            else:
                W = self.concept_predictor.weight[:n_valid]  # type: ignore
                predicted, topk_indices, topk_logits = self.linear_features_topk_streaming(
                    hidden,
                    W,
                    E,
                    k=k_features,  # type: ignore
                    block_size=self.block_size,
                    topk_on_logits=self.topk_on_logits,
                )

        else:
            # DENSE ALL CONCEPTS PATH (no top-k configured)
            E = self._get_embedding_weight()[:n_valid]

            if self.use_attention:
                query = self.concept_query_projection(hidden)
                predicted = self.attention_block_features(query, E, block_size=self.block_size)
            else:
                W = self.concept_predictor.weight[:n_valid]  # type: ignore
                predicted = self.linear_block_features(hidden, W, E, block_size=self.block_size)

        # Slice top-k for loss from larger top-k for features
        if (
            topk_indices is not None
            and self.topk is not None
            and self.topk_features is not None
            and self.topk_features > self.topk
        ):
            _, rerank_idx = torch.topk(topk_logits, self.topk, dim=-1)  # type: ignore
            topk_indices = torch.gather(topk_indices, -1, rerank_idx)
            topk_logits = torch.gather(topk_logits, -1, rerank_idx)  # type: ignore

        # --- Step 2: Optionally compute dense logits/weights for analysis ---
        if return_logits and not use_dense_intervention:
            # Dense logits were not already computed; compute them now (analysis only)
            E = self._get_embedding_weight()[:n_valid]

            if self.use_attention:
                query = self.concept_query_projection(hidden)
                concept_logits = self.blocked_logits(query, E, block_size=self.block_size)
            else:
                if self.factorize:
                    W = self._get_predictor_weight()[:n_valid]  # type: ignore
                    raw_logits = hidden @ W.T
                else:
                    raw_logits = self.concept_predictor(hidden)[..., :n_valid]  # type: ignore
                concept_logits = raw_logits.float().clamp(-15, 15)

            concept_weight = self._compute_weights(concept_logits, E)

        # Debug: log which path was taken (once per head)
        if not hasattr(self, "_logged_forward_path"):
            self._logged_forward_path = True
            path = (
                "dense_intervention"
                if use_dense_intervention
                else "factorized_topk"
                if (self.factorize and apply_topk)
                else "factorized_all"
                if self.factorize
                else "streaming_topk"
                if apply_topk
                else "dense_all"
            )
            logger.info(
                f"[ConceptHead] {'Unknown' if self.is_unknown else 'Known'} head: "
                f"path={path}, topk={self.topk}, topk_features={self.topk_features}, "
                f"n_concepts={self.n_concepts}, factorize={self.factorize}, apply_topk={apply_topk}"
            )

        # Debug: log topk slice (once)
        if (  # noqa: SIM102 - ignore complex condition
            topk_indices is not None
            and self.topk is not None
            and self.topk_features is not None
            and self.topk_features > self.topk
        ):
            if not hasattr(self, "_logged_topk_slice"):
                self._logged_topk_slice = True
                logger.info(
                    f"[ConceptHead] {'Unknown' if self.is_unknown else 'Known'} head: "
                    f"Sliced topk: {self.topk_features} features -> {self.topk} for loss"
                )

        # Apply sparse interventions if needed
        if has_interventions and not use_dense_intervention:
            assert intervene_ids is not None and intervene_vals is not None
            predicted = self._apply_sparse_interventions(predicted, hidden, intervene_ids, intervene_vals)

        return ConceptHeadOutput(
            features=predicted,
            gt_features=None,
            logits=concept_logits,
            predicted=predicted,
            weights=concept_weight,
            topk_indices=topk_indices,
            topk_logits=topk_logits,
            hidden=hidden.detach() if store_hidden else None,
        )

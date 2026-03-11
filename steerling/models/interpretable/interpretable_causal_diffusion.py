"""
Interpretable CausalDiffusionLM with concept decomposition (inference-only).

Wraps CausalDiffusionLM with known + unknown concept heads for:
- Concept attribution (which concepts contribute to predictions)
- Concept steering (intervene on concept activations)
- Embedding extraction (hidden, composed, known, unknown)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from steerling.configs.causal_diffusion import CausalDiffusionConfig
from steerling.configs.concept import ConceptConfig
from steerling.models.causal_diffusion import CausalDiffusionLM
from steerling.models.interpretable.concept_head import ConceptHead, ConceptHeadOutput
from steerling.models.interpretable.outputs import InterpretableOutput

logger = logging.getLogger(__name__)


class InterpretableCausalDiffusionLM(nn.Module):
    """
    Interpretable CausalDiffusionLM with concept decomposition heads.

    Wraps a CausalDiffusionLM and adds:
    - Known concept head: predicts known concepts from hidden states
    - Unknown concept head: captures residual features (optional)
    - Steering via concept interventions

    Args:
        config: CausalDiffusionConfig (model architecture)
        concept_config: ConceptConfig (concept decomposition)
        vocab_size: Vocabulary size
    """

    def __init__(
        self,
        config: CausalDiffusionConfig,
        concept_config: ConceptConfig,
        vocab_size: int,
    ):
        super().__init__()

        self.config = config
        self.concept_config = concept_config
        self.vocab_size = vocab_size

        # Base transformer
        self.transformer = CausalDiffusionLM(config, vocab_size)

        # Known concept head
        self.known_head = ConceptHead(
            n_concepts=concept_config.n_concepts,
            concept_dim=concept_config.concept_dim,
            n_embd=config.n_embd,
            is_unknown=False,
            use_attention=concept_config.use_attention_known,
            topk=concept_config.topk_known,
            topk_features=concept_config.topk_known_features,
            block_size=concept_config.block_size,
            pad_multiple=concept_config.pad_multiple,
            store_unknown_weights=False,
            apply_topk_to_unknown=False,
            topk_on_logits=concept_config.topk_on_logits,
        )

        # Unknown concept head (optional)
        if concept_config.use_unknown:
            if concept_config.n_unknown_concepts is None:
                raise ValueError("n_unknown_concepts must be set when use_unknown=True")

            self.unknown_head: ConceptHead | None = ConceptHead(
                n_concepts=concept_config.n_unknown_concepts,
                concept_dim=concept_config.concept_dim,
                n_embd=config.n_embd,
                is_unknown=True,
                use_attention=concept_config.use_attention_unknown,
                topk=concept_config.unknown_topk,
                block_size=concept_config.block_size,
                pad_multiple=concept_config.pad_multiple,
                store_unknown_weights=False,
                apply_topk_to_unknown=concept_config.apply_topk_to_unknown,
                topk_on_logits=concept_config.topk_on_logits,
                factorize=concept_config.factorize_unknown,
                factorize_rank=concept_config.factorize_rank,
            )
        else:
            self.unknown_head = None

    def forward(
        self,
        input_ids: Tensor,
        *,
        input_embeds: Tensor | None = None,
        intervene_known_ids: Tensor | None = None,
        intervene_known_vals: Tensor | None = None,
        intervene_unknown_ids: Tensor | None = None,
        intervene_unknown_vals: Tensor | None = None,
        minimal_output: bool = False,
        position_injection: Tensor | None = None,
        steering_inject_layer: int | None = None,
        steering_inject_alpha: float = 1.0,
        unknown_topk: int = 64,
    ) -> tuple[Tensor, InterpretableOutput]:
        """
        Forward pass with concept decomposition.

        Args:
            input_ids: Token IDs (B, T). May contain mask tokens.
            input_embeds: Pre-computed embeddings (B, T, D). Overrides input_ids.
            intervene_known_ids: Known concept IDs to intervene (B, T, K_int)
            intervene_known_vals: Intervention values for known (B, T, K_int)
            intervene_unknown_ids: Unknown concept IDs to intervene (B, T, K_int)
            intervene_unknown_vals: Intervention values for unknown (B, T, K_int)
            minimal_output: If True, skip some expensive computations
            position_injection: Per-position steering injection (B, T, D)
            steering_inject_layer: Inject at layers >= this
            steering_inject_alpha: Injection strength
            unknown_topk: Top-k for unknown head attribution

        Returns:
            logits: LM logits (B, T, V)
            outputs: InterpretableOutput with all decomposition components
        """
        need_dense_logits = not minimal_output

        # Forward through transformer
        if position_injection is not None and steering_inject_layer is not None:
            hidden = self._forward_with_injection(
                input_ids,
                input_embeds,
                position_injection,
                steering_inject_layer,
                steering_inject_alpha,
            )
        else:
            hidden = self.transformer(input_ids, input_embeds=input_embeds, return_hidden=True)

        # Known concept head
        known_out: ConceptHeadOutput = self.known_head(
            hidden,
            intervene_ids=intervene_known_ids,
            intervene_vals=intervene_known_vals,
            return_logits=need_dense_logits,
        )
        known_features = known_out.features.to(hidden.dtype)

        # Residual unknown
        unk = hidden - known_features.detach()

        # Unknown head
        unk_for_lm: Tensor = unk
        unknown_out: ConceptHeadOutput | None = None
        unk_hat: Tensor | None = None

        if self.unknown_head is not None:
            unknown_out = self.unknown_head(
                hidden.detach(),
                intervene_ids=intervene_unknown_ids,
                intervene_vals=intervene_unknown_vals,
                return_logits=not minimal_output and not self.unknown_head._is_large,
            )
            assert unknown_out is not None
            unk_hat = unknown_out.features.to(hidden.dtype)
            unk_for_lm = unk_hat.detach()

        # Epsilon true
        epsilon_true = None
        if self.unknown_head is not None and unk_hat is not None:
            epsilon_true = hidden.detach() - (known_out.predicted + unk_hat)

        # Epsilon correction
        epsilon = None
        if self.concept_config.use_epsilon_correction and intervene_known_ids is None:
            epsilon = hidden - (unk_for_lm + known_features)
            unk_for_lm = unk_for_lm + epsilon

        # Compose and project
        composed = unk_for_lm + known_features
        logits = self.transformer.lm_head(composed)

        # Unknown top-k for attribution
        _unk_topk_indices = unknown_out.topk_indices if unknown_out else None
        _unk_topk_logits = unknown_out.topk_logits if unknown_out else None

        if (
            not minimal_output
            and self.unknown_head is not None
            and unknown_out is not None
            and _unk_topk_indices is None
            and unknown_topk > 0
        ):
            with torch.no_grad():
                _unk_topk_indices, _unk_topk_logits = self._compute_unknown_topk(hidden, unknown_topk)

        outputs = InterpretableOutput(
            hidden=hidden,
            known_features=known_features,
            known_logits=known_out.logits,
            known_gt_features=known_out.gt_features,
            known_predicted=known_out.predicted,
            known_weights=known_out.weights,
            known_topk_indices=known_out.topk_indices,
            known_topk_logits=known_out.topk_logits,
            unk=unk,
            unk_hat=unk_hat,
            unk_for_lm=unk_for_lm,
            unknown_logits=unknown_out.logits if unknown_out else None,
            unknown_weights=unknown_out.weights if unknown_out else None,
            unknown_topk_indices=_unk_topk_indices,
            unknown_topk_logits=_unk_topk_logits,
            composed=composed,
            epsilon=epsilon,
            epsilon_true=epsilon_true,
        )

        return logits, outputs

    def _compute_unknown_topk(self, hidden: Tensor, unknown_topk: int) -> tuple[Tensor | None, Tensor | None]:
        """Compute unknown head top-k indices for attribution."""

        assert self.unknown_head is not None

        if self.unknown_head.factorize:
            if self.unknown_head.use_attention:
                _query = self.unknown_head.concept_query_projection(hidden.detach())
                _, indices, logits = self.unknown_head.attention_features_topk_factorized(
                    _query, k=unknown_topk, block_size=self.unknown_head.block_size
                )
            else:
                _, indices, logits = self.unknown_head.linear_features_topk_factorized(
                    hidden.detach(),
                    k=unknown_topk,
                    block_size=self.unknown_head.block_size,
                )
        else:
            _E = self.unknown_head._get_embedding_weight()[: self.unknown_head.n_concepts]
            if self.unknown_head.use_attention:
                _query = self.unknown_head.concept_query_projection(hidden.detach())
                _, indices, logits = self.unknown_head.attention_features_topk_streaming(
                    _query,
                    _E,
                    k=unknown_topk,
                    block_size=self.unknown_head.block_size,
                )
            else:
                _W = self.unknown_head.concept_predictor.weight[: self.unknown_head.n_concepts]  # type: ignore
                _, indices, logits = self.unknown_head.linear_features_topk_streaming(
                    hidden.detach(),
                    _W,
                    _E,
                    k=unknown_topk,
                    block_size=self.unknown_head.block_size,
                )

        return indices, logits

    def _forward_with_injection(
        self,
        input_ids: Tensor,
        input_embeds: Tensor | None,
        position_injection: Tensor,
        inject_layer: int,
        inject_alpha: float,
    ) -> Tensor:
        """Forward through transformer with steering injection at specified layers."""

        x = input_embeds if input_embeds is not None else self.transformer.tok_emb(input_ids)

        for i, block in enumerate(self.transformer.blocks):
            x = block(x)
            if (i + 1) >= inject_layer:
                x = x + inject_alpha * position_injection

        x = self.transformer.ln_f(x)
        return x

    @torch.no_grad()
    def intervene(
        self,
        input_ids: Tensor,
        known: dict[int, float] | None = None,
        unknown: dict[int, float] | None = None,
        positions: Tensor | None = None,
    ) -> tuple[Tensor, InterpretableOutput]:
        """
        Run inference with concept interventions.

        Args:
            input_ids: Input token IDs (B, T)
            known: Dict mapping known concept IDs to intervention strengths
            unknown: Dict mapping unknown concept IDs to intervention strengths
            positions: Bool mask of positions to intervene (B, T). Default: all.

        Returns:
            logits: LM logits (B, T, V)
            outputs: InterpretableOutput
        """

        B, T = input_ids.shape
        device = input_ids.device

        if positions is None:
            positions = torch.ones(B, T, dtype=torch.bool, device=device)

        int_known_ids, int_known_vals = None, None
        if known is not None and len(known) > 0:
            int_known_ids, int_known_vals = self._build_intervention_tensors(known, B, T, positions, device)

        int_unknown_ids, int_unknown_vals = None, None
        if unknown is not None and len(unknown) > 0:
            int_unknown_ids, int_unknown_vals = self._build_intervention_tensors(
                unknown, B, T, positions, device
            )

        return self(
            input_ids,
            intervene_known_ids=int_known_ids,
            intervene_known_vals=int_known_vals,
            intervene_unknown_ids=int_unknown_ids,
            intervene_unknown_vals=int_unknown_vals,
            minimal_output=False,
        )

    @staticmethod
    def _build_intervention_tensors(
        interventions: dict[int, float],
        B: int,
        T: int,
        positions: Tensor,
        device: torch.device,
    ) -> tuple[Tensor, Tensor]:
        """Build intervention tensors for concept steering."""

        K = len(interventions)
        concept_ids = list(interventions.keys())
        directions = list(interventions.values())

        ids = torch.full((B, T, K), -1, dtype=torch.long, device=device)
        vals = torch.zeros((B, T, K), dtype=torch.float32, device=device)

        concept_tensor = torch.tensor(concept_ids, device=device)
        direction_tensor = torch.tensor(directions, dtype=torch.float32, device=device)

        n_active = int(positions.sum().item())
        ids[positions] = concept_tensor.unsqueeze(0).expand(n_active, -1)
        vals[positions] = direction_tensor.unsqueeze(0).expand(n_active, -1)

        return ids, vals

    def get_num_params(self, non_embedding: bool = True) -> int:
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and hasattr(self.transformer, "tok_emb"):
            n_params -= self.transformer.tok_emb.weight.numel()
        return n_params

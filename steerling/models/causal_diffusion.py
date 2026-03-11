"""
CausalDiffusionLM backbone model (inference-only).

A block-causal diffusion transformer. This file contains the pure compute
graph with no training logic.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from steerling.configs.causal_diffusion import CausalDiffusionConfig
from steerling.models.layers.causal_diffusion_layers import CausalDiffusionBlock
from steerling.models.layers.primitives import RMSNorm


class CausalDiffusionLM(nn.Module):
    """
    CausalDiffusionLM transformer backbone with block-causal attention.

    Pure compute graph — no training code, no loss logic.

    Args:
        config: CausalDiffusionConfig with model hyperparameters
        vocab_size: Vocabulary size (including special tokens)
    """

    def __init__(self, config: CausalDiffusionConfig, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        # Token embeddings
        self.tok_emb = nn.Embedding(vocab_size, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([CausalDiffusionBlock(config) for _ in range(config.n_layers)])

        # Final layer norm
        if config.use_rms_norm:
            self.ln_f: nn.Module = RMSNorm(config)
        else:
            self.ln_f = nn.LayerNorm(config.n_embd)

        # Output projection
        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)

        # Weight tying
        if config.weight_sharing:
            self.tok_emb.weight = self.lm_head.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        input_embeds: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            input_ids: Token indices [B, T] (may contain mask tokens)
            input_embeds: Pre-computed embeddings [B, T, D]. If provided, input_ids is ignored.
            return_hidden: If True, return hidden states before lm_head.

        Returns:
            logits [B, T, vocab_size] or hidden_states [B, T, n_embd]
        """
        if input_embeds is not None:
            x = input_embeds
        elif input_ids is not None:
            x = self.tok_emb(input_ids)
        else:
            raise ValueError("Either input_ids or input_embeds must be provided")

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)

        if return_hidden:
            return x

        return self.lm_head(x)

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return number of parameters."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.tok_emb.weight.numel()
        return n_params

    def _restore_weight_tying(self) -> None:
        """Re-establish weight tying after to_empty() or device transfer."""
        if self.config.weight_sharing:
            self.tok_emb.weight = self.lm_head.weight

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize model weights (used for fresh models, not loaded checkpoints)."""
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "SCALE_INIT"):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)

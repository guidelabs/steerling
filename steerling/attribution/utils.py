"""Shared helpers for input feature attribution."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
from torch import Tensor

from steerling.configs.attribution import BaselineConfig, BaselineMode


def normalize_attributions(attributions: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """Normalize so attributions sum to 1 along ``dim`` (sign-preserving)."""
    total = attributions.sum(dim=dim, keepdim=True)
    sign = torch.sign(total)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    denom = torch.clamp(torch.abs(total), min=eps)
    return attributions / (sign * denom)


def resolve_baseline_token_id(
    config: BaselineConfig,
    *,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> int | None:
    """
    Resolve a baseline config to a concrete token ID.

    Token ids are passed in (the open model config does not carry them); the
    faithful attributor threads them from the generator. Returns None for ZERO.
    """
    if config.token_id is not None:
        return config.token_id
    if config.mode == BaselineMode.ZERO:
        return None
    if config.mode == BaselineMode.MASK:
        if mask_token_id is None:
            raise ValueError(
                "BaselineMode.MASK needs a mask token id. Pass mask_token_id or "
                "use BaselineConfig(token_id=<id>)."
            )
        return mask_token_id
    if config.mode == BaselineMode.PAD:
        if pad_token_id is None:
            raise ValueError(
                "BaselineMode.PAD needs a pad token id. Pass pad_token_id or "
                "use BaselineConfig(token_id=<id>)."
            )
        return pad_token_id
    raise ValueError(f"Unknown baseline mode: {config.mode}")


def get_baseline_embedding(
    backbone: Any,
    config: BaselineConfig,
    *,
    mask_token_id: int | None = None,
    pad_token_id: int | None = None,
) -> Tensor:
    """Return the [D] baseline embedding (detached, on the model device)."""
    token_id = resolve_baseline_token_id(config, mask_token_id=mask_token_id, pad_token_id=pad_token_id)
    device = next(backbone.parameters()).device
    if token_id is None:
        return torch.zeros(backbone.config.n_embd, device=device)
    token_tensor = torch.tensor([token_id], device=device, dtype=torch.long)
    with torch.no_grad():
        emb = backbone.transformer.tok_emb(token_tensor)  # [1, D]
    return emb.squeeze(0).detach()


# def _backbone_forward_fn(backbone: Any) -> Callable[[Tensor, Tensor], tuple[Tensor, Any]]:
#     """
#     Build the forward used by IG.

#     Interpretable backbones return (logits, outputs). Plain backbones return hidden,
#     to which we apply lm_head, with outputs = None. The interpolated embeddings are
#     injected via input_embeds; there is no teacher forcing at inference.
#     """
#     if hasattr(backbone, "known_head"):

#         def forward_fn(input_ids: Tensor, interp: Tensor) -> tuple[Tensor, Any]:
#             return backbone(input_ids=input_ids, input_embeds=interp, minimal_output=False)

#     else:

#         def forward_fn(input_ids: Tensor, interp: Tensor) -> tuple[Tensor, Any]:
#             hidden = backbone.transformer(input_ids=input_ids, input_embeds=interp, return_hidden=True)
#             return backbone.transformer.lm_head(hidden), None

#     return forward_fn


def _backbone_forward_fn(backbone: Any) -> Callable[[Tensor, Tensor], tuple[Tensor, Any]]:
    """IG forward: transformer hidden -> lm_head. Clean end-to-end gradient.
    Equals the model's output when epsilon correction reconstructs hidden (matches scalex),
    and avoids the concept-head detaches that otherwise break completeness."""

    def forward_fn(input_ids: Tensor, interp: Tensor) -> tuple[Tensor, Any]:
        hidden = backbone.transformer(input_ids=input_ids, input_embeds=interp, return_hidden=True)
        return backbone.transformer.lm_head(hidden), None

    return forward_fn


@torch.compiler.disable
def integrated_gradients(
    backbone: Any,
    input_ids: Tensor,
    baseline_embedding: Tensor,
    target_fn: Callable[[Tensor, Any], Tensor],
    n_steps: int = 1,
) -> Tensor:
    """
    Batched integrated gradients over input embeddings (right-Riemann).

    Attributes each input token to a scalar target from target_fn. n_steps=1 is
    Input x Gradient. baseline_embedding accepts [D], [T, D], or [B, T, D].
    See https://arxiv.org/abs/1703.01365.
    """
    B, T_in = input_ids.shape

    with torch.no_grad():
        input_embeds = backbone.transformer.tok_emb(input_ids)

    if baseline_embedding.dim() == 1:  # [D]
        baseline = baseline_embedding.view(1, 1, -1).expand(B, T_in, -1)
    elif baseline_embedding.dim() == 2:  # [T, D] per-position (faithful path)
        baseline = baseline_embedding.unsqueeze(0).expand(B, -1, -1)
    else:  # [B, T, D]
        baseline = baseline_embedding

    forward_fn = _backbone_forward_fn(backbone)
    delta = input_embeds - baseline
    grad_accum = torch.zeros_like(input_embeds, dtype=torch.float32)

    for step in range(n_steps):
        alpha = (step + 1) / n_steps  # right-Riemann
        with torch.inference_mode(False), torch.enable_grad():
            interp = (baseline + alpha * delta).detach().requires_grad_(True)
            logits, outputs = forward_fn(input_ids, interp)
            targets = target_fn(logits, outputs)
            grads = torch.autograd.grad(targets.sum(), interp, create_graph=False)[0]
        grad_accum.add_(grads)

    return (delta.float() * grad_accum / n_steps).sum(dim=-1)

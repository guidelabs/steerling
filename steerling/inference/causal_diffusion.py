"""
Steerling text generator.

Main user-facing API for:
- Text generation (confidence-based unmasking)
- Concept steering (intervene on concept activations)
- Embedding extraction
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention

import steerling.models.layers.causal_diffusion_layers as layers
from steerling.configs.causal_diffusion import CausalDiffusionConfig
from steerling.configs.concept import ConceptConfig
from steerling.configs.generation import GenerationConfig
from steerling.data.tokenizer import SteerlingTokenizer
from steerling.inference.checkpoint_utils import load_config, load_state_dict
from steerling.models.causal_diffusion import CausalDiffusionLM
from steerling.models.interpretable.interpretable_causal_diffusion import (
    InterpretableCausalDiffusionLM,
)

logger = logging.getLogger(__name__)


@dataclass
class GenerationOutput:
    """Output from generation."""

    text: str
    tokens: torch.Tensor
    prompt_tokens: int
    generated_tokens: int


class SteerlingGenerator:
    """
    Generator for Steerling models.

    Uses confidence-based unmasking by default, which produces better results
    than left-to-right decoding by filling positions where the model is most
    confident first.

    Example:
        generator = SteerlingGenerator.from_pretrained("guidelabs/steerling-8b")

        config = GenerationConfig(max_new_tokens=100, seed=42)
        text = generator.generate("Once upon a time", config)
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: SteerlingTokenizer,
        model_config: CausalDiffusionConfig,
        is_interpretable: bool = False,
        device: str | torch.device = "cuda",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.is_interpretable = is_interpretable
        self.device = torch.device(device)

        self.model.to(self.device)
        self.model.eval()

        self.mask_token_id = tokenizer.mask_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        self.diff_block_size = model_config.diff_block_size

        logger.info(f"SteerlingGenerator initialized on {self.device}")
        logger.info(f"Interpretable: {is_interpretable}")

    def __repr__(self) -> str:
        params = sum(p.numel() for p in self.model.parameters())
        return (
            f"SteerlingGenerator(\n"
            f"  params={params:,},\n"
            f"  device={self.device},\n"
            f"  interpretable={self.is_interpretable},\n"
            f"  diff_block_size={self.diff_block_size}\n"
            f")"
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = torch.bfloat16,
    ) -> SteerlingGenerator:
        """
        Load a Steerling model from HuggingFace Hub or local directory.

        Args:
            model_name_or_path: HuggingFace repo ID (e.g. "guidelabs/steerling-8b")
                or path to a local directory containing config.json + safetensors.
            device: Device to load model on ("cuda" or "cpu")
            dtype: Model dtype (default: bfloat16)

        Returns:
            SteerlingGenerator ready for inference
        """

        # Load config
        raw_config = load_config(model_name_or_path)

        # Extract model fields only (config.json may contain tokenizer, concept, etc.)
        model_fields = set(CausalDiffusionConfig.model_fields.keys())
        model_data = {k: v for k, v in raw_config.items() if k in model_fields}
        model_config = CausalDiffusionConfig.model_validate(model_data)

        # Determine if interpretable
        is_interpretable = raw_config.get("interpretable", False)
        concept_data = raw_config.get("concept")

        # Vocab size from config or tokenizer default
        vocab_size = raw_config.get("vocab_size", SteerlingTokenizer().vocab_size)

        tokenizer = SteerlingTokenizer()

        # Create model
        if is_interpretable and concept_data is not None:
            concept_config = ConceptConfig.model_validate(concept_data)
            model: nn.Module = InterpretableCausalDiffusionLM(
                config=model_config,
                concept_config=concept_config,
                vocab_size=vocab_size,
            )
        else:
            model = CausalDiffusionLM(
                config=model_config,
                vocab_size=vocab_size,
            )

        # Load weights
        state_dict = load_state_dict(model_name_or_path)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            # Weight tying may cause lm_head.weight to appear "missing"
            non_tying = [k for k in missing if "lm_head" not in k]
            if non_tying:
                logger.warning(f"Missing keys (non-tying): {non_tying}")
            else:
                logger.info("Only weight-tied keys missing (expected)")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

        # Restore weight tying if needed
        if hasattr(model, "transformer"):
            model.transformer._restore_weight_tying()  # type: ignore
        elif hasattr(model, "_restore_weight_tying"):
            model._restore_weight_tying()  # type: ignore

        # Cast dtype
        if dtype is not None:
            model = model.to(dtype=dtype)
            logger.info(f"Cast model to {dtype}")

        # Workaround: disable compiled flex_attention to avoid Triton kernel errors
        layers.compiled_flex_attention = flex_attention

        generator = cls(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            is_interpretable=is_interpretable,
            device=device,
        )
        return generator

    @torch.inference_mode()
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text from a prompt. Returns generated text only."""
        return self.generate_full(prompt, config).text

    @torch.inference_mode()
    def generate_full(self, prompt: str, config: GenerationConfig) -> GenerationOutput:
        """
        Generate text with full output details using confidence-based unmasking.

        Args:
            prompt: Input text prompt
            config: Generation configuration

        Returns:
            GenerationOutput with text, tokens, and counts
        """
        max_new_tokens = config.max_new_tokens
        temperature = config.temperature
        top_p = config.top_p
        use_entropy_sampling = config.use_entropy_sampling
        repetition_penalty = config.repetition_penalty
        tokens_per_step = config.tokens_per_step
        steer_known = config.steer_known
        steer_unknown = config.steer_unknown

        if config.seed is not None:
            torch.manual_seed(config.seed)

        # Tokenize prompt (no special tokens for generation)
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_ids)
        total_len = prompt_len + max_new_tokens

        # Initialize sequence with mask tokens
        mask_id = self.mask_token_id
        x = torch.full((1, total_len), mask_id, dtype=torch.long, device=self.device)
        if prompt_len > 0:
            x[0, :prompt_len] = torch.tensor(prompt_ids, dtype=torch.long, device=self.device)

        # Track regions
        is_prompt_mask = torch.zeros(total_len, dtype=torch.bool, device=self.device)
        is_prompt_mask[:prompt_len] = True
        gen_region = ~is_prompt_mask
        is_finalized = is_prompt_mask.clone()

        # Banned tokens
        banned_ids = {mask_id}
        if self.pad_token_id is not None:
            banned_ids.add(self.pad_token_id)

        # Build intervention tensors for steering
        int_known_ids, int_known_vals = None, None
        int_unknown_ids, int_unknown_vals = None, None

        if self.is_interpretable and steer_known:
            int_known_ids, int_known_vals = self._build_intervention_tensors(steer_known, total_len)
        if self.is_interpretable and steer_unknown:
            int_unknown_ids, int_unknown_vals = self._build_intervention_tensors(steer_unknown, total_len)

        # Generation loop
        tokens_generated = 0
        eos_id = self.eos_token_id

        while tokens_generated < max_new_tokens:
            still_masked = (x[0] == mask_id) & gen_region
            masked_indices = still_masked.nonzero(as_tuple=False).squeeze(-1)

            if masked_indices.numel() == 0:
                break
            if masked_indices.dim() == 0:
                masked_indices = masked_indices.unsqueeze(0)

            # Forward pass
            if self.is_interpretable:
                logits, _ = self.model(
                    x,
                    use_teacher_forcing=False,
                    intervene_known_ids=int_known_ids,
                    intervene_known_vals=int_known_vals,
                    intervene_unknown_ids=int_unknown_ids,
                    intervene_unknown_vals=int_unknown_vals,
                    minimal_output=True,
                )
            else:
                logits = self.model(x)

            masked_logits = logits[0, masked_indices].clone()

            # Eliminate special tokens
            for tid in banned_ids:
                masked_logits[:, tid] = -1e9

            # Repetition penalty
            if repetition_penalty != 1.0:
                finalized_tokens = x[0, is_finalized].tolist()
                for tok in set(finalized_tokens):
                    if tok not in banned_ids:
                        masked_logits[:, tok] /= repetition_penalty

            # Select top-k positions by confidence
            probs_for_conf = torch.softmax(masked_logits, dim=-1)
            confidences = probs_for_conf.max(dim=-1).values
            k = min(tokens_per_step, masked_indices.numel())
            _, selected_pos_indices = confidences.topk(k)

            # Fill selected positions
            for pos_idx in selected_pos_indices:
                seq_idx = int(masked_indices[pos_idx].item())
                pos_logits = masked_logits[pos_idx]

                # Temperature (entropy-adaptive or fixed)
                if use_entropy_sampling:
                    pos_probs_raw = torch.softmax(pos_logits, dim=-1)
                    sorted_probs, _ = torch.sort(pos_probs_raw, descending=True)
                    cumsum = torch.cumsum(sorted_probs, dim=-1)
                    effective_k = max((cumsum <= top_p).sum().item() + 1, 2)

                    entropy = -torch.sum(pos_probs_raw * torch.log(pos_probs_raw + 1e-10))
                    normalized_entropy = min(1.0, entropy.item() / math.log(effective_k))
                    adaptive_temp = 0.3 + 0.4 * normalized_entropy
                    pos_probs = torch.softmax(pos_logits / adaptive_temp, dim=-1)
                else:
                    pos_probs = torch.softmax(pos_logits / max(temperature, 1e-8), dim=-1)

                tok = self._sample_top_p(pos_probs, top_p)
                x[0, seq_idx] = tok
                is_finalized[seq_idx] = True
                tokens_generated += 1

                if eos_id is not None and tok == eos_id:
                    break

            if eos_id is not None and (x[0, gen_region] == eos_id).any():
                break

        # Extract generated tokens
        final_tokens = []
        for i in range(prompt_len, total_len):
            if is_finalized[i]:
                final_tokens.append(x[0, i].item())
            else:
                break

        text = self.tokenizer.decode(final_tokens)
        out_tokens = torch.tensor(prompt_ids + final_tokens, dtype=torch.long)

        return GenerationOutput(
            text=text,
            tokens=out_tokens,
            prompt_tokens=prompt_len,
            generated_tokens=len(final_tokens),
        )

    @torch.inference_mode()
    def get_embeddings(
        self,
        text: str,
        pooling: str = "mean",
        embedding_type: str = "composed",
    ) -> torch.Tensor:
        """
        Get embeddings for input text.

        Args:
            text: Input text
            pooling: "mean", "last", "first", or "none"
            embedding_type: "hidden", "composed", "known", or "unknown"

        Returns:
            Embedding tensor
        """
        token_ids = self.tokenizer.encode(text, add_special_tokens=False)
        x = torch.tensor([token_ids], dtype=torch.long, device=self.device)

        if self.is_interpretable:
            _, outputs = self.model(x, use_teacher_forcing=False, minimal_output=False)

            type_map = {
                "hidden": outputs.hidden,
                "composed": outputs.composed,
                "known": outputs.known_features,
                "unknown": outputs.unk_hat if outputs.unk_hat is not None else outputs.unk,
            }
            if embedding_type not in type_map:
                raise ValueError(
                    f"Unknown embedding_type: {embedding_type}. Options: {list(type_map.keys())}"
                )
            hidden = type_map[embedding_type]
        else:
            if embedding_type not in ("hidden", "composed"):
                raise ValueError(f"embedding_type='{embedding_type}' requires an interpretable model.")

            hidden_states: dict[str, torch.Tensor] = {}

            def hook_fn(module, input, output):
                hidden_states["hidden"] = output

            handle = self.model.ln_f.register_forward_hook(hook_fn)  # type: ignore
            try:
                _ = self.model(x)
            finally:
                handle.remove()
            hidden = hidden_states["hidden"]

        hidden = hidden.squeeze(0)  # (T, D)

        pool_map = {
            "mean": lambda h: h.mean(dim=0),
            "last": lambda h: h[-1],
            "first": lambda h: h[0],
            "none": lambda h: h,
        }
        if pooling not in pool_map:
            raise ValueError(f"Unknown pooling: {pooling}. Options: {list(pool_map.keys())}")
        return pool_map[pooling](hidden)

    def _build_intervention_tensors(
        self, interventions: dict[int, float], seq_len: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        K = len(interventions)
        ids = (
            torch.tensor(list(interventions.keys()), device=self.device)
            .view(1, 1, K)
            .expand(1, seq_len, K)
            .clone()
        )
        vals = (
            torch.tensor(list(interventions.values()), dtype=torch.float32, device=self.device)
            .view(1, 1, K)
            .expand(1, seq_len, K)
            .clone()
        )
        return ids, vals

    def _sample_top_p(self, probs: torch.Tensor, top_p: float) -> int:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        cutoff_mask = cumulative <= top_p
        cutoff_mask[0] = True
        cutoff_idx = min(cutoff_mask.sum().item() + 1, len(sorted_probs))
        truncated = sorted_probs[:cutoff_idx]
        truncated = truncated / truncated.sum()
        return int(sorted_indices[torch.multinomial(truncated, 1)].item())

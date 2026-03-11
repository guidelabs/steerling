"""
Steerling text generator.

Main user-facing API for:
- Text generation (confidence-based block unmasking)
- Embedding extraction
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import flex_attention

from transformers import AutoModel

import steerling.models.layers.causal_diffusion_layers as layers
from steerling.configs.causal_diffusion import CausalDiffusionConfig
from steerling.configs.concept import ConceptConfig
from steerling.data.tokenizer import SteerlingTokenizer
from steerling.inference.checkpoint_utils import load_config, load_state_dict
from steerling.models.causal_diffusion import CausalDiffusionLM
from steerling.models.interpretable.interpretable_causal_diffusion import (
    InterpretableCausalDiffusionLM,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Gumbel-max sampling for categorical distributions (float64 for quality)."""
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def _get_num_transfer_tokens(mask_index: torch.Tensor, steps: int) -> torch.Tensor:
    """Precompute how many tokens to unmask at each step (linear schedule)."""
    mask_num = mask_index.sum(dim=1, keepdim=True)
    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer = base.new_zeros(mask_num.size(0), steps) + base
    for i in range(mask_num.size(0)):
        num_transfer[i, : remainder[i]] += 1

    return num_transfer


class SteerlingGenerator:
    """
    Generator for Steerling models.

    Generates text by iteratively unmasking tokens block-by-block,
    selecting the most confident positions first within each block.

    Example:
        generator = SteerlingGenerator.from_pretrained("guidelabs/steerling-8b")
        out = generator.generate("Once upon a time", gen_length=128, steps=128)
        print(generator.decode(out))
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

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        device: str | torch.device = "cuda",
        dtype: torch.dtype | None = torch.bfloat16,
    ) -> SteerlingGenerator:
        """
        Load a Steerling model from HuggingFace Hub or local directory.

        Tries AutoModel.from_pretrained first (works when the repo contains
        HF-compatible modeling files with trust_remote_code). Falls back to
        direct checkpoint loading for repos that only have safetensors + config.
        """
        raw_config = load_config(model_name_or_path)
        is_interpretable = raw_config.get("interpretable", False)

        model_fields = set(CausalDiffusionConfig.model_fields.keys())
        model_data = {k: v for k, v in raw_config.items() if k in model_fields}
        model_config = CausalDiffusionConfig.model_validate(model_data)

        tokenizer = SteerlingTokenizer()

        # --- Try AutoModel (HF-native path) ---
        try:
            hf_model = AutoModel.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype=dtype or torch.bfloat16,
            )
            logger.info("Loaded model via AutoModel.from_pretrained")
            layers.compiled_flex_attention = flex_attention

            return cls(
                model=hf_model,
                tokenizer=tokenizer,
                model_config=model_config,
                is_interpretable=is_interpretable,
                device=device,
            )
        except Exception as e:
            logger.info(f"AutoModel loading failed ({e}), falling back to direct checkpoint loading")

        # --- Fallback: direct checkpoint loading ---
        concept_data = raw_config.get("concept")
        vocab_size = raw_config.get("vocab_size", tokenizer.vocab_size)

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

        state_dict = load_state_dict(model_name_or_path)

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            non_tying = [k for k in missing if "lm_head" not in k]
            if non_tying:
                logger.warning(f"Missing keys (non-tying): {non_tying}")
            else:
                logger.info("Only weight-tied keys missing (expected)")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")

        if hasattr(model, "transformer"):
            model.transformer._restore_weight_tying()  # type: ignore
        elif hasattr(model, "_restore_weight_tying"):
            model._restore_weight_tying()  # type: ignore

        if dtype is not None:
            model = model.to(dtype=dtype)
            logger.info(f"Cast model to {dtype}")

        layers.compiled_flex_attention = flex_attention

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            is_interpretable=is_interpretable,
            device=device,
        )

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        tokenizer: SteerlingTokenizer,
        device: str | torch.device = "cuda",
    ) -> SteerlingGenerator:
        """Wrap a pre-loaded model and tokenizer into a SteerlingGenerator."""
        hf_config = getattr(model, "config", None)

        if hf_config is not None:
            config_dict = {k: v for k, v in hf_config.to_dict().items()
                          if k in CausalDiffusionConfig.model_fields and k not in {"model_type", "transformers_version", "auto_map", "architectures"}}
            model_config = CausalDiffusionConfig.model_validate(config_dict)
            is_interpretable = getattr(hf_config, "interpretable", False)
        else:
            model_config = CausalDiffusionConfig()
            is_interpretable = False

        layers.compiled_flex_attention = flex_attention

        return cls(
            model=model,
            tokenizer=tokenizer,
            model_config=model_config,
            is_interpretable=is_interpretable,
            device=device,
        )

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str | torch.Tensor,
        steps: int = 128,
        gen_length: int = 128,
        block_length: int | None = None,
        temperature: float = 0.0,
        cfg_scale: float = 0.0,
        stop_tokens: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Generate text via block-by-block confidence-based unmasking.

        Within each block, tokens are unmasked over ``steps_per_block`` steps
        using a linear schedule. At each step the most confident masked
        positions are selected for unmasking. After completing a block, if
        any stop token is present the remaining blocks are skipped.

        Args:
            prompt: Input text string or token tensor of shape (B, L).
            steps: Total denoising steps (split evenly across blocks).
            gen_length: Number of tokens to generate (must be divisible by block_length).
            block_length: Block size. Defaults to config diff_block_size.
            temperature: Gumbel noise temperature (0 = greedy).
            cfg_scale: Classifier-free guidance scale (0 = disabled).
            stop_tokens: Token IDs that trigger early stop between blocks.

        Returns:
            Full sequence tensor (prompt + generated), shape (B, L + gen_length).
        """
        if block_length is None:
            block_length = self.diff_block_size
        mask_id = self.mask_token_id

        # Encode prompt
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        else:
            prompt_tensor = prompt.to(self.device)

        bsz = prompt_tensor.shape[0]
        prompt_len = prompt_tensor.shape[1]

        # Initialize: prompt + all-masked generation region
        x = torch.full(
            (bsz, prompt_len + gen_length), mask_id, dtype=torch.long, device=self.device
        )
        x[:, :prompt_len] = prompt_tensor

        prompt_index = x != mask_id

        assert gen_length % block_length == 0, (
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        )
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0, (
            f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        )
        steps_per_block = steps // num_blocks

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length

            block_mask_index = x[:, block_start:block_end] == mask_id
            num_transfer = _get_num_transfer_tokens(block_mask_index, steps_per_block)

            for step in range(steps_per_block):
                mask_index = x == mask_id

                # Forward pass with optional CFG
                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    logits = self._forward(x_)
                    cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                    logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
                else:
                    logits = self._forward(x)

                # Sample candidates
                logits_with_noise = _add_gumbel_noise(logits, temperature=temperature)
                x0 = torch.argmax(logits_with_noise, dim=-1)

                # Confidence = softmax probability of chosen token
                p = F.softmax(logits.float(), dim=-1)
                x0_p = torch.gather(p, dim=-1, index=x0.unsqueeze(-1)).squeeze(-1)

                # Only consider positions within current block
                x0_p[:, :block_start] = -np.inf
                x0_p[:, block_end:] = -np.inf

                x0 = torch.where(mask_index, x0, x)
                confidence = torch.where(mask_index, x0_p, -np.inf)

                # Unmask top-k most confident positions
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)
                for j in range(bsz):
                    k = num_transfer[j, step].item()
                    if k > 0:
                        _, select_index = torch.topk(confidence[j], k=k)
                        transfer_index[j, select_index] = True

                x[transfer_index] = x0[transfer_index]

            # After completing a block, check for stop tokens
            if stop_tokens:
                generated = x[:, prompt_len:block_end]
                if any((generated == t).any() for t in stop_tokens):
                    break

        return x

    def decode(self, output: torch.Tensor, prompt_len: int | None = None, skip_special: bool = True) -> str:
        """Decode generated tensor to text, stripping mask tokens."""
        ids = output[0, prompt_len:].tolist() if prompt_len is not None else output[0].tolist()

        ids = [t for t in ids if t != self.mask_token_id]
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special)

    # ------------------------------------------------------------------
    # Embeddings
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits, handling both HF and native models."""
        if self.is_interpretable:
            output = self.model(input_ids, minimal_output=True)
        else:
            output = self.model(input_ids)

        if isinstance(output, tuple):
            return output[0]
        return output

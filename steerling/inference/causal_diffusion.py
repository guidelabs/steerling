"""
Steerling text generator.

Main user-facing API for:
- Text generation (confidence-based block unmasking)
- Embedding extraction
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.nn.attention.flex_attention import flex_attention
from transformers import AutoModel

import steerling.models.layers.causal_diffusion_layers as layers
from steerling.configs.causal_diffusion import CausalDiffusionConfig
from steerling.configs.concept import ConceptConfig
from steerling.configs.generation import GenerationConfig
from steerling.configs.steering import SteeringConfig
from steerling.data.tokenizer import SteerlingTokenizer
from steerling.inference.checkpoint_utils import load_config, load_state_dict
from steerling.models.causal_diffusion import CausalDiffusionLM
from steerling.models.interpretable.interpretable_causal_diffusion import (
    InterpretableCausalDiffusionLM,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GenerationOutput:
    """Output from generation."""

    text: str
    tokens: torch.Tensor
    prompt_tokens: int
    generated_tokens: int


@dataclass
class GenerationStepInfo:
    """Information exposed at each generation step for attribution hooks.

    Passed to the step_callback in generate_full. All tensors are
    references to the forward-pass outputs (no copies), so the callback
    should extract what it needs before returning.

    Attributes:
        step: Denoising iteration index.
        logits: [1, T, V] logits from this forward pass.
        outputs: Model outputs (InterpretableTrainingOutput when
            minimal_output=True).
        committed_positions: [P] sequence indices committed this step.
        committed_token_ids: [P] token IDs committed this step.
    """

    step: int
    logits: torch.Tensor
    outputs: object
    committed_positions: torch.Tensor
    committed_token_ids: torch.Tensor


# Step callback type: called at each token commit during generation.
StepCallback = Callable[[GenerationStepInfo], None]


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


def _sample_top_p(
    probs: torch.Tensor, top_p: float, generator: torch.Generator | None = None
) -> torch.Tensor:
    """Nucleus (top-p) sampling from a 1-D probability distribution."""
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = (cumulative - sorted_probs) <= top_p
    keep[0] = True
    filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
    filtered = filtered / filtered.sum()
    return sorted_indices[torch.multinomial(filtered, 1, generator=generator)]


def _sample_token(
    logits_1d: torch.Tensor, config: GenerationConfig, generator: torch.Generator | None = None
) -> torch.Tensor:
    """
    Sample a single token from logits for one position.
    """
    if config.use_entropy_sampling:
        probs_raw = torch.softmax(logits_1d, dim=-1)
        sorted_probs, _ = torch.sort(probs_raw, descending=True)
        cumsum = torch.cumsum(sorted_probs, dim=-1)
        effective_k = max((cumsum <= config.top_p).sum().item() + 1, 2)
        entropy = -torch.sum(probs_raw * torch.log(probs_raw + 1e-10))
        normalized_entropy = min(1.0, entropy.item() / math.log(effective_k))
        temperature = 0.3 + 0.4 * normalized_entropy
        probs = torch.softmax(logits_1d / temperature, dim=-1)
        return _sample_top_p(probs, config.top_p, generator).squeeze(0)
    elif config.top_p > 0.0 and config.temperature > 0.0:
        probs = torch.softmax(logits_1d / config.temperature, dim=-1)
        return _sample_top_p(probs, config.top_p, generator).squeeze(0)
    else:
        return torch.argmax(logits_1d)


class SteerlingGenerator:
    """
    Generator for Steerling models.

    Generates text by iteratively unmasking tokens block-by-block,
    selecting the most confident positions first within each block.

    Example:
        generator = SteerlingGenerator.from_pretrained("guidelabs/steerling-8b")
        text = generator.generate("Once upon a time", GenerationConfig(max_new_tokens=128))
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

        # Tokens that should never be generated
        self._banned_ids = torch.tensor(
            [tokenizer.mask_token_id, tokenizer.pad_token_id],
            dtype=torch.long,
        )

        # EOT token for instruct models
        self._eot_id = getattr(tokenizer, "eot_id", None)

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

        instruct = raw_config.get("instruct", False)
        tokenizer = SteerlingTokenizer(instruct=instruct)

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
            model._restore_weight_tying()

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
            config_dict = {
                k: v
                for k, v in hf_config.to_dict().items()
                if k in CausalDiffusionConfig.model_fields
                and k not in {"model_type", "transformers_version", "auto_map", "architectures"}
            }
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

    @torch.inference_mode()
    def generate(self, prompt: str, config: GenerationConfig) -> str:
        """Generate text from a prompt. Returns generated text only."""
        return self.generate_full(prompt, config).text

    @torch.inference_mode()
    def generate_steered(
        self,
        prompt: str | torch.Tensor,
        config: GenerationConfig,
        steering: SteeringConfig,
    ) -> GenerationOutput:
        """
        Generate text while steering concept activations.

        Adds a concept direction to the residual stream at layers
        >= ``steering.inject_layer`` and/or suppresses positively aligned
        vocab via ``steering.relu_logit_mask``. Requires an interpretable model.
        """
        if not self.is_interpretable:
            raise ValueError("Steering requires an interpretable model.")
        return self.generate_full(prompt, config, steering=steering)

    @torch.inference_mode()
    def concept_top_tokens(self, concept_id: int, k: int = 15) -> list[tuple[str, float]]:
        """
        The vocabulary tokens a concept most promotes, with alignment scores.

        Projects the (unit-normalized) concept embedding onto the LM head. This
        shows what the concept actually does in the loaded weights, which is the
        reliable way to confirm an ID matches the concept you intend to steer.
        """
        if not self.is_interpretable:
            raise ValueError("Concept inspection requires an interpretable model.")
        cid = torch.tensor([concept_id], device=self.device)
        emb = self.model.known_head._get_embedding(cid).float()[0]
        emb = emb / (emb.norm() + 1e-12)
        alignment = self.model.transformer.lm_head.weight.float() @ emb
        values, indices = alignment.topk(k)
        return [(self.tokenizer.decode([int(i)]), float(v)) for i, v in zip(indices, values, strict=True)]

    @torch.inference_mode()
    def generate_full(
        self,
        prompt: str | torch.Tensor,
        config: GenerationConfig,
        steering: SteeringConfig | None = None,
        step_callback: StepCallback | None = None,
    ) -> GenerationOutput:
        """
        Generate text via block-by-block confidence-based unmasking.

        Within each block, tokens are unmasked over ``steps_per_block`` steps
        using a linear schedule. At each step the most confident masked
        positions are selected for unmasking. After completing a block, if
        any stop token is present the remaining blocks are skipped.

        Args:
            prompt: Input text string or token tensor of shape (B, L).
            config: Generation configuration.
            step_callback: Optional callback invoked each time tokens are committed.
                Called with (positions, token_ids, InterpretableOutput).
                Used by ConceptAttributor for faithful attribution.

        Returns:
            GenerationOutput with text, tokens, and counts.
        """
        gen_length = config.max_new_tokens
        steps = config.steps
        cfg_scale = config.cfg_scale

        # Create a dedicated generator for reproducibility
        generator: torch.Generator | None = None
        if config.seed is not None:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(config.seed)

        block_length = self.diff_block_size
        mask_id = self.mask_token_id

        # Steering: precompute injection direction/alpha and/or vocab suppression
        steer_direction: torch.Tensor | None = None
        steer_base_alpha: float = 1.0
        steer_inject_layer: int | None = None
        steer_schedule: str = "fixed"
        steer_cutoff: int = 32
        relu_suppression: torch.Tensor | None = None
        if steering is not None:
            if steering.has_injection:
                steer_direction, steer_base_alpha, steer_inject_layer = self._prepare_steering(steering)
                steer_schedule = steering.inject_alpha_schedule
                steer_cutoff = steering.cutoff_tokens
            if steering.has_logit_mask:
                assert steering.relu_logit_mask is not None
                relu_suppression = self._build_relu_mask(steering.relu_logit_mask)
        # Use full forward when step_callback needs InterpretableOutput
        need_outputs = step_callback is not None and self.is_interpretable

        # Encode prompt
        if isinstance(prompt, str):
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=self.device)
        else:
            prompt_tensor = prompt.to(self.device)
            prompt_ids = prompt_tensor[0].tolist()

        bsz = prompt_tensor.shape[0]
        prompt_len = prompt_tensor.shape[1]

        # Initialize: prompt + all-masked generation region
        x = torch.full((bsz, prompt_len + gen_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_len] = prompt_tensor

        prompt_index = x != mask_id

        assert (
            gen_length % block_length == 0
        ), f"max_new_tokens ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length

        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks

        # Stop tokens
        stop_tokens: list[int] = list(config.stop_tokens or [])

        # Banned token IDs on device
        banned_ids = self._banned_ids.to(self.device)

        # Track if generation should stop (EOT propagation)
        generation_done = False
        global_step = 0

        for block_idx in range(num_blocks):
            if generation_done:
                break

            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length

            block_mask_index = x[:, block_start:block_end] == mask_id
            num_transfer = _get_num_transfer_tokens(block_mask_index, steps_per_block)

            for step in range(steps_per_block):
                # Forward pass
                # Resolve injection strength under the alpha schedule, then inject
                # only at currently-masked positions (mask-aligned injection).
                current_alpha = 0.0
                if steer_direction is not None:
                    masked_remaining = int((x == mask_id).sum().item())
                    current_alpha = self._resolve_alpha(
                        steer_schedule, steer_base_alpha, masked_remaining, gen_length, steer_cutoff
                    )
                direction = steer_direction if current_alpha != 0.0 else None

                if cfg_scale > 0.0:
                    un_x = x.clone()
                    un_x[prompt_index] = mask_id
                    x_ = torch.cat([x, un_x], dim=0)
                    inj = self._position_injection(x_, direction)
                    logits = self._forward(x_, inj, steer_inject_layer, current_alpha)
                    cond_logits, uncond_logits = torch.chunk(logits, 2, dim=0)
                    logits = uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)
                    interp_outputs = None
                else:
                    # Single steered forward. return_outputs threads attribution
                    # through the SAME injected pass, so steering is still applied
                    # when a step_callback is set (was silently dropped before).
                    inj = self._position_injection(x, direction)
                    result = self._forward(
                        x, inj, steer_inject_layer, current_alpha, return_outputs=need_outputs
                    )
                    if need_outputs:
                        logits, interp_outputs = result
                    else:
                        logits, interp_outputs = result, None

                # Work only on masked positions within the current block
                for j in range(bsz):
                    masked_positions = (
                        (x[j, block_start:block_end] == mask_id).nonzero(as_tuple=False).squeeze(1)
                    )
                    if masked_positions.numel() == 0:
                        continue
                    # Offset to full sequence positions
                    masked_positions = masked_positions + block_start

                    all_logits = logits[j, masked_positions].float()

                    # Concept suppression: subtract strength * relu(concept-vocab alignment)
                    if relu_suppression is not None:
                        all_logits = all_logits - relu_suppression

                    # Repetition penalty
                    if config.repetition_penalty != 1.0:
                        vocab_size = all_logits.shape[-1]
                        repeated = x[j].unique()
                        repeated = repeated[(repeated >= 0) & (repeated < vocab_size)]
                        if repeated.numel() > 0:
                            values = all_logits[:, repeated]
                            penalized = torch.where(
                                values > 0,
                                values / config.repetition_penalty,
                                values * config.repetition_penalty,
                            )
                            all_logits[:, repeated] = penalized

                    # Suppress EOS token
                    all_logits[:, self.eos_token_id] = float("-inf")

                    # Suppress banned tokens (mask, pad)
                    all_logits.index_fill_(dim=-1, index=banned_ids, value=float("-inf"))

                    # Select most confident position
                    probs_for_conf = torch.softmax(all_logits, dim=-1)
                    confidences = probs_for_conf.max(dim=-1).values

                    k = int(num_transfer[j, step].item())
                    if k == 0:
                        continue
                    _, selected_pos_indices = confidences.topk(min(k, masked_positions.numel()))

                    # Sample token at each selected position
                    committed_positions: list[int] = []
                    committed_token_ids: list[int] = []
                    for pos_idx in selected_pos_indices:
                        row_logits = all_logits[pos_idx]
                        token = _sample_token(row_logits, config, generator)
                        pos = masked_positions[pos_idx]
                        x[j, pos] = token
                        committed_positions.append(int(pos))
                        committed_token_ids.append(int(token))

                        # EOT propagation: fill everything after with EOS
                        token_is_terminal = token == self._eot_id if self._eot_id is not None else False
                        if token_is_terminal:
                            x[j, pos + 1 :] = self.eos_token_id
                            generation_done = True
                            break

                    # Fire callback with committed positions
                    if step_callback is not None and interp_outputs is not None and committed_positions:
                        step_callback(
                            GenerationStepInfo(
                                step=global_step,
                                logits=logits,
                                outputs=interp_outputs,
                                committed_positions=torch.tensor(committed_positions, device=self.device),
                                committed_token_ids=torch.tensor(
                                    committed_token_ids, dtype=torch.long, device=self.device
                                ),
                            )
                        )

                    if generation_done:
                        break

                global_step += 1

                if generation_done:
                    break

            # After completing a block, check for stop tokens
            if not generation_done and stop_tokens:
                generated = x[:, prompt_len:block_end]
                if any((generated == t).any() for t in stop_tokens):
                    break

        # Extract generated tokens (strip special/terminal tokens)
        gen_ids = x[0, prompt_len:].tolist()
        terminal_tokens = set(stop_tokens)
        if self._eot_id is not None:
            terminal_tokens.add(self._eot_id)

        final_tokens = []
        for t in gen_ids:
            if t == mask_id:
                continue
            if t == self.eos_token_id:
                continue
            if t in terminal_tokens:
                break
            final_tokens.append(t)

        text = self.tokenizer.decode(final_tokens)

        return GenerationOutput(
            text=text,
            tokens=x[0],
            prompt_tokens=prompt_len,
            generated_tokens=len(final_tokens),
        )

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
            _, outputs = self.model(x, minimal_output=False)

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

    def _forward(
        self,
        input_ids: torch.Tensor,
        injection: torch.Tensor | None = None,
        inject_layer: int | None = None,
        inject_alpha: float = 1.0,
        return_outputs: bool = False,
    ):
        """Forward pass. Returns logits, or (logits, InterpretableOutput) when return_outputs=True.

        Threads steering injection into the model forward when provided.
        """
        if self.is_interpretable:
            if injection is not None and inject_layer is not None:
                output = self.model(
                    input_ids,
                    minimal_output=True,
                    position_injection=injection,
                    steering_inject_layer=inject_layer,
                    steering_inject_alpha=inject_alpha,
                )
            else:
                output = self.model(input_ids, minimal_output=True)
        else:
            output = self.model(input_ids)

        if isinstance(output, tuple):
            if return_outputs:
                return output[0], output[1]
            return output[0]
        if return_outputs:
            return output, None
        return output

    @staticmethod
    def _resolve_alpha(
        schedule: str,
        base_alpha: float,
        masked_remaining: int,
        max_new_tokens: int,
        cutoff_tokens: int,
    ) -> float:
        """
        Current injection alpha given the schedule and how many tokens are committed.

        committed = max_new_tokens - masked_remaining (counted across the full
        sequence). 'fixed' holds base_alpha. 'hard_cutoff' applies base_alpha while
        committed < cutoff_tokens, then drops to zero.
        """
        if schedule == "fixed":
            return base_alpha
        if schedule == "hard_cutoff":
            committed = max(max_new_tokens, 1) - masked_remaining
            return base_alpha if committed < cutoff_tokens else 0.0
        raise ValueError(f"Unknown inject_alpha_schedule: {schedule!r}")

    def _prepare_steering(self, steering: SteeringConfig) -> tuple[torch.Tensor, float, int]:
        """
        Compute the steering direction, base alpha, and injection layer.

        The direction is the L2-normalized sum of the concept embeddings, so a
        concept group composes into one unit direction. base_alpha scales it:
        with normalize_mai_lm_target, mai_lm_target reads in logit units
        (divided by the direction's peak LM-head alignment); otherwise it is the
        raw alpha. The effective top-token logit shift is base_alpha * peak.
        """
        head = self.model.known_head
        ids = torch.tensor(steering.concept_ids, device=self.device)
        emb = head._get_embedding(ids).float()  # (K, D)

        vec = emb.sum(dim=0)  # (D,)
        direction = vec / (vec.norm(p=2) + 1e-12)

        lm_weight = self.model.transformer.lm_head.weight.float()  # (V, D)
        peak = max(float((lm_weight @ direction).max().item()), 1e-6)
        if steering.normalize_mai_lm_target:
            base_alpha = steering.mai_lm_target / peak
        else:
            base_alpha = steering.mai_lm_target

        inject_layer = (
            steering.inject_layer if steering.inject_layer is not None else self.model_config.n_layers // 2
        )

        model_dtype = next(self.model.parameters()).dtype
        return direction.to(model_dtype), float(base_alpha), inject_layer

    def _position_injection(
        self, input_ids: torch.Tensor, direction: torch.Tensor | None
    ) -> torch.Tensor | None:
        """
        Build a (B, T, D) injection that is the direction at masked positions and
        zero elsewhere (mask-aligned injection). Rebuilt each step as the set of
        masked positions shrinks.
        """
        if direction is None:
            return None
        masked = input_ids == self.mask_token_id  # (B, T)
        inj = torch.zeros(*input_ids.shape, direction.shape[0], dtype=direction.dtype, device=self.device)
        inj[masked] = direction
        return inj

    def _build_relu_mask(self, relu_logit_mask: dict[int, float]) -> torch.Tensor:
        """
        Build a (V,) vocab suppression vector to subtract from logits.

        For each concept, alignment with the vocab is lm_head.weight @ embedding.
        Only positive alignment is suppressed (relu), so the mask removes tokens
        the concept promotes without boosting the rest.
        """
        head = self.model.known_head
        lm_weight = self.model.transformer.lm_head.weight.float()  # (V, D)

        ids = torch.tensor(list(relu_logit_mask.keys()), device=self.device)
        strengths = torch.tensor(list(relu_logit_mask.values()), device=self.device, dtype=torch.float32)

        emb = head._get_embedding(ids).float()  # (K, D)
        alignment = emb @ lm_weight.t()  # (K, V)
        return (strengths.unsqueeze(-1) * alignment.clamp_min(0.0)).sum(dim=0)  # (V,)

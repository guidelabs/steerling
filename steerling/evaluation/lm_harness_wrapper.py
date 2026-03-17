"""
LM Evaluation Harness wrapper for Steerling (causal diffusion) models.

Registers a ``steerling`` model with lm-eval-harness and implements
Monte-Carlo log-likelihood estimation and masked generation.
"""

from __future__ import annotations

import json
import logging
from contextlib import nullcontext, suppress
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from lm_eval import evaluator
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm

from steerling.configs.evaluation import get_task_settings
from steerling.configs.generation import GenerationConfig
from steerling.inference.causal_diffusion import SteerlingGenerator

logger = logging.getLogger(__name__)


@register_model("steerling")
class SteerlingLM(LM):
    """
    LM Evaluation Harness wrapper for Steerling models.

    Args:
        model_path: Path or HuggingFace repo ID
        batch_size: Evaluation batch size
        max_length: Maximum sequence length
        max_gen_toks: Maximum generation tokens
        device: Device string
        mc_num: Monte Carlo samples for likelihood estimation
        mc_batch_size: Batch size per MC forward pass
        cfg: Classifier-free guidance scale
        gen_length: Max tokens for generation tasks
        steps: Diffusion steps for generation tasks
    """

    def __init__(
        self,
        model_path: str = "guidelabs/steerling-8b",
        batch_size: int = 16,
        max_length: int = 2048,
        max_gen_toks: int = 1024,
        device: str = "cuda",
        mc_num: int = 128,
        mc_batch_size: int = 32,
        cfg: float = 0.0,
        gen_length: int | None = None,
        steps: int | None = None,
    ):
        super().__init__()

        self._batch_size = batch_size
        self._max_length = max_length
        self._max_gen_toks = gen_length or max_gen_toks
        self._device = device
        self._torch_device = torch.device(device)

        self.mc_num = mc_num
        self.mc_batch_size = mc_batch_size
        self.cfg = cfg
        self.steps = steps

        # Enable TF32
        if self._torch_device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            with suppress(Exception):
                torch.set_float32_matmul_precision("high")

        # Load model via SteerlingGenerator (handles AutoModel + fallback)
        gen = SteerlingGenerator.from_pretrained(
            model_path, device=device, dtype=torch.bfloat16
        )

        self.generator = gen
        self.model = gen.model
        self.tokenizer = gen.tokenizer
        self.is_interpretable = gen.is_interpretable
        self.diff_block_size = gen.diff_block_size
        self.mask_token_id = gen.mask_token_id
        self.pad_token_id = gen.pad_token_id
        self.eos_token_id = gen.eos_token_id
        self.vocab_size = gen.tokenizer.vocab_size

        logger.info(
            f"SteerlingLM initialized: interpretable={self.is_interpretable}, "
            f"device={device}, mc_num={mc_num}, cfg={cfg}"
        )

    # -- LM interface properties ---------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def device(self) -> str:
        return self._device

    @property
    def eot_token_id(self) -> int:
        return self.eos_token_id

    # -- Helpers -------------------------------------------------------------------

    def _get_amp_context(self):
        if self._torch_device.type == "cuda":
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _model_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits [B, T, V]."""
        with self._get_amp_context():
            if self.is_interpretable:
                output = self.model(input_ids, minimal_output=True)
            else:
                output = self.model(input_ids)

        if isinstance(output, tuple):
            return output[0]
        return output

    def _model_forward_with_cfg(
        self,
        input_ids: torch.Tensor,
        prompt_len: int,
        cfg_scale: float,
    ) -> torch.Tensor:
        """Forward with classifier-free guidance."""
        if cfg_scale <= 0:
            return self._model_forward(input_ids)

        uncond_ids = input_ids.clone()
        if prompt_len > 0:
            uncond_ids[:, :prompt_len] = self.mask_token_id

        combined = torch.cat([input_ids, uncond_ids], dim=0)
        logits = self._model_forward(combined)

        cond_logits, uncond_logits = logits.chunk(2, dim=0)
        return uncond_logits + (cfg_scale + 1) * (cond_logits - uncond_logits)

    def tok_encode(self, string: str) -> list[int]:
        # Normalize curly quotes and dashes
        string = string.replace("\u2019", "'").replace("\u2018", "'")
        string = string.replace("\u201c", '"').replace("\u201d", '"')
        string = string.replace("\u2013", "-").replace("\u2014", "-")
        string = string.replace("\u2026", "...")
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: list[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _encode_pair(self, context: str, continuation: str) -> tuple[list[int], list[int]]:
        """Encode context/continuation pair with boundary handling."""
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tok_encode(context + continuation)
        context_enc = self.tok_encode(context)
        continuation_enc = whole_enc[len(context_enc) :]
        return context_enc, continuation_enc

    # -- Masking -------------------------------------------------------------------

    def _forward_process(
        self,
        batch: torch.Tensor,
        prompt_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply random masking to continuation tokens for MC estimation."""
        bsz, seq_len = batch.shape
        cont_len = seq_len - prompt_len

        if cont_len <= 0:
            p_mask = torch.zeros(bsz, seq_len, device=batch.device)
            return batch, p_mask

        k = torch.randint(1, cont_len + 1, (), device=batch.device)

        x = torch.round(
            torch.linspace(float(k), k + (bsz - 1) * (cont_len / bsz), steps=bsz, device=batch.device)
        ).long()
        x = ((x - 1) % cont_len) + 1

        indices = torch.arange(cont_len, device=batch.device).repeat(bsz, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(bsz):
            perm = torch.randperm(cont_len, device=batch.device)
            is_mask[i] = is_mask[i][perm]

        noisy = batch.clone()
        noisy[:, prompt_len:] = torch.where(
            is_mask, torch.as_tensor(self.mask_token_id, device=batch.device), batch[:, prompt_len:]
        )

        p_per_row = (x.float() / cont_len).unsqueeze(1)
        p_mask = p_per_row.repeat(1, seq_len)

        return noisy, p_mask.clamp_min_(1e-6)

    # -- Memory management ---------------------------------------------------------

    def _clear_mask_caches(self) -> None:
        """Clear attention mask caches to prevent VRAM growth across variable-length inputs."""
        for module in self.model.modules():
            if hasattr(module, "_mask_cache"):
                cast(dict, module._mask_cache).clear()
            if hasattr(module, "_sdpa_mask_cache"):
                cast(dict, module._sdpa_mask_cache).clear()

    # -- Log-likelihood ------------------------------------------------------------

    @torch.no_grad()
    def _compute_mc_loglikelihood(
        self,
        context_tokens: list[int],
        continuation_tokens: list[int],
    ) -> tuple[float, bool]:
        """Monte Carlo log-likelihood estimation for masked models."""
        if self.mc_num <= 0:
            return 0.0, False

        seq = torch.tensor(
            context_tokens + continuation_tokens, dtype=torch.long, device=self._torch_device
        )
        prompt_len = len(context_tokens)
        cont_len = len(continuation_tokens)

        if cont_len == 0:
            return 0.0, True

        chunk = max(1, min(self.mc_batch_size, self.mc_num))
        total_loss = torch.zeros((), device=self._torch_device)
        n_chunks = 0
        processed = 0

        while processed < self.mc_num:
            bsz = min(chunk, self.mc_num - processed)
            batch = seq.unsqueeze(0).expand(bsz, -1).clone()

            noisy_batch, p_mask = self._forward_process(batch, prompt_len)
            logits = self._model_forward_with_cfg(noisy_batch, prompt_len, self.cfg)

            mask_indices = noisy_batch == self.mask_token_id
            if not mask_indices.any():
                processed += bsz
                continue

            labels = batch
            sel_logits = logits[mask_indices]
            sel_labels = labels[mask_indices]
            sel_p = p_mask[mask_indices].clamp_min_(1e-6)

            ce = F.cross_entropy(sel_logits, sel_labels, reduction="none")
            loss = (ce / sel_p).sum() / bsz

            total_loss += loss
            n_chunks += 1
            processed += bsz

        if n_chunks == 0:
            return 0.0, False

        log_likelihood = -(total_loss / n_chunks).item()
        return log_likelihood, False

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        """Compute log-likelihood for (context, continuation) pairs."""
        results = []

        with torch.inference_mode():
            for i, request in enumerate(tqdm(requests, desc="loglikelihood")):
                if i % 100 == 0:
                    self._clear_mask_caches()
                    torch.cuda.empty_cache()

                context, continuation = request.args
                context_tokens, continuation_tokens = self._encode_pair(
                    context if context else "", continuation
                )

                total_len = len(context_tokens) + len(continuation_tokens)
                if total_len > self.max_length:
                    results.append((float("-inf"), False))
                    continue

                if len(continuation_tokens) == 0:
                    results.append((0.0, False))
                    continue

                log_prob, is_greedy = self._compute_mc_loglikelihood(
                    context_tokens, continuation_tokens
                )
                results.append((log_prob, is_greedy))

        torch.cuda.empty_cache()
        return results

    def loglikelihood_rolling(self, requests: list[Instance]) -> list[float]:
        """Rolling log-likelihood (for perplexity)."""
        results: list[float] = []

        for request in requests:
            (text,) = request.args
            tokens = self.tok_encode(text)
            if len(tokens) > self.max_length:
                tokens = tokens[-self.max_length :]

            log_prob, _ = self._compute_mc_loglikelihood([], tokens)
            results.append(log_prob)

        return results

    # -- Generation ----------------------------------------------------------------

    def generate_until(self, requests: list[Instance]) -> list[str]:
        """Generate text using confidence-based masked unmasking."""
        results = []

        for request in tqdm(requests, desc="generate_until"):
            context, gen_kwargs = request.args
            until = gen_kwargs.get("until", [])
            max_gen_toks = gen_kwargs.get("max_gen_toks", self.max_gen_toks)
            temperature = gen_kwargs.get("temperature", 0.6)

            context_tokens = self.tok_encode(context) if context else []
            max_ctx = self.max_length - max_gen_toks
            if len(context_tokens) > max_ctx:
                context_tokens = context_tokens[-max_ctx:]

            # Encode stop strings to token IDs for block-level early stopping
            stop_token_ids = []
            for s in until:
                toks = self.tok_encode(s)
                if len(toks) == 1:
                    stop_token_ids.append(toks[0])
            stop_token_ids.append(self.eos_token_id)

            # Round gen_length up to nearest block_length multiple
            block_length = self.diff_block_size
            gen_length = ((max_gen_toks + block_length - 1) // block_length) * block_length
            steps = self.steps or gen_length

            prompt_tensor = torch.tensor(
                [context_tokens], dtype=torch.long, device=self._torch_device
            )

            gen_config = GenerationConfig(
                max_new_tokens=gen_length,
                steps=steps,
                temperature=temperature,
                cfg_scale=self.cfg,
                stop_tokens=stop_token_ids or None,
            )

            gen_output = self.generator.generate_full(prompt_tensor, gen_config)
            generated_text = gen_output.text

            # Truncate at stop strings
            for stop_str in until:
                if stop_str in generated_text:
                    generated_text = generated_text.split(stop_str)[0]
                    break

            results.append(generated_text.rstrip())

        return results


def run_evaluation(
    model_path: str,
    tasks: list[str],
    results_dir: str | Path | None = None,
    device: str = "cuda",
    task_overrides: dict[str, dict] | None = None,
) -> dict[str, Any]:
    """
    Run lm-eval-harness on a Steerling checkpoint.

    Args:
        model_path: HuggingFace repo ID or local path
        tasks: List of task names (e.g. ["hellaswag", "mmlu"])
        results_dir: Directory to save results (None = don't save)
        device: Device to run on
        task_overrides: Optional per-task setting overrides

    Returns:
        Results dictionary from lm-eval-harness
    """
    task_overrides = task_overrides or {}

    all_results = {}
    for task_name in tasks:
        settings = get_task_settings(task_name, task_overrides.get(task_name))

        logger.info(
            f"Task '{task_name}': num_fewshot={settings.num_fewshot}, "
            f"batch_size={settings.batch_size}, mc_num={settings.mc_num}, "
            f"cfg={settings.cfg}"
        )

        model = SteerlingLM(
            model_path=model_path,
            batch_size=settings.batch_size,
            mc_num=settings.mc_num,
            mc_batch_size=settings.mc_batch_size,
            cfg=settings.cfg,
            gen_length=settings.gen_length,
            steps=settings.steps,
            device=device,
        )

        results = evaluator.simple_evaluate(
            model=model,
            tasks=[task_name],
            num_fewshot=settings.num_fewshot,
            batch_size=settings.batch_size,
            device=device,
        )

        all_results[task_name] = results

        # Free GPU memory between tasks
        del model
        torch.cuda.empty_cache()

    if results_dir:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        for task_name, results in all_results.items():
            task_file = results_dir / f"{task_name}.json"
            with open(task_file, "w") as f:
                json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_dir}")

    return all_results

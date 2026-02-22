"""Tests for SteerlingGenerator with tiny model (CPU, no checkpoint)."""

import torch

from steerling.configs.generation import GenerationConfig
from steerling.inference.causal_diffusion import SteerlingGenerator
from steerling.models.causal_diffusion import CausalDiffusionLM


class TestSteerlingGenerator:
    def _make_generator(self, tiny_config, tokenizer, device):
        """Create a generator with random weights for testing."""
        model = CausalDiffusionLM(tiny_config, vocab_size=tokenizer.vocab_size)
        return SteerlingGenerator(
            model=model,
            tokenizer=tokenizer,
            model_config=tiny_config,
            is_interpretable=False,
            device=device,
        )

    def test_generate_produces_text(self, tiny_config, tokenizer, device):
        gen = self._make_generator(tiny_config, tokenizer, device)
        config = GenerationConfig(max_new_tokens=10, seed=42)
        output = gen.generate_full("Hello", config)

        assert output.generated_tokens > 0
        assert len(output.text) > 0
        assert output.prompt_tokens > 0

    def test_generate_deterministic(self, tiny_config, tokenizer, device):
        gen = self._make_generator(tiny_config, tokenizer, device)
        config = GenerationConfig(max_new_tokens=10, seed=42)

        text1 = gen.generate("Hello", config)
        text2 = gen.generate("Hello", config)
        assert text1 == text2

    def test_generate_respects_max_tokens(self, tiny_config, tokenizer, device):
        gen = self._make_generator(tiny_config, tokenizer, device)
        config = GenerationConfig(max_new_tokens=5, seed=42)
        output = gen.generate_full("Hello", config)
        assert output.generated_tokens <= 5

    def test_repr(self, tiny_config, tokenizer, device):
        gen = self._make_generator(tiny_config, tokenizer, device)
        r = repr(gen)
        assert "SteerlingGenerator" in r
        assert "params=" in r

    def test_embeddings(self, tiny_config, tokenizer, device):
        gen = self._make_generator(tiny_config, tokenizer, device)
        emb = gen.get_embeddings("Hello", pooling="mean", embedding_type="hidden")
        assert emb.shape == (tiny_config.n_embd,)
        assert not torch.isnan(emb).any()

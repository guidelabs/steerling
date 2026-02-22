"""Tests for model instantiation and forward pass."""

import torch

from steerling.models.causal_diffusion import CausalDiffusionLM


class TestCausalDiffusionLM:
    def test_instantiate(self, tiny_config, tokenizer):
        model = CausalDiffusionLM(tiny_config, vocab_size=tokenizer.vocab_size)
        assert model is not None

    def test_param_count(self, tiny_config, tokenizer):
        model = CausalDiffusionLM(tiny_config, vocab_size=tokenizer.vocab_size)
        n_params = model.get_num_params(non_embedding=False)
        assert n_params > 0

    def test_forward_shape(self, tiny_config, tokenizer, device):
        model = CausalDiffusionLM(tiny_config, vocab_size=tokenizer.vocab_size).to(device)
        model.eval()

        B, T = 2, 32
        input_ids = torch.randint(0, tokenizer.vocab_size, (B, T), device=device)

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (B, T, tokenizer.vocab_size)

    def test_forward_return_hidden(self, tiny_config, tokenizer, device):
        model = CausalDiffusionLM(tiny_config, vocab_size=tokenizer.vocab_size).to(device)
        model.eval()

        B, T = 1, 16
        input_ids = torch.randint(0, tokenizer.vocab_size, (B, T), device=device)

        with torch.no_grad():
            hidden = model(input_ids, return_hidden=True)

        assert hidden.shape == (B, T, tiny_config.n_embd)

    def test_weight_sharing(self, tiny_config, tokenizer):
        assert tiny_config.weight_sharing is True
        model = CausalDiffusionLM(tiny_config, vocab_size=tokenizer.vocab_size)
        assert model.tok_emb.weight.data_ptr() == model.lm_head.weight.data_ptr()

    def test_forward_with_mask_tokens(self, tiny_config, tokenizer, device):
        """Model should handle mask tokens (used during diffusion generation)."""
        model = CausalDiffusionLM(tiny_config, vocab_size=tokenizer.vocab_size).to(device)
        model.eval()

        B, T = 1, 32
        input_ids = torch.randint(0, tokenizer.vocab_size, (B, T), device=device)
        # Replace some positions with mask tokens
        input_ids[0, 16:] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(input_ids)

        assert logits.shape == (B, T, tokenizer.vocab_size)
        assert not torch.isnan(logits).any()

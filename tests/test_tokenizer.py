"""Tests for SteerlingTokenizer."""


class TestSteerlingTokenizer:
    def test_special_token_ids(self, tokenizer):
        assert tokenizer.vocab_size == 100281
        assert tokenizer.pad_token_id == 100277
        assert tokenizer.bos_token_id == 100278
        assert tokenizer.endofchunk_token_id == 100279
        assert tokenizer.mask_token_id == 100280
        assert tokenizer.eos_token_id == 100257

    def test_encode_decode_roundtrip(self, tokenizer):
        text = "Hello, world!"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)
        assert decoded == text

    def test_encode_with_special_tokens(self, tokenizer):
        text = "Hello"
        tokens = tokenizer.encode(text, add_special_tokens=True)
        assert tokens[0] == tokenizer.bos_token_id
        assert tokens[-1] == tokenizer.eos_token_id

    def test_encode_without_special_tokens(self, tokenizer):
        text = "Hello"
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert tokens[0] != tokenizer.bos_token_id

    def test_decode_filters_special_tokens(self, tokenizer):
        tokens = [tokenizer.bos_token_id, 9906, tokenizer.eos_token_id]  # 9906 = "Hello"
        decoded = tokenizer.decode(tokens)
        assert "Hello" in decoded

    def test_decode_filters_mask_tokens(self, tokenizer):
        tokens = [9906, tokenizer.mask_token_id, tokenizer.pad_token_id]
        decoded = tokenizer.decode(tokens)
        assert decoded == "Hello"

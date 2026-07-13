"""
Steerling tokenizer: tiktoken cl100k_base with custom special tokens.

Token layout (base):
    0 - 100256:  cl100k_base base vocabulary
    100257:      <|endoftext|> (EOS, built into tiktoken)
    100277:      <|pad|>
    100278:      <|bos|>
    100279:      <|endofchunk|>
    100280:      <|mask|>
    vocab_size:  100281

Token layout (instruct, extends base):
    100281:      <|start_header_id|>
    100282:      <|end_header_id|>
    100283:      <|eot_id|>
    vocab_size:  100284
"""

from __future__ import annotations

import numpy as np
import tiktoken
import torch


class SteerlingTokenizer:
    """
    Tokenizer for Steerling models.

    Uses tiktoken cl100k_base with custom special tokens.
    Pass ``instruct=True`` to include the 3 additional chat tokens
    used by the instruct model.
    """

    ENCODING_NAME = "cl100k_base"

    def __init__(self, instruct: bool = False):
        base_enc = tiktoken.get_encoding(self.ENCODING_NAME)
        base_vocab = base_enc.n_vocab  # 100277

        self._pad_token_id = base_vocab  # 100277
        self._bos_token_id = base_vocab + 1  # 100278
        self._endofchunk_token_id = base_vocab + 2  # 100279
        self._mask_token_id = base_vocab + 3  # 100280
        self._eos_token_id = base_enc._special_tokens["<|endoftext|>"]  # 100257
        self._instruct = instruct

        special_tokens = {
            **base_enc._special_tokens,
            "<|pad|>": self._pad_token_id,
            "<|bos|>": self._bos_token_id,
            "<|endofchunk|>": self._endofchunk_token_id,
            "<|mask|>": self._mask_token_id,
        }

        if instruct:
            self._start_header_id = base_vocab + 4  # 100281
            self._end_header_id = base_vocab + 5  # 100282
            self._eot_id = base_vocab + 6  # 100283
            self._vocab_size = base_vocab + 7  # 100284
            special_tokens.update(
                {
                    "<|start_header_id|>": self._start_header_id,
                    "<|end_header_id|>": self._end_header_id,
                    "<|eot_id|>": self._eot_id,
                }
            )
        else:
            self._start_header_id = None
            self._end_header_id = None
            self._eot_id = None
            self._vocab_size = base_vocab + 4  # 100281

        # Create encoding with custom special tokens
        self._tokenizer = tiktoken.Encoding(
            name=f"{self.ENCODING_NAME}_steerling",
            pat_str=base_enc._pat_str,
            mergeable_ranks=base_enc._mergeable_ranks,
            special_tokens=special_tokens,
        )

        self._special_token_ids = {
            self._pad_token_id,
            self._bos_token_id,
            self._eos_token_id,
            self._endofchunk_token_id,
            self._mask_token_id,
        }
        if instruct:
            self._special_token_ids.update(
                {
                    self._start_header_id,
                    self._end_header_id,
                    self._eot_id,
                }
            )

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Input text
            add_special_tokens: If True, prepend BOS and append EOS

        Returns:
            List of token IDs
        """

        tokens = self._tokenizer.encode(text, disallowed_special=())
        if add_special_tokens:
            tokens = [self._bos_token_id] + tokens + [self._eos_token_id]
        return tokens

    def decode(self, tokens: list[int] | np.ndarray | torch.Tensor, skip_special_tokens: bool = True) -> str:
        """
        Decode token IDs to text.

        Args:
            tokens: Token IDs (list, numpy array, or torch tensor)
            skip_special_tokens: If True, filter out special tokens before decoding

        Returns:
            Decoded text
        """

        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().numpy()
        if isinstance(tokens, np.ndarray):
            tokens = tokens.tolist()

        if skip_special_tokens:
            tokens = [int(t) for t in tokens if int(t) not in self._special_token_ids]
        else:
            tokens = [int(t) for t in tokens]
        return self._tokenizer.decode(tokens)

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return self._pad_token_id

    @property
    def bos_token_id(self) -> int:
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int:
        return self._eos_token_id

    @property
    def endofchunk_token_id(self) -> int:
        return self._endofchunk_token_id

    @property
    def mask_token_id(self) -> int:
        return self._mask_token_id

    @property
    def instruct(self) -> bool:
        return self._instruct

    @property
    def start_header_id(self) -> int | None:
        return self._start_header_id

    @property
    def end_header_id(self) -> int | None:
        return self._end_header_id

    @property
    def eot_id(self) -> int | None:
        return self._eot_id

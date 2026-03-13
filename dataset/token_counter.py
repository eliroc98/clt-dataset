"""
token_counter.py — Token-based text length measurement.

Supports tiktoken encodings, OpenAI model names, and HuggingFace tokenizers.
Falls back to whitespace split if no tokenizer library is available.
"""

from __future__ import annotations

import logging
from typing import Callable

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Compute text length in tokens using an LLM tokenizer.

    Supported tokenizer specs:
      - tiktoken encoding names: "cl100k_base", "o200k_base", etc.
      - OpenAI model names: "gpt-4o", "gpt-4", "gpt-3.5-turbo"
      - HuggingFace model IDs: "meta-llama/Llama-3.1-8B", etc.

    Falls back to whitespace split if no library is available.
    """

    def __init__(self, tokenizer: str = "cl100k_base"):
        self._name = tokenizer
        self._encode: Callable[[str], list[int]] | None = None
        self._load(tokenizer)

    def _load(self, tokenizer: str) -> None:
        try:
            import tiktoken
            try:
                enc = tiktoken.get_encoding(tokenizer)
                self._encode = enc.encode
                logger.info(f"  TokenCounter: tiktoken encoding '{tokenizer}'")
                return
            except ValueError:
                pass
            try:
                enc = tiktoken.encoding_for_model(tokenizer)
                self._encode = enc.encode
                logger.info(f"  TokenCounter: tiktoken encoding for model '{tokenizer}'")
                return
            except KeyError:
                pass
        except ImportError:
            pass

        try:
            from transformers import AutoTokenizer
            from dataset.auth import resolve_hf_token

            tok = AutoTokenizer.from_pretrained(
                tokenizer, trust_remote_code=True, token=resolve_hf_token() or None,
            )
            self._encode = tok.encode
            logger.info(f"  TokenCounter: HuggingFace tokenizer '{tokenizer}'")
            return
        except Exception:
            pass

        logger.warning(
            f"  TokenCounter: could not load '{tokenizer}'. Falling back to word count."
        )
        self._encode = None

    def count(self, text: str) -> int:
        if not text:
            return 0
        if self._encode is not None:
            return len(self._encode(text))
        return len(text.split())

    @property
    def name(self) -> str:
        return self._name


# ── Module-level singleton ─────────────────────────────────────────────────

_token_counter: TokenCounter | None = None


def get_token_counter() -> TokenCounter:
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter("cl100k_base")
    return _token_counter


def set_token_counter(tokenizer: str) -> TokenCounter:
    global _token_counter
    _token_counter = TokenCounter(tokenizer)
    return _token_counter


def token_length(text: str) -> int:
    return get_token_counter().count(text)

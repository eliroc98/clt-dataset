"""
local_llm.py — High-performance LLM inference via vLLM.

Uses vLLM's offline engine (PagedAttention + continuous batching) which
gives 5–20× throughput over standard HuggingFace generation on GPU.
Structured JSON output uses vLLM's native guided_json parameter instead
of the outlines library.

For gated models set HF_TOKEN in .env.huggingface — loaded via dataset.auth.
"""

from __future__ import annotations

import atexit
import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

# ── Module-level engine cache ─────────────────────────────────────────────
# Maps model_name → vllm.LLM instance.
_LLM_CACHE: dict[str, Any] = {}


def shutdown_engines() -> None:
    """Gracefully shut down all cached vLLM engines."""
    for name, llm in list(_LLM_CACHE.items()):
        try:
            engine_core = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
            if engine_core is not None and hasattr(engine_core, "shutdown"):
                engine_core.shutdown()
            del llm
            logger.info(f"Model '{name}' engine shut down.")
        except Exception as exc:
            logger.debug(f"Engine shutdown for '{name}': {exc}")
    _LLM_CACHE.clear()


atexit.register(shutdown_engines)


def _resolve_token() -> str | None:
    from dataset.auth import resolve_hf_token
    return resolve_hf_token() or None


def _patch_disabled_tqdm() -> None:
    """Fix vLLM 0.11 + huggingface_hub ≥1.6 ``DisabledTqdm`` crash.

    vLLM's ``DisabledTqdm.__init__`` unconditionally appends ``disable=True``
    to ``**kwargs``, but huggingface_hub already passes ``disable`` — causing
    a ``TypeError: multiple values for keyword argument 'disable'``.
    We patch the class to pop any existing ``disable`` before adding its own.
    """
    try:
        from vllm.model_executor.model_loader.weight_utils import DisabledTqdm
    except ImportError:
        return

    _orig_init = DisabledTqdm.__init__

    def _safe_init(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[no-untyped-def]
        kwargs.pop("disable", None)
        _orig_init(self, *args, **kwargs)

    if not getattr(DisabledTqdm, "_patched_disable", False):
        DisabledTqdm.__init__ = _safe_init  # type: ignore[method-assign]
        DisabledTqdm._patched_disable = True  # type: ignore[attr-defined]


def _patch_tokenizer_compat() -> None:
    """Add ``all_special_tokens_extended`` back for transformers ≥ 5.x.

    transformers 5.x removed this property but vLLM 0.11 still reads it
    during ``get_cached_tokenizer``.  We patch the base class once so that
    any tokenizer instance will fall back to ``all_special_tokens``.
    """
    from transformers import PreTrainedTokenizerBase  # type: ignore[attr-defined]

    if hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        return

    @property  # type: ignore[misc]
    def _all_special_tokens_extended(self):  # type: ignore[no-untyped-def]
        return self.all_special_tokens

    PreTrainedTokenizerBase.all_special_tokens_extended = (  # type: ignore[attr-defined]
        _all_special_tokens_extended
    )


def _schema_to_dict(json_schema: Any) -> dict:
    """Convert a Pydantic model class or dict to a JSON-schema dict."""
    if hasattr(json_schema, "model_json_schema"):
        return json_schema.model_json_schema()
    if isinstance(json_schema, dict):
        return json_schema
    raise TypeError(f"Unsupported json_schema type: {type(json_schema)}")


def load_model(
    model_name: str,
    *,
    device: str | None = None,
    gpu_memory_utilization: float = 0.7,
) -> Any:
    """
    Load and cache a vLLM LLM engine for *model_name*.

    Parameters
    ----------
    model_name
        HuggingFace model ID or local path.
    device
        CUDA device identifier, e.g. ``"cuda:1"``. When given, the engine
        is pinned to that single GPU via ``CUDA_VISIBLE_DEVICES``.

    Returns
    -------
    vllm.LLM
    """
    if model_name in _LLM_CACHE:
        return _LLM_CACHE[model_name]

    from vllm import LLM

    _patch_tokenizer_compat()
    _patch_disabled_tqdm()

    # Pin to a specific GPU when requested (e.g. "cuda:1" → device 1).
    # NOTE: CUDA_VISIBLE_DEVICES should ideally be set before any CUDA init
    # (see pipeline.py). This is a fallback for standalone usage.
    if device and device.startswith("cuda:"):
        gpu_id = device.split(":", 1)[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    token = _resolve_token()
    if token:
        os.environ.setdefault("HF_TOKEN", token)

    logger.info(f"Loading model '{model_name}' via vLLM (device={device})…")

    kwargs: dict[str, Any] = dict(
        model=model_name,
        tensor_parallel_size=1,
        dtype="float16",
        tokenizer_mode="auto",
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=gpu_memory_utilization
    )

    llm = LLM(**kwargs)
    _LLM_CACHE[model_name] = llm
    logger.info(f"Model '{model_name}' ready.")
    return llm


def generate_text(
    model_name: str,
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
    device: str | None = None,
    enable_thinking: bool | None = None,
    json_schema: Any | None = None,
) -> str:
    """
    Generate text from chat-style messages using vLLM.

    Parameters
    ----------
    model_name
        HuggingFace model ID or local path.
    messages
        Chat messages, e.g. [{"role": "system", "content": "…"}, …].
    temperature
        Sampling temperature. 0 = greedy decoding.
    max_new_tokens
        Maximum new tokens to generate.
    device
        CUDA device identifier, e.g. ``"cuda:1"``. Pins the engine to
        that GPU via ``CUDA_VISIBLE_DEVICES``.
    enable_thinking
        For models supporting chain-of-thought (e.g. Qwen3): False suppresses
        reasoning tokens. None uses the model default.
    json_schema
        When provided, constrains output to this JSON schema. Accepts a
        Pydantic BaseModel subclass or a JSON-schema dict.

    Returns
    -------
    str
        The generated assistant response text.
    """
    results = generate_text_batch(
        model_name, [messages],
        temperature=temperature,
        max_new_tokens=max_new_tokens,
        device=device,
        enable_thinking=enable_thinking,
        json_schema=json_schema,
    )
    return results[0]


def generate_text_batch(
    model_name: str,
    batch_messages: list[list[dict[str, str]]],
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
    device: str | None = None,
    enable_thinking: bool | None = None,
    json_schema: Any | None = None,
    gpu_memory_utilization: float = 0.7,
) -> list[str]:
    """
    Generate text for a batch of chat-message lists in one forward pass.

    vLLM processes all requests concurrently via continuous batching,
    which is significantly faster than sequential HuggingFace generation.

    Parameters
    ----------
    model_name
        HuggingFace model ID or local path.
    batch_messages
        List of chat message lists, one per example.
    temperature, max_new_tokens, device, enable_thinking, json_schema
        Same as generate_text().

    Returns
    -------
    list[str]
        Generated assistant response for each input, in the same order.
    """
    from vllm import SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

    llm = load_model(model_name, device=device, gpu_memory_utilization=gpu_memory_utilization)

    guided_json = _schema_to_dict(json_schema) if json_schema is not None else None

    params_kwargs: dict[str, Any] = dict(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )
    if guided_json is not None:
        params_kwargs["guided_decoding"] = GuidedDecodingParams(json=guided_json)
    else:
        logger.warning("No json_schema provided; output will not be constrained to JSON format.")
    params = SamplingParams(**params_kwargs)

    # Build chat_template_kwargs for thinking-mode control
    chat_kwargs: dict[str, Any] = {}
    if enable_thinking is not None:
        chat_kwargs["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    outputs = llm.chat(batch_messages, sampling_params=params, **chat_kwargs)
    return [o.outputs[0].text for o in outputs]

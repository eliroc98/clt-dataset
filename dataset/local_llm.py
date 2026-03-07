"""
local_llm.py — Local HuggingFace model inference with caching.

Loads a causal-LM from the HuggingFace Hub (or a local path) using
``transformers``, caches it in memory, and provides a simple
:func:`generate_text` interface that accepts chat-style messages.

For gated models (e.g. Llama, Gemma) set ``HF_TOKEN`` in
``.env.huggingface`` — the token is picked up via :mod:`dataset.auth`.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ── Module-level model cache ────────────────────────────────────────────────
# Maps model_name → (model, tokenizer) so we only load once per process.
_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}

# Maps model_name → outlines.Transformers wrapper (created lazily).
_OUTLINES_CACHE: dict[str, Any] = {}


def _apply_chat_template(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    enable_thinking: bool | None = None,
) -> str:
    """Apply the tokenizer's chat template, optionally suppressing thinking.

    For models that support ``enable_thinking`` (e.g. Qwen3) this passes
    the flag directly.  For others the parameter is silently ignored.
    """
    kwargs: dict[str, Any] = dict(tokenize=False, add_generation_prompt=True)
    if enable_thinking is not None:
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        # Tokenizer doesn't accept enable_thinking — fall back.
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def _strip_thinking(text: str) -> str:
    """Remove ``<think>…</think>`` reasoning blocks from model output."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _get_outlines_model(model_name: str, *, device: str | None = None) -> Any:
    """Return a cached ``outlines.Transformers`` wrapper for *model_name*."""
    if model_name in _OUTLINES_CACHE:
        return _OUTLINES_CACHE[model_name]

    import outlines

    model, tokenizer = load_model(model_name, device=device)
    outlines_model = outlines.from_transformers(model, tokenizer)
    _OUTLINES_CACHE[model_name] = outlines_model
    return outlines_model


def _resolve_token() -> str | None:
    """Return the HF token if available, else ``None``."""
    from dataset.auth import resolve_hf_token

    return resolve_hf_token() or None


def load_model(
    model_name: str,
    *,
    device: str | None = None,
    torch_dtype: Any | None = None,
) -> tuple[Any, Any]:
    """
    Load and cache a HuggingFace causal-LM and its tokenizer.

    Parameters
    ----------
    model_name : str
        A HuggingFace model ID (e.g. ``"meta-llama/Llama-3.1-8B-Instruct"``)
        or a local path.
    device : str | None
        ``"cuda"``, ``"cuda:0"``, ``"cuda:1"``, ``"mps"``, ``"cpu"``,
        or ``None`` for auto-detection. Specific GPU indices (e.g.
        ``"cuda:1"``) pin the model to that device.
    torch_dtype : torch.dtype | None
        ``None`` → float16 on GPU / MPS, float32 on CPU.

    Returns
    -------
    (model, tokenizer)

    Notes
    -----
    If ``HF_TOKEN`` is set, it is forwarded as ``token=`` to every
    ``from_pretrained`` call so gated models authenticate on the first
    attempt (no 401 round-trips).
    """
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = _resolve_token()

    # ── Resolve device ──────────────────────────────────────────────
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if torch_dtype is None:
        torch_dtype = torch.float16 if not device.startswith("cpu") else torch.float32

    logger.info(f"Loading model '{model_name}' on {device} ({torch_dtype})…")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=True, token=token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # device_map="auto" distributes across all GPUs (requires accelerate).
    # For any specific device we skip device_map and use .to(device) below,
    # which avoids the accelerate dependency.
    if device == "auto":
        _device_map: Any = "auto"
    else:
        # For a specific CUDA device, MPS, or CPU: load without device_map
        # (avoids the `accelerate` requirement) and move below with .to(device).
        _device_map = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map=_device_map,
        trust_remote_code=True,
        token=token,
    )

    # Move to device for backends that don't use device_map (MPS, CPU)
    if _device_map is None:
        model = model.to(device)

    model.eval()
    _MODEL_CACHE[model_name] = (model, tokenizer)
    logger.info(f"Model '{model_name}' loaded successfully on {device}")
    return model, tokenizer


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
    Generate text from chat-style messages using a local HuggingFace model.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    messages : list[dict]
        Chat messages, e.g. ``[{"role": "system", "content": "…"}, …]``.
    temperature : float
        Sampling temperature. 0 = greedy decoding.
    max_new_tokens : int
        Maximum number of new tokens to generate.
    device : str | None
        Device override (auto-detected if ``None``).
    enable_thinking : bool | None
        Controls reasoning/thinking mode for models that support it
        (e.g. Qwen3). ``False`` suppresses chain-of-thought reasoning
        so the model outputs the answer directly. ``None`` uses the
        model default.
    json_schema : type[BaseModel] | dict | None
        When provided, constrains decoding so the output is guaranteed
        to conform to this JSON schema.  Accepts a Pydantic ``BaseModel``
        subclass, a JSON-schema dict, or ``None`` (no constraint).
        Uses the `outlines` library under the hood.

    Returns
    -------
    str
        The generated text (assistant response only, no special tokens).
    """
    # ── Structured-output path (outlines) ───────────────────────────
    if json_schema is not None:
        import outlines
        from outlines.inputs import Chat

        outlines_model = _get_outlines_model(model_name, device=device)
        schema_type = outlines.json_schema(json_schema)
        generator = outlines.Generator(outlines_model, schema_type)

        gen_kwargs: dict[str, Any] = dict(max_new_tokens=max_new_tokens)
        if temperature > 0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = temperature
        else:
            gen_kwargs["do_sample"] = False

        result = generator(Chat(messages), **gen_kwargs)
        return result if isinstance(result, str) else str(result)

    # ── Free-form path ──────────────────────────────────────────────
    import torch

    model, tokenizer = load_model(model_name, device=device)

    # ── Build input ─────────────────────────────────────────────────
    input_text = _apply_chat_template(
        tokenizer, messages, enable_thinking=enable_thinking,
    )
    
    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    # ── Generation kwargs ───────────────────────────────────────────
    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    # ── Generate ────────────────────────────────────────────────────
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return _strip_thinking(decoded)


def generate_text_batch(
    model_name: str,
    batch_messages: list[list[dict[str, str]]],
    *,
    temperature: float = 0.0,
    max_new_tokens: int = 2048,
    device: str | None = None,
    enable_thinking: bool | None = None,
    json_schema: Any | None = None,
) -> list[str]:
    """
    Generate text for a batch of chat-message lists in a single forward pass.

    Parameters
    ----------
    model_name : str
        HuggingFace model ID or local path.
    batch_messages : list[list[dict]]
        A list of chat message lists, one per example.
    temperature : float
        Sampling temperature. 0 = greedy decoding.
    max_new_tokens : int
        Maximum number of new tokens to generate per example.
    device : str | None
        Device override (auto-detected if ``None``).
    enable_thinking : bool | None
        Controls reasoning/thinking mode for models that support it
        (e.g. Qwen3). ``False`` suppresses chain-of-thought reasoning.
        ``None`` uses the model default.
    json_schema : type[BaseModel] | dict | None
        When provided, constrains decoding to this JSON schema.

    Returns
    -------
    list[str]
        The generated assistant response for each input, in the same order.
    """
    # ── Structured-output path (outlines) ───────────────────────────
    if json_schema is not None:
        # Outlines Generator does not natively batch over Chat inputs,
        # so we loop — the logits processor still guarantees schema
        # compliance on every call.
        return [
            generate_text(
                model_name,
                msgs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                device=device,
                enable_thinking=enable_thinking,
                json_schema=json_schema,
            )
            for msgs in batch_messages
        ]

    # ── Free-form batched path ──────────────────────────────────────
    import torch

    model, tokenizer = load_model(model_name, device=device)

    # Build padded batch input
    input_texts = [
        _apply_chat_template(
            tokenizer, msgs, enable_thinking=enable_thinking,
        )
        for msgs in batch_messages
    ]

    tokenizer.padding_side = "left"
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
    ).to(model.device)

    gen_kwargs: dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    if temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)

    input_length = inputs["input_ids"].shape[1]
    results = []
    for out in output_ids:
        new_tokens = out[input_length:]
        decoded = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        results.append(_strip_thinking(decoded))
    return results

#!/usr/bin/env python3
"""
Semi-synthetic dataset construction from minimally-changing prompts.

Implements the pipeline described in dataset_construction.md:

  1. **Extract**: decompose real prompts (streamed from official instruction-
     tuning datasets) into *task specification templates* (with slots) and
     *options* (slot fillers) — via a few-shot LLM pipeline that produces
     structured output, or via manual/augmented entries.
  2. **Store**: persist templates and options in tidy structured datasets,
     annotated with their taxonomy level.
  3. **Store option compatibility**: record which options are compatible with
     which templates/task types so recombined prompts remain sensical.
     Options may be compatible with *multiple* task types and templates.
  4. **Augment options**: programmatically expand the option pool (e.g. any
     integer for word-count constraints).
  5. **Generate**: synthetically produce prompts by selecting N templates and
     filling them with compatible options, with control over prompt length
     and number of templates combined (density).

All lengths are computed in **tokens** (not characters) using a configurable
tokenizer (tiktoken encodings or HuggingFace AutoTokenizer).

Leverages:
  - Taxonomy from ``collect_task_types.py``  (instruction types + patterns)
  - Dataset registry from ``collect_task_types.py`` (streaming loaders)

Usage
─────
  # Full pipeline: extract → augment → generate
  python dataset/construct_dataset.py --n 5000 --density 3 --seed 42

  # With a specific tokenizer
  python dataset/construct_dataset.py --n 5000 --tokenizer cl100k_base

  # Extract templates via local HF model
  python dataset/construct_dataset.py --extract-only --model meta-llama/Llama-3.1-8B-Instruct

  # Extract from specific datasets only
  python dataset/construct_dataset.py --extract-only --datasets Alpaca DollyV2

  # Generate from previously extracted templates
  python dataset/construct_dataset.py --generate-only --n 10000 --density 4
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import math
import os
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Protocol

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR / "taxonomy".mkdir(exist_ok=True)

TAXONOMY_PATH = OUTPUT_DIR / "taxonomy" / "taxonomy.json"
TEMPLATES_PATH = OUTPUT_DIR / "templates.json"
OPTIONS_PATH = OUTPUT_DIR / "options.json"
GENERATED_PATH = OUTPUT_DIR / "constructed_prompts.jsonl"

# Taxonomy levels
LEVELS = ("content_task", "format", "style", "content_constraint", "meta")
CONSTRAINT_LEVELS = ("format", "style", "content_constraint", "meta")

sys.path.insert(0, str(BASE_DIR.parent))
from dataset.collect_task_types import (
    DATASET_REGISTRY,
    load_hf_dataset,
    load_github_data,
    load_kcif_data,
    extract_instruction_text,
)
from dataset.auth import resolve_hf_token  # loads .env files as a side-effect
from dataset.local_llm import generate_text as _local_generate


# ═══════════════════════════════════════════════════════════════════════════
# 0.  TOKENIZER-BASED LENGTH
# ═══════════════════════════════════════════════════════════════════════════

class TokenCounter:
    """
    Compute text length in tokens using an LLM tokenizer.

    Supported tokenizer specifications:
      - tiktoken encoding names: ``"cl100k_base"``, ``"o200k_base"``, etc.
      - OpenAI model names:      ``"gpt-4o"``, ``"gpt-4"``, ``"gpt-3.5-turbo"``
        (resolved to the corresponding tiktoken encoding).
      - HuggingFace model IDs:   ``"meta-llama/Llama-3.1-8B"``, etc.
        (loaded via ``transformers.AutoTokenizer``).

    Falls back to a simple whitespace split (``len(text.split())``) if no
    tokenizer library is available — with a warning.
    """

    def __init__(self, tokenizer: str = "cl100k_base"):
        self._name = tokenizer
        self._encode: Callable[[str], list[int]] | None = None
        self._load(tokenizer)

    def _load(self, tokenizer: str) -> None:
        # ── Try tiktoken first ───────────────────────────────────────
        try:
            import tiktoken

            # Direct encoding name
            try:
                enc = tiktoken.get_encoding(tokenizer)
                self._encode = enc.encode
                logger.info(f"  TokenCounter: using tiktoken encoding '{tokenizer}'")
                return
            except ValueError:
                pass

            # Model name → encoding
            try:
                enc = tiktoken.encoding_for_model(tokenizer)
                self._encode = enc.encode
                logger.info(
                    f"  TokenCounter: using tiktoken encoding for model '{tokenizer}'"
                )
                return
            except KeyError:
                pass
        except ImportError:
            pass

        # ── Try transformers AutoTokenizer ───────────────────────────
        try:
            from transformers import AutoTokenizer
            from dataset.local_llm import _resolve_token

            tok = AutoTokenizer.from_pretrained(
                tokenizer, trust_remote_code=True, token=_resolve_token(),
            )
            self._encode = tok.encode
            logger.info(
                f"  TokenCounter: using HuggingFace tokenizer '{tokenizer}'"
            )
            return
        except Exception:
            pass

        # ── Fallback: whitespace split ───────────────────────────────
        logger.warning(
            f"  TokenCounter: could not load tokenizer '{tokenizer}'. "
            "Falling back to whitespace-based word count."
        )
        self._encode = None

    def count(self, text: str) -> int:
        """Return the number of tokens in *text*."""
        if not text:
            return 0
        if self._encode is not None:
            return len(self._encode(text))
        return len(text.split())

    @property
    def name(self) -> str:
        return self._name


# Module-level default (overridden by CLI / callers).
_token_counter: TokenCounter | None = None


def get_token_counter() -> TokenCounter:
    """Return the module-level ``TokenCounter``, creating a default if needed."""
    global _token_counter
    if _token_counter is None:
        _token_counter = TokenCounter("cl100k_base")
    return _token_counter


def set_token_counter(tokenizer: str) -> TokenCounter:
    """Set the module-level ``TokenCounter`` to use *tokenizer*."""
    global _token_counter
    _token_counter = TokenCounter(tokenizer)
    return _token_counter


def token_length(text: str) -> int:
    """Convenience: token count of *text* using the current counter."""
    return get_token_counter().count(text)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TaskTemplate:
    """
    A task specification template with slots.

    Example:
      text = "Write {option_content}"
      slots = ["option_content"]
      task_type = "creative_writing"
      level = "content_task"
      token_length = 2  (length of the fixed text in tokens, excluding slots)
    """
    id: str
    text: str                           # template text with {slot} placeholders
    slots: list[str]                    # names of the slots
    task_type: str                      # taxonomy entry name
    level: str                          # content_task | format | style | ...
    token_length: int = 0               # length of fixed text in tokens
    source: str = ""                    # which dataset / method produced this
    compatible_with: list[str] = field(default_factory=list)
    # Which other template task_types this can be combined with


@dataclass
class Option:
    """
    A concrete value that fills a slot in a TaskTemplate.

    Example:
      value = "a long email template that invites participants to a meeting"
      slot = "option_content"
      compatible_task_types = ["creative_writing", "email"]
      token_length = 12
    """
    id: str
    value: str
    slot: str                           # which slot this fills
    compatible_task_types: list[str]    # which task types this option fits
    compatible_templates: list[str]     # template IDs this can be used with
    token_length: int = 0
    source: str = ""                    # "extracted" | "augmented" | "manual"
    tags: list[str] = field(default_factory=list)  # e.g. ["topic", "numeric"]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  TEMPLATE / OPTION EXTRACTION VIA FEW-SHOT LLM PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

# ── Utility IDs ──────────────────────────────────────────────────────────

def _template_id(task_type: str, text: str) -> str:
    """Deterministic short ID for a template."""
    key = f"{task_type}|{text}"
    return hashlib.md5(key.encode()).hexdigest()[:10]


def _option_id(slot: str, value: str) -> str:
    key = f"{slot}|{value}"
    return hashlib.md5(key.encode()).hexdigest()[:10]


# ── LLM extraction prompts ──────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """\
You are an expert prompt analyst. Given a user-facing instruction prompt, \
decompose it into its constituent **task specification templates** and \
**options** (the concrete values that fill slots in the templates).

Return a JSON object with two keys:
  "templates": [ ... ],
  "options":   [ ... ]

Each template object has:
  "text":      the template string with {slot_name} placeholders replacing options
  "slots":     list of slot names used in the template
  "task_type": the taxonomy label that best describes this template (from the provided list)
  "level":     one of "content_task", "format", "style", "content_constraint", "meta"

Each option object has:
  "value":                the concrete value extracted from the prompt
  "slot":                 which slot name this fills
  "compatible_task_types": list of taxonomy task types this option could sensibly fill \
(include ALL plausible task types, not just the one in this prompt)

Rules:
- A single prompt may contain MULTIPLE templates (e.g. a task + several constraints).
- Templates should be GENERAL: replace specific values with {slot} placeholders.
- Options should be the SPECIFIC values removed from the prompt.
- An option may be compatible with MULTIPLE task types — list all plausible ones.
- Use the taxonomy labels provided; do not invent new ones.
- Only output valid JSON — no markdown fences, no commentary.
"""

_EXTRACTION_FEW_SHOT = [
    {
        "role": "user",
        "content": (
            'Taxonomy labels: ["creative_writing", "email", "number_words", '
            '"keyword_inclusion", "no_commas", "summarization", "question_answering", '
            '"number_sentences", "all_lowercase"]\n\n'
            "Prompt:\n"
            '"Write a long email template that invites a group of participants to a meeting, '
            'with at least 500 words. The email must include the keywords "correlated" '
            'and "experiencing" and should not use any commas."'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "templates": [
                    {
                        "text": "Write {description}",
                        "slots": ["description"],
                        "task_type": "email",
                        "level": "content_task",
                    },
                    {
                        "text": "with at least {n} words",
                        "slots": ["n"],
                        "task_type": "number_words",
                        "level": "format",
                    },
                    {
                        "text": 'The {document_type} must include the keywords {keywords}',
                        "slots": ["document_type", "keywords"],
                        "task_type": "keyword_inclusion",
                        "level": "content_constraint",
                    },
                    {
                        "text": "should not use any {punctuation}",
                        "slots": ["punctuation"],
                        "task_type": "no_commas",
                        "level": "content_constraint",
                    },
                ],
                "options": [
                    {
                        "value": "a long email template that invites a group of participants to a meeting",
                        "slot": "description",
                        "compatible_task_types": [
                            "email",
                            "creative_writing",
                        ],
                    },
                    {
                        "value": "500",
                        "slot": "n",
                        "compatible_task_types": [
                            "number_words",
                            "number_sentences",
                            "number_paragraphs",
                        ],
                    },
                    {
                        "value": "email",
                        "slot": "document_type",
                        "compatible_task_types": [
                            "keyword_inclusion",
                            "keyword_frequency",
                        ],
                    },
                    {
                        "value": '"correlated" and "experiencing"',
                        "slot": "keywords",
                        "compatible_task_types": [
                            "keyword_inclusion",
                            "keyword_frequency",
                        ],
                    },
                    {
                        "value": "commas",
                        "slot": "punctuation",
                        "compatible_task_types": [
                            "no_commas",
                            "punctuation_or_token_exclusion",
                        ],
                    },
                ],
            },
            ensure_ascii=False,
        ),
    },
    {
        "role": "user",
        "content": (
            'Taxonomy labels: ["question_answering", "number_sentences", '
            '"all_lowercase", "end_checker", "summarization", "explanation"]\n\n'
            "Prompt:\n"
            '"Explain how photosynthesis works in exactly 5 sentences. Write '
            "everything in lowercase. End your response with the exact phrase: "
            '"that is all."'
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps(
            {
                "templates": [
                    {
                        "text": "Explain {topic}",
                        "slots": ["topic"],
                        "task_type": "explanation",
                        "level": "content_task",
                    },
                    {
                        "text": "in exactly {n} sentences",
                        "slots": ["n"],
                        "task_type": "number_sentences",
                        "level": "format",
                    },
                    {
                        "text": "Write everything in lowercase",
                        "slots": [],
                        "task_type": "all_lowercase",
                        "level": "style",
                    },
                    {
                        "text": 'End your response with the exact phrase: "{phrase}"',
                        "slots": ["phrase"],
                        "task_type": "end_checker",
                        "level": "style",
                    },
                ],
                "options": [
                    {
                        "value": "how photosynthesis works",
                        "slot": "topic",
                        "compatible_task_types": [
                            "explanation",
                            "question_answering",
                            "summarization",
                        ],
                    },
                    {
                        "value": "5",
                        "slot": "n",
                        "compatible_task_types": [
                            "number_sentences",
                            "number_words",
                            "number_paragraphs",
                        ],
                    },
                    {
                        "value": "that is all",
                        "slot": "phrase",
                        "compatible_task_types": ["end_checker"],
                    },
                ],
            },
            ensure_ascii=False,
        ),
    },
]


def _build_extraction_messages(
    prompt: str, taxonomy_labels: list[str]
) -> list[dict[str, str]]:
    """Build the chat messages for one LLM extraction call."""
    labels_str = json.dumps(taxonomy_labels)
    return [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        *_EXTRACTION_FEW_SHOT,
        {
            "role": "user",
            "content": (
                f"Taxonomy labels: {labels_str}\n\n"
                f"Prompt:\n\"{prompt}\""
            ),
        },
    ]


def _call_llm(
    messages: list[dict[str, str]],
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    device: str | None = None,
) -> str:
    """Call a local HuggingFace model and return the raw text response."""
    return _local_generate(
        model,
        messages,
        temperature=temperature,
        max_new_tokens=max_tokens,
        device=device,
    )


def _parse_llm_extraction(
    raw: str,
    taxonomy: dict,
) -> tuple[list[TaskTemplate], list[Option]]:
    """Parse the JSON returned by the LLM into TaskTemplate / Option objects."""
    # Strip possible markdown fences
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    data = json.loads(raw)

    valid_levels = set(LEVELS)
    valid_task_types = set(taxonomy.keys())

    templates: list[TaskTemplate] = []
    options: list[Option] = []

    for t in data.get("templates", []):
        task_type = t.get("task_type", "unknown")
        level = t.get("level", "content_task")
        text = t.get("text", "")
        slots = t.get("slots", [])

        # Validate against taxonomy
        if task_type not in valid_task_types:
            logger.debug(f"  Skipping unknown task_type '{task_type}' from LLM output")
            continue
        if level not in valid_levels:
            level = taxonomy.get(task_type, {}).get("level", "content_task")

        # Compute token length of the fixed frame (text minus slot placeholders)
        fixed_text = text
        for s in slots:
            fixed_text = fixed_text.replace(f"{{{s}}}", "")
        tlen = token_length(fixed_text)

        tid = _template_id(task_type, text)
        tmpl = TaskTemplate(
            id=tid,
            text=text,
            slots=slots,
            task_type=task_type,
            level=level,
            token_length=tlen,
            source="llm_extracted",
        )
        templates.append(tmpl)

    for o in data.get("options", []):
        value = o.get("value", "")
        slot = o.get("slot", "")
        compat_types = o.get("compatible_task_types", [])

        # Filter to known task types
        compat_types = [ct for ct in compat_types if ct in valid_task_types]
        if not compat_types:
            compat_types = ["_universal"]

        oid = _option_id(slot, value)
        opt = Option(
            id=oid,
            value=value,
            slot=slot,
            compatible_task_types=compat_types,
            compatible_templates=[
                t.id for t in templates if slot in t.slots
            ],
            token_length=token_length(value),
            source="llm_extracted",
        )
        options.append(opt)

    return templates, options


def extract_templates_from_prompt_llm(
    prompt: str,
    taxonomy: dict,
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str | None = None,
) -> tuple[list[TaskTemplate], list[Option]]:
    """
    Decompose a single prompt into task templates and options using a local
    HuggingFace model.

    The model receives a few-shot prompt with the full taxonomy label list
    and returns structured JSON.
    """
    taxonomy_labels = list(taxonomy.keys())
    messages = _build_extraction_messages(prompt, taxonomy_labels)

    try:
        raw = _call_llm(
            messages,
            model=model,
            temperature=0.0,
            device=device,
        )
        return _parse_llm_extraction(raw, taxonomy)
    except Exception as e:
        logger.warning(f"  LLM extraction failed for prompt: {e}")
        return [], []


def extract_templates_from_dataset(
    prompts: list[dict],
    taxonomy: dict,
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str | None = None,
    batch_size: int = 1,
) -> tuple[list[TaskTemplate], list[Option]]:
    """
    Extract templates and options from a list of prompt records via a local
    HuggingFace model.

    Each record should have at least a ``"prompt"`` field.
    Results are de-duplicated and compatibility info is merged.
    """
    all_templates: dict[str, TaskTemplate] = {}
    all_options: dict[str, Option] = {}

    total = len(prompts)
    for idx, rec in enumerate(prompts):
        prompt_text = rec.get("prompt", "")
        if not prompt_text:
            continue

        tmpls, opts = extract_templates_from_prompt_llm(
            prompt_text,
            taxonomy,
            model=model,
            device=device,
        )

        for t in tmpls:
            if t.id not in all_templates:
                all_templates[t.id] = t
        for o in opts:
            if o.id not in all_options:
                all_options[o.id] = o
            else:
                # Merge compatibility info — options can be compatible
                # with multiple task types and templates.
                existing = all_options[o.id]
                for tt in o.compatible_task_types:
                    if tt not in existing.compatible_task_types:
                        existing.compatible_task_types.append(tt)
                for tid in o.compatible_templates:
                    if tid not in existing.compatible_templates:
                        existing.compatible_templates.append(tid)

        if (idx + 1) % 50 == 0 or (idx + 1) == total:
            logger.info(f"  Extracted {idx+1}/{total} prompts")

    return list(all_templates.values()), list(all_options.values())


# ═══════════════════════════════════════════════════════════════════════════
# 3.  OPTION AUGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

# Rules for augmenting numeric options: (slot_name, task_type) → (min, max, step)
_NUMERIC_AUGMENTATION_RULES: dict[str, dict[str, tuple[int, int, int]]] = {
    "number_words": {
        "creative_writing": (100, 1000, 50),
        "summarization": (50, 500, 50),
        "question_answering": (50, 300, 50),
        "explanation": (100, 500, 50),
        "email": (50, 300, 50),
        "_default": (50, 500, 50),
    },
    "number_sentences": {
        "creative_writing": (3, 20, 1),
        "summarization": (2, 10, 1),
        "question_answering": (1, 10, 1),
        "_default": (3, 12, 1),
    },
    "number_paragraphs": {
        "creative_writing": (2, 8, 1),
        "summarization": (1, 5, 1),
        "explanation": (2, 6, 1),
        "_default": (2, 6, 1),
    },
    "number_bullets": {
        "brainstorming": (3, 15, 1),
        "data_analysis": (3, 10, 1),
        "_default": (3, 8, 1),
    },
}

# Keyword pools for augmenting keyword inclusion/frequency options
_KEYWORD_AUGMENTATION_POOL = [
    "innovation", "sustainability", "resilience", "algorithm",
    "framework", "paradigm", "ecosystem", "synergy", "leverage",
    "holistic", "empirical", "catalyst", "trajectory", "nuance",
    "infrastructure", "benchmark", "convergence", "disruption",
    "perspective", "methodology", "hypothesis", "correlation",
    "phenomenon", "principle", "strategy", "mechanism", "dimension",
    "architecture", "optimization", "dynamic", "equilibrium",
]

_LANGUAGE_AUGMENTATION = [
    "English", "French", "Spanish", "German", "Italian",
    "Portuguese", "Japanese", "Chinese", "Korean", "Arabic",
    "Russian", "Hindi", "Dutch", "Swedish", "Polish",
]

_END_PHRASE_AUGMENTATION = [
    "hope this helps", "thank you", "the end", "fin", "peace",
    "sincerely", "best regards", "yours truly", "that is all",
    "in conclusion", "to summarize",
]

_LETTER_AUGMENTATION = list("aeioustrnlc")

_FORBIDDEN_WORD_AUGMENTATION = [
    "very", "really", "things", "stuff", "nice", "good", "bad",
    "important", "interesting", "basically", "actually", "literally",
    "amazing", "awesome", "great", "fantastic", "wonderful",
    "terrible", "horrible", "absolutely", "definitely", "extremely",
]


def augment_options(
    existing_options: list[Option],
    templates: list[TaskTemplate],
    rng: random.Random,
    max_augmented_per_slot: int = 20,
) -> list[Option]:
    """
    Augment the option pool with programmatically generated variations.

    For numeric slots → generate a range of sensible integers.
    For keyword slots → sample from curated word pools.
    For language slots → expand from the language list.

    Augmented options are made compatible with **all** relevant task types
    (not just one).
    """
    augmented: list[Option] = []
    seen_ids: set[str] = {o.id for o in existing_options}

    # Group templates by task_type for compatibility lookup
    templates_by_task: dict[str, list[str]] = defaultdict(list)
    for t in templates:
        templates_by_task[t.task_type].append(t.id)

    # ── Numeric augmentation ─────────────────────────────────────────
    for constraint_name, task_rules in _NUMERIC_AUGMENTATION_RULES.items():
        # Collect ALL task types this numeric option could apply to
        all_compatible_tasks = [
            tt for tt in task_rules if tt != "_default"
        ]
        for task_type, (lo, hi, step) in task_rules.items():
            if task_type == "_default":
                continue
            for val in range(lo, hi + 1, step):
                val_str = str(val)
                oid = _option_id("n", f"{constraint_name}_{task_type}_{val_str}")
                if oid in seen_ids:
                    continue
                seen_ids.add(oid)

                # Compatible with the constraint type AND all task types
                # that share the same numeric slot
                compat_tasks = [constraint_name] + all_compatible_tasks
                compat_tasks = list(dict.fromkeys(compat_tasks))  # dedupe, preserve order

                opt = Option(
                    id=oid,
                    value=val_str,
                    slot="n",
                    compatible_task_types=compat_tasks,
                    compatible_templates=templates_by_task.get(constraint_name, []),
                    token_length=token_length(val_str),
                    source="augmented",
                    tags=["numeric", task_type],
                )
                augmented.append(opt)

    # ── Keyword augmentation ─────────────────────────────────────────
    for kw in _KEYWORD_AUGMENTATION_POOL:
        oid = _option_id("keyword", kw)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        opt = Option(
            id=oid,
            value=kw,
            slot="keyword",
            compatible_task_types=["keyword_inclusion", "keyword_frequency"],
            compatible_templates=(
                templates_by_task.get("keyword_inclusion", [])
                + templates_by_task.get("keyword_frequency", [])
            ),
            token_length=token_length(kw),
            source="augmented",
            tags=["keyword"],
        )
        augmented.append(opt)

    # ── Keyword combination augmentation (for keyword_inclusion) ─────
    for _ in range(max_augmented_per_slot):
        k = rng.randint(2, 4)
        kws = rng.sample(_KEYWORD_AUGMENTATION_POOL, k)
        combo = ", ".join(kws)
        oid = _option_id("keywords", combo)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        opt = Option(
            id=oid,
            value=combo,
            slot="keywords",
            compatible_task_types=["keyword_inclusion"],
            compatible_templates=templates_by_task.get("keyword_inclusion", []),
            token_length=token_length(combo),
            source="augmented",
            tags=["keyword_combo"],
        )
        augmented.append(opt)

    # ── Language augmentation ────────────────────────────────────────
    for lang in _LANGUAGE_AUGMENTATION:
        oid = _option_id("language", lang)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        opt = Option(
            id=oid,
            value=lang,
            slot="language",
            compatible_task_types=["response_language"],
            compatible_templates=templates_by_task.get("response_language", []),
            token_length=token_length(lang),
            source="augmented",
            tags=["language"],
        )
        augmented.append(opt)

    # ── End-phrase augmentation ──────────────────────────────────────
    for phrase in _END_PHRASE_AUGMENTATION:
        oid = _option_id("phrase", phrase)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        opt = Option(
            id=oid,
            value=phrase,
            slot="phrase",
            compatible_task_types=["end_checker"],
            compatible_templates=templates_by_task.get("end_checker", []),
            token_length=token_length(phrase),
            source="augmented",
            tags=["phrase"],
        )
        augmented.append(opt)

    # ── Letter augmentation ─────────────────────────────────────────
    for letter in _LETTER_AUGMENTATION:
        oid = _option_id("letter", letter)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        opt = Option(
            id=oid,
            value=letter,
            slot="letter",
            compatible_task_types=["letter_frequency"],
            compatible_templates=templates_by_task.get("letter_frequency", []),
            token_length=token_length(letter),
            source="augmented",
            tags=["letter"],
        )
        augmented.append(opt)

    # ── Forbidden words augmentation ────────────────────────────────
    for _ in range(max_augmented_per_slot):
        k = rng.randint(2, 4)
        words = rng.sample(_FORBIDDEN_WORD_AUGMENTATION, k)
        combo = ", ".join(words)
        oid = _option_id("words", combo)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        opt = Option(
            id=oid,
            value=combo,
            slot="words",
            compatible_task_types=["forbidden_words", "forbidden_content",
                                   "punctuation_or_token_exclusion"],
            compatible_templates=templates_by_task.get("forbidden_words", []),
            token_length=token_length(combo),
            source="augmented",
            tags=["forbidden_words"],
        )
        augmented.append(opt)

    logger.info(f"  Augmented {len(augmented)} new options")
    return augmented


# ═══════════════════════════════════════════════════════════════════════════
# 4.  TEMPLATE & OPTION STORE
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class TemplateStore:
    """
    Tidy storage for templates and options with compatibility tracking.

    Supports:
      - Querying templates by task type and level
      - Finding compatible options for a given template
      - Options compatible with **multiple** task types and templates
      - Merging extracted and augmented data
    """
    templates: dict[str, TaskTemplate] = field(default_factory=dict)
    options: dict[str, Option] = field(default_factory=dict)

    # Indices for fast lookup
    _by_task_type: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    _by_level: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    _by_slot: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))

    def add_template(self, tmpl: TaskTemplate) -> None:
        if tmpl.id in self.templates:
            return
        self.templates[tmpl.id] = tmpl
        self._by_task_type[tmpl.task_type].append(tmpl.id)
        self._by_level[tmpl.level].append(tmpl.id)

    def add_option(self, opt: Option) -> None:
        if opt.id in self.options:
            existing = self.options[opt.id]
            # Merge compatibility — options may be compatible with many
            # task types and templates.
            for tt in opt.compatible_task_types:
                if tt not in existing.compatible_task_types:
                    existing.compatible_task_types.append(tt)
            for tid in opt.compatible_templates:
                if tid not in existing.compatible_templates:
                    existing.compatible_templates.append(tid)
            return
        self.options[opt.id] = opt
        self._by_slot[opt.slot].append(opt.id)

    def add_templates(self, templates: list[TaskTemplate]) -> None:
        for t in templates:
            self.add_template(t)

    def add_options(self, options: list[Option]) -> None:
        for o in options:
            self.add_option(o)

    def get_templates_for_task(self, task_type: str) -> list[TaskTemplate]:
        return [self.templates[tid] for tid in self._by_task_type.get(task_type, [])]

    def get_templates_for_level(self, level: str) -> list[TaskTemplate]:
        return [self.templates[tid] for tid in self._by_level.get(level, [])]

    def get_options_for_slot(self, slot: str) -> list[Option]:
        return [self.options[oid] for oid in self._by_slot.get(slot, [])]

    def get_compatible_options(self, template: TaskTemplate) -> dict[str, list[Option]]:
        """
        Return ``{slot_name: [compatible options]}`` for a template.

        An option is compatible if:
          - the template's task_type appears in the option's
            ``compatible_task_types``, OR
          - the template's id appears in the option's
            ``compatible_templates``, OR
          - the option has ``_universal`` in ``compatible_task_types``.

        Falls back to *all* options for a slot if none match directly.
        """
        result: dict[str, list[Option]] = {}
        for slot in template.slots:
            slot_options = self.get_options_for_slot(slot)
            compatible = [
                o for o in slot_options
                if (
                    template.task_type in o.compatible_task_types
                    or template.id in o.compatible_templates
                    or "_universal" in o.compatible_task_types
                )
            ]
            # If no directly compatible options, allow all options for this slot
            if not compatible:
                compatible = slot_options
            result[slot] = compatible
        return result

    def summary(self) -> dict:
        # Compute how many options are compatible with >1 task type
        multi_compat = sum(
            1 for o in self.options.values()
            if len(o.compatible_task_types) > 1
        )
        return {
            "n_templates": len(self.templates),
            "n_options": len(self.options),
            "n_options_multi_compatible": multi_compat,
            "templates_by_level": {
                lv: len(tids) for lv, tids in self._by_level.items()
            },
            "options_by_slot": {
                slot: len(oids) for slot, oids in self._by_slot.items()
            },
            "task_types": sorted(self._by_task_type.keys()),
        }

    def save(
        self,
        templates_path: Path = TEMPLATES_PATH,
        options_path: Path = OPTIONS_PATH,
    ) -> None:
        """Persist to JSON files."""
        templates_data = []
        for t in self.templates.values():
            templates_data.append({
                "id": t.id,
                "text": t.text,
                "slots": t.slots,
                "task_type": t.task_type,
                "level": t.level,
                "token_length": t.token_length,
                "source": t.source,
                "compatible_with": t.compatible_with,
            })
        with open(templates_path, "w") as f:
            json.dump(templates_data, f, indent=2, ensure_ascii=False)
        logger.info(
            f"  Templates saved → {templates_path} ({len(templates_data)} entries)"
        )

        options_data = []
        for o in self.options.values():
            options_data.append({
                "id": o.id,
                "value": o.value,
                "slot": o.slot,
                "compatible_task_types": o.compatible_task_types,
                "compatible_templates": o.compatible_templates,
                "token_length": o.token_length,
                "source": o.source,
                "tags": o.tags,
            })
        with open(options_path, "w") as f:
            json.dump(options_data, f, indent=2, ensure_ascii=False)
        logger.info(
            f"  Options saved → {options_path} ({len(options_data)} entries)"
        )

    @classmethod
    def load(
        cls,
        templates_path: Path = TEMPLATES_PATH,
        options_path: Path = OPTIONS_PATH,
    ) -> "TemplateStore":
        """Load from JSON files."""
        store = cls()

        if templates_path.exists():
            with open(templates_path) as f:
                for item in json.load(f):
                    # Support both old char_length and new token_length keys
                    if "char_length" in item and "token_length" not in item:
                        item["token_length"] = item.pop("char_length")
                    elif "char_length" in item:
                        del item["char_length"]
                    tmpl = TaskTemplate(**item)
                    store.add_template(tmpl)
            logger.info(
                f"  Loaded {len(store.templates)} templates from {templates_path}"
            )

        if options_path.exists():
            with open(options_path) as f:
                for item in json.load(f):
                    if "char_length" in item and "token_length" not in item:
                        item["token_length"] = item.pop("char_length")
                    elif "char_length" in item:
                        del item["char_length"]
                    opt = Option(**item)
                    store.add_option(opt)
            logger.info(
                f"  Loaded {len(store.options)} options from {options_path}"
            )

        return store


# ═══════════════════════════════════════════════════════════════════════════
# 5.  SYNTHETIC PROMPT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class GeneratedPrompt:
    """One generated prompt with full provenance."""
    id: int
    prompt: str
    task_template_id: str
    constraint_template_ids: list[str]
    option_ids: list[str]
    active_labels: list[str]          # taxonomy labels active in this prompt
    task: str
    constraints: list[str]
    density: int                       # total active features
    token_length: int                  # total length in tokens
    combo_id: str


def _combo_hash(task: str, constraints: list[str], option_vals: list[str]) -> str:
    key = task + "|" + "|".join(sorted(constraints)) + "|" + "|".join(sorted(option_vals))
    return hashlib.md5(key.encode()).hexdigest()[:12]


class SyntheticGenerator:
    """
    Generate prompts by composing templates with compatible options.

    Parameters
    ----------
    store       : TemplateStore with templates and options
    taxonomy    : taxonomy dict (from taxonomy.json)
    density     : number of templates to combine (1 = task only, 2+ = task + constraints)
    min_tokens  : minimum token length for generated prompts (0 = no limit)
    max_tokens  : maximum token length (0 = no limit)
    seed        : random seed
    """

    def __init__(
        self,
        store: TemplateStore,
        taxonomy: dict,
        density: int = 3,
        min_tokens: int = 0,
        max_tokens: int = 0,
        seed: int = 42,
    ):
        self.store = store
        self.taxonomy = taxonomy
        self.density = density
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.rng = random.Random(seed)

        # Precompute available content_task and constraint templates
        self.task_templates = store.get_templates_for_level("content_task")
        self.constraint_templates: list[TaskTemplate] = []
        for lv in CONSTRAINT_LEVELS:
            self.constraint_templates.extend(store.get_templates_for_level(lv))

        # Usage tracking for coverage balancing
        self._task_usage: Counter = Counter()
        self._constraint_usage: Counter = Counter()
        self._seen_combos: set[str] = set()

        logger.info(
            f"  Generator ready: {len(self.task_templates)} task templates, "
            f"{len(self.constraint_templates)} constraint templates, "
            f"density={density}"
        )

    def _weight(self, name: str, usage: Counter) -> float:
        return 1.0 / (1 + usage[name])

    def _fill_template(self, tmpl: TaskTemplate) -> tuple[str, list[str]]:
        """
        Fill a template's slots with compatible options.
        Returns (filled_text, list_of_option_ids).
        """
        if not tmpl.slots:
            return tmpl.text, []

        compatible = self.store.get_compatible_options(tmpl)
        filled = tmpl.text
        option_ids: list[str] = []

        for slot in tmpl.slots:
            opts = compatible.get(slot, [])
            if not opts:
                # No options available — leave a reasonable default
                if slot == "n":
                    val = str(self.rng.randint(3, 10))
                elif slot == "constraints":
                    val = ""
                else:
                    val = "..."
                filled = filled.replace(f"{{{slot}}}", val, 1)
            else:
                chosen = self.rng.choice(opts)
                filled = filled.replace(f"{{{slot}}}", chosen.value, 1)
                option_ids.append(chosen.id)

        return filled, option_ids

    def generate_one(self) -> GeneratedPrompt | None:
        """
        Generate a single prompt by:
          1. Picking a content_task template (coverage-weighted)
          2. Picking (density-1) constraint templates (coverage-weighted)
          3. Filling all slots with compatible options
          4. Checking length constraints (in tokens)
        """
        if not self.task_templates:
            return None

        # Pick task template
        task_weights = [self._weight(t.task_type, self._task_usage)
                        for t in self.task_templates]
        task_tmpl = self.rng.choices(self.task_templates, weights=task_weights, k=1)[0]

        # Pick constraint templates
        n_constraints = max(self.density - 1, 0)
        chosen_constraints: list[TaskTemplate] = []

        if n_constraints > 0 and self.constraint_templates:
            available = list(self.constraint_templates)
            avail_weights = [self._weight(c.task_type, self._constraint_usage)
                             for c in available]

            # Avoid duplicate constraint types; at most 2 per level
            level_counts: Counter = Counter()
            seen_types: set[str] = set()

            for _ in range(n_constraints):
                if not available:
                    break
                picked = self.rng.choices(available, weights=avail_weights, k=1)[0]

                # Check level cap
                if level_counts[picked.level] >= 2:
                    # Try to find another from a different level
                    alt = [(a, w) for a, w in zip(available, avail_weights)
                           if level_counts[a.level] < 2 and a.task_type not in seen_types]
                    if alt:
                        alt_items, alt_wts = zip(*alt)
                        picked = self.rng.choices(alt_items, weights=alt_wts, k=1)[0]
                    else:
                        continue

                # Avoid duplicate constraint types
                if picked.task_type in seen_types:
                    alt = [(a, w) for a, w in zip(available, avail_weights)
                           if a.task_type not in seen_types and level_counts[a.level] < 2]
                    if alt:
                        alt_items, alt_wts = zip(*alt)
                        picked = self.rng.choices(alt_items, weights=alt_wts, k=1)[0]
                    else:
                        continue

                chosen_constraints.append(picked)
                level_counts[picked.level] += 1
                seen_types.add(picked.task_type)

                # Remove from available pool
                idx = available.index(picked)
                available.pop(idx)
                avail_weights.pop(idx)

        # Fill templates
        task_text, task_option_ids = self._fill_template(task_tmpl)
        constraint_texts: list[str] = []
        constraint_option_ids: list[str] = []
        constraint_template_ids: list[str] = []

        for ct in chosen_constraints:
            ct_text, ct_opts = self._fill_template(ct)
            # Don't add empty constraints
            ct_text = ct_text.strip()
            if ct_text:
                constraint_texts.append(ct_text)
                constraint_option_ids.extend(ct_opts)
                constraint_template_ids.append(ct.id)

        # Compose final prompt
        constraints_block = "\n".join(constraint_texts)

        # If the task template has {constraints}, fill it
        if "{constraints}" in task_text:
            prompt = task_text.replace("{constraints}", constraints_block).strip()
        else:
            # Append constraints after the task
            if constraints_block:
                prompt = f"{task_text}\n\n{constraints_block}".strip()
            else:
                prompt = task_text.strip()

        # Token-length check
        tok_len = token_length(prompt)
        if self.min_tokens > 0 and tok_len < self.min_tokens:
            return None  # Too short — caller should retry
        if self.max_tokens > 0 and tok_len > self.max_tokens:
            return None  # Too long — caller should retry

        # Build labels
        active_labels = [task_tmpl.task_type] + [c.task_type for c in chosen_constraints]
        constraints_list = [c.task_type for c in chosen_constraints]
        all_option_ids = task_option_ids + constraint_option_ids
        combo_id = _combo_hash(task_tmpl.task_type, constraints_list,
                               [self.store.options[oid].value
                                for oid in all_option_ids if oid in self.store.options])

        # Update usage counters
        self._task_usage[task_tmpl.task_type] += 1
        for c in chosen_constraints:
            self._constraint_usage[c.task_type] += 1
        self._seen_combos.add(combo_id)

        return GeneratedPrompt(
            id=0,  # assigned by caller
            prompt=prompt,
            task_template_id=task_tmpl.id,
            constraint_template_ids=constraint_template_ids,
            option_ids=all_option_ids,
            active_labels=active_labels,
            task=task_tmpl.task_type,
            constraints=constraints_list,
            density=len(active_labels),
            token_length=tok_len,
            combo_id=combo_id,
        )

    def generate_batch(self, n: int, max_retries: int = 3) -> list[GeneratedPrompt]:
        """
        Generate n prompts, respecting length constraints.
        Retries up to max_retries times per prompt if length is out of range.
        """
        results: list[GeneratedPrompt] = []
        for i in range(n):
            prompt = None
            for _ in range(max_retries):
                prompt = self.generate_one()
                if prompt is not None:
                    break
            if prompt is not None:
                prompt.id = i
                results.append(prompt)

            if (i + 1) % 1000 == 0:
                logger.info(f"  …{i+1}/{n} generated")

        return results

    def coverage_report(self) -> dict:
        return {
            "unique_combos": len(self._seen_combos),
            "task_usage": dict(self._task_usage.most_common()),
            "constraint_usage": dict(self._constraint_usage.most_common()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 6.  FULL PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def _load_dataset_prompts(
    datasets: list[str] | None = None,
) -> list[dict]:
    """
    Stream prompts from official instruction-tuning datasets using the
    registry and loaders from ``collect_task_types.py``.

    Returns a list of ``{"prompt": ..., "source": ...}`` dicts suitable
    for ``extract_templates_from_dataset()``.
    """
    registry = DATASET_REGISTRY
    if datasets:
        registry = {k: v for k, v in registry.items() if k in datasets}

    prompts: list[dict] = []

    for ds_name, config in registry.items():
        if config.get("source") == "skip":
            logger.info(f"  Skipping {ds_name}")
            continue

        logger.info(f"  Loading: {ds_name}")
        try:
            if config["source"] == "huggingface":
                samples = load_hf_dataset(config)
            elif config["source"] == "github_raw":
                samples = load_github_data(config)
            elif config["source"] == "kcif_github":
                samples = load_kcif_data(config)
            else:
                logger.warning(f"    Unknown source type: {config['source']}")
                continue
        except Exception as e:
            logger.error(f"    Error loading {ds_name}: {e}")
            continue

        if not samples:
            logger.warning(f"    No samples for {ds_name}")
            continue

        instruction_fields = config.get("instruction_fields", [])
        ds_count = 0
        for sample in samples:
            text = extract_instruction_text(sample, instruction_fields)
            if text.strip():
                prompts.append({"prompt": text, "source": ds_name})
                ds_count += 1

        logger.info(f"    {ds_name}: {len(samples)} samples → {ds_count} prompts")

    logger.info(f"  Total prompts loaded: {len(prompts)}")
    return prompts


def run_extraction(
    taxonomy: dict,
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str | None = None,
    datasets: list[str] | None = None,
) -> TemplateStore:
    """
    Extract templates from real dataset prompts via a local HuggingFace model.

    Streams prompts from official instruction-tuning datasets (using the
    registry in ``collect_task_types.py``), then runs few-shot LLM
    extraction on each prompt to decompose it into templates and options.
    """
    store = TemplateStore()

    logger.info(f"Streaming prompts from official datasets for extraction (model={model})…")
    prompts = _load_dataset_prompts(datasets)

    if prompts:
        ext_tmpls, ext_opts = extract_templates_from_dataset(
            prompts, taxonomy,
            model=model,
            device=device,
        )
        store.add_templates(ext_tmpls)
        store.add_options(ext_opts)
        logger.info(f"  Extracted: {len(ext_tmpls)} templates, {len(ext_opts)} options")
    else:
        logger.warning("No prompts found from any dataset; store will be empty.")

    logger.info(f"  Total store: {len(store.templates)} templates, {len(store.options)} options")
    return store


def run_augmentation(store: TemplateStore, seed: int = 42) -> None:
    """Step 4: Augment the option pool."""
    logger.info("Augmenting options…")
    rng = random.Random(seed)
    augmented = augment_options(
        list(store.options.values()),
        list(store.templates.values()),
        rng,
    )
    store.add_options(augmented)
    logger.info(f"  Store after augmentation: {len(store.templates)} templates, {len(store.options)} options")


def run_generation(
    store: TemplateStore,
    taxonomy: dict,
    n: int = 5000,
    density: int = 3,
    min_tokens: int = 0,
    max_tokens: int = 0,
    seed: int = 42,
    output_path: Path = GENERATED_PATH,
) -> list[GeneratedPrompt]:
    """Generate synthetic prompts."""
    logger.info(f"Generating {n} prompts (density={density})…")

    gen = SyntheticGenerator(
        store=store,
        taxonomy=taxonomy,
        density=density,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        seed=seed,
    )
    results = gen.generate_batch(n)

    # Save to JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            record = {
                "id": r.id,
                "prompt": r.prompt,
                "task_template_id": r.task_template_id,
                "constraint_template_ids": r.constraint_template_ids,
                "option_ids": r.option_ids,
                "active_labels": r.active_labels,
                "task": r.task,
                "constraints": r.constraints,
                "density": r.density,
                "token_length": r.token_length,
                "combo_id": r.combo_id,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info(f"  Saved {len(results)} prompts → {output_path}")

    # Coverage report
    cov = gen.coverage_report()
    logger.info(f"  Unique combos: {cov['unique_combos']}")
    cov_path = output_path.with_suffix(".coverage.json")
    with open(cov_path, "w") as f:
        json.dump(cov, f, indent=2)

    return results


# ═══════════════════════════════════════════════════════════════════════════
# 7.  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Semi-synthetic dataset construction from minimally-changing prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Full pipeline
  python dataset/construct_dataset.py --n 5000 --density 3

  # Extract only (requires local HF model)
  python dataset/construct_dataset.py --extract-only --model meta-llama/Llama-3.1-8B-Instruct

  # Extract from specific datasets
  python dataset/construct_dataset.py --extract-only --datasets Alpaca DollyV2

  # Generate with token-length constraints
  python dataset/construct_dataset.py --n 10000 --density 4 --min-tokens 20 --max-tokens 200

  # Use a specific tokenizer
  python dataset/construct_dataset.py --tokenizer cl100k_base --n 5000
""",
    )
    parser.add_argument("--n", type=int, default=5000, help="Number of prompts to generate")
    parser.add_argument("--density", type=int, default=3,
                        help="Number of templates per prompt (1=task only, 3=task+2 constraints)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-tokens", type=int, default=0,
                        help="Minimum prompt length in tokens (0=no limit)")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Maximum prompt length in tokens (0=no limit)")
    parser.add_argument("--output", type=str, default=None, help="Output JSONL path")

    # Tokenizer
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Tokenizer name: tiktoken encoding (cl100k_base, o200k_base), "
                             "OpenAI model name, or HuggingFace model id")

    # Local HuggingFace model for extraction
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID for local template extraction "
                             "(e.g. meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token override for gated models. "
                             "If omitted, HF_TOKEN is read from .env.huggingface "
                             "in the project root.")
    parser.add_argument("--device", type=str, default=None,
                        help="Device for local model inference: cpu, cuda, cuda:0, cuda:1, mps, "
                             "or None for auto-detection.")

    # Dataset selection
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Specific dataset names to stream from (default: all). "
                             "Names must match keys in DATASET_REGISTRY.")

    # Mode flags
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract and store templates, don't generate")
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate from previously extracted templates")

    args = parser.parse_args()

    # Initialise tokenizer
    set_token_counter(args.tokenizer)
    tc = get_token_counter()
    logger.info(f"Tokenizer: {tc.name}")

    # Load taxonomy
    if not TAXONOMY_PATH.exists():
        logger.error(f"Taxonomy not found at {TAXONOMY_PATH}. Run collect_task_types.py first.")
        sys.exit(1)
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

    output_path = Path(args.output) if args.output else GENERATED_PATH

    # ── Extract (always needed for generate) ─────────────────────────
    if not args.generate_only:
        # Set HF_TOKEN from CLI if provided
        if args.hf_token:
            os.environ["HF_TOKEN"] = args.hf_token

        store = run_extraction(
            taxonomy,
            model=args.model,
            device=args.device,
            datasets=args.datasets,
        )
        run_augmentation(store, seed=args.seed)
        store.save()
    else:
        store = TemplateStore.load()
        if not store.templates:
            logger.error("No templates found. Run without --generate-only first.")
            sys.exit(1)

    if args.extract_only:
        summary = store.summary()
        logger.info(f"Extraction complete: {json.dumps(summary, indent=2)}")
        return

    # ── Generate ─────────────────────────────────────────────────────
    run_generation(
        store, taxonomy,
        n=args.n,
        density=args.density,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

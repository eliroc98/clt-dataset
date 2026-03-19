"""
option_taxonomy.py — Semantic classification of options into types.

Defines a taxonomy of option types that cuts across slot names, enabling
cross-template compatibility. For example, an option of type TOPICAL_SUBJECT
can fill both a {topic} slot and a {subject} slot.

Each option type has a set of compatible slot names (many-to-many) and a
rule-based classifier with LLM fallback for ambiguous cases.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from dataset.schema import Option

logger = logging.getLogger(__name__)


# ── Option type definitions ──────────────────────────────────────────────

@dataclass(frozen=True)
class OptionTypeEntry:
    """One entry in the option taxonomy."""
    name: str
    description: str
    compatible_slots: frozenset[str]


OPTION_TYPES: dict[str, OptionTypeEntry] = {
    "entity_reference": OptionTypeEntry(
        name="entity_reference",
        description="Named entities: people, organizations, places, works of art, products, events",
        compatible_slots=frozenset({
            "topic", "person", "subject", "title", "content", "description",
            "category", "example", "source",
        }),
    ),
    "topical_subject": OptionTypeEntry(
        name="topical_subject",
        description="Abstract topics, domains, themes, disciplines, subject areas",
        compatible_slots=frozenset({
            "topic", "subject", "description", "content", "title", "category",
        }),
    ),
    "textual_content": OptionTypeEntry(
        name="textual_content",
        description="Passages, paragraphs, documents, long-form text, code blocks",
        compatible_slots=frozenset({
            "text", "passage", "context", "description", "content", "source",
            "code", "example", "question",
        }),
    ),
    "quantitative": OptionTypeEntry(
        name="quantitative",
        description="Numbers, counts, percentages, measurements, quantities",
        compatible_slots=frozenset({
            "number", "content",
        }),
    ),
    "linguistic": OptionTypeEntry(
        name="linguistic",
        description="Languages, tones, styles, registers, voices",
        compatible_slots=frozenset({
            "language", "tone", "style", "format",
        }),
    ),
    "structural": OptionTypeEntry(
        name="structural",
        description="Output formats, structures, document types",
        compatible_slots=frozenset({
            "format", "text_type", "style", "content",
        }),
    ),
    "keyword_phrase": OptionTypeEntry(
        name="keyword_phrase",
        description="Individual words, short phrases, tags, labels",
        compatible_slots=frozenset({
            "keyword", "topic", "category", "option", "content", "subject",
        }),
    ),
    "temporal": OptionTypeEntry(
        name="temporal",
        description="Dates, times, durations, time periods",
        compatible_slots=frozenset({
            "content", "topic", "example",
        }),
    ),
    "instructional": OptionTypeEntry(
        name="instructional",
        description="Task descriptions, directives, process instructions",
        compatible_slots=frozenset({
            "task", "description", "content", "role",
        }),
    ),
    "example_instance": OptionTypeEntry(
        name="example_instance",
        description="Concrete examples, sample items, specific instances",
        compatible_slots=frozenset({
            "example", "content", "option", "completion",
        }),
    ),
}


# ── Rule-based slot → option_type mapping ────────────────────────────────

_SLOT_TO_TYPE: dict[str, str] = {
    "person": "entity_reference",
    "topic": "topical_subject",
    "subject": "topical_subject",
    "passage": "textual_content",
    "context": "textual_content",
    "code": "textual_content",
    "text": "textual_content",
    "number": "quantitative",
    "language": "linguistic",
    "tone": "linguistic",
    "style": "linguistic",
    "format": "structural",
    "text_type": "structural",
    "keyword": "keyword_phrase",
    "task": "instructional",
    "role": "instructional",
    "example": "example_instance",
    "completion": "example_instance",
    "option": "example_instance",
}


def classify_option_rule(option: Option) -> str | None:
    """Classify an option using rule-based heuristics. Returns None if ambiguous."""
    # Direct slot name mapping
    if option.slot in _SLOT_TO_TYPE:
        return _SLOT_TO_TYPE[option.slot]

    # Value-based heuristics for ambiguous slots
    value = option.value.strip()

    # Numeric check
    if re.match(r"^\d+(\.\d+)?$", value):
        return "quantitative"

    # Very long values are likely textual content
    if option.token_length > 50:
        return "textual_content"

    # Very short values (1-2 words) are likely keywords
    if option.token_length <= 2 and option.slot in {"content", "description"}:
        return "keyword_phrase"

    return None


def classify_options_batch(
    options: list[Option],
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str | None = None,
    batch_size: int = 40,
    gpu_memory_utilization: float = 0.7,
) -> dict[str, str]:
    """Classify options into option types.

    Returns a dict mapping option_id → option_type_name.
    Uses rule-based classification first, then LLM for ambiguous cases.
    """
    results: dict[str, str] = {}
    needs_llm: list[Option] = []

    # Rule-based pass
    for opt in options:
        otype = classify_option_rule(opt)
        if otype:
            results[opt.id] = otype
        else:
            needs_llm.append(opt)

    logger.info(
        f"Option classification: {len(results)} rule-based, "
        f"{len(needs_llm)} need LLM"
    )

    if not needs_llm:
        return results

    # LLM classification for ambiguous options
    from dataset.local_llm import generate_text_batch as _batch_gen

    type_names = list(OPTION_TYPES.keys())
    type_descriptions = "\n".join(
        f"- {t.name}: {t.description}" for t in OPTION_TYPES.values()
    )

    system_prompt = (
        "Classify each option value into exactly one of these types:\n"
        f"{type_descriptions}\n\n"
        "Output a JSON object mapping each value to its type name.\n"
        "Example: {\"climate change\": \"topical_subject\", \"Python\": \"keyword_phrase\"}\n"
        "Output valid JSON only."
    )

    for i in range(0, len(needs_llm), batch_size):
        batch = needs_llm[i: i + batch_size]
        values_dict = {opt.value[:200]: opt.id for opt in batch}
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(
                list(values_dict.keys()), ensure_ascii=False
            )},
        ]

        try:
            raws = _batch_gen(
                model, [messages],
                temperature=0.0, device=device,
                enable_thinking=False,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            raw = raws[0] if raws else ""
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            mapping = json.loads(raw)
        except Exception as e:
            logger.warning(f"  LLM classification batch failed: {e}")
            # Fallback: assign "topical_subject" for content slots
            for opt in batch:
                results[opt.id] = "topical_subject"
            continue

        for value_key, opt_id in values_dict.items():
            classified_type = mapping.get(value_key, "topical_subject")
            if classified_type in OPTION_TYPES:
                results[opt_id] = classified_type
            else:
                results[opt_id] = "topical_subject"

    return results


def get_compatible_slots_for_type(option_type: str) -> frozenset[str]:
    """Return the set of slot names compatible with an option type."""
    entry = OPTION_TYPES.get(option_type)
    return entry.compatible_slots if entry else frozenset()

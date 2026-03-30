"""
extractor.py — LLM extraction of task templates and options.

Decomposes raw instruction-tuning prompts into TaskTemplate / Option objects
using a local model with structured JSON output.

Pipeline:
  Stage A — Segmentation: split prompts into clauses, classify each with a
            taxonomy label (via segmenter.py).
  Stage B — LLM Extraction: for each segment, extract templates and options
            via batched LLM calls with structured JSON output.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from tqdm import tqdm

from dataset.schema import (
    LEVELS, LEVEL_REMAP,
    TaskTemplate, Option,
    ExtractionResult,
    template_id, option_id,
    FEW_SHOT_EXAMPLES_PATH,
    SEGMENTS_PATH,
)
from dataset.token_counter import token_length
from dataset.local_llm import generate_text as _local_generate, generate_text_batch as _local_generate_batch

logger = logging.getLogger(__name__)

from dataset.fix_slots import (
    CANONICAL_PREFERRED_SLOTS,
    _normalize_slot_names,
    _detect_self_replicating_slots,
    _expand_list_options,
    _merge_numbered_slots,
    _drop_duplicate_slot_templates,
    _remove_broken_options,
    normalize_existing,
)

import sys


def _flatten_taxonomy_labels(taxonomy: dict) -> list[str]:
    """Collect all leaf label names from a nested taxonomy dict.

    The taxonomy has structure like:
        {"task_type": {"information_tasks": {"question_answering": {...}, ...}, ...},
         "format_constraint": {"length_constraint": {...}, ...}}

    This returns the deepest dict-key names that map to non-dict values or
    leaf dicts (containing 'level'/'description' but no nested sub-dicts).
    Falls back to top-level keys when a branch has no nested dicts.
    """
    labels: list[str] = []

    def _collect(d: dict) -> None:
        for k, v in d.items():
            if k.startswith("_"):
                continue
            if not isinstance(v, dict):
                continue
            # Check if v has nested dict children (i.e. is a branch)
            children = {ck: cv for ck, cv in v.items()
                        if not ck.startswith("_") and isinstance(cv, dict)}
            # A branch has at least one child whose value is also a dict-of-dicts
            has_deeper = any(
                any(isinstance(gv, dict) for gk, gv in cv.items() if not gk.startswith("_"))
                for cv in children.values()
            )
            if has_deeper:
                _collect(v)
            elif children:
                # Children are leaf dicts — collect their names
                labels.extend(children.keys())
            else:
                # v itself is a leaf (has metadata keys like level/description)
                labels.append(k)

    _collect(taxonomy)
    # Deduplicate preserving order
    seen: set[str] = set()
    unique = []
    for l in labels:
        if l not in seen:
            seen.add(l)
            unique.append(l)
    return unique or list(taxonomy.keys())


def _label_in_group(label: str, group: dict) -> bool:
    """Check if *label* appears as a key anywhere in a nested taxonomy group."""
    for k, v in group.items():
        if k == label:
            return True
        if isinstance(v, dict) and _label_in_group(label, v):
            return True
    return False


# ── Extraction system prompt ──────────────────────────────────────────────

_EXTRACTION_SYSTEM_PROMPT = """\
You are a prompt analyst. Decompose a user instruction into **templates** \
(reusable patterns with {slot} placeholders) and **options** (the concrete values \
that fill those slots).

Output a single JSON object:
{"templates": [{"text": "...", "slots": ["slot_name", ...], "task_type": "...", "level": "..."}],
 "options":   [{"value": "...", "slot": "slot_name", "compatible_task_types": ["...", ...]}]}

TEMPLATES — split one prompt into multiple templates when it mixes task + constraints:
- "text": fixed framing with every variable part replaced by a {slot_name} placeholder.
- "slots": list of placeholder names that appear in "text" (must match exactly).
- "task_type": one label from the provided taxonomy.
- "level": classify by role:
    task_type                → the core task being requested
    format_constraint        → any verifiable/measurable rule on the output: structure (bullets, \
table, JSON), length (word/sentence/paragraph count), casing (all caps, lowercase), \
forbidden words or elements, keyword frequency or inclusion, required markers or phrases, \
output encoding (markdown, JSON block). \
Rule of thumb: if you can verify it by counting or checking the output text → format_constraint.
    content_style_constraint → tone, voice, style, register, or audience \
(e.g. "Shakespearean style", "formal tone", "aimed at children") — \
NOT verifiable by mechanical text inspection.
    process_directive        → instructions about HOW to reason or process \
(step-by-step, chain-of-thought, repeat the request first, …)

OPTIONS — CRITICAL RULES FOR OPTION BOUNDARIES:
- "value": the COMPLETE literal text from the prompt filling this slot. \
The option value must include all contextual framing that makes it a coherent \
reference. For example, "the book 'what a beautiful day'" is ONE option value \
(not just "what a beautiful day"). Similarly, a full paragraph provided as context \
is ONE option value, not multiple fragments.
- "slot": the slot name this value fills (see SLOT NAMING below).
- "compatible_task_types": ALL taxonomy labels where this value could plausibly appear.

LONG CONTENT AS SINGLE OPTIONS:
- When a prompt contains a paragraph, passage, code block, list of choices, or any \
multi-sentence content that serves as input/context for the task, capture the ENTIRE \
content span as a single option value. Do NOT fragment it.
- Use a FUNCTIONAL ROLE slot name (see SLOT NAMING below).
- Examples:
  ✓ Prompt: "Summarize the following: The quick brown fox jumped over the lazy dog. \
It was a sunny day and the fox was feeling adventurous."
    → template: "Summarize the following: {source_passage}"
    → option: slot="source_passage", value="The quick brown fox jumped over the lazy dog. \
It was a sunny day and the fox was feeling adventurous."
  ✓ Prompt: "Classify this text: 'Machine learning is a subset of AI that...'"
    → template: "Classify this text: {source_passage}"
    → option: slot="source_passage", value="Machine learning is a subset of AI that..."
  ✓ Prompt: "Given the following problem: Find the shortest path in a graph..."
    → template: "Given the following problem: {problem_statement}"
    → option: slot="problem_statement", value="Find the shortest path in a graph..."

SLOT NAMING — name each slot with the most SPECIFIC descriptive `snake_case` name \
that captures what KIND of value goes there. Be precise — the name should predict \
what values fit.
- SPECIFIC IS BETTER: `interviewee` is better than `person`, `academic_subject` is \
better than `topic`, `recipe_ingredient` is better than `content`, \
`target_language` is better than `language`.
- FUNCTIONAL ROLE SLOTS for multi-sentence/paragraph content. Use one of these:
  `source_passage` — text to read/analyze/summarize
  `problem_statement` — a task/puzzle/problem to solve
  `code_snippet` — code to analyze/fix/extend
  `example_context` — a worked example or test case
  `factual_background` — encyclopedic or factual background info
  `input_data` — structured input (tables, lists, JSON)
- KEPT CANONICALS for these specific categories:
  `text_type` — text-genre words (essay, poem, letter, report, article, story, summary, \
speech, blog post, review, script, email, memo). ALWAYS extract these as {text_type}.
  `unit` — linguistic/structural units (character, word, sentence, syllable, paragraph, \
line, token). ALWAYS extract these as {unit} when they refer to a swappable unit.
  `number` — any count, quantity, or numeric constraint (300, five, 3-5).
- ALL slot names are SINGULAR. For list-valued slots, emit one option per item \
(all sharing the same slot name), not one option with a comma-separated string. \
Example: keywords "python" and "NLP" → two options both with slot="keyword", \
NOT one option slot="keyword" value="python, NLP".
- CORRECT vs WRONG examples:
  ✓ slot="interviewee" for "the CEO", "a local farmer"
  ✗ slot="person" or "topic" — too generic, can't predict what fits
  ✓ slot="academic_subject" for "organic chemistry", "medieval history"
  ✗ slot="topic" — too generic for a clear academic discipline
  ✓ slot="writing_tone" for "formal", "casual", "sarcastic"
  ✗ slot="formal" with value="formal" — slot must describe the type, not the value
  ✓ slot="text_type" for "essay", "poem", "letter", "paragraph"
  ✗ template "Write an essay about {academic_subject}" with no text_type slot — \
"essay" should be {text_type}
  ✓ slot="unit" for "character", "word", "sentence", "syllable"
  ✗ template "The first character should be uppercase" with no unit slot — \
"character" should be {unit}
  ✓ slot="source_passage" for a full paragraph serving as context
  ✗ slot="text" or "passage" — use the functional role name
  ✓ "Write a summary of the book 'what a beautiful day'" \
    → template: "Write a summary of {literary_work}" \
    → option: slot="literary_work", value="the book 'what a beautiful day'"
- NEVER use a single letter (a, b, p, q, …) — use the full semantic name.
- NEVER use numbered variants (option_1/option_2, n1/n2) — group into one slot.
- NEVER name a slot the same as its value (e.g. slot="formal" value="formal" is wrong; \
use `writing_tone` or `writing_style`).

REUSABILITY IS MANDATORY — every template must be a pattern that works with MANY \
different option values. Ask yourself: "can I plug in 10 different values for each \
slot and still get a coherent, useful prompt?" If not, do NOT extract it.

WHEN NOT TO DECOMPOSE:
- If the segment is a factual paragraph, encyclopedic text, problem statement, \
worked example, or any self-contained block of content, the ENTIRE text is ONE \
option value for a single functional-role slot. Do NOT break it into sub-parts. \
The template should be minimal (e.g. just "{source_passage}" or \
"Given the following context: {factual_background}") with the text as the option.
- Test-case inputs, specific answers, worked-example data, and other values that \
are meaningful ONLY in the context of one specific problem MUST stay as literal \
text — do NOT extract them as separate options.
- Example: a prompt about swapping cards "a, b, c" should keep "abc", "acb", "bac" \
as literal text, NOT extract each permutation as a separate option.
- If a template has a specific sentence fragment that only makes sense for one \
topic (e.g. "The x86 family is a bit different."), it is NOT reusable — make the \
whole thing a {source_passage} or {factual_background} instead.

TIGHTLY-COUPLED SLOTS:
- When multiple values in a prompt are semantically linked (e.g. an author + their \
book title + a quote from that book), they SHOULD be combined into fewer, larger \
slots rather than many small ones. Prefer a functional-role slot ({source_passage}, \
{factual_background}) over separate {author}, {title}, {quote} slots when the values \
are meaningless without each other.
- Rule: if changing one slot value requires changing another to stay coherent, \
merge them into a single slot.

MAXIMIZE GENERALITY — extract as many REUSABLE slots as possible:
- If a specific word or phrase in the template could be replaced with a different \
value to produce a valid prompt of the same type, it SHOULD be a slot.
- TEXT-TYPE WORDS (essay, poem, letter, report, article, paragraph, story, summary, \
speech, blog post, review, script, email, memo) → ALWAYS make these {text_type}.
  ✓ "The essay should be a minimum of {number} words" → \
"The {text_type} should be a minimum of {number} words"
  ✓ "Write an essay about climate change" → "Write a {text_type} about {academic_subject}"
- LINGUISTIC UNIT WORDS (character, word, sentence, syllable, paragraph, line, \
letter, token) → ALWAYS make these {unit} when they refer to a structural unit \
that could be swapped.
  ✓ "The first character should be in uppercase" → \
"The first {unit} should be in {text_casing}"
  ✓ "alternate capitalization of characters" → \
"alternate {text_casing} of {unit}s"
- "How many people in the room are more than six feet tall?" → \
"How many {entity_type} in {location} are more than {number} {measurement_unit}?" \
— use specific slot names that predict what values fit.

Output valid JSON only — no markdown fences, no commentary.\
"""


# ── Dataset-specific few-shot cache ──────────────────────────────────────

_few_shots_cache: dict[str, list[dict]] | None = None


def _load_few_shots() -> dict[str, list[dict]]:
    global _few_shots_cache
    if _few_shots_cache is not None:
        return _few_shots_cache
    if not FEW_SHOT_EXAMPLES_PATH.exists():
        _few_shots_cache = {}
        return _few_shots_cache
    try:
        with open(FEW_SHOT_EXAMPLES_PATH) as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError("Expected a JSON object at the top level")
        _few_shots_cache = data
        logger.info(f"  Loaded few-shot examples for {list(data.keys())}")
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning(f"  Could not load few-shot examples: {exc}")
        _few_shots_cache = {}
    return _few_shots_cache


def _get_few_shot_messages(
    dataset_name: str | None,
    taxonomy_labels: list[str],
) -> list[dict[str, str]]:
    """Return dataset-specific few-shot turns, or [] for zero-shot."""
    if not dataset_name:
        return []
    store = _load_few_shots()
    examples = store.get(dataset_name, [])
    if not examples:
        return []
    labels_str = json.dumps(taxonomy_labels)
    messages: list[dict[str, str]] = []
    for ex in examples:
        prompt_text = ex.get("prompt", "")
        annotation = ex.get("annotation", {})
        if not prompt_text or not annotation:
            continue
        messages.append({
            "role": "user",
            "content": f"Taxonomy labels: {labels_str}\n\nPrompt:\n\"{prompt_text}\"",
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps(annotation, ensure_ascii=False),
        })
    return messages


def _build_messages(
    prompt: str,
    taxonomy_labels: list[str],
    dataset_name: str | None = None,
    segment_hint: str | None = None,
    slot_vocabulary: str = "",
) -> list[dict[str, str]]:
    labels_str = json.dumps(taxonomy_labels)
    few_shot = _get_few_shot_messages(dataset_name, taxonomy_labels)
    hint_line = (
        f"[This segment has been pre-classified as: {segment_hint} — use this as context only, do NOT extract it as a slot or option.]\n\n"
        if segment_hint else ""
    )
    vocab_section = f"\n\n{slot_vocabulary}" if slot_vocabulary else ""
    return [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        *few_shot,
        {"role": "user", "content": f"Taxonomy labels: {labels_str}{vocab_section}\n\n{hint_line}Prompt:\n\"{prompt}\""},
    ]


# ── JSON repair ───────────────────────────────────────────────────────────

def _repair_json(raw: str) -> dict:
    """Best-effort extraction and repair of JSON from LLM output."""
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    brace_start = raw.find("{")
    if brace_start == -1:
        raise ValueError("No JSON object found in LLM output")

    depth = 0
    in_string = False
    escape = False
    end = brace_start
    for i in range(brace_start, len(raw)):
        ch = raw[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    candidate = raw[brace_start: end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    cleaned = candidate
    fixups = [
        lambda s: re.sub(r"(?<=[{,\[])\s*'([^']+?)'\s*:", r' "\1":', s),
        lambda s: re.sub(r",\s*([}\]])", r"\1", s),
        lambda s: re.sub(r'(?<=[{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r' "\1":', s),
        lambda s: re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", s),
    ]
    for fix in fixups:
        cleaned = fix(cleaned)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass

    # Truncation repair
    patched = cleaned
    quotes = re.findall(r'(?<!\\)"', patched)
    if len(quotes) % 2 == 1:
        last_quote = patched.rfind('"')
        truncate_at = max(
            patched.rfind(",", 0, last_quote),
            patched.rfind("[", 0, last_quote),
            patched.rfind("{", 0, last_quote),
        )
        if truncate_at > 0:
            patched = patched[:truncate_at] if patched[truncate_at] == "," else patched[:truncate_at + 1]

    patched = re.sub(r",\s*$", "", patched)
    open_brackets = patched.count("[") - patched.count("]")
    open_braces = patched.count("{") - patched.count("}")
    if open_brackets > 0 or open_braces > 0:
        patched += "]" * max(open_brackets, 0)
        patched += "}" * max(open_braces, 0)
        patched = re.sub(r",\s*([}\]])", r"\1", patched)
        try:
            return json.loads(patched)
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from LLM output (length={len(raw)}): {raw[:200]}…"
    )


# Slot → most likely compatible taxonomy levels.
# ── Slot → level inference ────────────────────────────────────────────────
#
# With specific slot naming, we infer the level from the option taxonomy when
# available (via embedding similarity), or fall back to broad heuristics.

# Minimal keyword hints — used only when no taxonomy is available.
_LEVEL_KEYWORDS: dict[str, list[str]] = {
    "format_constraint": [
        "number", "count", "length", "limit", "unit", "casing",
        "format", "layout", "structure", "frequency",
    ],
    "content_style_constraint": [
        "tone", "style", "voice", "register", "audience", "mood",
        "writing_tone", "writing_style",
    ],
    "process_directive": [
        "reasoning", "chain_of_thought", "step",
    ],
}


def _infer_compatible_types_from_slot(slot: str, taxonomy: dict | None = None) -> list[str]:
    """Infer compatible_task_types from the slot name when the LLM output is invalid.

    Uses keyword hints as a lightweight fallback. When a taxonomy is available,
    returns all leaf labels to be maximally permissive — the option taxonomy's
    compatible_slots mechanism handles precise matching at generation time.
    """
    slot_lower = slot.strip().lower()

    if taxonomy is not None:
        all_leaves = _flatten_taxonomy_labels(taxonomy)
        # Check keyword hints first for a targeted result
        for level, keywords in _LEVEL_KEYWORDS.items():
            if any(kw in slot_lower for kw in keywords):
                if level in taxonomy and isinstance(taxonomy[level], dict):
                    group_leaves = [l for l in all_leaves if _label_in_group(l, taxonomy[level])]
                    if group_leaves:
                        return group_leaves
                return [level]
        # Default: return all task_type leaves (most common level)
        if "task_type" in taxonomy and isinstance(taxonomy["task_type"], dict):
            group_leaves = [l for l in all_leaves if _label_in_group(l, taxonomy["task_type"])]
            if group_leaves:
                return group_leaves
        return all_leaves

    # No taxonomy — use keyword hints
    for level, keywords in _LEVEL_KEYWORDS.items():
        if any(kw in slot_lower for kw in keywords):
            return [level]
    return ["task_type"]


def _parse_llm_extraction(
    raw: str,
    taxonomy: dict,
    *,
    merge_numbered: bool = False,
) -> tuple[list[TaskTemplate], list[Option]]:
    """Parse JSON from LLM into TaskTemplate / Option objects with normalization."""
    data = _repair_json(raw)
    raw_templates = data.get("templates", [])
    raw_options = data.get("options", [])

    raw_templates, raw_options = _normalize_slot_names(raw_templates, raw_options)
    raw_options = _detect_self_replicating_slots(raw_options, raw_templates)
    raw_options = _expand_list_options(raw_options)
    if merge_numbered:
        raw_templates, raw_options = _merge_numbered_slots(raw_templates, raw_options)
        raw_templates = _drop_duplicate_slot_templates(raw_templates)
    raw_options = _remove_broken_options(raw_options)

    valid_levels = set(LEVELS)
    valid_task_types = set(_flatten_taxonomy_labels(taxonomy))

    templates: list[TaskTemplate] = []
    options: list[Option] = []

    for t in raw_templates:
        task_type = t.get("task_type", "unknown")
        level = t.get("level", "task_type")
        text = t.get("text", "")
        slots = t.get("slots", [])

        if task_type not in valid_task_types:
            logger.debug(f"  Skipping unknown task_type '{task_type}'")
            continue

        # Remap old level names if present
        level = LEVEL_REMAP.get(level, level)
        if level not in valid_levels:
            # Infer level from the top-level taxonomy group this label belongs to
            inferred_level = "task_type"
            for group_name in taxonomy:
                if group_name.startswith("_"):
                    continue
                group = taxonomy[group_name]
                if isinstance(group, dict) and _label_in_group(task_type, group):
                    inferred_level = group_name
                    break
            level = inferred_level

        if level == "task_type" and not slots:
            logger.warning(f"  task_type template with 0 slots: '{text[:80]}…'")
        if len(slots) > 6:
            logger.warning(f"  Template with {len(slots)} slots (may be over-fragmented): '{text[:80]}…'")

        fixed_text = text
        for s in slots:
            fixed_text = fixed_text.replace(f"{{{s}}}", "")
        tlen = token_length(fixed_text)

        tid = template_id(task_type, text)
        templates.append(TaskTemplate(
            id=tid, text=text, slots=slots, task_type=task_type,
            level=level, token_length=tlen, source="llm_extracted",
        ))

    for o in raw_options:
        value = o.get("value", "")
        slot = o.get("slot", "")
        compat_types = o.get("compatible_task_types", [])

        if slot and value and slot.lower() == value.lower().strip():
            logger.warning(f"  Slot name equals value: slot='{slot}', value='{value}'")

        compat_types = [ct for ct in compat_types if ct in valid_task_types]
        if not compat_types:
            # Infer from slot semantics rather than giving up with _universal
            compat_types = _infer_compatible_types_from_slot(slot, taxonomy)

        oid = option_id(slot, value)
        options.append(Option(
            id=oid, value=value, slot=slot,
            compatible_task_types=compat_types,
            compatible_templates=[t.id for t in templates if slot in t.slots],
            token_length=token_length(value),
            source="llm_extracted",
        ))

    return templates, options


# ── Reusability filter ────────────────────────────────────────────────────

def _filter_non_reusable(
    templates: list[TaskTemplate],
    options: list[Option],
) -> tuple[list[TaskTemplate], list[Option]]:
    """Remove templates that are not reusable patterns.

    Catches:
    - Templates where the fixed (non-slot) text is too specific to one topic
      (high ratio of fixed text to slot text, with domain-specific words)
    - Templates that are just "{slot}" with no framing at all
    - Templates with very long fixed text that's clearly a passage fragment
    """
    kept_templates: list[TaskTemplate] = []
    removed_ids: set[str] = set()

    for t in templates:
        text = t.text.strip()
        # Remove slot placeholders to get the fixed framing
        fixed = re.sub(r"\{[\w]+\}", "", text).strip()
        fixed_words = fixed.split()

        # Template is ONLY a slot placeholder — not a useful pattern
        if not fixed and len(t.slots) == 1:
            removed_ids.add(t.id)
            logger.info(f"  Dropping bare-slot template: {text!r}")
            continue

        # Fixed text is very long (>20 words) — likely a passage fragment,
        # not a reusable pattern
        if len(fixed_words) > 20:
            removed_ids.add(t.id)
            logger.info(
                f"  Dropping non-reusable template (fixed text too long: "
                f"{len(fixed_words)} words): {text[:80]!r}…"
            )
            continue

        kept_templates.append(t)

    if not removed_ids:
        return templates, options

    # Remove options that were ONLY linked to removed templates
    kept_options: list[Option] = []
    for o in options:
        o.compatible_templates = [
            tid for tid in o.compatible_templates if tid not in removed_ids
        ]
        kept_options.append(o)

    return kept_templates, kept_options


def _filter_undercovered_templates(
    templates: list[TaskTemplate],
    options: list[Option],
) -> tuple[list[TaskTemplate], list[Option]]:
    """Remove templates where all slots are singletons (only 1 option value).

    A template whose every slot has exactly one known value cannot produce any
    variation during generation — it is effectively a hard-coded example (e.g.
    a specific author/title/quote triple) rather than a reusable pattern.
    """
    # Build slot → set of distinct option values
    slot_values: dict[str, set[str]] = defaultdict(set)
    for o in options:
        slot_values[o.slot].add(o.value)

    kept: list[TaskTemplate] = []
    removed_ids: set[str] = set()

    for t in templates:
        if not t.slots:
            kept.append(t)
            continue
        # Count how many slots have >1 distinct value
        variable_slots = sum(
            1 for s in t.slots
            if len(slot_values.get(s, set())) > 1
        )
        if variable_slots == 0:
            # Every slot has 0 or 1 option → no variation possible
            removed_ids.add(t.id)
            logger.info(
                f"  Dropping singleton-slot template (0/{len(t.slots)} "
                f"slots have >1 option): {t.text[:80]!r}"
            )
        else:
            kept.append(t)

    if not removed_ids:
        return templates, options

    logger.info(f"  Singleton-slot filter removed {len(removed_ids)} templates")

    # Clean up option→template links
    for o in options:
        o.compatible_templates = [
            tid for tid in o.compatible_templates if tid not in removed_ids
        ]

    return kept, options


# ── Merge helper ──────────────────────────────────────────────────────────

def _merge_into_store(
    all_templates: dict[str, TaskTemplate],
    all_options: dict[str, Option],
    tmpls: list[TaskTemplate],
    opts: list[Option],
) -> None:
    for t in tmpls:
        if t.id not in all_templates:
            all_templates[t.id] = t
    for o in opts:
        if o.id not in all_options:
            all_options[o.id] = o
        else:
            existing = all_options[o.id]
            for tt in o.compatible_task_types:
                if tt not in existing.compatible_task_types:
                    existing.compatible_task_types.append(tt)
            for tid in o.compatible_templates:
                if tid not in existing.compatible_templates:
                    existing.compatible_templates.append(tid)


# ── Public extraction API ─────────────────────────────────────────────────

def extract_templates_from_prompt_llm(
    prompt: str,
    taxonomy: dict,
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str | None = None,
    dataset_name: str | None = None,
) -> tuple[list[TaskTemplate], list[Option]]:
    """Decompose a single prompt into templates and options via LLM."""
    taxonomy_labels = _flatten_taxonomy_labels(taxonomy)
    messages = _build_messages(prompt, taxonomy_labels, dataset_name)
    try:
        raw = _local_generate(
            model, messages,
            temperature=0.0, device=device,
            enable_thinking=False, json_schema=ExtractionResult,
        )
        return _parse_llm_extraction(raw, taxonomy)
    except Exception as e:
        logger.warning(f"  LLM extraction failed: {e}")
        return [], []


def extract_templates_from_dataset(
    prompts: list[dict],
    taxonomy: dict,
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str | None = None,
    batch_size: int = 32,
    gpu_memory_utilization: float = 0.7,
    skip_segmentation: bool = False,
) -> tuple[list[TaskTemplate], list[Option]]:
    """
    Extract templates and options from a list of prompt records.

    Each record must have a "prompt" field; optional "source" field enables
    dataset-specific few-shot examples. Uses batched vLLM inference.

    Pipeline: Stage A (segmentation) → Stage B (LLM extraction per segment).

    skip_segmentation
        When True, load Stage A results from segments.jsonl (if it exists)
        instead of re-running the segmenter. Useful to iterate on Stage B
        without paying the segmentation cost again.
    """
    return _extract_segmented_llm(
        prompts, taxonomy,
        model=model, device=device, batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
        skip_segmentation=skip_segmentation,
    )


def _load_segments_from_disk() -> list[list] | None:
    """Load segments from SEGMENTS_PATH if it exists. Returns None if not found."""
    from dataset.schema import Segment
    if not SEGMENTS_PATH.exists():
        return None
    segments_by_prompt: dict[str, list] = {}
    with open(SEGMENTS_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            seg = Segment(
                span_text=d["span_text"],
                taxonomy_label=d["taxonomy_label"],
                level=d["level"],
                source_prompt=d["source_prompt"],
                classification_method=d["classification_method"],
            )
            segments_by_prompt.setdefault(seg.source_prompt, []).append(seg)
    return list(segments_by_prompt.values())


def _extract_segmented_llm(
    prompts: list[dict],
    taxonomy: dict,
    *,
    model: str,
    device: str | None,
    batch_size: int,
    gpu_memory_utilization: float = 0.7,
    skip_segmentation: bool = False,
) -> tuple[list[TaskTemplate], list[Option]]:
    """Segmentation → LLM extraction per segment.

    Stage A: segment and classify all prompts (via segmenter.py).
             Skipped if skip_segmentation=True and segments.jsonl already exists.
    Stage B: for each segment, extract templates/options via batched LLM calls.
    """
    from dataset.segmenter import segment_and_classify_batch
    from dataset.taxonomy.collect_task_types import Taxonomy

    taxonomy_obj = Taxonomy.from_dict(taxonomy)

    all_templates: dict[str, TaskTemplate] = {}
    all_options: dict[str, Option] = {}

    valid_records = [r for r in prompts if r.get("prompt", "").strip()]
    prompt_texts = [r["prompt"] for r in valid_records]
    source_map = {r["prompt"]: r.get("source") for r in valid_records}

    # ── Stage A: segment and classify all prompts ─────────────────────
    if skip_segmentation:
        cached = _load_segments_from_disk()
        if cached is not None:
            logger.info(
                f"  Skipping segmentation — loaded {sum(len(s) for s in cached)} "
                f"segments from {SEGMENTS_PATH}"
            )
            all_prompt_segments = cached
        else:
            logger.warning(
                f"  --skip-segmentation requested but {SEGMENTS_PATH} not found; "
                f"running segmentation."
            )
            skip_segmentation = False

    if not skip_segmentation:
        logger.info(
            f"Segmented LLM extraction: segmenting {len(prompt_texts)} prompts "
            f"(model={model})…"
        )
        all_prompt_segments = segment_and_classify_batch(
            prompt_texts, taxonomy_obj,
            model=model, device=device, batch_size=batch_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )
        with open(SEGMENTS_PATH, "w", encoding="utf-8") as f:
            for segments in all_prompt_segments:
                for seg in segments:
                    f.write(json.dumps({
                        "span_text": seg.span_text,
                        "taxonomy_label": seg.taxonomy_label,
                        "level": seg.level,
                        "source_prompt": seg.source_prompt,
                        "classification_method": seg.classification_method,
                    }, ensure_ascii=False) + "\n")
        logger.info(f"  Segmentation results saved to {SEGMENTS_PATH}")

    # ── Stage B: LLM extraction per segment (batched) ─────────────────
    # The option taxonomy starts from any prior run saved on disk, then grows
    # batch by batch: after each batch, newly extracted options are added to
    # the in-memory taxonomy and the updated context is used for the next batch.
    from dataset.option_taxonomy import OptionTaxonomy
    opt_taxonomy = OptionTaxonomy.load()
    if opt_taxonomy.types:
        logger.info(
            f"  Option taxonomy loaded from disk: {len(opt_taxonomy.types)} types."
        )

    # Flatten all segments
    taxonomy_labels = _flatten_taxonomy_labels(taxonomy)
    flat_segments = []

    for segments in all_prompt_segments:
        for seg in segments:
            if not seg.span_text.strip():
                continue
            flat_segments.append(seg)

    logger.info(
        f"  Extracting templates from {len(flat_segments)} segments "
        f"via LLM (batch_size={batch_size})…"
    )

    # Batch LLM calls — taxonomy context is rebuilt after each batch
    failures = 0
    n_batches = (len(flat_segments) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(flat_segments), batch_size), total=n_batches, desc="Extracting", unit="batch")
    for batch_start in pbar:
        batch_segs = flat_segments[batch_start: batch_start + batch_size]
        new_opts_this_seg: list[Option] = []

        # Per-segment taxonomy retrieval — each segment gets relevant types
        batch_msgs = [
            _build_messages(
                seg.span_text,
                taxonomy_labels,
                source_map.get(seg.source_prompt),
                segment_hint=f"{seg.taxonomy_label} ({seg.level})",
                slot_vocabulary=opt_taxonomy.to_prompt_context(
                    query=seg.span_text,
                ),
            )
            for seg in batch_segs
        ]

        try:
            raws = _local_generate_batch(
                model, batch_msgs,
                temperature=0.0, device=device,
                enable_thinking=False, json_schema=ExtractionResult,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except Exception as e:
            logger.warning(f"  Batch extraction failed: {e}")
            raws = [""] * len(batch_msgs)

        for seg, raw in zip(batch_segs, raws):
            if not raw:
                failures += 1
                continue
            try:
                tmpls, opts = _parse_llm_extraction(
                    raw, taxonomy, merge_numbered=True,
                )
                # Override level/task_type from segmentation when LLM doesn't match
                for t in tmpls:
                    if t.task_type not in taxonomy:
                        t.task_type = seg.taxonomy_label
                    if t.level not in set(LEVELS):
                        t.level = seg.level
            except Exception as e:
                logger.warning(f"  Parse failed for segment: {e}")
                failures += 1
                continue
            tmpls, opts = _filter_non_reusable(tmpls, opts)
            if not tmpls and not opts:
                failures += 1
            _merge_into_store(all_templates, all_options, tmpls, opts)
            new_opts_this_seg.extend(opts)

        # Update in-memory taxonomy with options extracted from this batch,
        # so the next batch sees the most current slot vocabulary as context
        opt_taxonomy.update_from_options(new_opts_this_seg)

    logger.info(
        f"Segmented LLM extraction done: {len(all_templates)} templates, "
        f"{len(all_options)} options, {failures} failures"
    )
    final_t, final_o = list(all_templates.values()), list(all_options.values())
    final_t, final_o = _filter_undercovered_templates(final_t, final_o)
    return final_t, final_o


def _extract_llm_single_pass(
    prompts: list[dict],
    taxonomy: dict,
    *,
    model: str,
    device: str | None,
    batch_size: int,
    gpu_memory_utilization: float = 0.7,
) -> tuple[list[TaskTemplate], list[Option]]:
    """Original single-pass LLM extraction (A/B test path)."""
    all_templates: dict[str, TaskTemplate] = {}
    all_options: dict[str, Option] = {}
    taxonomy_labels = _flatten_taxonomy_labels(taxonomy)
    failures = 0

    valid_records = [r for r in prompts if r.get("prompt", "").strip()]
    all_msgs = [
        _build_messages(r["prompt"], taxonomy_labels, dataset_name=r.get("source"))
        for r in valid_records
    ]

    logger.info(
        f"Sending {len(all_msgs)} prompts to vLLM for extraction "
        f"(model={model})…"
    )
    try:
        raws = _local_generate_batch(
            model, all_msgs,
            temperature=0.0, device=device,
            enable_thinking=False, json_schema=ExtractionResult,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    except Exception as e:
        logger.warning(f"  Batch extraction failed: {e}")
        raws = [""] * len(valid_records)

    for raw in tqdm(raws, desc="Parsing extractions", unit="prompt"):
        if not raw:
            failures += 1
            continue
        try:
            tmpls, opts = _parse_llm_extraction(raw, taxonomy, merge_numbered=True)
        except Exception as e:
            logger.warning(f"  Parse failed: {e}")
            failures += 1
            continue
        if not tmpls and not opts:
            failures += 1
        _merge_into_store(all_templates, all_options, tmpls, opts)

    logger.info(
        f"Extraction done: {len(all_templates)} templates, "
        f"{len(all_options)} options, {failures} failures"
    )
    final_t, final_o = list(all_templates.values()), list(all_options.values())
    final_t, final_o = _filter_undercovered_templates(final_t, final_o)
    return final_t, final_o


# reclassify_exotic_slots removed — with specific slot naming, we no longer
# collapse exotic names back to generic canonicals. The option taxonomy
# handles slot grouping instead.

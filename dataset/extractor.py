"""
extractor.py — Few-shot LLM extraction of task templates and options.

Decomposes raw instruction-tuning prompts into TaskTemplate / Option objects
using a local model with structured JSON output.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from typing import Any

from tqdm import tqdm

from dataset.schema import (
    LEVELS, LEVEL_REMAP,
    TaskTemplate, Option,
    ExtractionResult,
    template_id, option_id,
    FEW_SHOT_EXAMPLES_PATH,
)
from dataset.token_counter import token_length
from dataset.local_llm import generate_text as _local_generate, generate_text_batch as _local_generate_batch

logger = logging.getLogger(__name__)

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

OPTIONS:
- "value": the literal text from the prompt filling this slot — never a {placeholder}.
- "slot": the canonical slot name this value fills.
- "compatible_task_types": ALL taxonomy labels where this value could plausibly appear.

SLOT NAMING — strictly follow these rules:
- Preferred names: `topic`, `description`, `text`, `passage`, `question`, `context`, \
`source`, `code`, `keywords`, `options`, `completions`, `language`, `programming_language`, \
`text_type`, `person`, `n` (for any number).
- NEVER use a single letter (a, b, p, q, …) — use the full semantic name.
- NEVER use numbered variants (option_1/option_2, n1/n2) — group into one slot \
(e.g. `options`, `number_list`).
- NEVER name a slot the same as its value (e.g. slot="formal" value="formal" is wrong; \
use `tone` or `style`).
- Use `{code}` for full code blocks, not just variable names.

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
) -> list[dict[str, str]]:
    labels_str = json.dumps(taxonomy_labels)
    few_shot = _get_few_shot_messages(dataset_name, taxonomy_labels)
    return [
        {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
        *few_shot,
        {"role": "user", "content": f"Taxonomy labels: {labels_str}\n\nPrompt:\n\"{prompt}\""},
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


# ── Post-processing normalization ─────────────────────────────────────────

_EXACT_SLOT_CANONICAL_MAP: dict[str, str] = {
    "value": "n",
    "total": "n",
    "amount": "n",
    "count": "n",
    "number": "n",
}

_TRAILING_NUMBERED_SLOT_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9]*?)(?:_)?(\d+)$")
_NUMBERED_N_SLOT_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9]*?)?(\d+)$")


def _canonicalize_slot_name(slot: str) -> str | None:
    """Return a canonical slot name for known alias patterns."""
    if slot in _EXACT_SLOT_CANONICAL_MAP:
        return _EXACT_SLOT_CANONICAL_MAP[slot]

    match = _NUMBERED_N_SLOT_RE.fullmatch(slot)
    if match:
        return match.group(1)

    match = _TRAILING_NUMBERED_SLOT_RE.fullmatch(slot)
    if match:
        return match.group(1)

    return None

_SINGLE_LETTER_MAP: dict[str, str] = {
    "p": "passage", "s": "sentence", "t": "topic",
    "a": "option", "b": "option", "c": "option", "d": "option", "e": "option",
    "f": "function_code", "g": "group", "h": "heading",
    "i": "item", "j": "item", "l": "language", "n": "number",
    "q": "question", "v": "value", "w": "word",
}
_SINGLE_LETTER_MAP.update({k.upper(): v for k, v in _SINGLE_LETTER_MAP.items()})

_TASK_TYPE_TO_DEFAULT_SLOT: dict[str, str] = {
    # task_type entries
    "question_answering": "topic",
    "fact_verification": "topic",
    "information_extraction": "topic",
    "summarization": "topic",
    "mathematical_reasoning": "topic",
    "logical_deductive_reasoning": "topic",
    "commonsense_reasoning": "topic",
    "argumentation": "topic",
    "prediction": "topic",
    "creative_writing": "topic",
    "text_completion": "topic",
    "dialogue_generation": "topic",
    "translation": "topic",
    "rewriting_paraphrasing": "topic",
    "communication_writing": "topic",
    "classification": "category",
    "ranking_comparison": "topic",
    "data_analysis": "topic",
    "code_generation": "programming_task",
    "conversion": "topic",
    "planning": "topic",
    "brainstorming": "topic",
    "role_playing": "topic",
    "explanation": "topic",
    # format_constraint / content_style_constraint entries
    "length_constraint": "n",
    "structure_constraint": "n",
    "keyword_inclusion": "keywords",
    "keyword_frequency": "keyword",
    "forbidden_words": "words",
    "response_language": "language",
    "casing_constraint": "topic",
    "tone_constraint": "topic",
    "audience_constraint": "topic",
}


def _normalize_slot_names(
    templates: list[dict], options: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Rename known bad slot names to canonical forms."""
    all_slots: set[str] = set()
    for t in templates:
        all_slots.update(t.get("slots", []))
    for o in options:
        s = o.get("slot", "")
        if s:
            all_slots.add(s)

    # Build a slot rename map from known canonical and single-letter aliases.
    rename: dict[str, str] = {}
    for slot in all_slots:
        canonical_slot = _canonicalize_slot_name(slot)
        if canonical_slot is not None:
            rename[slot] = canonical_slot
        elif len(slot) == 1 and slot in _SINGLE_LETTER_MAP:
            rename[slot] = _SINGLE_LETTER_MAP[slot]

    if not rename:
        return templates, options

    # Rename mapped slots in templates, including many-to-one merges.
    rename_count = 0
    for t in templates:
        t_slots = t.get("slots", [])
        applicable = {s: rename[s] for s in t_slots if s in rename}
        if not applicable:
            continue
        text = t.get("text", "")
        for old, new in applicable.items():
            text = text.replace(f"{{{old}}}", f"{{{new}}}")

        # Keep slot names unique while preserving first-seen order.
        new_slots: list[str] = []
        seen: set[str] = set()
        for s in t_slots:
            mapped = applicable.get(s, s)
            if mapped not in seen:
                new_slots.append(mapped)
                seen.add(mapped)

        t["slots"] = new_slots
        t["text"] = text
        rename_count += len(applicable)

    # Rename all mapped option slots; multiple options can share one slot name.
    opt_rename_count = 0
    for o in options:
        slot = o.get("slot", "")
        if slot in rename:
            o["slot"] = rename[slot]
            opt_rename_count += 1

    # Emit a summary only when at least one rename occurred.
    if rename_count or opt_rename_count:
        logger.info(
            f"  Slot normalization: {rename_count} template, {opt_rename_count} option renames"
        )
    return templates, options


def _detect_self_replicating_slots(
    options: list[dict], templates: list[dict],
) -> list[dict]:
    """Fix options where slot name equals the value."""
    template_task_types = [t.get("task_type", "") for t in templates]
    for o in options:
        slot = o.get("slot", "")
        value = o.get("value", "")
        if not slot or not value:
            continue
        value_lower = value.lower().strip()
        is_self_rep = (
            slot.lower() == value_lower
            or (len(value.split()) < 3 and slot.lower() in value_lower)
        )
        if not is_self_rep:
            continue
        new_slot = None
        for tt in (o.get("compatible_task_types", []) + template_task_types):
            if tt in _TASK_TYPE_TO_DEFAULT_SLOT:
                candidate = _TASK_TYPE_TO_DEFAULT_SLOT[tt]
                if candidate.lower() != value_lower:
                    new_slot = candidate
                    break
        if not new_slot:
            new_slot = "content"
        logger.info(f"  Self-replicating slot: '{slot}'='{value}' → '{new_slot}'")
        for t in templates:
            t_slots = t.get("slots", [])
            if slot in t_slots:
                t["slots"] = [new_slot if s == slot else s for s in t_slots]
                t["text"] = t.get("text", "").replace(f"{{{slot}}}", f"{{{new_slot}}}")
        o["slot"] = new_slot
    return options


def _merge_numbered_slots(
    templates: list[dict], options: list[dict], *, join_values: bool = True,
) -> tuple[list[dict], list[dict]]:
    """Merge numbered slot variants (n1, n2 → number_list) into single compound slots."""
    numbered_re = re.compile(r"^(.+?)(\d+)$")
    base_groups: dict[str, list[str]] = defaultdict(list)

    all_slots: set[str] = set()
    for t in templates:
        all_slots.update(t.get("slots", []))

    for slot in sorted(all_slots):
        m = numbered_re.match(slot)
        if m:
            base = m.group(1).rstrip("_")
            base_groups[base].append(slot)

    merge_map: dict[str, str] = {}
    merged_bases: dict[str, str] = {}
    for base, slots in base_groups.items():
        if len(slots) < 2:
            continue
        if base in ("n", ""):
            merged_name = "number_list"
        else:
            merged_name = f"{base}_list" if not base.endswith("s") else base
        merged_bases[base] = merged_name
        for s in slots:
            merge_map[s] = merged_name

    if not merge_map:
        return templates, options

    logger.info(
        f"  Merging numbered slots: {len(merge_map)} slots across {len(merged_bases)} bases"
    )

    for t in templates:
        t_slots = t.get("slots", [])
        slots_to_merge = [s for s in t_slots if s in merge_map]
        if not slots_to_merge:
            continue
        text = t.get("text", "")
        new_slots: list[str] = []
        seen_merged: set[str] = set()
        for s in t_slots:
            if s in merge_map:
                merged = merge_map[s]
                if merged not in seen_merged:
                    new_slots.append(merged)
                    seen_merged.add(merged)
                    text = text.replace(f"{{{s}}}", f"{{{merged}}}", 1)
                else:
                    for pattern in [f", {{{s}}}", f"{{{s}}}, ", f"{{{s}}}"]:
                        if pattern in text:
                            text = text.replace(pattern, "", 1)
                            break
            else:
                new_slots.append(s)
        t["slots"] = new_slots
        t["text"] = text

    merged_options_by_target: dict[str, list[dict]] = defaultdict(list)
    kept_options: list[dict] = []
    for o in options:
        if o.get("slot", "") in merge_map:
            merged_options_by_target[merge_map[o["slot"]]].append(o)
        else:
            kept_options.append(o)

    if join_values:
        for target, group in merged_options_by_target.items():
            joined_value = ", ".join(o.get("value", "") for o in group)
            merged_opt = dict(group[0])
            merged_opt["slot"] = target
            merged_opt["value"] = joined_value
            all_types: list[str] = []
            for o in group:
                for tt in o.get("compatible_task_types", []):
                    if tt not in all_types:
                        all_types.append(tt)
            merged_opt["compatible_task_types"] = all_types
            kept_options.append(merged_opt)
    else:
        for target, group in merged_options_by_target.items():
            for o in group:
                o["slot"] = target
                kept_options.append(o)

    return templates, kept_options


def _sync_template_slots(templates: list[dict]) -> list[dict]:
    """Ensure slot lists match {placeholder} references in template text."""
    placeholder_re = re.compile(r"\{(\w+)\}")
    added_total = removed_total = 0
    for t in templates:
        slots = list(t.get("slots", []))
        text = t.get("text", "")
        placeholders = set(placeholder_re.findall(text))
        slot_set = set(slots)
        missing = placeholders - slot_set
        phantom = slot_set - placeholders
        if missing or phantom:
            new_slots = [s for s in slots if s not in phantom]
            new_slots.extend(sorted(missing))
            t["slots"] = new_slots
            added_total += len(missing)
            removed_total += len(phantom)
    if added_total or removed_total:
        logger.info(f"  Slot sync: added {added_total}, removed {removed_total} phantom slots")
    return templates


def _remove_broken_options(options: list[dict]) -> list[dict]:
    """Remove options whose value contains unresolved {placeholder} patterns."""
    placeholder_re = re.compile(r"\{\w+\}")
    kept = [o for o in options if not placeholder_re.search(o.get("value", ""))]
    removed = len(options) - len(kept)
    if removed:
        logger.info(f"  Removed {removed} options with placeholder values")
    return kept


# ── Parse LLM extraction output ───────────────────────────────────────────

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
    #_detect_self_replicating_slots(raw_options, raw_templates)
    if merge_numbered:
        raw_templates, raw_options = _merge_numbered_slots(raw_templates, raw_options)

    valid_levels = set(LEVELS)
    valid_task_types = set(taxonomy.keys())

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
            level = taxonomy.get(task_type, {}).get("level", "task_type")

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
            compat_types = ["_universal"]

        oid = option_id(slot, value)
        options.append(Option(
            id=oid, value=value, slot=slot,
            compatible_task_types=compat_types,
            compatible_templates=[t.id for t in templates if slot in t.slots],
            token_length=token_length(value),
            source="llm_extracted",
        ))

    return templates, options


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
    taxonomy_labels = list(taxonomy.keys())
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
) -> tuple[list[TaskTemplate], list[Option]]:
    """
    Extract templates and options from a list of prompt records.

    Each record must have a "prompt" field; optional "source" field enables
    dataset-specific few-shot examples. Uses batched vLLM inference.
    """
    all_templates: dict[str, TaskTemplate] = {}
    all_options: dict[str, Option] = {}
    taxonomy_labels = list(taxonomy.keys())
    failures = 0

    # Filter valid prompts and build all messages upfront
    valid_records = [r for r in prompts if r.get("prompt", "").strip()]
    all_msgs = [
        _build_messages(r["prompt"], taxonomy_labels, dataset_name=r.get("source"))
        for r in valid_records
    ]

    # Send everything to vLLM at once — its internal scheduler handles
    # continuous batching far more efficiently than manual chunking.
    logger.info(
        f"Sending {len(all_msgs)} prompts to vLLM for extraction "
        f"(model={model})…"
    )
    try:
        raws = _local_generate_batch(
            model, all_msgs,
            temperature=0.0, device=device,
            enable_thinking=False, json_schema=ExtractionResult,
        )
    except Exception as e:
        logger.warning(f"  Batch extraction failed: {e}")
        raws = [""] * len(valid_records)

    # Post-process results
    for raw in tqdm(raws, desc="Parsing extractions", unit="prompt"):
        if not raw:
            failures += 1
            continue
        try:
            tmpls, opts = _parse_llm_extraction(raw, taxonomy)
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
    return list(all_templates.values()), list(all_options.values())


# ── Normalization helpers (for --normalize-existing) ──────────────────────

def normalize_existing(
    raw_templates: list[dict],
    raw_options: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Run the full normalization pipeline on raw template/option dicts.

    Used by the --normalize-existing CLI flag to clean up previously
    extracted data without re-running LLM inference.
    """
    from dataset.schema import template_id as _tid, option_id as _oid

    raw_templates, raw_options = _normalize_slot_names(raw_templates, raw_options)
    _detect_self_replicating_slots(raw_options, raw_templates)
    raw_templates, raw_options = _merge_numbered_slots(
        raw_templates, raw_options, join_values=False,
    )
    raw_templates = _sync_template_slots(raw_templates)
    raw_options = _remove_broken_options(raw_options)

    # Deduplicate templates
    seen_texts: dict[str, str] = {}
    removed_ids: set[str] = set()
    kept_templates: list[dict] = []
    for t in raw_templates:
        text = t.get("text", "")
        tid = t.get("id", "")
        if text in seen_texts:
            removed_ids.add(tid)
        else:
            seen_texts[text] = tid
            kept_templates.append(t)
    if removed_ids:
        for o in raw_options:
            compat = o.get("compatible_templates", [])
            if any(tid in removed_ids for tid in compat):
                o["compatible_templates"] = [tid for tid in compat if tid not in removed_ids]
        logger.info(f"  Deduplicated: removed {len(removed_ids)} duplicate templates")
    raw_templates = kept_templates

    # Remove orphaned / empty options
    template_slots: set[str] = set()
    for t in raw_templates:
        template_slots.update(t.get("slots", []))
    orphaned = empty = 0
    kept_options: list[dict] = []
    for o in raw_options:
        value = o.get("value", "")
        if not value.strip():
            empty += 1
            continue
        if o.get("slot", "") not in template_slots:
            orphaned += 1
            continue
        kept_options.append(o)
    if orphaned or empty:
        logger.info(f"  Removed {orphaned} orphaned options, {empty} empty-value options")
    raw_options = kept_options

    # Deduplicate options
    seen_opts: dict[tuple[str, str], dict] = {}
    final_options: list[dict] = []
    removed_opts = 0
    for o in raw_options:
        key = (o.get("slot", ""), o.get("value", ""))
        if key in seen_opts:
            existing = seen_opts[key]
            for tt in o.get("compatible_task_types", []):
                if tt not in existing.get("compatible_task_types", []):
                    existing.setdefault("compatible_task_types", []).append(tt)
            for tid in o.get("compatible_templates", []):
                if tid not in existing.get("compatible_templates", []):
                    existing.setdefault("compatible_templates", []).append(tid)
            removed_opts += 1
        else:
            seen_opts[key] = o
            final_options.append(o)
    if removed_opts:
        logger.info(f"  Deduplicated: removed {removed_opts} duplicate options")
    raw_options = final_options

    # Rebuild IDs
    for t in raw_templates:
        t["id"] = _tid(t.get("task_type", ""), t.get("text", ""))
    for o in raw_options:
        o["id"] = _oid(o.get("slot", ""), o.get("value", ""))

    return raw_templates, raw_options

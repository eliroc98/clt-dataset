"""
fix_slots.py — Slot and template normalization pipeline.

Houses all data and functions for canonicalising slot names, fixing template
levels, merging numbered variants, deduplicating, and cleaning options.
Serves two roles:

  1. Normalization module imported by extractor.py so that every extraction
     pass immediately benefits from the full cleaning pipeline.
  2. Standalone CLI for normalizing existing templates.json / options.json
     (and few_shot_examples.json) without re-running LLM inference.

Run from repo root:
    python -m dataset.fix_slots
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from pathlib import Path

from dataset.schema import template_id as _tid, option_id as _oid

logger = logging.getLogger(__name__)

DATASET_DIR = Path(__file__).parent
OUTPUT_DIR = DATASET_DIR / "output"
TEMPLATES_PATH = OUTPUT_DIR / "templates.json"
OPTIONS_PATH = OUTPUT_DIR / "options.json"
FEW_SHOT_PATH = DATASET_DIR / "few_shot_examples.json"


# ── Canonical slot names ───────────────────────────────────────────────────

CANONICAL_PREFERRED_SLOTS: frozenset[str] = frozenset({
    "topic", "description", "text", "passage", "question", "context",
    "source", "code", "keyword", "option", "completion", "language",
    "programming_language", "text_type", "person", "number",
    # extended canonical names accepted during normalization
    "tone", "style", "format", "role", "category", "title",
    "subject", "task", "content", "example",
    # NE-derived slot names (from extractor._NE_TO_SLOT)
    "date", "location", "event", "law", "cardinal", "ordinal",
    "quantity", "percent", "money", "time",
    "country_city_state", "nationality_religion_political_group",
    "building_airport_highway_bridge", "object_vehicle_food",
    "book_song_art", "company_agency_institution",
})
_CANONICAL_PREFERRED_SLOTS = CANONICAL_PREFERRED_SLOTS   # private alias


# ── Exact-alias slot canonicalization ─────────────────────────────────────

_EXACT_SLOT_CANONICAL_MAP: dict[str, str] = {
    "n":      "number",
    "value":  "number",
    "total":  "number",
    "amount": "number",
    "count":  "number",
}

_TRAILING_NUMBERED_SLOT_RE = re.compile(r"^([a-zA-Z][a-zA-Z0-9_]*?)(?:_)?(\d+)$")
_NUMBERED_N_SLOT_RE         = re.compile(r"^([a-zA-Z][a-zA-Z0-9_]*?)?(\d+)$")


def _canonicalize_slot_name(slot: str) -> str | None:
    """Return a canonical slot name for known alias patterns, or None if unchanged."""
    if slot in _EXACT_SLOT_CANONICAL_MAP:
        return _EXACT_SLOT_CANONICAL_MAP[slot]

    m = _NUMBERED_N_SLOT_RE.fullmatch(slot)
    if m:
        prefix = m.group(1)
        return prefix.rstrip("_") if prefix else None

    m = _TRAILING_NUMBERED_SLOT_RE.fullmatch(slot)
    if m:
        prefix = m.group(1)
        return prefix.rstrip("_") if prefix else None

    return None


# ── Single-letter slot canonicalization ───────────────────────────────────

_SINGLE_LETTER_MAP: dict[str, str] = {
    "p": "passage",  "s": "sentence", "t": "topic",
    "a": "option",   "b": "option",   "c": "option",   "d": "option",   "e": "option",
    "f": "function_code", "g": "group", "h": "heading",
    "i": "item",     "j": "item",     "k": "keyword",  "l": "language", "m": "method",
    "n": "number",   "o": "option",
    "q": "question", "r": "role",     "u": "unit",     "v": "value",    "w": "word",
    "x": "variable", "y": "variable", "z": "variable",
}
_SINGLE_LETTER_MAP.update({k.upper(): v for k, v in _SINGLE_LETTER_MAP.items()})


# ── Compound slot canonicalization ────────────────────────────────────────
#
# Most exotic slot names are qualified canonical names, e.g.
# `task_description` → `description`, `input_text` → `text`.
# We check the last underscore-separated component against a safe canonical
# suffix list and strip the qualifier.
#
# `text` is excluded from suffix matching because `text_type` is itself
# canonical and would be wrongly collapsed.

_SUFFIX_CANONICAL: frozenset[str] = frozenset({
    "topic", "description", "passage", "question", "context", "source",
    "code", "keyword", "language", "person", "number", "option",
    "completion", "tone", "style", "format", "role", "category",
    "title", "subject", "task", "content", "example",
    # multi-word forms matched as two-component suffixes
    "text_type", "programming_language",
})

# Manual overrides for cases where suffix matching would give the wrong result.
_COMPOUND_SLOT_REMAP: dict[str, str] = {
    # language qualifiers → programming_language when code context is clear
    "code_language":         "programming_language",
    "coding_language":       "programming_language",
    "script_language":       "programming_language",
    "program_language":      "programming_language",
    # bare `text` qualifiers — excluded from suffix set to protect `text_type`
    "input_text":            "text",
    "output_text":           "text",
    "original_text":         "text",
    "body_text":             "text",
    "sample_text":           "text",
    "given_text":            "text",
    "provided_text":         "text",
    "source_text":           "text",
    "target_text":           "text",
    # person qualifiers
    "historical_person":     "person",
    "famous_person":         "person",
    "public_figure":         "person",
    # over-qualified topic slots
    "research_topic":        "topic",
    "essay_topic":           "topic",
    "discussion_topic":      "topic",
    "main_topic":            "topic",
    "programming_task":      "topic",
}


def _canonicalize_compound_slot(slot: str) -> str | None:
    """Map a compound slot name to a canonical form, or return None if unknown.

    Priority:
    1. Manual override (_COMPOUND_SLOT_REMAP).
    2. Two-component suffix (e.g. ``text_type``, ``programming_language``).
    3. Single-component suffix (e.g. ``description``, ``topic``).
    """
    lower = slot.lower()
    if lower in _COMPOUND_SLOT_REMAP:
        return _COMPOUND_SLOT_REMAP[lower]

    parts = lower.split("_")
    if len(parts) < 2:
        return None

    two_suffix = "_".join(parts[-2:])
    if two_suffix in _SUFFIX_CANONICAL:
        return two_suffix

    one_suffix = parts[-1]
    if one_suffix in _SUFFIX_CANONICAL:
        return one_suffix

    return None


# ── Task-type → default slot fallback ─────────────────────────────────────

_TASK_TYPE_TO_DEFAULT_SLOT: dict[str, str] = {
    "question_answering":          "topic",
    "fact_verification":           "topic",
    "information_extraction":      "topic",
    "summarization":               "topic",
    "mathematical_reasoning":      "topic",
    "logical_deductive_reasoning": "topic",
    "commonsense_reasoning":       "topic",
    "argumentation":               "topic",
    "prediction":                  "topic",
    "creative_writing":            "topic",
    "text_completion":             "topic",
    "dialogue_generation":         "topic",
    "translation":                 "topic",
    "rewriting_paraphrasing":      "topic",
    "communication_writing":       "topic",
    "classification":              "category",
    "ranking_comparison":          "topic",
    "data_analysis":               "topic",
    "code_generation":             "topic",
    "conversion":                  "topic",
    "planning":                    "topic",
    "brainstorming":               "topic",
    "role_playing":                "topic",
    "explanation":                 "topic",
    "length_constraint":           "number",
    "structure_constraint":        "number",
    "keyword_inclusion":           "keyword",
    "keyword_frequency":           "keyword",
    "forbidden_words":             "keyword",
    "response_language":           "language",
    "casing_constraint":           "topic",
    "tone_constraint":             "tone",
    "audience_constraint":         "topic",
}


# ── Self-replicating slot detection ───────────────────────────────────────

_SELF_REP_VALUE_TO_SLOT: dict[str, str] = {
    # Tone / style / register descriptors
    "formal":        "tone",
    "informal":      "tone",
    "professional":  "tone",
    "casual":        "tone",
    "humorous":      "tone",
    "friendly":      "tone",
    "serious":       "tone",
    "sarcastic":     "tone",
    "neutral":       "tone",
    "polite":        "tone",
    "academic":      "tone",
    "analytical":    "tone",
    "creative":      "style",
    # Format / structure descriptors
    "bullet":        "text_type",
    "bullets":       "text_type",
    "list":          "text_type",
    "json":          "text_type",
    "table":         "text_type",
    "markdown":      "text_type",
    # Plurals / near-synonyms of canonical slot names
    "people":        "person",
    "persons":       "person",
    "character":     "person",
    "characters":    "person",
    "name":          "person",
    "names":         "person",
    "topics":        "topic",
    "thing":         "topic",
    "things":        "topic",
    "item":          "topic",
    "items":         "topic",
    "word":          "keyword",
    "words":         "keyword",
    "tag":           "keyword",
    "tags":          "keyword",
    "hashtag":       "keyword",
    "hashtags":      "keyword",
    "section":       "description",
    "sections":      "description",
    "sentence":      "description",
    "sentences":     "description",
    "paragraph":     "description",
    "paragraphs":    "description",
    "postscript":    "description",
}


def _detect_self_replicating_slots(
    options: list[dict], templates: list[dict],
) -> list[dict]:
    """Remap options where slot name equals the value to a canonical slot.

    Three cases:
    1. Slot is already canonical → the value is the slot name itself
       (meaningless); drop the option.
    2. Slot is a semantic descriptor (e.g. ``"formal"``, ``"people"``) →
       look it up in _SELF_REP_VALUE_TO_SLOT for a proper slot, keep value.
    3. Fallback: use _TASK_TYPE_TO_DEFAULT_SLOT, then ``"description"``.
    """
    template_task_types = [t.get("task_type", "") for t in templates]
    kept: list[dict] = []
    for o in options:
        slot = o.get("slot", "")
        value = o.get("value", "")
        if not slot or not value:
            kept.append(o)
            continue
        value_lower = value.lower().strip()
        is_self_rep = (
            slot.lower() == value_lower
            or (len(value.split()) < 3 and slot.lower() in value_lower)
        )
        if not is_self_rep:
            kept.append(o)
            continue

        # Case 1: already canonical → value is meaningless, drop.
        if slot.lower() in _CANONICAL_PREFERRED_SLOTS:
            logger.info(
                f"  Dropping uninformative self-replicating option: "
                f"slot='{slot}' value='{value}'"
            )
            continue

        # Case 2: direct semantic remap.
        new_slot = (
            _SELF_REP_VALUE_TO_SLOT.get(value_lower)
            or _SELF_REP_VALUE_TO_SLOT.get(slot.lower())
        )

        # Case 3: task-type fallback.
        if not new_slot:
            for tt in (o.get("compatible_task_types", []) + template_task_types):
                if tt in _TASK_TYPE_TO_DEFAULT_SLOT:
                    candidate = _TASK_TYPE_TO_DEFAULT_SLOT[tt]
                    if candidate.lower() != value_lower:
                        new_slot = candidate
                        break

        if not new_slot:
            new_slot = "description"

        logger.info(
            f"  Self-replicating slot: '{slot}'='{value}' → slot='{new_slot}'"
        )
        for t in templates:
            if slot in t.get("slots", []):
                t["slots"] = [new_slot if s == slot else s for s in t["slots"]]
                t["text"] = t.get("text", "").replace(
                    f"{{{slot}}}", f"{{{new_slot}}}"
                )
        o["slot"] = new_slot
        kept.append(o)
    return kept


# ── List-option expansion ─────────────────────────────────────────────────

_LIST_SLOTS: frozenset[str] = frozenset({"keyword", "option", "completion"})
_LIST_SPLIT_RE = re.compile(r"\s*,\s*")


def _expand_list_options(options: list[dict]) -> list[dict]:
    """Split comma-separated values for list-type slots into individual options.

    e.g. slot="keyword" value="python, NLP, transformers"
         → three options each with a single keyword.

    Only applied to slots in ``_LIST_SLOTS``; other slots are left untouched.
    """
    expanded: list[dict] = []
    split_count = 0
    for o in options:
        slot = o.get("slot", "")
        value = o.get("value", "")
        if slot not in _LIST_SLOTS or "," not in value:
            expanded.append(o)
            continue
        items = [v.strip() for v in _LIST_SPLIT_RE.split(value) if v.strip()]
        if len(items) < 2:
            expanded.append(o)
            continue
        for item in items:
            new_opt = dict(o)
            new_opt["value"] = item
            expanded.append(new_opt)
        split_count += 1
    if split_count:
        logger.info(f"  Expanded {split_count} comma-separated list options")
    return expanded


# ── Core normalization functions ───────────────────────────────────────────

def _normalize_slot_names(
    templates: list[dict], options: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Rename bad slot names to canonical forms.

    Passes (in priority order):
    1. Exact alias map (_EXACT_SLOT_CANONICAL_MAP).
    2. Single-letter map (_SINGLE_LETTER_MAP).
    3. Compound suffix / manual override (_canonicalize_compound_slot).
    Slots already in CANONICAL_PREFERRED_SLOTS are skipped.
    """
    all_slots: set[str] = set()
    for t in templates:
        all_slots.update(t.get("slots", []))
    for o in options:
        s = o.get("slot", "")
        if s:
            all_slots.add(s)

    rename: dict[str, str] = {}
    for slot in all_slots:
        if slot in _CANONICAL_PREFERRED_SLOTS:
            continue
        canonical = _canonicalize_slot_name(slot)
        if canonical is not None:
            rename[slot] = canonical
        elif len(slot) == 1 and slot in _SINGLE_LETTER_MAP:
            rename[slot] = _SINGLE_LETTER_MAP[slot]
        else:
            compound = _canonicalize_compound_slot(slot)
            if compound is not None:
                rename[slot] = compound

    if not rename:
        return templates, options

    rename_count = 0
    for t in templates:
        t_slots = t.get("slots", [])
        applicable = {s: rename[s] for s in t_slots if s in rename}
        if not applicable:
            continue
        text = t.get("text", "")
        for old, new in applicable.items():
            text = text.replace(f"{{{old}}}", f"{{{new}}}")
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

    opt_rename_count = 0
    for o in options:
        slot = o.get("slot", "")
        if slot in rename:
            o["slot"] = rename[slot]
            opt_rename_count += 1

    if rename_count or opt_rename_count:
        logger.info(
            f"  Slot normalization: {rename_count} template, "
            f"{opt_rename_count} option renames"
        )
    return templates, options


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
        f"  Merging numbered slots: {len(merge_map)} slots "
        f"across {len(merged_bases)} bases"
    )

    for t in templates:
        t_slots = t.get("slots", [])
        if not any(s in merge_map for s in t_slots):
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
        logger.info(
            f"  Slot sync: added {added_total}, removed {removed_total} phantom slots"
        )
    return templates


def _remove_broken_options(options: list[dict]) -> list[dict]:
    """Remove options whose value contains unresolved {placeholder} patterns."""
    placeholder_re = re.compile(r"\{\w+\}")
    kept = [o for o in options if not placeholder_re.search(o.get("value", ""))]
    removed = len(options) - len(kept)
    if removed:
        logger.info(f"  Removed {removed} options with placeholder values")
    return kept


def _drop_duplicate_slot_templates(templates: list[dict]) -> list[dict]:
    """Remove templates where a merge left the same placeholder appearing twice."""
    ph_re = re.compile(r"\{(\w+)\}")
    kept = []
    dropped = 0
    for t in templates:
        placeholders = ph_re.findall(t.get("text", ""))
        if len(placeholders) != len(set(placeholders)):
            dropped += 1
            continue
        kept.append(t)
    if dropped:
        logger.info(
            f"  Dropped {dropped} templates with duplicate slot references after merge"
        )
    return kept


# ── ID-specific template fixes ────────────────────────────────────────────

# Templates where {constraint} slot should be renamed {format_constraint}.
CONSTRAINT_SLOT_RENAMES: set[str] = {
    "369c0ad2aa",  # "do not use any {constraint} in your response."
    "e5617457ac",  # "…should not contain any {constraint}."
}

# Templates whose `level` field is misclassified.
LEVEL_FIXES: dict[str, str] = {
    "949ef12351": "format_constraint",        # "without using {case_type}"
    "7054774d53": "format_constraint",        # "Do not contain {keywords}…"
    "e28f5a789b": "format_constraint",        # "Include the word {keywords} at least…"
    "27bd8f4114": "format_constraint",        # "use the word {word1} at least {n1} times…"
    "99d574dacd": "format_constraint",        # "The word "{forbidden_word}" should not…"
    "b943f4a7ab": "format_constraint",        # "Do not include the following {keywords}…"
    "f2de5b52d3": "format_constraint",        # "don't use any {forbidden_element}"
    "7cae2d01fc": "format_constraint",        # "You can use {text_type} ticks such as ```."
    "e845227cc7": "content_style_constraint", # "Make it {description}."
}


def fix_templates(templates: list[dict]) -> tuple[list[dict], int, int]:
    """Apply ID-specific level corrections and slot renames."""
    slot_fixes = level_fixes = 0
    for t in templates:
        tid = t.get("id", "")
        if tid in LEVEL_FIXES and t.get("level") != LEVEL_FIXES[tid]:
            t["level"] = LEVEL_FIXES[tid]
            level_fixes += 1
        if tid in CONSTRAINT_SLOT_RENAMES:
            t["text"] = t["text"].replace("{constraint}", "{format_constraint}")
            t["slots"] = [
                "format_constraint" if s == "constraint" else s
                for s in t.get("slots", [])
            ]
            slot_fixes += 1
    return templates, slot_fixes, level_fixes


def deduplicate_templates(
    templates: list[dict],
) -> tuple[list[dict], dict[str, str], int]:
    """Remove exact-text duplicates; merge compatible_with lists.

    Also removes the wrong-level copy of known cross-level duplicates
    (e.g. "First repeat the request…" — keep process_directive).

    Returns (deduped_templates, replace_map, removed_count) where
    replace_map maps removed-id → kept-id for updating option references.
    """
    same_level: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for t in templates:
        same_level[(t["text"], t["level"])].append(t)

    replace_map: dict[str, str] = {}
    for (_, _), group in same_level.items():
        if len(group) > 1:
            keep = group[0]
            all_compat = set(keep.get("compatible_with", []))
            for dup in group[1:]:
                all_compat.update(dup.get("compatible_with", []))
                replace_map[dup["id"]] = keep["id"]
            keep["compatible_with"] = sorted(all_compat)

    # Cross-level dedup: keep process_directive over format_constraint
    text_to_all: dict[str, list[dict]] = defaultdict(list)
    for t in templates:
        text_to_all[t["text"]].append(t)

    for text, group in text_to_all.items():
        if len(group) < 2:
            continue
        levels = {e["level"] for e in group}
        if (
            levels == {"process_directive", "format_constraint"}
            and "First repeat the request" in text
        ):
            keep = next(e for e in group if e["level"] == "process_directive")
            for dup in group:
                if dup["id"] != keep["id"] and dup["id"] not in replace_map:
                    replace_map[dup["id"]] = keep["id"]

    ids_to_remove = set(replace_map.keys())
    deduped = [t for t in templates if t["id"] not in ids_to_remove]
    return deduped, replace_map, len(ids_to_remove)


# ── Option fixes ──────────────────────────────────────────────────────────

def fix_options(
    options: list[dict],
    replace_map: dict[str, str] | None = None,
) -> tuple[list[dict], int]:
    """Apply option-level fixes.

    - markdown with slot="text_type" → slot="format"
    - Update compatible_templates refs after template deduplication.
    """
    fixes = 0
    for o in options:
        # markdown is an output encoding → belongs under `format`, not `text_type`
        if o.get("value") == "markdown" and o.get("slot") == "text_type":
            o["slot"] = "format"
            fixes += 1

        if replace_map:
            old_compat = o.get("compatible_templates", [])
            new_compat_seen: set[str] = set()
            new_compat: list[str] = []
            for tid in old_compat:
                mapped = replace_map.get(tid, tid)
                if mapped not in new_compat_seen:
                    new_compat_seen.add(mapped)
                    new_compat.append(mapped)
            if new_compat != old_compat:
                o["compatible_templates"] = new_compat
                fixes += 1
    return options, fixes


# ── Few-shot fixes ────────────────────────────────────────────────────────

def fix_few_shot(data: dict) -> tuple[dict, int, int]:
    """Fix slot names and level classifications in few_shot_examples.json."""
    slot_fixes = level_fixes = 0

    LEVEL_CORRECTIONS: list[tuple[str, str]] = [
        ("and in all capital letters", "format_constraint"),
        ("should appear {n} or more times in your response", "format_constraint"),
        ("should appear {number} or more times in your response", "format_constraint"),
        (
            "Your entire response should be in {language}, "
            "and in all capital letters.",
            "format_constraint",
        ),
        ("Include {metric} in {unit}.", "format_constraint"),
    ]

    for examples in data.values():
        for ex in examples:
            annotation = ex.get("annotation", {})
            for t in annotation.get("templates", []):
                text = t.get("text", "")
                # Rename residual {n} → {number}
                if "{n}" in text:
                    t["text"] = text.replace("{n}", "{number}")
                    slot_fixes += 1
                    text = t["text"]
                slots = t.get("slots", [])
                if "n" in slots:
                    t["slots"] = ["number" if s == "n" else s for s in slots]
                    slot_fixes += 1
                for substr, correct_level in LEVEL_CORRECTIONS:
                    if substr in text and t.get("level") != correct_level:
                        t["level"] = correct_level
                        level_fixes += 1
                        break
            for o in annotation.get("options", []):
                if o.get("slot") == "n":
                    o["slot"] = "number"
                    slot_fixes += 1

    return data, slot_fixes, level_fixes


# ── Unified normalization pipeline ────────────────────────────────────────

def normalize_existing(
    raw_templates: list[dict],
    raw_options: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Run the full normalization pipeline on raw template/option dicts.

    Combines:
    - General slot canonicalization (alias map, single-letter, compound suffix)
    - Self-replicating slot detection and remapping
    - List-option expansion
    - Numbered slot merging
    - Slot/placeholder sync
    - Broken-option removal
    - ID-specific level and slot fixes
    - Sophisticated deduplication (with cross-level conflict resolution)
    - Option-specific fixes (markdown→format, compatible_templates update)
    - Orphaned/empty option removal
    - Option deduplication
    - ID rebuild
    """
    # General normalization
    raw_templates, raw_options = _normalize_slot_names(raw_templates, raw_options)
    # DEPRECATED: removed from normalize_existing in v2 — structural cause
    # eliminated by ablation-based extraction (pass 2: self-replicating slots)
    # raw_options = _detect_self_replicating_slots(raw_options, raw_templates)
    raw_options = _expand_list_options(raw_options)
    # DEPRECATED: removed from normalize_existing in v2 (passes 4+5)
    # Numbered slots came from LLM enumerating slot instances; ablation assigns
    # independently per span so numbering never arises.
    # raw_templates, raw_options = _merge_numbered_slots(
    #     raw_templates, raw_options, join_values=False,
    # )
    # raw_templates = _drop_duplicate_slot_templates(raw_templates)
    raw_templates = _sync_template_slots(raw_templates)
    raw_options = _remove_broken_options(raw_options)

    # ID-specific fixes
    raw_templates, t_slot_fixes, t_level_fixes = fix_templates(raw_templates)
    if t_slot_fixes or t_level_fixes:
        logger.info(
            f"  ID-specific fixes: {t_slot_fixes} slot, {t_level_fixes} level"
        )

    # Deduplicate templates (handles same-level and cross-level conflicts)
    raw_templates, replace_map, removed = deduplicate_templates(raw_templates)
    if removed:
        logger.info(f"  Deduplicated: removed {removed} duplicate templates")

    # Option-level fixes (update compatible_templates refs, markdown→format)
    raw_options, o_fixes = fix_options(raw_options, replace_map=replace_map)
    if o_fixes:
        logger.info(f"  Option fixes: {o_fixes} updates")

    # Remove orphaned and empty options
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
        logger.info(
            f"  Removed {orphaned} orphaned options, {empty} empty-value options"
        )
    raw_options = kept_options

    # Deduplicate options (merge compatible lists)
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


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    print("Loading templates.json…")
    templates = json.loads(TEMPLATES_PATH.read_text())
    options   = json.loads(OPTIONS_PATH.read_text())
    n_t, n_o  = len(templates), len(options)

    templates, options = normalize_existing(templates, options)

    TEMPLATES_PATH.write_text(json.dumps(templates, ensure_ascii=False, indent=2))
    OPTIONS_PATH.write_text(json.dumps(options, ensure_ascii=False, indent=2))
    print(
        f"  {n_t} → {len(templates)} templates, "
        f"{n_o} → {len(options)} options written."
    )

    if FEW_SHOT_PATH.exists():
        print("Loading few_shot_examples.json…")
        few_shot = json.loads(FEW_SHOT_PATH.read_text())
        few_shot, fs_slot_fixes, fs_level_fixes = fix_few_shot(few_shot)
        FEW_SHOT_PATH.write_text(json.dumps(few_shot, ensure_ascii=False, indent=2))
        print(
            f"  {fs_slot_fixes} slot fixes, {fs_level_fixes} level fixes "
            f"written to few_shot_examples.json"
        )

    print("Done.")


if __name__ == "__main__":
    main()

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
    _CANONICAL_PREFERRED_SLOTS,
    _TASK_TYPE_TO_DEFAULT_SLOT,
    _normalize_slot_names,
    _detect_self_replicating_slots,
    _expand_list_options,
    _merge_numbered_slots,
    _drop_duplicate_slot_templates,
    _remove_broken_options,
    normalize_existing,
)

import sys

# spaCy is optional — required only for two-stage extraction
try:
    import spacy as _spacy
    _SPACY_AVAILABLE = True
except ImportError:
    _SPACY_AVAILABLE = False

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

SLOT NAMING — you MUST use a canonical name. Inventing new slot names is the last resort.
- CANONICAL names (use one of these whenever it fits the semantics):
  `topic`, `description`, `text`, `passage`, `question`, `context`, `source`, `code`,
  `keyword`, `option`, `completion`, `language`, `programming_language`,
  `text_type`, `person`, `number` (for any count/quantity), `tone`, `style`, `format`,
  `role`, `category`, `title`, `subject`, `task`, `content`, `example`.
- ALL slot names are SINGULAR. For list-valued slots, emit one option per item \
(all sharing the same slot name), not one option with a comma-separated string. \
Example: keywords "python" and "NLP" → two options both with slot="keyword", \
NOT one option slot="keyword" value="python, NLP".
- DEFAULT FALLBACKS: when no specific canonical name fits:
  → free-form content, instructions, or multi-sentence text → `description`
  → a subject matter, domain, or theme → `topic`
  → a brief value that doesn't fit anything else → `content`
- CORRECT vs WRONG examples:
  ✓ slot="description" for "a detailed explanation of the project"
  ✗ slot="project_description" — never qualify a canonical name with a prefix
  ✓ slot="topic" for "climate change", "machine learning", "tax law"
  ✗ slot="research_topic" or "essay_topic" — use plain `topic`
  ✓ slot="tone" for "formal", "casual", "sarcastic"
  ✗ slot="formal" with value="formal" — slot must describe the type, not the value
  ✓ two options slot="keyword" value="python" and slot="keyword" value="NLP"
  ✗ one option slot="keyword" value="python, NLP"
- NEVER use a single letter (a, b, p, q, …) — use the full semantic name.
- NEVER use numbered variants (option_1/option_2, n1/n2) — group into one slot \
(e.g. `option`, `number_list`).
- NEVER name a slot the same as its value (e.g. slot="formal" value="formal" is wrong; \
use `tone` or `style`).
- NEVER qualify a canonical name with a prefix or suffix (no `task_description`, \
`source_language`, `input_text`, `target_code`). Use the bare canonical name.
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


# ── Two-stage: Stage B ablation extraction ────────────────────────────────

def extract_from_segment(
    segment: "Segment",
    taxonomy: dict,
    nlp: "Any",                    # spaCy Language object; caller loads once
    taxonomy_obj: "Any | None" = None,  # Taxonomy object for pattern checks
) -> "tuple[TaskTemplate | None, list[Option]]":
    """Extract a TaskTemplate and Options from a pre-classified Segment.

    Algorithm:
    1. Parse span_text with spaCy.
    2. Build candidate spans in ablation priority order.
    3. For each candidate span, check if ablating it still Taxonomy.matches().
    4. Commit spans that pass; keep failed spans as literals.
    5. Tag degenerate results with source="ablation_degenerate".
    """
    from dataset.schema import TaskTemplate, Option, template_id, option_id
    from dataset.token_counter import token_length

    span_text = segment.span_text.strip()
    if not span_text:
        return None, []

    # ── Named entity → slot name mapping ─────────────────────────────
    _NE_TO_SLOT: dict[str, str] = {
        "DATE":         "date",
        "PERSON":       "person",
        "GPE":          "country_city_state",
        "LOC":          "location",
        "NORP":         "nationality_religion_political_group",
        "FAC":          "building_airport_highway_bridge",
        "PRODUCT":      "object_vehicle_food",
        "EVENT":        "event",
        "WORK_OF_ART":  "book_song_art",
        "LAW":          "law",
        "ORG":          "company_agency_institution",
        "LANGUAGE":     "language",
        "CARDINAL":     "cardinal",
        "ORDINAL":      "ordinal",
        "QUANTITY":     "quantity",
        "PERCENT":      "percent",
        "MONEY":        "money",
        "TIME":         "time",
    }

    # ── Quoted string regex ───────────────────────────────────────────
    _QUOTED_RE = re.compile(r"""(?:\"[^\"]+\"|'[^']+'|`[^`]+`)""")
    # ── Numeric + unit regex ──────────────────────────────────────────
    _NUM_UNIT_RE = re.compile(
        r"""\b\d+\s*(?:words?|sentences?|paragraphs?|characters?|"""
        r"""items?|bullets?|lines?|pages?|times?)\b""",
        re.I,
    )

    doc = nlp(span_text)
    task_type = segment.taxonomy_label

    # ── Build candidate spans (most-specific first) ───────────────────
    candidates: list[tuple[str, str]] = []  # (span_text, slot_name)

    # 1. Named entities
    for ent in doc.ents:
        slot = _NE_TO_SLOT.get(ent.label_, None)
        if slot:
            candidates.append((ent.text, slot))

    # 2. Quoted strings
    for m in _QUOTED_RE.finditer(span_text):
        inner = m.group(0)[1:-1]  # strip quotes
        if inner.strip():
            candidates.append((inner, "keyword"))

    # 3. Numeric + unit expressions
    for m in _NUM_UNIT_RE.finditer(span_text):
        candidates.append((m.group(0), "number"))

    # 4. Noun chunks (longest first, skip if already covered)
    nc_list = sorted(doc.noun_chunks, key=lambda nc: len(nc.text), reverse=True)
    for nc in nc_list:
        nc_text = nc.text.strip()
        if not nc_text:
            continue
        # Determine slot from POS of first non-article token
        slot = _TASK_TYPE_TO_DEFAULT_SLOT.get(task_type, "description")
        for tok in nc:
            if tok.pos_ == "ADJ" and tok.dep_ in ("amod", "nsubj"):
                slot = "style"
                break
        candidates.append((nc_text, slot))

    # 5. Prepositional phrase subtrees
    for tok in doc:
        if tok.dep_ == "prep":
            subtree_text = " ".join(t.text for t in tok.subtree).strip()
            if subtree_text and subtree_text != tok.text:
                candidates.append((subtree_text, "description"))

    # 6. Adverbial modifiers
    for tok in doc:
        if tok.dep_ == "advmod":
            candidates.append((tok.text, "style"))

    # ── Greedy ablation loop ──────────────────────────────────────────
    current_text = span_text
    committed_options: list[tuple[str, str]] = []   # (value, slot)

    # Track committed spans to avoid double-ablating
    committed_spans: set[str] = set()

    # Deduplicate candidates while preserving order
    seen_spans: set[str] = set()
    dedup_candidates: list[tuple[str, str]] = []
    for span, slot in candidates:
        key = span.strip()
        if key and key not in seen_spans and len(key) > 1:
            seen_spans.add(key)
            dedup_candidates.append((span, slot))

    for span_val, slot_name in dedup_candidates:
        if span_val in committed_spans:
            continue
        if span_val not in current_text:
            continue

        candidate_text = current_text.replace(span_val, f"{{{slot_name}}}", 1)

        # Check: does the ablated template still match the taxonomy label?
        passes = False
        if taxonomy_obj is not None:
            try:
                passes = taxonomy_obj.matches(candidate_text, task_type)
            except Exception:
                passes = False
        else:
            # If no taxonomy_obj, accept ablation based on heuristic:
            # the template still contains at least one recognizable verb
            verb_re = re.compile(r"\b(write|describe|explain|translate|summarize|classify|list|generate|create|analyze|solve|convert|compare|rank|plan|brainstorm)\b", re.I)
            passes = bool(verb_re.search(candidate_text))

        if passes:
            current_text = candidate_text
            committed_options.append((span_val, slot_name))
            committed_spans.add(span_val)

    if not current_text.strip():
        return None, []

    # ── Build TaskTemplate ────────────────────────────────────────────
    placeholder_re = re.compile(r"\{(\w+)\}")
    slots = list(dict.fromkeys(placeholder_re.findall(current_text)))  # deduplicated

    # Count non-placeholder tokens for degenerate check
    stripped = placeholder_re.sub("", current_text)
    non_placeholder_tokens = len(stripped.split())

    source = "ablation"
    if non_placeholder_tokens < 3 and slots:
        source = "ablation_degenerate"

    tid = template_id(task_type, current_text)
    level = segment.level
    fixed_text = placeholder_re.sub("", current_text)
    tlen = token_length(fixed_text)

    tmpl = TaskTemplate(
        id=tid,
        text=current_text,
        slots=slots,
        task_type=task_type,
        level=level,
        token_length=tlen,
        source=source,
    )

    # ── Build Options ─────────────────────────────────────────────────
    options: list[Option] = []
    for val, slot in committed_options:
        if not val.strip():
            continue
        oid = option_id(slot, val)
        options.append(Option(
            id=oid,
            value=val,
            slot=slot,
            compatible_task_types=[task_type],
            compatible_templates=[tid],
            token_length=token_length(val),
            source="ablation",
        ))

    return tmpl, options


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
    use_two_stage: bool = True,
    taxonomy_obj: "Any | None" = None,
    gpu_memory_utilization: float = 0.7,
) -> tuple[list[TaskTemplate], list[Option]]:
    """
    Extract templates and options from a list of prompt records.

    Each record must have a "prompt" field; optional "source" field enables
    dataset-specific few-shot examples. Uses batched vLLM inference.

    Parameters
    ----------
    use_two_stage
        When True (default): run Stage A (segmentation) + Stage B (ablation).
        When False: use the original single-pass LLM path (for A/B testing).
    taxonomy_obj
        Taxonomy object from collect_task_types. Required for accurate ablation
        checks in Stage B. If None, a fallback heuristic is used.
    """
    if use_two_stage:
        return _extract_two_stage(
            prompts, taxonomy,
            model=model, device=device, batch_size=batch_size,
            taxonomy_obj=taxonomy_obj,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    else:
        return _extract_llm_single_pass(
            prompts, taxonomy,
            model=model, device=device, batch_size=batch_size,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def _extract_two_stage(
    prompts: list[dict],
    taxonomy: dict,
    *,
    model: str,
    device: str | None,
    batch_size: int,
    taxonomy_obj: "Any | None",
    gpu_memory_utilization: float = 0.7,
) -> tuple[list[TaskTemplate], list[Option]]:
    """Two-stage extraction: segmentation → ablation."""
    from dataset.segmenter import segment_and_classify_batch
    from dataset.taxonomy.collect_task_types import Taxonomy

    if not _SPACY_AVAILABLE:
        logger.error(
            "spaCy not available — falling back to single-pass LLM extraction. "
            "Install spaCy: pip install spacy && python -m spacy download en_core_web_sm"
        )
        raise RuntimeError("spaCy is required for two-stage extraction but is not installed.")

    # Build Taxonomy object if not provided
    if taxonomy_obj is None:
        taxonomy_obj = Taxonomy.from_dict(taxonomy)

    nlp = _spacy.load("en_core_web_sm")#, disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])

    all_templates: dict[str, TaskTemplate] = {}
    all_options: dict[str, Option] = {}

    valid_records = [r for r in prompts if r.get("prompt", "").strip()]
    prompt_texts = [r["prompt"] for r in valid_records]

    logger.info(
        f"Two-stage extraction: segmenting {len(prompt_texts)} prompts "
        f"(model={model})…"
    )

    # Stage A: segment and classify all prompts
    all_prompt_segments = segment_and_classify_batch(
        prompt_texts, taxonomy_obj,
        model=model, device=device, batch_size=batch_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # Save segmentation results to JSONL
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
    
    

    # Stage B: ablation-based extraction per segment
    failures = 0
    for segments in all_prompt_segments:
        for seg in segments:
            try:
                tmpl, opts = extract_from_segment(
                    seg, taxonomy, nlp, taxonomy_obj=taxonomy_obj
                )
            except Exception as exc:
                logger.warning(f"  extract_from_segment failed: {exc}")
                failures += 1
                continue
            if tmpl:
                _merge_into_store(all_templates, all_options, [tmpl], opts)

    logger.info(
        f"Two-stage extraction done: {len(all_templates)} templates, "
        f"{len(all_options)} options, {failures} failures"
    )
    return list(all_templates.values()), list(all_options.values())


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
    taxonomy_labels = list(taxonomy.keys())
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
    return list(all_templates.values()), list(all_options.values())


# ── LLM batch slot reclassification ──────────────────────────────────────

_RECLASSIFY_SYSTEM_PROMPT = """\
You are a slot name normalizer for a prompt template extraction system.

Map each slot name to the most semantically appropriate canonical equivalent.

Canonical slots:
topic, description, text, passage, question, context, source, code,
keyword, option, completion, language, programming_language, text_type,
person, number, tone, style, format, role, category, title, subject,
task, content, example

Rules:
- Choose the canonical that best captures the SEMANTIC ROLE of the variable \
in a prompt template (e.g. "research_topic" → "topic").
- If a slot is a qualified canonical name, use the bare canonical.
- Default: free-form content → "description"; subject/domain/theme → "topic".

Output ONLY a JSON object {"slot_name": "canonical_name", ...} — no commentary.\
"""

_RECLASSIFY_VALID_TARGETS: frozenset[str] = CANONICAL_PREFERRED_SLOTS | frozenset({
    "tone", "style", "format", "role", "category",
    "title", "subject", "task", "content", "example",
})


def reclassify_exotic_slots(
    exotic_slots: list[str],
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str | None = None,
    batch_size: int = 40,
) -> dict[str, str]:
    """Map exotic slot names to canonical equivalents via LLM.

    Intended as a post-processing step on accumulated unique slot names.
    Returns a dict mapping exotic_slot → canonical_slot; slots already in
    CANONICAL_PREFERRED_SLOTS are skipped.

    Typical usage:
        all_opts = json.loads(Path("dataset/output/options.json").read_text())
        exotic = [s for s in {o["slot"] for o in all_opts}
                  if s not in CANONICAL_PREFERRED_SLOTS]
        remap = reclassify_exotic_slots(exotic, model=..., device="cuda")
        raw_t, raw_o = normalize_existing(raw_templates, raw_opts)
        # then apply remap via a second _normalize_slot_names call
    """
    if not exotic_slots:
        return {}

    remap: dict[str, str] = {}
    n_batches = (len(exotic_slots) + batch_size - 1) // batch_size
    for i in range(0, len(exotic_slots), batch_size):
        batch = exotic_slots[i: i + batch_size]
        messages = [
            {"role": "system", "content": _RECLASSIFY_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(batch, ensure_ascii=False)},
        ]
        try:
            raw = _local_generate(
                model, messages,
                temperature=0.0, device=device,
                enable_thinking=False,
            )
            mapping = _repair_json(raw)
        except Exception as e:
            logger.warning(
                f"  Reclassify batch {i // batch_size + 1}/{n_batches} failed: {e}"
            )
            continue

        for slot, target in mapping.items():
            if slot not in batch or not isinstance(target, str):
                continue
            target = target.strip().lower()
            if target in _RECLASSIFY_VALID_TARGETS and target != slot.lower():
                remap[slot] = target

    logger.info(
        f"  Slot reclassification: {len(remap)}/{len(exotic_slots)} slots remapped"
    )
    return remap

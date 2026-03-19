"""
segmenter.py — Stage A of the two-stage extraction pipeline.

Splits a prompt into clauses and classifies each clause with a taxonomy label
using a single LLM call per prompt. The LLM receives the full taxonomy
(labels, levels, descriptions) and performs both segmentation and
classification in one shot.
"""

from __future__ import annotations

import json
import logging
import re
from enum import Enum
from typing import Any, Literal

from tqdm import tqdm

from pydantic import BaseModel as _BaseModel, create_model

from dataset.schema import Segment, LEVELS
from dataset.taxonomy.collect_task_types import Taxonomy

logger = logging.getLogger(__name__)


# ── Pydantic models for structured LLM output ────────────────────────────

# Static fallback (used only when no taxonomy is available)
class _SegmentItem(_BaseModel):
    span_text: str
    taxonomy_label: str
    level: str


class _SegmentationResult(_BaseModel):
    segments: list[_SegmentItem]


def _build_segment_model(valid_labels: list[str]) -> type[_BaseModel]:
    """Build a Pydantic model with taxonomy labels and levels constrained as enums.

    This allows vLLM's guided decoding to restrict output to valid labels only,
    eliminating hallucinated labels.
    """
    LabelEnum = Enum("LabelEnum", {lbl: lbl for lbl in valid_labels})  # type: ignore[misc]
    LevelEnum = Enum("LevelEnum", {lv: lv for lv in LEVELS})  # type: ignore[misc]

    DynSegmentItem = create_model(
        "_DynSegmentItem",
        span_text=(str, ...),
        taxonomy_label=(LabelEnum, ...),
        level=(LevelEnum, ...),
    )

    DynSegmentationResult = create_model(
        "_DynSegmentationResult",
        segments=(list[DynSegmentItem], ...),  # type: ignore[valid-type]
    )

    return DynSegmentationResult


# ── Taxonomy description builder ─────────────────────────────────────────

def _build_taxonomy_description(taxonomy: Taxonomy) -> str:
    """Build a human-readable description of the taxonomy for the LLM prompt."""
    lines: list[str] = []
    by_level: dict[str, list[str]] = {}
    for entry in taxonomy:
        by_level.setdefault(entry.level, []).append(
            f"  - {entry.name}" + (f": {entry.description}" if entry.description else "")
        )
    for level in LEVELS:
        entries = by_level.get(level, [])
        if entries:
            lines.append(f"\n## {level}")
            lines.extend(entries)
    return "\n".join(lines)


_SYSTEM_PROMPT = """\
You are an instruction segmenter and classifier.

Given a prompt (an instruction given to an LLM), you must:
1. Identify the core task and any distinct constraints or directives.
2. Split the prompt into segments — one for the core task (including any \
context, reference text, or elaboration that belongs to it) and one for each \
distinct constraint or directive from a different taxonomy axis.
3. Classify each segment with the most appropriate taxonomy label and its level.

## Key principles

**Context, reference text, and task operands stay with the task.**
When a prompt contains reference material (a passage, document, code snippet, \
data, or background information) that the task operates on, that material is \
part of the task segment. Do NOT split reference text into separate segments. \
This includes: passages provided for QA, input text to classify, option lists \
that define valid answers, and data to analyze or extract from.

**Elaboration is not a separate segment.**
Sentences that elaborate, clarify, or add detail to the *same* task (without \
introducing a constraint from a different taxonomy axis) remain in the task \
segment. Only create a new segment when text introduces an orthogonal \
constraint from a different level (format_constraint, content_style_constraint, \
or process_directive).

**One segment per distinct constraint axis.**
Split only when you encounter a genuinely new constraint type — e.g., a length \
limit, a formatting requirement, a style directive. Multiple sentences \
describing the same task or same constraint type belong together.

## Examples

Prompt: "Write a poem about the sea using 50 characters. The poem should be in rhyming couplets targeting children."
Segments:
- "Write a poem about the sea."
  taxonomy_label: "creative_writing"
  level: "task_type"
- "using 50 characters."
  taxonomy_label: "length_constraint"
  level: "format_constraint"
- "The poem should be in rhyming couplets"
  taxonomy_label: "structure_constraint"
  level: "format_constraint"
- "targeting children."
  taxonomy_label: "audience_constraint"
  level: "content_style_constraint"

Prompt: "Write a 300+ word summary of the wikipedia page \\"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\\". Do not use any commas and highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*."
Segments:
- "Write a summary of the wikipedia page \\"https://en.wikipedia.org/wiki/Raymond_III,_Count_of_Tripoli\\"."
  taxonomy_label: "summarization"
  level: "task_type"
  hint: this segment does not contain the length constraint, which is a separate segment.
- "300+ word"
  taxonomy_label: "length_constraint"
  level: "format_constraint"
- "Do not use any commas."
  taxonomy_label: "forbidden_words"
  level: "content_style_constraint"
- "highlight at least 3 sections that has titles in markdown format, for example *highlighted section part 1*, *highlighted section part 2*, *highlighted section part 3*."
  taxonomy_label: "output_syntax_format"
  level: "format_constraint"

Prompt: "Make a list of 10 ways to help students improve their study skills.\\n\\nOutput:"
Segments:
- "Make a list of 10 ways to help students improve their study skills.\\n\\nOutput:"
  taxonomy_label: "brainstorming"
  level: "task_type"
  hint: "Output:" is part of the task, not a separate constraint.

Prompt: "I am planning a trip to Japan, and I would like to write an itinerary for my journey in a Shakespearean style."
Segments:
- "I am planning a trip to Japan, and I would like thee to write an itinerary for my journey"
  taxonomy_label: "planning"
  level: "task_type"
- "in a Shakespearean style."
  taxonomy_label: "tone_constraint"
  level: "content_style_constraint"

Prompt: "Task: Find out what are the key topics in the document? output \\"topic 1\\", \\"topic 2\\", ... , \\"topic n\\".\\n\\nThe United States has withdrawn from the Paris Climate Agreement.\\n\\n"
Segments:
- "Find out what are the key topics in the document?\\n\\nThe United States has withdrawn from the Paris Climate Agreement.\\n\\n"
  taxonomy_label: "information_extraction"
  level: "task_type"
  hint: the question and the context passage are part of the same task segment.
- "output \\"topic 1\\", \\"topic 2\\", ... , \\"topic n\\"."
  taxonomy_label: "structure_constraint"
  level: "format_constraint"

Prompt: "How many field goals did the Lions score?\\nTo start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Josh Freeman found Arrelious Benn on a 13-yard touchdown pass."
Segments:
- "How many field goals did the Lions score?\\nTo start the season, the Lions traveled south to Tampa, Florida to take on the Tampa Bay Buccaneers. The Lions scored first in the first quarter with a 23-yard field goal by Jason Hanson. The Buccaneers tied it up with a 38-yard field goal by Connor Barth, then took the lead when Josh Freeman found Arrelious Benn on a 13-yard touchdown pass."
  taxonomy_label: "question_answering"
  level: "task_type"
  hint: the entire passage is context for the question — it is ONE segment, not many.

Prompt: "Create an Excel macro that automatically detects any changes made to a specific column in a worksheet. If the changes are within a certain range of values in that column, execute a function that copies the row and pastes it into a new worksheet. The function should only copy rows that contain values that match a certain criteria specified in a separate sheet. The macro should be able to handle multiple modifications at once and should include error handling to prevent any unexpected behavior."
Segments:
- "Create an Excel macro that automatically detects any changes made to a specific column in a worksheet. If the changes are within a certain range of values in that column, execute a function that copies the row and pastes it into a new worksheet. The function should only copy rows that contain values that match a certain criteria specified in a separate sheet. The macro should be able to handle multiple modifications at once and should include error handling to prevent any unexpected behavior."
  taxonomy_label: "code_generation"
  level: "task_type"
  hint: all sentences elaborate the same code_generation task. There are no format or style constraints, so this is a single segment.

Prompt: "Pick one category for the following text. The options are - company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work. Michael DenDekker - Michael G. DenDekker (born July 11 1961) is an assemblyman for the state of New York's 34th district which includes the neighborhoods of Woodside Jackson Heights and East Elmhurst all in the borough/county of Queens."
Segments:
- "Pick one category for the following text. The options are - company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work. Michael DenDekker - Michael G. DenDekker (born July 11 1961) is an assemblyman for the state of New York's 34th district which includes the neighborhoods of Woodside Jackson Heights and East Elmhurst all in the borough/county of Queens."
  taxonomy_label: "classification"
  level: "task_type"
  hint: the task instruction, the list of options, and the input text to classify are ALL part of the same task — this is ONE segment. The option list and input text are not separate constraints, they are operands of the classification task.

Prompt: "What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?\\nWithout Coriolis Effect the global winds would blow north to south or south to north. But Coriolis makes them blow northeast to southwest or the reverse in the Northern Hemisphere. The winds blow northwest to southeast or the reverse in the southern hemisphere."
Segments:
- "What phenomenon makes global winds blow northeast to southwest or the reverse in the northern hemisphere and northwest to southeast or the reverse in the southern hemisphere?\\nWithout Coriolis Effect the global winds would blow north to south or south to north. But Coriolis makes them blow northeast to southwest or the reverse in the Northern Hemisphere. The winds blow northwest to southeast or the reverse in the southern hemisphere."
  taxonomy_label: "question_answering"
  level: "task_type"
  hint: the passage after the question is context the model should use to answer — it is NOT a separate task. Keep question + context as one segment.

## Rules
- Each segment's span_text must be a verbatim substring of the original prompt.
- Choose the single best taxonomy label for each segment from the available labels.
- The level must match the taxonomy entry's level.
- Do not invent text that is not in the prompt.
- Prefer fewer, larger segments over many small ones. Only split when there is \
a genuinely distinct constraint from a different taxonomy axis.

Output a JSON object with a "segments" array. Each element has:
  - "span_text": the verbatim text from the prompt
  - "taxonomy_label": one of the available labels
  - "level": one of task_type, format_constraint, content_style_constraint, \
process_directive
"""


# ── Public API ────────────────────────────────────────────────────────────

def segment_and_classify(
    prompt: str,
    taxonomy: Taxonomy,
    *,
    model: str | None = None,
    device: str | None = None,
) -> list[Segment]:
    """Segment and classify a single prompt via LLM."""
    results = segment_and_classify_batch(
        [prompt], taxonomy, model=model, device=device,
    )
    return results[0]


def segment_and_classify_batch(
    prompts: list[str],
    taxonomy: Taxonomy,
    *,
    model: str | None = None,
    device: str | None = None,
    batch_size: int = 64,
    gpu_memory_utilization: float = 0.7,
) -> list[list[Segment]]:
    """Segment and classify a batch of prompts using a single LLM call.

    The LLM receives the full taxonomy and performs both clause splitting
    and classification for each prompt.

    Parameters
    ----------
    prompts
        Raw instruction strings.
    taxonomy
        Taxonomy object with entries to classify against.
    model
        Model ID for the LLM. If None, falls back to simple heuristic.
    device
        Device string passed to the LLM.
    batch_size
        Maximum number of prompts per LLM batch call.
    """
    if not model:
        logger.error("No model provided; returning each prompt as a single 'description' segment.")
        raise ValueError("Model name must be provided for LLM-based segmentation.")

    from dataset.local_llm import generate_text_batch

    taxonomy_desc = _build_taxonomy_description(taxonomy)
    valid_labels = set(taxonomy._entries.keys())
    label_level_map = {e.name: e.level for e in taxonomy}

    # Build a dynamic Pydantic model with enum-constrained labels and levels
    # so vLLM's guided decoding only allows valid taxonomy values.
    segment_model = _build_segment_model(sorted(valid_labels))

    all_segments: list[list[Segment]] = []

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    pbar = tqdm(range(0, len(prompts), batch_size), total=n_batches, desc="Segmenting", unit="batch")
    for batch_start in pbar:
        batch_prompts = prompts[batch_start: batch_start + batch_size]

        messages_batch = [
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"## Available taxonomy labels\n{taxonomy_desc}\n\n"
                        f"## Prompt to segment\n\"{prompt}\""
                    ),
                },
            ]
            for prompt in batch_prompts
        ]

        # Thinking mode and guided JSON decoding can conflict in vLLM:
        # thinking tokens (<think>…</think>) aren't valid JSON, so guided
        # decoding may reject them.  Try with both enabled first; on failure
        # retry with guided decoding disabled — the prompt instructions +
        # _parse_llm_output validation still enforce the schema.
        try:
            raws = generate_text_batch(
                model, messages_batch,
                temperature=0.0, device=device,
                enable_thinking=True, json_schema=segment_model,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except Exception:
            logger.info(
                "  Thinking + guided decoding failed; "
                "retrying with guided decoding disabled."
            )
            try:
                raws = generate_text_batch(
                    model, messages_batch,
                    temperature=0.0, device=device,
                    enable_thinking=True, json_schema=None,
                    gpu_memory_utilization=gpu_memory_utilization,
                )
            except Exception as exc:
                logger.error(f"  LLM segmentation batch failed: {exc}")
                raise RuntimeError(f"LLM segmentation batch failed: {exc}") from exc

        for prompt, raw in zip(batch_prompts, raws):
            segments = _parse_llm_output(raw, prompt, valid_labels, label_level_map)
            all_segments.append(segments)

    total = sum(len(s) for s in all_segments)
    n_ok = sum(
        1 for segs in all_segments for seg in segs
        if seg.classification_method == "llm"
    )
    if total:
        logger.info(
            f"  Segmentation: {total} clauses from {len(prompts)} prompts; "
            f"LLM classified: {n_ok}/{total} ({100 * n_ok // total}%)"
        )

    return all_segments


_THINKING_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(raw: str) -> str:
    """Remove <think>…</think> blocks that Qwen3 emits when thinking is enabled."""
    return _THINKING_RE.sub("", raw).strip()


def _parse_llm_output(
    raw: str,
    source_prompt: str,
    valid_labels: set[str],
    label_level_map: dict[str, str],
) -> list[Segment]:
    """Parse the LLM JSON output into Segment objects."""
    if not raw:
        return [_fallback_segment(source_prompt)]

    # Strip thinking tokens if present (Qwen3 with enable_thinking=True)
    cleaned = _strip_thinking(raw) if isinstance(raw, str) else raw

    try:
        obj = json.loads(cleaned) if isinstance(cleaned, str) else cleaned
        items = obj.get("segments", []) if isinstance(obj, dict) else []
    except Exception:
        return [_fallback_segment(source_prompt)]

    if not items:
        return [_fallback_segment(source_prompt)]

    segments: list[Segment] = []
    for item in items:
        span = str(item.get("span_text", "")).strip()
        label = str(item.get("taxonomy_label", "")).strip()
        level = str(item.get("level", "")).strip()

        if not span:
            continue

        if label in valid_labels and level in LEVELS:
            # Enforce consistency: use the taxonomy's own level for this label
            segments.append(Segment(
                span_text=span,
                taxonomy_label=label,
                level=label_level_map.get(label, level),
                source_prompt=source_prompt,
                classification_method="llm",
            ))
        else:
            segments.append(Segment(
                span_text=span,
                taxonomy_label="description",
                level="task_type",
                source_prompt=source_prompt,
                classification_method="llm_invalid",
            ))

    # Merge adjacent segments that share the same label and level,
    # then absorb context-like task_type segments into their neighbor task.
    segments = _merge_adjacent(segments)
    segments = _absorb_context_segments(segments)

    return segments or [_fallback_segment(source_prompt)]


def _fallback_segment(source_prompt: str) -> Segment:
    """Return a single fallback segment for the entire prompt."""
    return Segment(
        span_text=source_prompt,
        taxonomy_label="description",
        level="task_type",
        source_prompt=source_prompt,
        classification_method="llm_failed",
    )


# Labels that typically indicate the model mis-segmented context/input text
# as a separate "task" rather than keeping it with the actual task.
_CONTEXT_LABELS = frozenset({
    "information_extraction", "description", "explanation",
})


def _merge_adjacent(segments: list[Segment]) -> list[Segment]:
    """Merge consecutive segments that share the same taxonomy_label and level.

    This is a safety net that collapses residual over-segmentation — e.g., when
    the model splits a single task description into multiple segments all labeled
    ``code_generation / task_type``.
    """
    if len(segments) <= 1:
        return segments

    merged: list[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        if (
            seg.taxonomy_label == prev.taxonomy_label
            and seg.level == prev.level
            and seg.classification_method == prev.classification_method
        ):
            # Combine span texts
            merged[-1] = Segment(
                span_text=prev.span_text + " " + seg.span_text,
                taxonomy_label=prev.taxonomy_label,
                level=prev.level,
                source_prompt=prev.source_prompt,
                classification_method=prev.classification_method,
            )
        else:
            merged.append(seg)

    return merged


def _absorb_context_segments(segments: list[Segment]) -> list[Segment]:
    """Absorb task_type segments that look like context into a neighboring task.

    Catches the pattern where the model splits "question + passage" or
    "classify + input text" into two task_type segments — e.g.,
    question_answering + information_extraction, or classification +
    information_extraction.  The context-like segment should be absorbed
    into the real task segment.
    """
    if len(segments) <= 1:
        return segments

    # Only act when ALL segments are task_type (no format/style constraints
    # were split off — those are legitimate splits).
    all_task_type = all(s.level == "task_type" for s in segments)
    if not all_task_type:
        # Still absorb context segments adjacent to a task, but leave
        # non-task_type segments untouched.
        return _absorb_context_into_task(segments)

    # All segments are task_type: find the "real" task (non-context label)
    # and absorb everything else into it.
    real_tasks = [s for s in segments if s.taxonomy_label not in _CONTEXT_LABELS]
    if len(real_tasks) == 1:
        # One clear task, absorb all others into it
        task = real_tasks[0]
        combined_span = " ".join(s.span_text for s in segments)
        return [Segment(
            span_text=combined_span,
            taxonomy_label=task.taxonomy_label,
            level=task.level,
            source_prompt=task.source_prompt,
            classification_method=task.classification_method,
        )]

    return segments


def _absorb_context_into_task(segments: list[Segment]) -> list[Segment]:
    """Absorb task_type context segments into the nearest real task segment.

    Handles prompts where format/style constraints were correctly split off,
    but context text was still detached from the task — e.g.,
    [classification, topic_scope, information_extraction] should become
    [classification (with input text absorbed), topic_scope].

    Context segments are absorbed even if non-task_type segments appear
    in between (those are kept in place as legitimate constraint splits).
    """
    # First pass: identify which segments are context vs real tasks
    real_task_idx: int | None = None
    for i, seg in enumerate(segments):
        if seg.level == "task_type" and seg.taxonomy_label not in _CONTEXT_LABELS:
            real_task_idx = i
            break

    if real_task_idx is None:
        return segments

    # Collect context segments to absorb
    context_indices: list[int] = []
    for i, seg in enumerate(segments):
        if i == real_task_idx:
            continue
        if seg.level == "task_type" and seg.taxonomy_label in _CONTEXT_LABELS:
            context_indices.append(i)

    if not context_indices:
        return segments

    # Build result: absorb context into the real task, keep others
    task_seg = segments[real_task_idx]
    context_spans = [segments[j].span_text for j in context_indices]
    combined_span = task_seg.span_text + " " + " ".join(context_spans)

    skip = set(context_indices)
    result: list[Segment] = []
    for i, seg in enumerate(segments):
        if i in skip:
            continue
        if i == real_task_idx:
            result.append(Segment(
                span_text=combined_span,
                taxonomy_label=task_seg.taxonomy_label,
                level=task_seg.level,
                source_prompt=task_seg.source_prompt,
                classification_method=task_seg.classification_method,
            ))
        else:
            result.append(seg)

    return result

"""
validator.py — Post-extraction cross-validation.

Run after run_normalization(), before run_augmentation() in pipeline.py.
Three tests:
  1. Substitution (round-trip): fill slots with random options; LLM classifies
     whether the filled prompt still matches the claimed task type.
  2. Template collision + merge logic: same-text templates → merge or flag.
  3. Slot coverage: flag (template, slot) pairs with < K compatible options.

Failures are warnings — the pipeline does not halt.
Output: dataset/output/validation_report.json
"""

from __future__ import annotations

import json
import logging
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dataset.schema import ValidationReport, TaskTemplate, Option
from dataset.store import TemplateStore

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent / "output"

_CLASSIFICATION_SYSTEM_PROMPT = """\
You are a task-type classifier. Given an instruction prompt and a claimed task type label, \
decide whether the prompt genuinely belongs to that task type.

Reply with ONLY a JSON object: {"match": true} or {"match": false}.
Do not explain.
"""


# ── Public API ────────────────────────────────────────────────────────────

def run_post_extraction_validation(
    store: TemplateStore,
    taxonomy_obj: "Any",  # Taxonomy object from collect_task_types
    *,
    n_fillings: int = 5,
    min_slot_coverage: int = 3,
    output_path: Path | None = None,
    model: str | None = None,
    device: str | None = None,
    gpu_memory_utilization: float = 0.7,
) -> ValidationReport:
    """Run all three validation tests and return a ValidationReport.

    Parameters
    ----------
    store
        The TemplateStore after normalization.
    taxonomy_obj
        A Taxonomy object (from collect_task_types.Taxonomy) — used as
        fallback when no LLM model is available.
    n_fillings
        Number of random slot-fillings per template for the substitution test.
    min_slot_coverage
        Minimum number of compatible options per slot for a template to pass
        the coverage test.
    output_path
        Where to write the JSON report. Defaults to output/validation_report.json.
    model
        HuggingFace model ID for LLM-based substitution classification.
        When None, falls back to regex-based Taxonomy.matches().
    device
        CUDA device for vLLM.
    gpu_memory_utilization
        Fraction of GPU memory for vLLM.
    """
    templates = list(store.templates.values())
    options = list(store.options.values())

    logger.info(
        f"Running post-extraction validation: "
        f"{len(templates)} templates, {len(options)} options"
    )

    # ── Build lookup structures ───────────────────────────────────────
    options_by_slot: dict[str, list[Option]] = defaultdict(list)
    for o in options:
        options_by_slot[o.slot].append(o)

    # ── Test 1: substitution round-trip ───────────────────────────────
    substitution_failures = _test_substitution(
        templates, options_by_slot, taxonomy_obj, n_fillings,
        model=model, device=device,
        gpu_memory_utilization=gpu_memory_utilization,
    )

    # ── Test 2: template collision + merge logic ───────────────────────
    same_type_merges, cross_type_merges, semantic_collisions = _test_collision(
        templates, options_by_slot
    )

    # ── Test 3: slot coverage ─────────────────────────────────────────
    undercovered_slots = _test_slot_coverage(
        templates, options_by_slot, min_slot_coverage
    )

    # ── Degenerate templates ──────────────────────────────────────────
    degenerate_templates = [
        {"template_id": t.id, "template_text": t.text}
        for t in templates
        if t.source == "ablation_degenerate"
    ]

    report = ValidationReport(
        substitution_failures=substitution_failures,
        same_type_merges=same_type_merges,
        cross_type_merges=cross_type_merges,
        semantic_collisions=semantic_collisions,
        undercovered_slots=undercovered_slots,
        degenerate_templates=degenerate_templates,
        n_templates_tested=len(templates),
        n_options_tested=len(options),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    _log_summary(report)

    if output_path is None:
        output_path = OUTPUT_DIR / "validation_report.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _write_report(report, output_path)

    return report


# ── Test 1: substitution ──────────────────────────────────────────────────

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")


def _fill_slots(template_text: str, slot_values: dict[str, str]) -> str:
    """Replace {slot} placeholders with provided values."""
    result = template_text
    for slot, value in slot_values.items():
        result = result.replace(f"{{{slot}}}", value)
    return result


def _test_substitution(
    templates: list[TaskTemplate],
    options_by_slot: dict[str, list[Option]],
    taxonomy_obj: "Any",
    n_fillings: int,
    *,
    model: str | None = None,
    device: str | None = None,
    gpu_memory_utilization: float = 0.7,
) -> list[dict]:
    """Test 1: fill slots N times and check that filled text still matches task type.

    Uses LLM classification when *model* is provided, otherwise falls back
    to regex-based Taxonomy.matches().
    """
    failures: list[dict] = []
    rng = random.Random(42)

    task_type_templates = [t for t in templates if t.level == "task_type" and t.slots]

    # Generate all filled prompts first
    filled_items: list[tuple[TaskTemplate, str]] = []
    for tmpl in task_type_templates:
        if not tmpl.task_type:
            continue
        for _ in range(n_fillings):
            slot_values: dict[str, str] = {}
            for slot in tmpl.slots:
                candidates = options_by_slot.get(slot, [])
                compat = [
                    o for o in candidates
                    if tmpl.task_type in o.compatible_task_types
                ] or candidates
                if not compat:
                    slot_values[slot] = slot
                else:
                    slot_values[slot] = rng.choice(compat).value
            filled = _fill_slots(tmpl.text, slot_values)
            filled_items.append((tmpl, filled))

    if not filled_items:
        logger.info("  Substitution test: no task_type templates with slots to test")
        return failures

    # Classify — LLM batch or regex fallback
    if model:
        match_results = _classify_substitutions_llm(
            filled_items, model=model, device=device,
            gpu_memory_utilization=gpu_memory_utilization,
        )
    else:
        match_results = []
        for tmpl, filled in filled_items:
            try:
                match_results.append(taxonomy_obj.matches(filled, tmpl.task_type))
            except Exception:
                match_results.append(False)

    # Aggregate per template
    idx = 0
    for tmpl in task_type_templates:
        if not tmpl.task_type:
            continue
        successes = sum(1 for r in match_results[idx:idx + n_fillings] if r)
        idx += n_fillings

        fail_rate = 1.0 - (successes / n_fillings) if n_fillings > 0 else 0.0
        if fail_rate > 0.4:
            failures.append({
                "template_id": tmpl.id,
                "template_text": tmpl.text,
                "task_type": tmpl.task_type,
                "fail_rate": round(fail_rate, 2),
                "n_tested": n_fillings,
            })

    if failures:
        logger.warning(
            f"  Substitution test: {len(failures)}/{len(task_type_templates)} "
            f"task_type templates are fragile"
        )
    else:
        logger.info(
            f"  Substitution test: all {len(task_type_templates)} "
            f"task_type templates pass"
        )
    return failures


_VALIDATION_BATCH_SIZE = 500  # sub-batch size for LLM classification


def _classify_substitutions_llm(
    filled_items: list[tuple[TaskTemplate, str]],
    *,
    model: str,
    device: str | None,
    gpu_memory_utilization: float,
) -> list[bool]:
    """Use an LLM to classify whether each filled prompt matches its task type.

    Splits the work into sub-batches of _VALIDATION_BATCH_SIZE to avoid
    OOM crashes when the full set is too large for a single vLLM call.
    """
    from dataset.local_llm import generate_text_batch
    from pydantic import BaseModel

    class MatchResponse(BaseModel):
        match: bool

    messages_batch = []
    for tmpl, filled in filled_items:
        messages_batch.append([
            {"role": "system", "content": _CLASSIFICATION_SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Instruction prompt:\n\"{filled}\"\n\n"
                f"Claimed task type: \"{tmpl.task_type}\"\n\n"
                f"Does this prompt belong to the \"{tmpl.task_type}\" task type?"
            )},
        ])

    logger.info(f"  LLM substitution classification: {len(messages_batch)} prompts…")

    all_raws: list[str] = []
    n_batches = (len(messages_batch) + _VALIDATION_BATCH_SIZE - 1) // _VALIDATION_BATCH_SIZE
    for batch_idx in range(n_batches):
        start = batch_idx * _VALIDATION_BATCH_SIZE
        end = min(start + _VALIDATION_BATCH_SIZE, len(messages_batch))
        sub_batch = messages_batch[start:end]
        try:
            raws = generate_text_batch(
                model, sub_batch,
                temperature=0.0, device=device,
                enable_thinking=False,
                json_schema=MatchResponse,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            all_raws.extend(raws)
        except Exception as e:
            logger.warning(
                f"  LLM classification failed on batch {batch_idx + 1}/{n_batches}: {e}; "
                f"marking {len(sub_batch)} items as fail"
            )
            all_raws.extend([""] * len(sub_batch))

    results: list[bool] = []
    for raw in all_raws:
        raw = raw.strip()
        # Try to extract {"match": true/false}
        try:
            # Handle thinking tags if present
            raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            data = json.loads(raw)
            results.append(bool(data.get("match", False)))
        except (json.JSONDecodeError, AttributeError):
            # Fallback: look for true/false in text
            results.append("true" in raw.lower() and "false" not in raw.lower())

    return results


# ── Test 2: collision + merge ──────────────────────────────────────────────

def _fill_most_frequent(
    template: TaskTemplate,
    options_by_slot: dict[str, list[Option]],
) -> str:
    """Fill a template with the first available option for each slot."""
    text = template.text
    for slot in template.slots:
        candidates = options_by_slot.get(slot, [])
        if candidates:
            text = text.replace(f"{{{slot}}}", candidates[0].value, 1)
        else:
            text = text.replace(f"{{{slot}}}", f"__{slot}__", 1)
    return text


def _test_collision(
    templates: list[TaskTemplate],
    options_by_slot: dict[str, list[Option]],
) -> tuple[list[dict], list[dict], list[dict]]:
    """Test 2: find same-text templates and decide merge vs. flag."""
    # Fill all templates
    filled_texts: dict[str, str] = {}
    for t in templates:
        filled_texts[t.id] = _fill_most_frequent(t, options_by_slot)

    # Group by (filled_text, task_type) for same-type collision
    by_text_type: dict[tuple[str, str], list[TaskTemplate]] = defaultdict(list)
    # Group by filled_text only for cross-type collision
    by_text_only: dict[str, list[TaskTemplate]] = defaultdict(list)

    for t in templates:
        ft = filled_texts[t.id]
        by_text_type[(ft, t.task_type)].append(t)
        by_text_only[ft].append(t)

    same_type_merges: list[dict] = []
    for (_, _), group in by_text_type.items():
        if len(group) > 1:
            keep = group[0]
            for dup in group[1:]:
                same_type_merges.append({
                    "t1_id": keep.id,
                    "t2_id": dup.id,
                    "t1_text": keep.text,
                    "t2_text": dup.text,
                })

    cross_type_merges: list[dict] = []
    semantic_collisions: list[dict] = []

    # IDs already flagged as same-type duplicates
    same_type_dup_ids = {m["t2_id"] for m in same_type_merges}

    for filled_text, group in by_text_only.items():
        if len(group) < 2:
            continue
        distinct_types = list({t.task_type for t in group})
        if len(distinct_types) < 2:
            continue  # same task_type — handled above

        # Filter out already-flagged same-type dups
        group = [t for t in group if t.id not in same_type_dup_ids]
        if len(group) < 2:
            continue

        # Check if slot names are the same across all templates in group
        slot_sets = [frozenset(t.slots) for t in group]
        all_same_slots = len(set(slot_sets)) == 1

        if all_same_slots:
            # Same slots = mergeable cross-type templates
            keep = group[0]
            for other in group[1:]:
                cross_type_merges.append({
                    "t1_id": keep.id,
                    "t2_id": other.id,
                    "t1_task_type": keep.task_type,
                    "t2_task_type": other.task_type,
                    "filled_text": filled_text,
                })
        else:
            # Different slot names despite same filled text → semantic collision
            for i, t1 in enumerate(group):
                for t2 in group[i + 1:]:
                    semantic_collisions.append({
                        "t1_id": t1.id,
                        "t2_id": t2.id,
                        "filled_text": filled_text,
                        "t1_slots": t1.slots,
                        "t2_slots": t2.slots,
                    })

    logger.info(
        f"  Collision test: {len(same_type_merges)} same-type merges, "
        f"{len(cross_type_merges)} cross-type merges, "
        f"{len(semantic_collisions)} semantic collisions"
    )
    return same_type_merges, cross_type_merges, semantic_collisions


# ── Test 3: slot coverage ─────────────────────────────────────────────────

def _test_slot_coverage(
    templates: list[TaskTemplate],
    options_by_slot: dict[str, list[Option]],
    min_coverage: int,
) -> list[dict]:
    """Test 3: flag (template, slot) pairs where coverage < min_coverage.

    Zero-slot templates are valid and are not flagged.
    """
    undercovered: list[dict] = []

    for tmpl in templates:
        if not tmpl.slots:
            continue  # zero-slot template — valid, skip
        for slot in tmpl.slots:
            all_opts = options_by_slot.get(slot, [])
            compat = [
                o for o in all_opts
                if tmpl.task_type in o.compatible_task_types
            ] or all_opts
            n = len(compat)
            if n < min_coverage:
                undercovered.append({
                    "template_id": tmpl.id,
                    "template_text": tmpl.text,
                    "slot": slot,
                    "option_count": n,
                })

    if undercovered:
        logger.warning(
            f"  Coverage test: {len(undercovered)} (template, slot) pairs "
            f"with < {min_coverage} options"
        )
    else:
        logger.info(
            f"  Coverage test: all slotted templates have ≥ {min_coverage} options"
        )
    return undercovered


# ── Output ────────────────────────────────────────────────────────────────

def _log_summary(report: ValidationReport) -> None:
    logger.info(
        f"Validation summary — {report.n_templates_tested} templates tested:\n"
        f"  substitution_failures: {len(report.substitution_failures)}\n"
        f"  same_type_merges:      {len(report.same_type_merges)}\n"
        f"  cross_type_merges:     {len(report.cross_type_merges)}\n"
        f"  semantic_collisions:   {len(report.semantic_collisions)}\n"
        f"  undercovered_slots:    {len(report.undercovered_slots)}\n"
        f"  degenerate_templates:  {len(report.degenerate_templates)}"
    )


def _write_report(report: ValidationReport, path: Path) -> None:
    import dataclasses
    data = dataclasses.asdict(report)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    logger.info(f"  Validation report written → {path}")

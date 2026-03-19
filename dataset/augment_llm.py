"""
augment_llm.py — LLM-based semantic option augmentation.

Generates semantically varied alternatives for extracted options using a local
LLM. For each seed option, produces shortened, alternative, and generalized
variations that can fill the same template slots.

Usage
─────
  # Augment all content-slot options
  python -m dataset.augment_llm

  # Limit to N seed options
  python -m dataset.augment_llm --max-options 500

  # Use a specific model and device
  python -m dataset.augment_llm --model meta-llama/Llama-3.1-8B-Instruct --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dataset.schema import (
    Option, AugmentationResult,
    TEMPLATES_PATH, OPTIONS_PATH, AUGMENTED_OPTIONS_PATH,
    option_id,
)
from dataset.token_counter import token_length, set_token_counter
from dataset.store import TemplateStore
from dataset.local_llm import generate_text_batch

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Slots that contain semantic content worth augmenting
_CONTENT_SLOTS = frozenset({
    "topic", "description", "text", "passage", "question", "context",
    "content", "subject", "title", "source", "code", "example", "keyword",
    "person", "role", "task", "category",
})

# ── Augmentation prompt ──────────────────────────────────────────────────

_AUGMENTATION_SYSTEM_PROMPT = """\
You are an option augmenter for a prompt template system. Given an option \
value and its slot name (and optionally the template it came from), generate \
semantically varied alternatives.

For each option, produce 3-5 variations of these types:
- "shortened": A shorter form that preserves the core meaning. \
Strip surrounding context like "the book", "the following paragraph", etc. \
If the value is already short, skip this type.
- "alternative": A different concrete value of the SAME semantic type. \
E.g., for a book title, give another book title; for a country, give another country.
- "generalized": A more abstract or generic form. \
E.g., "the book 'War and Peace'" → "a classic Russian novel".

Rules:
- Each variation must be usable as a drop-in replacement in the same template slot.
- Do NOT repeat the original value.
- Keep alternative values realistic and diverse.
- For code, produce different but syntactically valid code of similar complexity.
- For passages/text, produce different content about different topics.

Output valid JSON only:
{"variations": [{"value": "...", "variation_type": "shortened|alternative|generalized"}, ...]}
"""


def _select_seed_options(
    store: TemplateStore,
    max_options: int | None = None,
) -> list[tuple[Option, str]]:
    """Select options suitable for LLM augmentation.

    Returns (option, template_context) pairs where template_context is the
    template text this option was originally extracted from (for context).
    """
    seeds: list[tuple[Option, str]] = []

    for opt in store.options.values():
        # Only augment content-slot options
        if opt.slot not in _CONTENT_SLOTS:
            continue
        # Skip very short (single token) or very long options
        if opt.token_length < 2 or opt.token_length > 500:
            continue
        # Skip programmatically augmented options
        if opt.source == "augmented" or opt.source_option_id is not None:
            continue

        # Find a template context for this option
        template_context = ""
        for tid in opt.compatible_templates:
            tmpl = store.templates.get(tid)
            if tmpl:
                template_context = tmpl.text
                break

        seeds.append((opt, template_context))

    # Sort by diversity: spread across slots
    seeds.sort(key=lambda x: (x[0].slot, x[0].value))

    if max_options is not None and len(seeds) > max_options:
        # Sample evenly across slots
        from collections import defaultdict
        import random
        rng = random.Random(42)
        by_slot: dict[str, list] = defaultdict(list)
        for s in seeds:
            by_slot[s[0].slot].append(s)
        selected: list[tuple[Option, str]] = []
        per_slot = max(1, max_options // len(by_slot))
        for slot_seeds in by_slot.values():
            rng.shuffle(slot_seeds)
            selected.extend(slot_seeds[:per_slot])
        seeds = selected[:max_options]

    return seeds


def _build_augmentation_messages(
    option: Option,
    template_context: str,
) -> list[dict[str, str]]:
    """Build chat messages for augmenting a single option."""
    user_content = f"Slot: {option.slot}\nValue: {option.value}"
    if template_context:
        user_content += f"\nTemplate: {template_context}"
    return [
        {"role": "system", "content": _AUGMENTATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _parse_augmentation_result(
    raw: str,
    seed_option: Option,
) -> list[Option]:
    """Parse LLM output into augmented Option objects."""
    import re
    raw = raw.strip()
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract JSON object
        brace_start = raw.find("{")
        if brace_start == -1:
            return []
        try:
            # Find matching closing brace
            depth = 0
            end = brace_start
            for i in range(brace_start, len(raw)):
                if raw[i] == "{":
                    depth += 1
                elif raw[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            data = json.loads(raw[brace_start:end + 1])
        except (json.JSONDecodeError, IndexError):
            return []

    variations = data.get("variations", [])
    results: list[Option] = []
    seen_values: set[str] = {seed_option.value.lower().strip()}

    for var in variations:
        value = var.get("value", "").strip()
        variation_type = var.get("variation_type", "alternative")
        if not value:
            continue
        # Skip duplicates of the original
        if value.lower().strip() in seen_values:
            continue
        seen_values.add(value.lower().strip())

        oid = option_id(seed_option.slot, value)
        results.append(Option(
            id=oid,
            value=value,
            slot=seed_option.slot,
            compatible_task_types=list(seed_option.compatible_task_types),
            compatible_templates=list(seed_option.compatible_templates),
            token_length=token_length(value),
            source="llm_augmented",
            tags=[variation_type],
            source_option_id=seed_option.id,
        ))

    return results


def augment_options_llm(
    store: TemplateStore,
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str | None = None,
    batch_size: int = 32,
    max_options: int | None = None,
    gpu_memory_utilization: float = 0.7,
) -> list[Option]:
    """Augment options via LLM and return the new options.

    Does NOT modify the store — caller decides what to do with results.
    """
    seeds = _select_seed_options(store, max_options=max_options)
    if not seeds:
        logger.warning("No seed options selected for augmentation.")
        return []

    logger.info(f"LLM augmentation: {len(seeds)} seed options (batch_size={batch_size})")

    all_augmented: list[Option] = []

    for batch_start in range(0, len(seeds), batch_size):
        batch = seeds[batch_start: batch_start + batch_size]
        batch_msgs = [
            _build_augmentation_messages(opt, ctx) for opt, ctx in batch
        ]

        try:
            raws = generate_text_batch(
                model, batch_msgs,
                temperature=0.3,  # some creativity for variations
                device=device,
                enable_thinking=False,
                json_schema=AugmentationResult,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except Exception as e:
            logger.warning(f"  Augmentation batch failed: {e}")
            continue

        for (seed_opt, _ctx), raw in zip(batch, raws):
            if not raw:
                continue
            try:
                new_opts = _parse_augmentation_result(raw, seed_opt)
                all_augmented.extend(new_opts)
            except Exception as e:
                logger.warning(f"  Parse augmentation failed: {e}")

        if (batch_start + batch_size) % (batch_size * 10) == 0:
            logger.info(
                f"  …{batch_start + len(batch)}/{len(seeds)} seeds processed, "
                f"{len(all_augmented)} variations so far"
            )

    logger.info(
        f"LLM augmentation done: {len(all_augmented)} variations "
        f"from {len(seeds)} seeds"
    )
    return all_augmented


# ── CLI ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-based semantic option augmentation",
    )
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-options", type=int, default=None,
                        help="Max seed options to augment (default: all content-slot options)")
    parser.add_argument("--gpu-memory", type=float, default=0.7)
    parser.add_argument("--tokenizer", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct")
    args = parser.parse_args()

    set_token_counter(args.tokenizer)

    # Load base extraction results (not augmented)
    store = TemplateStore.load()
    if not store.options:
        logger.error("No options found in options.json. Run extraction first.")
        sys.exit(1)

    logger.info(f"Loaded {len(store.templates)} templates, {len(store.options)} options")

    augmented = augment_options_llm(
        store,
        model=args.model,
        device=args.device,
        batch_size=args.batch_size,
        max_options=args.max_options,
        gpu_memory_utilization=args.gpu_memory,
    )

    if not augmented:
        logger.warning("No augmented options produced.")
        return

    # Load existing augmented options (if any) and merge
    existing_augmented: list[dict] = []
    if AUGMENTED_OPTIONS_PATH.exists():
        with open(AUGMENTED_OPTIONS_PATH) as f:
            existing_augmented = json.load(f)
        logger.info(f"  Loaded {len(existing_augmented)} existing augmented options")

    existing_ids = {item["id"] for item in existing_augmented}
    new_entries = []
    for opt in augmented:
        if opt.id not in existing_ids:
            new_entries.append({
                "id": opt.id, "value": opt.value, "slot": opt.slot,
                "compatible_task_types": opt.compatible_task_types,
                "compatible_templates": opt.compatible_templates,
                "token_length": opt.token_length, "source": opt.source,
                "tags": opt.tags, "source_option_id": opt.source_option_id,
            })

    combined = existing_augmented + new_entries
    with open(AUGMENTED_OPTIONS_PATH, "w") as f:
        json.dump(combined, f, indent=2, ensure_ascii=False)
    logger.info(
        f"Saved {len(combined)} total augmented options "
        f"({len(new_entries)} new) → {AUGMENTED_OPTIONS_PATH}"
    )


if __name__ == "__main__":
    main()

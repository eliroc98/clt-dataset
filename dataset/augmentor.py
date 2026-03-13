"""
augmentor.py — Programmatic option-pool augmentation.

Expands the option pool with numeric, keyword, language, and other values
compatible with the current taxonomy labels.
"""

from __future__ import annotations

import random
from collections import defaultdict

from dataset.schema import Option, option_id
from dataset.token_counter import token_length

# ── Numeric augmentation rules ────────────────────────────────────────────
# Keys are taxonomy constraint labels; values map task_type → (min, max, step).

_NUMERIC_AUGMENTATION_RULES: dict[str, dict[str, tuple[int, int, int]]] = {
    "length_constraint": {
        # word counts
        "creative_writing":    (100, 1000, 50),
        "summarization":       (50,  500,  50),
        "question_answering":  (50,  300,  50),
        "explanation":         (100, 500,  50),
        "communication_writing": (50, 300, 50),
        "_default":            (50,  500,  50),
    },
    "structure_constraint": {
        # bullet / section counts
        "brainstorming": (3, 15, 1),
        "data_analysis": (3, 10, 1),
        "_default":      (3,  8, 1),
    },
}

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
    templates: list[Option],  # kept for compat; actually list[TaskTemplate]
    rng: random.Random,
    max_augmented_per_slot: int = 20,
) -> list[Option]:
    """
    Augment the option pool with programmatically generated variations.

    Produces numeric values (for length/structure constraints), keyword
    combinations, language names, end-phrases, letters, and forbidden-word
    combinations.  All augmented options carry taxonomy-aligned
    ``compatible_task_types``.
    """
    augmented: list[Option] = []
    seen_ids: set[str] = {o.id for o in existing_options}

    templates_by_task: dict[str, list[str]] = defaultdict(list)
    for t in templates:
        if hasattr(t, "task_type") and hasattr(t, "id"):
            templates_by_task[t.task_type].append(t.id)

    # ── Numeric ──────────────────────────────────────────────────────
    for constraint_label, task_rules in _NUMERIC_AUGMENTATION_RULES.items():
        all_compatible = [tt for tt in task_rules if tt != "_default"]
        for task_type, (lo, hi, step) in task_rules.items():
            if task_type == "_default":
                continue
            for val in range(lo, hi + 1, step):
                val_str = str(val)
                oid = option_id("n", f"{constraint_label}_{task_type}_{val_str}")
                if oid in seen_ids:
                    continue
                seen_ids.add(oid)
                compat_tasks = list(dict.fromkeys([constraint_label] + all_compatible))
                augmented.append(Option(
                    id=oid, value=val_str, slot="n",
                    compatible_task_types=compat_tasks,
                    compatible_templates=templates_by_task.get(constraint_label, []),
                    token_length=token_length(val_str),
                    source="augmented", tags=["numeric", task_type],
                ))

    # ── Individual keywords ───────────────────────────────────────────
    for kw in _KEYWORD_AUGMENTATION_POOL:
        oid = option_id("keyword", kw)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        augmented.append(Option(
            id=oid, value=kw, slot="keyword",
            compatible_task_types=["keyword_inclusion", "keyword_frequency"],
            compatible_templates=(
                templates_by_task.get("keyword_inclusion", [])
                + templates_by_task.get("keyword_frequency", [])
            ),
            token_length=token_length(kw),
            source="augmented", tags=["keyword"],
        ))

    # ── Keyword combos ────────────────────────────────────────────────
    for _ in range(max_augmented_per_slot):
        k = rng.randint(2, 4)
        kws = rng.sample(_KEYWORD_AUGMENTATION_POOL, k)
        combo = ", ".join(kws)
        oid = option_id("keywords", combo)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        augmented.append(Option(
            id=oid, value=combo, slot="keywords",
            compatible_task_types=["keyword_inclusion"],
            compatible_templates=templates_by_task.get("keyword_inclusion", []),
            token_length=token_length(combo),
            source="augmented", tags=["keyword_combo"],
        ))

    # ── Languages ─────────────────────────────────────────────────────
    for lang in _LANGUAGE_AUGMENTATION:
        oid = option_id("language", lang)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        augmented.append(Option(
            id=oid, value=lang, slot="language",
            compatible_task_types=["response_language"],
            compatible_templates=templates_by_task.get("response_language", []),
            token_length=token_length(lang),
            source="augmented", tags=["language"],
        ))

    # ── End phrases ───────────────────────────────────────────────────
    for phrase in _END_PHRASE_AUGMENTATION:
        oid = option_id("phrase", phrase)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        augmented.append(Option(
            id=oid, value=phrase, slot="phrase",
            compatible_task_types=["end_with"],
            compatible_templates=templates_by_task.get("end_with", []),
            token_length=token_length(phrase),
            source="augmented", tags=["phrase"],
        ))

    # ── Letters ───────────────────────────────────────────────────────
    for letter in _LETTER_AUGMENTATION:
        oid = option_id("letter", letter)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        augmented.append(Option(
            id=oid, value=letter, slot="letter",
            compatible_task_types=["letter_frequency"],
            compatible_templates=templates_by_task.get("letter_frequency", []),
            token_length=token_length(letter),
            source="augmented", tags=["letter"],
        ))

    # ── Forbidden word combos ─────────────────────────────────────────
    for _ in range(max_augmented_per_slot):
        k = rng.randint(2, 4)
        words = rng.sample(_FORBIDDEN_WORD_AUGMENTATION, k)
        combo = ", ".join(words)
        oid = option_id("words", combo)
        if oid in seen_ids:
            continue
        seen_ids.add(oid)
        augmented.append(Option(
            id=oid, value=combo, slot="words",
            compatible_task_types=["forbidden_words"],
            compatible_templates=templates_by_task.get("forbidden_words", []),
            token_length=token_length(combo),
            source="augmented", tags=["forbidden_words"],
        ))

    import logging
    logging.getLogger(__name__).info(f"  Augmented {len(augmented)} new options")
    return augmented

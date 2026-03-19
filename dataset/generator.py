"""
generator.py — Synthetic prompt generation by composing templates with options.
"""

from __future__ import annotations

import hashlib
import logging
import random
from collections import Counter

from dataset.schema import (
    TaskTemplate, Option, GeneratedPrompt,
    CONSTRAINT_LEVELS,
)
from dataset.store import TemplateStore
from dataset.token_counter import token_length

logger = logging.getLogger(__name__)


def _combo_hash(task: str, constraints: list[str], option_vals: list[str]) -> str:
    key = task + "|" + "|".join(sorted(constraints)) + "|" + "|".join(sorted(option_vals))
    return hashlib.md5(key.encode()).hexdigest()[:12]


class SyntheticGenerator:
    """
    Generate prompts by composing templates with compatible options.

    Parameters
    ----------
    store       TemplateStore with templates and options.
    taxonomy    Taxonomy dict (from taxonomy.json).
    density     Number of templates per prompt (1 = task only, 2+ = task + constraints).
    min_tokens  Minimum token length (0 = no limit).
    max_tokens  Maximum token length (0 = no limit).
    seed        Random seed.
    """

    def __init__(
        self,
        store: TemplateStore,
        taxonomy: dict,
        density: int = 3,
        min_tokens: int = 0,
        max_tokens: int = 0,
        seed: int = 42,
        compatibility_index: dict[str, list[str]] | None = None,
    ):
        self.store = store
        self.taxonomy = taxonomy
        self.compatibility_index = compatibility_index
        self.density = density
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.rng = random.Random(seed)

        self.task_templates = store.get_templates_for_level("task_type")
        self.constraint_templates: list[TaskTemplate] = []
        for lv in CONSTRAINT_LEVELS:
            self.constraint_templates.extend(store.get_templates_for_level(lv))

        self._task_usage: Counter = Counter()
        self._constraint_usage: Counter = Counter()
        self._seen_combos: set[str] = set()

        logger.info(
            f"  Generator ready: {len(self.task_templates)} task templates, "
            f"{len(self.constraint_templates)} constraint templates, density={density}"
        )

    def _weight(self, name: str, usage: Counter) -> float:
        return 1.0 / (1 + usage[name])

    def _fill_template(self, tmpl: TaskTemplate) -> tuple[str, list[str]]:
        """Fill a template's slots with compatible options."""
        if not tmpl.slots:
            return tmpl.text, []

        if self.compatibility_index is not None:
            compatible = self.store.get_semantically_compatible_options(
                tmpl, compatibility_index=self.compatibility_index,
            )
        else:
            compatible = self.store.get_compatible_options(tmpl)
        filled = tmpl.text
        option_ids: list[str] = []

        for slot in tmpl.slots:
            opts = compatible.get(slot, [])
            if not opts:
                val = str(self.rng.randint(3, 10)) if slot == "n" else ("" if slot == "constraints" else "...")
                filled = filled.replace(f"{{{slot}}}", val, 1)
            else:
                chosen = self.rng.choice(opts)
                filled = filled.replace(f"{{{slot}}}", chosen.value, 1)
                option_ids.append(chosen.id)

        return filled, option_ids

    def generate_one(self) -> GeneratedPrompt | None:
        """
        Generate a single prompt:
          1. Pick a task_type template (coverage-weighted)
          2. Pick (density-1) constraint templates (coverage-weighted)
          3. Fill slots with compatible options
          4. Check token-length constraints
        """
        if not self.task_templates:
            return None

        task_weights = [self._weight(t.task_type, self._task_usage) for t in self.task_templates]
        task_tmpl = self.rng.choices(self.task_templates, weights=task_weights, k=1)[0]

        n_constraints = max(self.density - 1, 0)
        chosen_constraints: list[TaskTemplate] = []

        if n_constraints > 0 and self.constraint_templates:
            available = list(self.constraint_templates)
            avail_weights = [self._weight(c.task_type, self._constraint_usage) for c in available]
            level_counts: Counter = Counter()
            seen_types: set[str] = set()

            for _ in range(n_constraints):
                if not available:
                    break
                picked = self.rng.choices(available, weights=avail_weights, k=1)[0]

                if level_counts[picked.level] >= 2:
                    alt = [
                        (a, w) for a, w in zip(available, avail_weights)
                        if level_counts[a.level] < 2 and a.task_type not in seen_types
                    ]
                    if alt:
                        alt_items, alt_wts = zip(*alt)
                        picked = self.rng.choices(alt_items, weights=alt_wts, k=1)[0]
                    else:
                        continue

                if picked.task_type in seen_types:
                    alt = [
                        (a, w) for a, w in zip(available, avail_weights)
                        if a.task_type not in seen_types and level_counts[a.level] < 2
                    ]
                    if alt:
                        alt_items, alt_wts = zip(*alt)
                        picked = self.rng.choices(alt_items, weights=alt_wts, k=1)[0]
                    else:
                        continue

                chosen_constraints.append(picked)
                level_counts[picked.level] += 1
                seen_types.add(picked.task_type)

                idx = available.index(picked)
                available.pop(idx)
                avail_weights.pop(idx)

        task_text, task_option_ids = self._fill_template(task_tmpl)
        constraint_texts: list[str] = []
        constraint_option_ids: list[str] = []
        constraint_template_ids: list[str] = []

        for ct in chosen_constraints:
            ct_text, ct_opts = self._fill_template(ct)
            ct_text = ct_text.strip()
            if ct_text:
                constraint_texts.append(ct_text)
                constraint_option_ids.extend(ct_opts)
                constraint_template_ids.append(ct.id)

        constraints_block = "\n".join(constraint_texts)
        if "{constraints}" in task_text:
            prompt = task_text.replace("{constraints}", constraints_block).strip()
        else:
            prompt = f"{task_text}\n\n{constraints_block}".strip() if constraints_block else task_text.strip()

        tok_len = token_length(prompt)
        if self.min_tokens > 0 and tok_len < self.min_tokens:
            return None
        if self.max_tokens > 0 and tok_len > self.max_tokens:
            return None

        active_labels = [task_tmpl.task_type] + [c.task_type for c in chosen_constraints]
        constraints_list = [c.task_type for c in chosen_constraints]
        all_option_ids = task_option_ids + constraint_option_ids
        combo_id = _combo_hash(
            task_tmpl.task_type, constraints_list,
            [self.store.options[oid].value for oid in all_option_ids if oid in self.store.options],
        )

        self._task_usage[task_tmpl.task_type] += 1
        for c in chosen_constraints:
            self._constraint_usage[c.task_type] += 1
        self._seen_combos.add(combo_id)

        return GeneratedPrompt(
            id=0,
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
        """Generate n prompts, retrying up to max_retries if length constraints aren't met."""
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
                logger.info(f"  …{i + 1}/{n} generated")
        return results

    def coverage_report(self) -> dict:
        return {
            "unique_combos": len(self._seen_combos),
            "task_usage": dict(self._task_usage.most_common()),
            "constraint_usage": dict(self._constraint_usage.most_common()),
        }

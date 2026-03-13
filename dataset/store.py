"""
store.py — TemplateStore: storage and retrieval for templates and options.

Supports:
  - Querying templates by task type and level
  - Finding compatible options for a given template
  - Merging extracted and augmented data
  - Backward-compatible loading (remaps old level names to current taxonomy names)
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from dataset.schema import (
    TaskTemplate, Option,
    TEMPLATES_PATH, OPTIONS_PATH,
    LEVEL_REMAP,
    template_id, option_id,
)

logger = logging.getLogger(__name__)


@dataclass
class TemplateStore:
    """
    Tidy storage for templates and options with compatibility tracking.
    """
    templates: dict[str, TaskTemplate] = field(default_factory=dict)
    options: dict[str, Option] = field(default_factory=dict)

    _by_task_type: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _by_level: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    _by_slot: dict[str, list[str]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def add_template(self, tmpl: TaskTemplate) -> None:
        if tmpl.id in self.templates:
            return
        self.templates[tmpl.id] = tmpl
        self._by_task_type[tmpl.task_type].append(tmpl.id)
        self._by_level[tmpl.level].append(tmpl.id)

    def add_option(self, opt: Option) -> None:
        if opt.id in self.options:
            existing = self.options[opt.id]
            for tt in opt.compatible_task_types:
                if tt not in existing.compatible_task_types:
                    existing.compatible_task_types.append(tt)
            for tid in opt.compatible_templates:
                if tid not in existing.compatible_templates:
                    existing.compatible_templates.append(tid)
            return
        self.options[opt.id] = opt
        self._by_slot[opt.slot].append(opt.id)

    def add_templates(self, templates: list[TaskTemplate]) -> None:
        for t in templates:
            self.add_template(t)

    def add_options(self, options: list[Option]) -> None:
        for o in options:
            self.add_option(o)

    def get_templates_for_task(self, task_type: str) -> list[TaskTemplate]:
        return [self.templates[tid] for tid in self._by_task_type.get(task_type, [])]

    def get_templates_for_level(self, level: str) -> list[TaskTemplate]:
        return [self.templates[tid] for tid in self._by_level.get(level, [])]

    def get_options_for_slot(self, slot: str) -> list[Option]:
        return [self.options[oid] for oid in self._by_slot.get(slot, [])]

    def get_compatible_options(self, template: TaskTemplate) -> dict[str, list[Option]]:
        """
        Return {slot_name: [compatible options]} for a template.

        An option is compatible if the template's task_type appears in the
        option's compatible_task_types, the template's id appears in
        compatible_templates, or the option has "_universal" as a type.
        Falls back to all options for a slot if none match directly.
        """
        result: dict[str, list[Option]] = {}
        for slot in template.slots:
            slot_options = self.get_options_for_slot(slot)
            compatible = [
                o for o in slot_options
                if (
                    template.task_type in o.compatible_task_types
                    or template.id in o.compatible_templates
                    or "_universal" in o.compatible_task_types
                )
            ]
            result[slot] = compatible if compatible else slot_options
        return result

    def summary(self) -> dict:
        multi_compat = sum(
            1 for o in self.options.values() if len(o.compatible_task_types) > 1
        )
        return {
            "n_templates": len(self.templates),
            "n_options": len(self.options),
            "n_options_multi_compatible": multi_compat,
            "templates_by_level": {
                lv: len(tids) for lv, tids in self._by_level.items()
            },
            "options_by_slot": {
                slot: len(oids) for slot, oids in self._by_slot.items()
            },
            "task_types": sorted(self._by_task_type.keys()),
        }

    def save(
        self,
        templates_path: Path = TEMPLATES_PATH,
        options_path: Path = OPTIONS_PATH,
    ) -> None:
        """Persist templates and options to JSON files."""
        templates_data = [
            {
                "id": t.id, "text": t.text, "slots": t.slots,
                "task_type": t.task_type, "level": t.level,
                "token_length": t.token_length, "source": t.source,
                "compatible_with": t.compatible_with,
            }
            for t in self.templates.values()
        ]
        with open(templates_path, "w") as f:
            json.dump(templates_data, f, indent=2, ensure_ascii=False)
        logger.info(f"  Templates saved → {templates_path} ({len(templates_data)} entries)")

        options_data = [
            {
                "id": o.id, "value": o.value, "slot": o.slot,
                "compatible_task_types": o.compatible_task_types,
                "compatible_templates": o.compatible_templates,
                "token_length": o.token_length, "source": o.source, "tags": o.tags,
            }
            for o in self.options.values()
        ]
        with open(options_path, "w") as f:
            json.dump(options_data, f, indent=2, ensure_ascii=False)
        logger.info(f"  Options saved → {options_path} ({len(options_data)} entries)")

    @classmethod
    def load(
        cls,
        templates_path: Path = TEMPLATES_PATH,
        options_path: Path = OPTIONS_PATH,
    ) -> "TemplateStore":
        """Load from JSON files, remapping old level names to current taxonomy names."""
        store = cls()

        if templates_path.exists():
            with open(templates_path) as f:
                for item in json.load(f):
                    # Migrate char_length → token_length
                    if "char_length" in item and "token_length" not in item:
                        item["token_length"] = item.pop("char_length")
                    elif "char_length" in item:
                        del item["char_length"]
                    # Remap old level names
                    item["level"] = LEVEL_REMAP.get(item.get("level", ""), item.get("level", "task_type"))
                    store.add_template(TaskTemplate(**item))
            logger.info(f"  Loaded {len(store.templates)} templates from {templates_path}")

        if options_path.exists():
            with open(options_path) as f:
                for item in json.load(f):
                    if "char_length" in item and "token_length" not in item:
                        item["token_length"] = item.pop("char_length")
                    elif "char_length" in item:
                        del item["char_length"]
                    store.add_option(Option(**item))
            logger.info(f"  Loaded {len(store.options)} options from {options_path}")

        return store

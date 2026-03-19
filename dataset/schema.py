"""
schema.py — Shared constants, dataclasses, and Pydantic models.

Imported by all other dataset modules; has no internal dependencies.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel as _BaseModel

# ── Paths ─────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

TAXONOMY_PATH = OUTPUT_DIR / "taxonomy" / "taxonomy.json"
SEGMENTS_PATH = OUTPUT_DIR / "segments.jsonl"
TEMPLATES_PATH = OUTPUT_DIR / "templates.json"
OPTIONS_PATH = OUTPUT_DIR / "options.json"
GENERATED_PATH = OUTPUT_DIR / "constructed_prompts.jsonl"
FEW_SHOT_EXAMPLES_PATH = OUTPUT_DIR / "few_shot_examples.json"

# ── Taxonomy levels (aligned with taxonomy.json) ──────────────────────────

LEVELS = ("task_type", "format_constraint", "content_style_constraint", "process_directive")
CONSTRAINT_LEVELS = ("format_constraint", "content_style_constraint", "process_directive")

# Remap old level names → new names (for backward-compat loading)
LEVEL_REMAP: dict[str, str] = {
    "content_task":       "task_type",
    "format":             "format_constraint",
    "style":              "content_style_constraint",
    "content_constraint": "content_style_constraint",
    "meta":               "process_directive",
}


# ── ID helpers ────────────────────────────────────────────────────────────

def template_id(task_type: str, text: str) -> str:
    return hashlib.md5(f"{task_type}|{text}".encode()).hexdigest()[:10]


def option_id(slot: str, value: str) -> str:
    return hashlib.md5(f"{slot}|{value}".encode()).hexdigest()[:10]


# ── Data structures ───────────────────────────────────────────────────────

@dataclass
class TaskTemplate:
    """A task specification template with {slot} placeholders."""
    id: str
    text: str
    slots: list[str]
    task_type: str
    level: str
    token_length: int = 0
    source: str = ""
    compatible_with: list[str] = field(default_factory=list)


@dataclass
class Option:
    """A concrete value that fills a slot in a TaskTemplate."""
    id: str
    value: str
    slot: str
    compatible_task_types: list[str]
    compatible_templates: list[str]
    token_length: int = 0
    source: str = ""
    tags: list[str] = field(default_factory=list)


@dataclass
class GeneratedPrompt:
    """One generated prompt with full provenance."""
    id: int
    prompt: str
    task_template_id: str
    constraint_template_ids: list[str]
    option_ids: list[str]
    active_labels: list[str]
    task: str
    constraints: list[str]
    density: int
    token_length: int
    combo_id: str


@dataclass
class Segment:
    """A clause from a prompt, labelled with a taxonomy category."""
    span_text: str
    taxonomy_label: str          # key in taxonomy dict
    level: str                   # one of LEVELS
    source_prompt: str
    classification_method: str   # "regex" | "verb_canon" | "llm"


@dataclass
class ValidationReport:
    """Results of post-extraction cross-validation."""
    substitution_failures: list[dict]   # {template_id, template_text, fail_rate}
    same_type_merges: list[dict]         # {t1_id, t2_id, merged_id}
    cross_type_merges: list[dict]        # {t1_id, t2_id, merged_id}
    semantic_collisions: list[dict]      # {t1_id, t2_id, filled_text}
    undercovered_slots: list[dict]       # {template_id, slot, option_count}
    degenerate_templates: list[dict]     # {template_id, template_text}
    n_templates_tested: int
    n_options_tested: int
    timestamp: str


# ── Pydantic schema for structured LLM output ─────────────────────────────

class _TemplateItem(_BaseModel):
    text: str
    slots: list[str]
    task_type: str
    level: str


class _OptionItem(_BaseModel):
    value: str
    slot: str
    compatible_task_types: list[str]


class ExtractionResult(_BaseModel):
    templates: list[_TemplateItem]
    options: list[_OptionItem]

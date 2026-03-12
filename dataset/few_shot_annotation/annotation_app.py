#!/usr/bin/env python3
"""
Annotation web app for building few-shot examples.

Opens a local browser UI at http://localhost:8765 for annotating prompts:
  - Select text spans from a prompt → mark as template or option value
  - Specify slot names and compatible task types
  - Saves annotations to dataset/output/few_shot_examples.json

Usage:
    python dataset/annotation_app.py
    python dataset/annotation_app.py --port 8888
    python dataset/annotation_app.py --no-browser
    python dataset/annotation_app.py --output path/to/my_annotations.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
TAXONOMY_PATH = OUTPUT_DIR / "taxonomy" / "taxonomy.json"
DEFAULT_EXAMPLES_PATH = OUTPUT_DIR / "few_shot_examples.json"

LEVELS = ["content_task", "format", "style", "content_constraint", "meta"]

# ─────────────────────────────────────────────────────────────────────────────
# Dataset registry  (same streaming logic as annotate_few_shot.py)
# ─────────────────────────────────────────────────────────────────────────────

try:
    sys.path.insert(0, str(BASE_DIR.parent))
    from dataset.collect_task_types import (
        DATASET_REGISTRY,
        load_hf_dataset,
        load_github_data,
        load_kcif_data,
        extract_instruction_text,
        Taxonomy,
        VERB_CANON,
        TASK_OBJECT_MAP,
        CANONICAL_VERB_TO_TASK,
        _VERB_PHRASE_RE,
        _ARTICLES,
        _STOPWORDS,
        _SKIP_ADJECTIVES,
    )
    from dataset.auth import resolve_hf_token  # loads .env as side-effect
    _REGISTRY_AVAILABLE = True
except Exception as _imp_err:
    DATASET_REGISTRY: dict = {}
    _REGISTRY_AVAILABLE = False
    print(f"WARNING: Could not import dataset registry: {_imp_err}")
    print("         /api/load_dataset will not work.")


def _stream_prompts(ds_name: str, config: dict, n: int = 20) -> list[str]:
    """Stream up to *n* random prompts — identical logic to annotate_few_shot."""
    source = config.get("source", "")
    print(f"  Streaming from {ds_name} ({source}) …", flush=True)
    if source == "huggingface":
        cfg = {**config, "sample_size": n * 4}
        samples = load_hf_dataset(cfg, max_samples=n * 4)
    elif source == "github_raw":
        samples = load_github_data(config)
    elif source == "kcif_github":
        samples = load_kcif_data(config)
    else:
        raise ValueError(f"Unknown source '{source}' for dataset '{ds_name}'")

    fields = config.get("instruction_fields", [])
    texts: list[str] = []
    for sample in samples:
        t = extract_instruction_text(sample, fields).strip()
        if t:
            texts.append(t)

    random.shuffle(texts)
    return texts[:n]


# ─────────────────────────────────────────────────────────────────────────────
# Task-type suggestion engine
# ─────────────────────────────────────────────────────────────────────────────

_cached_taxonomy: Taxonomy | None = None


def _get_taxonomy() -> Taxonomy:
    """Return a Taxonomy pre-loaded with seed entries + saved taxonomy."""
    global _cached_taxonomy
    if _cached_taxonomy is not None:
        return _cached_taxonomy
    if not _REGISTRY_AVAILABLE:
        _cached_taxonomy = None  # type: ignore[assignment]
        return None  # type: ignore[return-value]
    taxonomy = Taxonomy()
    # Merge entries from the saved taxonomy.json for richer detection
    if TAXONOMY_PATH.exists():
        try:
            saved = json.loads(TAXONOMY_PATH.read_text())
            for name, info in saved.items():
                lvl = info.get("level", "content_task")
                if name not in taxonomy:
                    # Create with a pattern derived from the entry name so
                    # detect() can actually match it in prompt text.
                    name_pat = name.replace("_", r"[\s_]")
                    pat = re.compile(rf"\b{name_pat}\b", re.I)
                    entry = taxonomy.get_or_create(name, lvl)
                    entry.patterns.append(pat)
                else:
                    # Seed entry already has patterns — just ensure level
                    taxonomy.get_or_create(name, lvl)
        except (json.JSONDecodeError, OSError):
            pass
    _cached_taxonomy = taxonomy
    return taxonomy


def _suggest_for_text(text: str) -> dict:
    """
    Analyse *text* (a prompt or template) and return suggested task types
    and levels using the taxonomy pattern-matching + verb-phrase extraction.
    """
    taxonomy = _get_taxonomy()
    if taxonomy is None:
        return {"suggestions": []}

    scored: dict[str, dict] = {}  # name → {level, score, source}

    # 1. Pattern-based detection from the Taxonomy's compiled regexes
    detected = taxonomy.detect(text)
    for name, hit in detected.items():
        if hit:
            entry = taxonomy[name]
            scored[name] = {
                "name": name,
                "level": entry.level,
                "score": 2,
                "source": "pattern",
            }

    # 2. Verb-phrase extraction (same logic as enrich_from_instructions)
    try:
        for m in _VERB_PHRASE_RE.finditer(text):
            raw_verb = m.group(1).lower()
            canon = VERB_CANON.get(raw_verb)
            if not canon:
                continue
            following = m.group(2).lower().split()
            obj_noun: str | None = None
            for w in following:
                w_clean = re.sub(r"[^a-z]", "", w)
                if not w_clean or len(w_clean) < 3:
                    continue
                if w_clean in _ARTICLES or w_clean in _STOPWORDS or w_clean in _SKIP_ADJECTIVES:
                    continue
                obj_noun = w_clean
                break

            task_name: str | None = None
            if obj_noun and obj_noun in TASK_OBJECT_MAP:
                task_name = TASK_OBJECT_MAP[obj_noun]
            if task_name is None and canon in CANONICAL_VERB_TO_TASK:
                task_name = CANONICAL_VERB_TO_TASK[canon]
            if task_name is None:
                snippet = f"{raw_verb} {obj_noun}" if obj_noun else raw_verb
                for entry in taxonomy:
                    if entry.matches(snippet):
                        task_name = entry.name
                        break

            if task_name:
                if task_name in scored:
                    scored[task_name]["score"] += 1
                else:
                    lvl = taxonomy[task_name].level if task_name in taxonomy else "content_task"
                    scored[task_name] = {
                        "name": task_name,
                        "level": lvl,
                        "score": 1,
                        "source": "verb",
                    }
    except Exception:
        pass  # verb extraction is best-effort

    suggestions = sorted(scored.values(), key=lambda s: -s["score"])
    return {"suggestions": suggestions}


# ─────────────────────────────────────────────────────────────────────────────
# Auto-load prompts from all datasets
# ─────────────────────────────────────────────────────────────────────────────

def _autoload_all(n_per_dataset: int) -> dict:
    """
    Load *n_per_dataset* prompts from every dataset in the registry.
    Returns the merged examples dict.
    """
    examples = _load_examples()
    total_added = 0

    for ds_name in sorted(DATASET_REGISTRY.keys()):
        config = DATASET_REGISTRY[ds_name]
        existing_texts = {item["prompt"] for item in examples.get(ds_name, [])}
        have = len(existing_texts)
        if have >= n_per_dataset:
            print(f"  {ds_name}: already have {have} prompts — skipping")
            continue
        need = n_per_dataset - have
        try:
            prompts = _stream_prompts(ds_name, config, n=need)
        except Exception as exc:
            print(f"  {ds_name}: error — {exc}")
            continue

        new = [p for p in prompts if p not in existing_texts]
        if not new:
            continue

        if ds_name not in examples:
            examples[ds_name] = []
        for p in new:
            examples[ds_name].append({"prompt": p, "annotation": None})
        total_added += len(new)
        print(f"  {ds_name}: +{len(new)} prompts loaded")

    if total_added:
        _save_examples(examples)
    print(f"Auto-load complete: {total_added} new prompts across {len(DATASET_REGISTRY)} datasets.")
    return examples


# ─────────────────────────────────────────────────────────────────────────────
# HTML / CSS / JS  (single-page app, embedded as a string)
# ─────────────────────────────────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Few-Shot Annotation Tool</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:       #16161e;
  --bg2:      #1f1f2e;
  --bg3:      #252537;
  --border:   #353550;
  --text:     #cdd6f4;
  --muted:    #6c7086;
  --blue:     #89b4fa;
  --green:    #a6e3a1;
  --yellow:   #f9e2af;
  --red:      #f38ba8;
  --purple:   #cba6f7;
  --teal:     #94e2d5;
  --orange:   #fab387;
}

body {
  font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  background: var(--bg);
  color: var(--text);
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  font-size: 14px;
}

/* ── Header ─────────────────────────────────────────────────── */
#header {
  background: var(--bg2);
  border-bottom: 1px solid var(--border);
  padding: 10px 18px;
  display: flex;
  align-items: center;
  gap: 16px;
  flex-shrink: 0;
}
#header h1 { font-size: 15px; color: var(--blue); font-weight: 600; white-space: nowrap; }
#header .spacer { flex: 1; }
#status-msg {
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 4px;
  min-width: 120px;
  text-align: center;
  transition: opacity 0.4s;
}
#status-msg.ok   { background: #1e3a2a; color: var(--green); border: 1px solid #2a5c3a; }
#status-msg.err  { background: #3a1e1e; color: var(--red);   border: 1px solid #5c2a2a; }
#status-msg.info { background: #1e2a40; color: var(--blue);  border: 1px solid #2a3a5c; }

/* ── Main layout ─────────────────────────────────────────────── */
#body-wrap {
  display: flex;
  flex: 1;
  overflow: hidden;
}

/* ── Sidebar ─────────────────────────────────────────────────── */
#sidebar {
  width: 220px;
  min-width: 150px;
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg2);
  flex-shrink: 0;
}
#sidebar-header {
  padding: 10px 12px 6px;
  display: flex;
  align-items: center;
  gap: 8px;
  border-bottom: 1px solid var(--border);
}
#sidebar-header span { font-size: 11px; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; flex: 1; }
#btn-add-prompt {
  background: var(--blue);
  color: var(--bg);
  border: none;
  border-radius: 4px;
  padding: 2px 7px;
  font-size: 16px;
  cursor: pointer;
  line-height: 1.4;
  font-weight: bold;
}
#btn-add-prompt:hover { filter: brightness(1.2); }

#prompt-list {
  overflow-y: auto;
  flex: 1;
  padding: 4px 0;
}
.ds-group-label {
  font-size: 10px;
  color: var(--muted);
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.07em;
  padding: 8px 12px 3px;
}
.prompt-item {
  padding: 6px 12px;
  cursor: pointer;
  border-left: 3px solid transparent;
  transition: background 0.15s;
  font-size: 12px;
  color: var(--muted);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.prompt-item:hover { background: var(--bg3); color: var(--text); }
.prompt-item.active { background: var(--bg3); border-left-color: var(--blue); color: var(--text); }
.prompt-item.done   { border-left-color: var(--green); color: var(--green); }
.prompt-item.done.active { background: var(--bg3); }

/* ── Prompt area ─────────────────────────────────────────────── */
#prompt-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  border-right: 1px solid var(--border);
  min-width: 0;
}
#prompt-area-header {
  padding: 8px 14px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}
#prompt-area-header .label { font-size: 11px; color: var(--muted); font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; flex: 1; }
#nav-prev, #nav-next {
  background: var(--bg3);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 3px 9px;
  cursor: pointer;
  font-size: 13px;
}
#nav-prev:hover, #nav-next:hover { background: var(--border); }
#prompt-counter { font-size: 11px; color: var(--muted); }

#prompt-display {
  flex: 1;
  overflow-y: auto;
  padding: 16px 18px;
  line-height: 1.7;
  white-space: pre-wrap;
  word-break: break-word;
  user-select: text;
  cursor: text;
  font-size: 13.5px;
}
#prompt-display.empty {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--muted);
  font-style: italic;
}

/* Selection toolbar */
#selection-bar {
  border-top: 1px solid var(--border);
  padding: 8px 14px;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  min-height: 40px;
}
#sel-text-preview {
  flex: 1;
  font-size: 12px;
  color: var(--muted);
  font-style: italic;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
#sel-text-preview.has-sel { color: var(--yellow); font-style: normal; font-weight: 500; }
.sel-btn {
  background: var(--bg3);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 4px 10px;
  cursor: pointer;
  font-size: 12px;
  white-space: nowrap;
  transition: background 0.1s;
}
.sel-btn:hover { background: var(--border); }
.sel-btn.to-tmpl:hover { border-color: var(--purple); color: var(--purple); }
.sel-btn.to-opt:hover  { border-color: var(--teal);   color: var(--teal);   }

/* ── Annotation panel ────────────────────────────────────────── */
#annotation-panel {
  width: 360px;
  min-width: 300px;
  max-width: 420px;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  background: var(--bg2);
  flex-shrink: 0;
}

/* Scrollable inner */
#annotation-scroll {
  flex: 1;
  overflow-y: auto;
  padding: 14px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.ann-section { display: flex; flex-direction: column; gap: 8px; }
.ann-section h3 {
  font-size: 11px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.07em;
  padding-bottom: 4px;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 6px;
}
.ann-section h3 .badge {
  background: var(--bg3);
  color: var(--blue);
  font-size: 10px;
  padding: 1px 5px;
  border-radius: 8px;
  font-weight: 700;
}

/* Form fields */
.field { display: flex; flex-direction: column; gap: 4px; }
.field label { font-size: 11px; color: var(--muted); }

input[type=text], textarea, select {
  background: var(--bg3);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 6px 9px;
  font-size: 12.5px;
  font-family: inherit;
  width: 100%;
  outline: none;
  transition: border-color 0.15s;
}
input[type=text]:focus, textarea:focus, select:focus { border-color: var(--blue); }
textarea { resize: vertical; min-height: 56px; }

.row { display: flex; gap: 8px; }
.row .field { flex: 1; }

/* Detected slots */
#tmpl-slots-preview {
  font-size: 11px;
  color: var(--teal);
  min-height: 16px;
}

.add-btn {
  background: var(--blue);
  color: var(--bg);
  border: none;
  border-radius: 5px;
  padding: 6px 12px;
  cursor: pointer;
  font-weight: 600;
  font-size: 12px;
  align-self: flex-start;
  transition: filter 0.1s;
}
.add-btn:hover { filter: brightness(1.15); }
.add-btn.green { background: var(--green); }

/* Items list */
.item-list { display: flex; flex-direction: column; gap: 6px; }
.item-card {
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px 10px;
  font-size: 12px;
  position: relative;
}
.item-card .item-main { font-weight: 500; color: var(--text); word-break: break-word; }
.item-card .item-meta { color: var(--muted); margin-top: 3px; font-size: 11px; }
.item-card .item-slots { color: var(--purple); margin-top: 2px; font-size: 11px; }
.item-card .item-compat { color: var(--teal); margin-top: 2px; font-size: 11px; }
.item-card .del-btn {
  position: absolute; top: 6px; right: 6px;
  background: none; border: none; color: var(--muted); cursor: pointer; font-size: 13px;
  padding: 0 3px; border-radius: 3px;
  transition: color 0.1s;
}
.item-card .del-btn:hover { color: var(--red); }

/* Footer bar */
#footer {
  border-top: 1px solid var(--border);
  padding: 10px 14px;
  display: flex;
  align-items: center;
  gap: 8px;
  flex-shrink: 0;
  background: var(--bg2);
}
.btn-save  { background: var(--green);  color: var(--bg); border: none; border-radius: 5px; padding: 7px 18px; font-weight: 600; cursor: pointer; font-size: 13px; }
.btn-skip  { background: var(--bg3);    color: var(--muted); border: 1px solid var(--border); border-radius: 5px; padding: 6px 14px; cursor: pointer; font-size: 12px; }
.btn-clear { background: var(--bg3);    color: var(--red);   border: 1px solid var(--border); border-radius: 5px; padding: 6px 14px; cursor: pointer; font-size: 12px; }
.btn-save:hover  { filter: brightness(1.1); }
.btn-skip:hover  { background: var(--border); }
.btn-clear:hover { background: var(--border); }

/* Modal */
#modal-backdrop {
  display: none;
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.6);
  z-index: 100;
  align-items: center;
  justify-content: center;
}
#modal-backdrop.open { display: flex; }
#modal {
  background: var(--bg2);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 22px 24px;
  width: 480px;
  max-width: 95vw;
  display: flex;
  flex-direction: column;
  gap: 14px;
  box-shadow: 0 8px 40px rgba(0,0,0,0.5);
}
#modal h2 { font-size: 14px; color: var(--blue); }
#modal .modal-actions { display: flex; gap: 8px; justify-content: flex-end; }
#modal .modal-actions button {
  border: 1px solid var(--border);
  background: var(--bg3);
  color: var(--text);
  border-radius: 5px;
  padding: 6px 14px;
  cursor: pointer;
  font-size: 12px;
}
#modal .modal-actions .btn-confirm {
  background: var(--blue); color: var(--bg); border-color: var(--blue); font-weight: 600;
}
#modal .modal-actions button:hover { filter: brightness(1.15); }

/* Multi-select pill list */
#compat-pills {
  display: flex;
  flex-wrap: wrap;
  gap: 5px;
  max-height: 100px;
  overflow-y: auto;
  padding: 4px;
  background: var(--bg3);
  border: 1px solid var(--border);
  border-radius: 5px;
}
.pill {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  cursor: pointer;
  border: 1px solid var(--border);
  background: var(--bg);
  color: var(--muted);
  transition: all 0.1s;
  user-select: none;
}
.pill.selected { background: var(--teal); color: var(--bg); border-color: var(--teal); font-weight: 600; }
.pill:hover:not(.selected) { border-color: var(--blue); color: var(--text); }

/* Highlight selected text in prompt */
::selection { background: rgba(137,180,250,0.25); }

/* ── Load-from-registry panel ────────────────────────────────── */
#load-panel {
  border-top: 1px solid var(--border);
  padding: 10px 12px;
  display: flex;
  flex-direction: column;
  gap: 6px;
  flex-shrink: 0;
  background: var(--bg2);
}
#load-panel-hdr {
  font-size: 10px;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
#load-panel select, #load-panel input[type=number] {
  width: 100%;
  background: var(--bg3);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 5px 7px;
  font-size: 12px;
  font-family: inherit;
  outline: none;
}
#load-panel select:focus, #load-panel input:focus { border-color: var(--blue); }
.load-row { display: flex; gap: 6px; }
.load-row input[type=number] { width: 60px; flex-shrink: 0; }
#btn-load-ds {
  flex: 1;
  background: var(--blue);
  color: var(--bg);
  border: none;
  border-radius: 4px;
  padding: 5px 8px;
  font-size: 12px;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
}
#btn-load-ds:hover  { filter: brightness(1.15); }
#btn-load-ds:disabled { opacity: 0.5; cursor: default; }
#load-msg {
  font-size: 11px;
  color: var(--muted);
  min-height: 14px;
  transition: color 0.2s;
}
#load-msg.ok  { color: var(--green); }
#load-msg.err { color: var(--red); }

/* ── Suggestions ──────────────────────────────────────────────── */
#suggest-area {
  display: flex;
  flex-wrap: wrap;
  gap: 4px;
  min-height: 0;
  transition: min-height 0.2s;
}
#suggest-area:empty { display: none; }
#suggest-label {
  font-size: 10px;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  width: 100%;
  margin-bottom: 2px;
  display: none;
}
#suggest-label.visible { display: block; }
.suggest-chip {
  font-size: 11px;
  padding: 2px 8px;
  border-radius: 10px;
  cursor: pointer;
  border: 1px solid var(--purple);
  background: rgba(203,166,247,0.1);
  color: var(--purple);
  transition: all 0.12s;
  user-select: none;
  white-space: nowrap;
}
.suggest-chip:hover {
  background: var(--purple);
  color: var(--bg);
}
.suggest-chip .chip-level {
  font-size: 9px;
  opacity: 0.7;
  margin-left: 3px;
}

/* ── Load All button ──────────────────────────────────────────── */
#btn-load-all {
  width: 100%;
  background: var(--bg3);
  color: var(--teal);
  border: 1px solid var(--border);
  border-radius: 4px;
  padding: 5px 8px;
  font-size: 11px;
  font-weight: 600;
  cursor: pointer;
  white-space: nowrap;
  margin-top: 4px;
}
#btn-load-all:hover { background: var(--border); border-color: var(--teal); }
#btn-load-all:disabled { opacity: 0.5; cursor: default; }

/* ── Option auto-compat banner ────────────────────────────────── */
#auto-compat-hint {
  font-size: 10px;
  color: var(--muted);
  font-style: italic;
  min-height: 14px;
}
</style>
</head>
<body>

<!-- Header -->
<div id="header">
  <h1>&#9998; Few-Shot Annotation Tool</h1>
  <div class="spacer"></div>
  <div id="status-msg" class="info">Loading…</div>
</div>

<!-- Main body -->
<div id="body-wrap">

  <!-- Sidebar: prompt list -->
  <div id="sidebar">
    <div id="sidebar-header">
      <span>Prompts</span>
      <button id="btn-add-prompt" title="Add new prompt" onclick="openAddPromptModal()">+</button>
    </div>
    <div id="prompt-list">
      <div style="padding:14px 12px; color:var(--muted); font-size:12px; font-style:italic;">No prompts loaded.<br>Click + to add one.</div>
    </div>

    <!-- Load prompts from dataset registry -->
    <div id="load-panel">
      <div id="load-panel-hdr">Load from registry</div>
      <select id="load-ds-select"><option value="">&#8212; dataset &#8212;</option></select>
      <div class="load-row">
        <input type="number" id="load-n" value="15" min="1" max="500"
               title="Number of prompts to stream">
        <button id="btn-load-ds" onclick="loadDataset()">Load</button>
      </div>
      <div id="load-msg"></div>
      <button id="btn-load-all" onclick="loadAllDatasets()">Load All Datasets</button>
    </div>
  </div>

  <!-- Prompt display area -->
  <div id="prompt-area">
    <div id="prompt-area-header">
      <span class="label">Prompt</span>
      <span id="prompt-counter"></span>
      <button id="nav-prev" onclick="navigate(-1)" title="Previous prompt">&#8249;</button>
      <button id="nav-next" onclick="navigate(+1)" title="Next prompt">&#8250;</button>
    </div>
    <div id="prompt-display" class="empty">← Select or add a prompt to begin</div>
    <div id="selection-bar">
      <span id="sel-text-preview">Select text in the prompt above…</span>
      <button class="sel-btn to-tmpl" onclick="fillTemplateFromSelection()" title="Copy selection into Template text field">→ Template</button>
      <button class="sel-btn to-opt"  onclick="fillOptionFromSelection()"   title="Copy selection into Option value field">→ Option</button>
    </div>
  </div>

  <!-- Annotation panel -->
  <div id="annotation-panel">
    <div id="annotation-scroll">

      <!-- Templates section -->
      <div class="ann-section" id="sec-templates">
        <h3>Templates <span class="badge" id="tmpl-badge">0</span></h3>

        <div class="field">
          <label>Template text — use {slot_name} for variable parts</label>
          <textarea id="tmpl-text" rows="2"
            placeholder='e.g.  Write {description} in {n} words'
            oninput="onTmplTextChange()"></textarea>
          <div id="tmpl-slots-preview"></div>
          <div id="suggest-label">Suggested task types</div>
          <div id="suggest-area"></div>
        </div>

        <div class="row">
          <div class="field">
            <label>Task type</label>
            <select id="tmpl-task-type">
              <option value="">— select —</option>
            </select>
          </div>
          <div class="field">
            <label>Level</label>
            <select id="tmpl-level">
              <option value="">— select —</option>
            </select>
          </div>
        </div>

        <button class="add-btn" onclick="addTemplate()">+ Add Template</button>
        <div class="item-list" id="templates-list"></div>
      </div>

      <!-- Divider -->
      <hr style="border-color: var(--border);">

      <!-- Options section -->
      <div class="ann-section" id="sec-options">
        <h3>Options <span class="badge" id="opts-badge">0</span></h3>

        <div class="field">
          <label>Option value (select from prompt or type)</label>
          <input type="text" id="opt-value" placeholder="e.g.  500">
        </div>

        <div class="row">
          <div class="field">
            <label>Slot name</label>
            <input type="text" id="opt-slot" placeholder="e.g.  n" list="slot-datalist" autocomplete="off">
            <datalist id="slot-datalist"></datalist>
          </div>
        </div>

        <div class="field">
          <label>Compatible task types <span style="color:var(--muted)">(click to toggle)</span></label>
          <div id="auto-compat-hint"></div>
          <div id="compat-pills"></div>
        </div>

        <button class="add-btn green" onclick="addOption()">+ Add Option</button>
        <div class="item-list" id="options-list"></div>
      </div>

    </div><!-- /annotation-scroll -->

    <!-- Footer: save / skip -->
    <div id="footer">
      <button class="btn-save"  onclick="saveAnnotation()">&#10003; Save</button>
      <button class="btn-skip"  onclick="skipPrompt()">Skip</button>
      <button class="btn-clear" onclick="clearAnnotation()">Clear</button>
    </div>

  </div><!-- /annotation-panel -->
</div><!-- /body-wrap -->

<!-- ── Add Prompt modal ───────────────────────────────────────── -->
<div id="modal-backdrop">
  <div id="modal">
    <h2>Add prompt</h2>
    <div class="field">
      <label>Dataset name</label>
      <input type="text" id="modal-dataset" placeholder="e.g.  IFEval" list="ds-datalist" autocomplete="off">
      <datalist id="ds-datalist"></datalist>
    </div>
    <div class="field">
      <label>Prompt text</label>
      <textarea id="modal-prompt" rows="6" placeholder="Paste the full prompt here…"></textarea>
    </div>
    <div class="modal-actions">
      <button onclick="closeModal()">Cancel</button>
      <button class="btn-confirm" onclick="confirmAddPrompt()">Add Prompt</button>
    </div>
  </div>
</div>

<!-- ── JavaScript ─────────────────────────────────────────────── -->
<script>

// ═══════════════════════════════════════════════════════════════
// State
// ═══════════════════════════════════════════════════════════════

let taskTypes = [];
let levels    = [];
// examples: { dataset: [ {prompt, annotation} ] }
let examples  = {};
// Flat ordered list: [ {dataset, prompt, annotation, localIdx} ]
let promptList = [];
let currentPIdx = -1;   // index into promptList
let selectedText = '';
// annotation being built for current prompt
let curAnnotation = { templates: [], options: [] };

// ═══════════════════════════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════════════════════════

async function boot() {
  try {
    const res  = await fetch('/api/state');
    const data = await res.json();
    taskTypes = data.task_types || [];
    levels    = data.levels    || [];
    examples  = data.examples  || {};
    populateSelectFromList('tmpl-task-type', taskTypes);
    populateLevels();
    buildPromptList();
    renderSidebar();
    if (promptList.length > 0) selectPrompt(0);
    setStatus('Ready', 'ok');
    initDatasets();
  } catch (e) {
    setStatus('Failed to load state', 'err');
  }
}

function populateSelectFromList(selectId, items) {
  const sel = document.getElementById(selectId);
  items.forEach(v => {
    const opt = document.createElement('option');
    opt.value = opt.textContent = v;
    sel.appendChild(opt);
  });
}

function populateLevels() {
  const sel = document.getElementById('tmpl-level');
  levels.forEach(v => {
    const opt = document.createElement('option');
    opt.value = opt.textContent = v;
    sel.appendChild(opt);
  });
}

function buildPromptList() {
  promptList = [];
  for (const [dataset, items] of Object.entries(examples)) {
    items.forEach((item, localIdx) => {
      promptList.push({ dataset, prompt: item.prompt, annotation: item.annotation, localIdx });
    });
  }
}

// ═══════════════════════════════════════════════════════════════
// Sidebar / navigation
// ═══════════════════════════════════════════════════════════════

function renderSidebar() {
  const listEl = document.getElementById('prompt-list');
  listEl.innerHTML = '';

  if (promptList.length === 0) {
    listEl.innerHTML = '<div style="padding:14px 12px; color:var(--muted); font-size:12px; font-style:italic;">No prompts loaded.<br>Click + to add one.</div>';
    return;
  }

  let lastDs = null;
  promptList.forEach((item, idx) => {
    if (item.dataset !== lastDs) {
      lastDs = item.dataset;
      const lbl = document.createElement('div');
      lbl.className = 'ds-group-label';
      lbl.textContent = item.dataset;
      listEl.appendChild(lbl);
    }
    const el = document.createElement('div');
    el.className = 'prompt-item' + (item.annotation ? ' done' : '') + (idx === currentPIdx ? ' active' : '');
    el.id = 'pi-' + idx;
    el.textContent = '#' + (item.localIdx + 1) + ' ' + truncate(item.prompt, 35);
    el.onclick = () => selectPrompt(idx);
    listEl.appendChild(el);
  });

  // Populate dataset datalist for the modal
  const dsDl = document.getElementById('ds-datalist');
  dsDl.innerHTML = '';
  Object.keys(examples).forEach(ds => {
    const opt = document.createElement('option');
    opt.value = ds;
    dsDl.appendChild(opt);
  });
}

function selectPrompt(idx) {
  if (idx < 0 || idx >= promptList.length) return;
  currentPIdx = idx;
  const item = promptList[idx];

  // Load annotation (existing or fresh)
  const existing = item.annotation;
  curAnnotation = existing && existing.templates
    ? JSON.parse(JSON.stringify(existing))
    : { templates: [], options: [] };

  // Update prompt display
  const display = document.getElementById('prompt-display');
  display.classList.remove('empty');
  display.textContent = item.prompt;

  // Update counter
  const ds = item.dataset;
  const dsItems = examples[ds] || [];
  document.getElementById('prompt-counter').textContent =
    ds + '  •  ' + (item.localIdx + 1) + ' / ' + dsItems.length;

  // Clear selection
  clearSelection();

  // Render annotation forms
  renderTemplates();
  renderOptions();
  renderCompatPills();
  updateSlotDatalist();

  // Highlight active sidebar item
  document.querySelectorAll('.prompt-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });

  // Clear form fields
  document.getElementById('tmpl-text').value = '';
  document.getElementById('tmpl-slots-preview').textContent = '';
  document.getElementById('tmpl-task-type').value = '';
  document.getElementById('tmpl-level').value = '';
  document.getElementById('opt-value').value = '';
  document.getElementById('opt-slot').value = '';

  // Fetch suggestions based on the full prompt text
  if (item.prompt && !existing) {
    fetchSuggestions(item.prompt);
  }
}

function navigate(delta) {
  selectPrompt(currentPIdx + delta);
}

// ═══════════════════════════════════════════════════════════════
// Text selection
// ═══════════════════════════════════════════════════════════════

document.getElementById('prompt-display').addEventListener('mouseup', () => {
  const sel = window.getSelection();
  const text = sel ? sel.toString().trim() : '';
  selectedText = text;
  const preview = document.getElementById('sel-text-preview');
  if (text) {
    preview.textContent = '\u201c' + truncate(text, 80) + '\u201d';
    preview.classList.add('has-sel');
  } else {
    preview.textContent = 'Select text in the prompt above\u2026';
    preview.classList.remove('has-sel');
  }
});

function clearSelection() {
  selectedText = '';
  const preview = document.getElementById('sel-text-preview');
  preview.textContent = 'Select text in the prompt above\u2026';
  preview.classList.remove('has-sel');
}

function fillTemplateFromSelection() {
  if (!selectedText) { setStatus('Select text first', 'info'); return; }
  document.getElementById('tmpl-text').value = selectedText;
  onTmplTextChange();
  document.getElementById('tmpl-text').focus();
}

function fillOptionFromSelection() {
  if (!selectedText) { setStatus('Select text first', 'info'); return; }
  document.getElementById('opt-value').value = selectedText;
  document.getElementById('opt-slot').focus();
}

// ═══════════════════════════════════════════════════════════════
// Template helpers
// ═══════════════════════════════════════════════════════════════

function detectSlots(text) {
  const matches = [...text.matchAll(/\{(\w+)\}/g)].map(m => m[1]);
  // deduplicate while preserving order
  return [...new Set(matches)];
}

let _suggestTimer = null;

function onTmplTextChange() {
  const text = document.getElementById('tmpl-text').value;
  const slots = detectSlots(text);
  const el = document.getElementById('tmpl-slots-preview');
  el.textContent = slots.length ? 'Slots detected: ' + slots.join(', ') : '';

  // Debounced suggestion
  clearTimeout(_suggestTimer);
  if (text.trim().length > 5) {
    _suggestTimer = setTimeout(() => fetchSuggestions(text.trim()), 400);
  } else {
    clearSuggestions();
  }
}

async function fetchSuggestions(text) {
  // Also include the full prompt text for richer context
  const promptText = currentPIdx >= 0 ? promptList[currentPIdx].prompt : '';
  const combined = promptText ? text + ' ' + promptText : text;
  try {
    const res = await fetch('/api/suggest', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: combined }),
    });
    const data = await res.json();
    renderSuggestions(data.suggestions || []);
  } catch (e) {
    clearSuggestions();
  }
}

function renderSuggestions(suggestions) {
  const area = document.getElementById('suggest-area');
  const label = document.getElementById('suggest-label');
  area.innerHTML = '';
  if (!suggestions.length) {
    label.classList.remove('visible');
    return;
  }
  label.classList.add('visible');
  suggestions.slice(0, 8).forEach(s => {
    const chip = document.createElement('span');
    chip.className = 'suggest-chip';
    chip.innerHTML = escHtml(s.name) + '<span class="chip-level">' + escHtml(s.level) + '</span>';
    chip.onclick = () => applySuggestion(s.name, s.level);
    area.appendChild(chip);
  });

  // Auto-fill the first suggestion if dropdowns are empty
  const ttSel = document.getElementById('tmpl-task-type');
  const lvlSel = document.getElementById('tmpl-level');
  if (!ttSel.value && suggestions.length) {
    applySuggestion(suggestions[0].name, suggestions[0].level);
  }
}

function applySuggestion(taskType, level) {
  const ttSel = document.getElementById('tmpl-task-type');
  const lvlSel = document.getElementById('tmpl-level');
  // Ensure option exists in dropdown (might be a newly discovered type)
  if (!Array.from(ttSel.options).some(o => o.value === taskType)) {
    const opt = document.createElement('option');
    opt.value = opt.textContent = taskType;
    ttSel.appendChild(opt);
    taskTypes.push(taskType);
    renderCompatPills();
  }
  ttSel.value = taskType;
  lvlSel.value = level;
  setStatus('Suggested: ' + taskType + ' (' + level + ')', 'info');
}

function clearSuggestions() {
  document.getElementById('suggest-area').innerHTML = '';
  document.getElementById('suggest-label').classList.remove('visible');
}

function addTemplate() {
  const text      = document.getElementById('tmpl-text').value.trim();
  const task_type = document.getElementById('tmpl-task-type').value;
  const level     = document.getElementById('tmpl-level').value;

  if (!text)      { setStatus('Template text is required', 'err'); return; }
  if (!task_type) { setStatus('Task type is required', 'err'); return; }
  if (!level)     { setStatus('Level is required', 'err'); return; }

  const slots = detectSlots(text);
  curAnnotation.templates.push({ text, slots, task_type, level });

  // Clear form
  document.getElementById('tmpl-text').value = '';
  document.getElementById('tmpl-slots-preview').textContent = '';
  document.getElementById('tmpl-task-type').value = '';
  document.getElementById('tmpl-level').value = '';

  renderTemplates();
  updateSlotDatalist();
  clearSuggestions();
  autoPreselectCompat();
  setStatus('Template added', 'ok');
}

function removeTemplate(i) {
  curAnnotation.templates.splice(i, 1);
  renderTemplates();
  updateSlotDatalist();
  autoPreselectCompat();
}

function renderTemplates() {
  const list = document.getElementById('templates-list');
  document.getElementById('tmpl-badge').textContent = curAnnotation.templates.length;
  list.innerHTML = '';
  curAnnotation.templates.forEach((t, i) => {
    const card = document.createElement('div');
    card.className = 'item-card';
    card.innerHTML =
      '<div class="item-main">' + escHtml(t.text) + '</div>' +
      '<div class="item-meta">' + escHtml(t.task_type) + '  &middot;  ' + escHtml(t.level) + '</div>' +
      (t.slots.length ? '<div class="item-slots">slots: ' + t.slots.map(escHtml).join(', ') + '</div>' : '') +
      '<button class="del-btn" title="Remove" onclick="removeTemplate(' + i + ')">&#x2715;</button>';
    list.appendChild(card);
  });
}

// ═══════════════════════════════════════════════════════════════
// Option helpers
// ═══════════════════════════════════════════════════════════════

// Track which task-type pills are selected for the option form
let selectedCompat = new Set();

function renderCompatPills() {
  const container = document.getElementById('compat-pills');
  container.innerHTML = '';
  taskTypes.forEach(tt => {
    const pill = document.createElement('span');
    pill.className = 'pill' + (selectedCompat.has(tt) ? ' selected' : '');
    pill.textContent = tt;
    pill.onclick = () => toggleCompat(tt, pill);
    container.appendChild(pill);
  });
}

// Auto-pre-select compatible task types based on existing templates
function autoPreselectCompat() {
  const hint = document.getElementById('auto-compat-hint');
  if (!curAnnotation.templates.length) {
    hint.textContent = '';
    return;
  }
  // Collect task types from all added templates
  const templateTypes = new Set(curAnnotation.templates.map(t => t.task_type));
  selectedCompat = new Set(templateTypes);
  renderCompatPills();
  hint.textContent = 'Auto-selected from templates: ' + [...templateTypes].join(', ');
}

function toggleCompat(tt, pill) {
  if (selectedCompat.has(tt)) {
    selectedCompat.delete(tt);
    pill.classList.remove('selected');
  } else {
    selectedCompat.add(tt);
    pill.classList.add('selected');
  }
}

function updateSlotDatalist() {
  const dl = document.getElementById('slot-datalist');
  dl.innerHTML = '';
  const allSlots = new Set(curAnnotation.templates.flatMap(t => t.slots));
  allSlots.forEach(s => {
    const opt = document.createElement('option');
    opt.value = s;
    dl.appendChild(opt);
  });
}

function addOption() {
  const value = document.getElementById('opt-value').value.trim();
  const slot  = document.getElementById('opt-slot').value.trim();

  if (!value) { setStatus('Option value is required', 'err'); return; }
  if (!slot)  { setStatus('Slot name is required', 'err'); return; }

  const compat = [...selectedCompat];
  curAnnotation.options.push({
    value,
    slot,
    compatible_task_types: compat,
  });

  // Clear form
  document.getElementById('opt-value').value = '';
  document.getElementById('opt-slot').value = '';
  selectedCompat.clear();
  renderCompatPills();
  renderOptions();
  setStatus('Option added', 'ok');
}

function removeOption(i) {
  curAnnotation.options.splice(i, 1);
  renderOptions();
}

function renderOptions() {
  const list = document.getElementById('options-list');
  document.getElementById('opts-badge').textContent = curAnnotation.options.length;
  list.innerHTML = '';
  curAnnotation.options.forEach((o, i) => {
    const card = document.createElement('div');
    card.className = 'item-card';
    card.innerHTML =
      '<div class="item-main">' + escHtml(o.value) + '</div>' +
      '<div class="item-meta">slot: <b>' + escHtml(o.slot) + '</b></div>' +
      (o.compatible_task_types.length
        ? '<div class="item-compat">' + o.compatible_task_types.map(escHtml).join(', ') + '</div>'
        : '') +
      '<button class="del-btn" title="Remove" onclick="removeOption(' + i + ')">&#x2715;</button>';
    list.appendChild(card);
  });
}

// ═══════════════════════════════════════════════════════════════
// Save / skip / clear
// ═══════════════════════════════════════════════════════════════

async function saveAnnotation() {
  if (currentPIdx < 0) { setStatus('No prompt selected', 'err'); return; }
  if (!curAnnotation.templates.length) { setStatus('Add at least one template', 'err'); return; }

  const item = promptList[currentPIdx];
  const payload = {
    dataset:      item.dataset,
    prompt_index: item.localIdx,
    prompt:       item.prompt,
    annotation:   curAnnotation,
  };

  try {
    const res = await fetch('/api/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (data.ok) {
      // Update local state
      promptList[currentPIdx].annotation = JSON.parse(JSON.stringify(curAnnotation));
      examples[item.dataset][item.localIdx].annotation = JSON.parse(JSON.stringify(curAnnotation));
      renderSidebar();
      setStatus('Saved \u2713', 'ok');
    } else {
      setStatus('Save failed: ' + (data.error || '?'), 'err');
    }
  } catch (e) {
    setStatus('Save failed: ' + e.message, 'err');
  }
}

function skipPrompt() {
  if (currentPIdx < promptList.length - 1) {
    selectPrompt(currentPIdx + 1);
  } else {
    setStatus('Already at last prompt', 'info');
  }
}

function clearAnnotation() {
  curAnnotation = { templates: [], options: [] };
  renderTemplates();
  renderOptions();
  selectedCompat.clear();
  renderCompatPills();
  setStatus('Cleared', 'info');
}

// ═══════════════════════════════════════════════════════════════
// Add prompt modal
// ═══════════════════════════════════════════════════════════════

function openAddPromptModal() {
  document.getElementById('modal-dataset').value = '';
  document.getElementById('modal-prompt').value  = '';
  document.getElementById('modal-backdrop').classList.add('open');
  setTimeout(() => document.getElementById('modal-dataset').focus(), 50);
}

function closeModal() {
  document.getElementById('modal-backdrop').classList.remove('open');
}

async function confirmAddPrompt() {
  const dataset = document.getElementById('modal-dataset').value.trim();
  const prompt  = document.getElementById('modal-prompt').value.trim();
  if (!dataset) { setStatus('Dataset name required', 'err'); return; }
  if (!prompt)  { setStatus('Prompt text required',  'err'); return; }

  try {
    const res = await fetch('/api/add_prompt', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset, prompt }),
    });
    const data = await res.json();
    if (data.ok) {
      // Update local state
      if (!examples[dataset]) examples[dataset] = [];
      const localIdx = examples[dataset].length;
      examples[dataset].push({ prompt, annotation: null });
      promptList.push({ dataset, prompt, annotation: null, localIdx });
      renderSidebar();
      closeModal();
      selectPrompt(promptList.length - 1);
      setStatus('Prompt added', 'ok');
    } else {
      setStatus('Error: ' + (data.error || '?'), 'err');
    }
  } catch (e) {
    setStatus('Error: ' + e.message, 'err');
  }
}

// Close modal on backdrop click
document.getElementById('modal-backdrop').addEventListener('click', function(e) {
  if (e.target === this) closeModal();
});

// ═══════════════════════════════════════════════════════════════
// Keyboard shortcuts
// ═══════════════════════════════════════════════════════════════

document.addEventListener('keydown', e => {
  // Ctrl/Cmd+S → save
  if ((e.ctrlKey || e.metaKey) && e.key === 's') {
    e.preventDefault();
    saveAnnotation();
  }
  // Escape → close modal
  if (e.key === 'Escape') closeModal();
  // ArrowLeft / ArrowRight while not in an input → navigate
  const tag = document.activeElement.tagName;
  if (tag !== 'INPUT' && tag !== 'TEXTAREA' && tag !== 'SELECT') {
    if (e.key === 'ArrowLeft')  { e.preventDefault(); navigate(-1); }
    if (e.key === 'ArrowRight') { e.preventDefault(); navigate(+1); }
  }
});

// ═══════════════════════════════════════════════════════════════
// Utils
// ═══════════════════════════════════════════════════════════════

function setStatus(msg, type) {
  const el = document.getElementById('status-msg');
  el.textContent = msg;
  el.className = type;
  el.style.opacity = '1';
  if (type === 'ok' || type === 'info') {
    clearTimeout(el._timer);
    el._timer = setTimeout(() => { el.style.opacity = '0'; }, 3000);
  }
}

function truncate(str, n) {
  return str.length > n ? str.slice(0, n) + '\u2026' : str;
}

function escHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

// ═══════════════════════════════════════════════════════════════
// Load-from-registry panel
// ═══════════════════════════════════════════════════════════════

async function loadAllDatasets() {
  const n   = parseInt(document.getElementById('load-n').value, 10) || 15;
  const btn = document.getElementById('btn-load-all');
  const msgEl = document.getElementById('load-msg');
  btn.disabled = true;
  msgEl.className = '';
  msgEl.textContent = 'Loading all datasets\u2026';
  setStatus('Loading all datasets\u2026', 'info');

  try {
    const res = await fetch('/api/autoload', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ n }),
    });
    const data = await res.json();
    if (data.ok) {
      // Reload full state
      const stateRes = await fetch('/api/state');
      const state = await stateRes.json();
      examples = state.examples || {};
      buildPromptList();
      renderSidebar();
      if (promptList.length > 0 && currentPIdx < 0) selectPrompt(0);
      msgEl.className = 'ok';
      msgEl.textContent = '+' + data.total_added + ' prompts from ' + data.n_datasets + ' datasets';
      setStatus('Loaded ' + data.total_added + ' prompts from all datasets', 'ok');
    } else {
      msgEl.className = 'err';
      msgEl.textContent = 'Error: ' + (data.error || '?');
    }
  } catch (e) {
    msgEl.className = 'err';
    msgEl.textContent = 'Error: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

async function initDatasets() {
  try {
    const res  = await fetch('/api/datasets');
    const data = await res.json();
    const sel  = document.getElementById('load-ds-select');
    if (!data.datasets || !data.datasets.length) {
      document.getElementById('load-panel').style.display = 'none';
      return;
    }
    data.datasets.forEach(name => {
      const opt = document.createElement('option');
      opt.value = opt.textContent = name;
      sel.appendChild(opt);
      // also populate the modal datalist
      const dlOpt = document.createElement('option');
      dlOpt.value = name;
      document.getElementById('ds-datalist').appendChild(dlOpt);
    });
  } catch (e) {
    document.getElementById('load-panel').style.display = 'none';
  }
}

async function loadDataset() {
  const ds  = document.getElementById('load-ds-select').value;
  const n   = parseInt(document.getElementById('load-n').value, 10) || 15;
  if (!ds) { setStatus('Select a dataset first', 'info'); return; }

  const msgEl = document.getElementById('load-msg');
  const btn   = document.getElementById('btn-load-ds');
  msgEl.className = '';
  msgEl.textContent = 'Streaming\u2026 (may take a while)';
  btn.disabled = true;

  try {
    const res  = await fetch('/api/load_dataset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ dataset: ds, n }),
    });
    const data = await res.json();
    if (data.ok) {
      const added = data.prompts.length;
      if (!examples[ds]) examples[ds] = [];
      data.prompts.forEach(p => {
        const localIdx = examples[ds].length;
        examples[ds].push({ prompt: p, annotation: null });
        promptList.push({ dataset: ds, prompt: p, annotation: null, localIdx });
      });
      renderSidebar();
      msgEl.className = 'ok';
      msgEl.textContent = added ? `+${added} loaded` : 'No new prompts';
      if (added > 0 && currentPIdx < 0) selectPrompt(0);
      setStatus(`Loaded ${added} prompts from ${ds}`, 'ok');
    } else {
      msgEl.className = 'err';
      msgEl.textContent = 'Error: ' + (data.error || '?');
      setStatus('Load failed: ' + (data.error || '?'), 'err');
    }
  } catch (e) {
    msgEl.className = 'err';
    msgEl.textContent = 'Error: ' + e.message;
    setStatus('Load failed: ' + e.message, 'err');
  } finally {
    btn.disabled = false;
  }
}

// ═══════════════════════════════════════════════════════════════
// Start
// ═══════════════════════════════════════════════════════════════
boot();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────────────────────────────────────
# HTTP server
# ─────────────────────────────────────────────────────────────────────────────

# Will be set by main() before the server starts
EXAMPLES_PATH: Path = DEFAULT_EXAMPLES_PATH


def _load_taxonomy() -> dict:
    if TAXONOMY_PATH.exists():
        try:
            return json.loads(TAXONOMY_PATH.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _load_examples() -> dict:
    if EXAMPLES_PATH.exists():
        try:
            data = json.loads(EXAMPLES_PATH.read_text())
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def _save_examples(data: dict) -> None:
    EXAMPLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    EXAMPLES_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False))


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # silence default request logging
        pass

    # ── routing ───────────────────────────────────────────────────────────

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, "text/html; charset=utf-8", HTML.encode())
        elif path == "/api/state":
            taxonomy = _load_taxonomy()
            self._json({
                "task_types": sorted(taxonomy.keys()),
                "levels":     LEVELS,
                "examples":   _load_examples(),
            })
        elif path == "/api/datasets":
            self._json({"datasets": sorted(DATASET_REGISTRY.keys()) if _REGISTRY_AVAILABLE else []})
        else:
            self._send(404, "text/plain", b"Not found")

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            payload = json.loads(body)
        except (ValueError, json.JSONDecodeError) as exc:
            self._json({"ok": False, "error": str(exc)}, status=400)
            return

        if path == "/api/save":
            self._handle_save(payload)
        elif path == "/api/add_prompt":
            self._handle_add_prompt(payload)
        elif path == "/api/load_dataset":
            self._handle_load_dataset(payload)
        elif path == "/api/suggest":
            self._handle_suggest(payload)
        elif path == "/api/autoload":
            self._handle_autoload(payload)
        else:
            self._send(404, "text/plain", b"Not found")

    # ── handlers ─────────────────────────────────────────────────────────

    def _handle_save(self, payload: dict) -> None:
        dataset = str(payload.get("dataset", "")).strip()
        idx = payload.get("prompt_index")
        annotation = payload.get("annotation")
        prompt = str(payload.get("prompt", "")).strip()

        if not dataset or annotation is None:
            self._json({"ok": False, "error": "Missing dataset or annotation"}, status=400)
            return

        examples = _load_examples()

        if dataset not in examples:
            examples[dataset] = []

        if isinstance(idx, int) and 0 <= idx < len(examples[dataset]):
            examples[dataset][idx]["annotation"] = annotation
        else:
            # New entry (shouldn't normally hit this path)
            examples[dataset].append({"prompt": prompt, "annotation": annotation})

        _save_examples(examples)
        self._json({"ok": True})

    def _handle_add_prompt(self, payload: dict) -> None:
        dataset = str(payload.get("dataset", "")).strip()
        prompt = str(payload.get("prompt", "")).strip()

        if not dataset or not prompt:
            self._json({"ok": False, "error": "Missing dataset or prompt"}, status=400)
            return

        examples = _load_examples()
        if dataset not in examples:
            examples[dataset] = []
        examples[dataset].append({"prompt": prompt, "annotation": None})
        _save_examples(examples)
        self._json({"ok": True})

    def _handle_load_dataset(self, payload: dict) -> None:
        if not _REGISTRY_AVAILABLE:
            self._json({"ok": False, "error": "Dataset registry not available (import failed)"}, status=503)
            return

        dataset = str(payload.get("dataset", "")).strip()
        n = min(max(int(payload.get("n", 15)), 1), 500)

        if not dataset:
            self._json({"ok": False, "error": "Missing dataset name"}, status=400)
            return

        config = DATASET_REGISTRY.get(dataset)
        if config is None:
            self._json({"ok": False, "error": f"Dataset '{dataset}' not found in registry"}, status=404)
            return

        try:
            prompts = _stream_prompts(dataset, config, n=n)
        except Exception as exc:
            self._json({"ok": False, "error": str(exc)})
            return

        # Deduplicate against what is already saved
        examples = _load_examples()
        if dataset not in examples:
            examples[dataset] = []
        existing_texts = {item["prompt"] for item in examples[dataset]}
        new_prompts = [p for p in prompts if p not in existing_texts]

        for p in new_prompts:
            examples[dataset].append({"prompt": p, "annotation": None})

        if new_prompts:
            _save_examples(examples)

        self._json({"ok": True, "prompts": new_prompts, "added": len(new_prompts)})

    def _handle_suggest(self, payload: dict) -> None:
        text = str(payload.get("text", "")).strip()
        if not text:
            self._json({"suggestions": []})
            return
        result = _suggest_for_text(text)
        self._json(result)

    def _handle_autoload(self, payload: dict) -> None:
        if not _REGISTRY_AVAILABLE:
            self._json({"ok": False, "error": "Dataset registry not available"}, status=503)
            return
        n = min(max(int(payload.get("n", 15)), 1), 500)
        examples = _autoload_all(n)
        total = sum(len(v) for v in examples.values())
        self._json({"ok": True, "total_added": total, "n_datasets": len(examples)})

    # ── helpers ───────────────────────────────────────────────────────────

    def _send(self, status: int, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data, ensure_ascii=False).encode()
        self._send(status, "application/json", body)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    global EXAMPLES_PATH

    parser = argparse.ArgumentParser(
        prog="annotation_app.py",
        description="Web UI for annotating few-shot examples.",
    )
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_EXAMPLES_PATH, metavar="PATH",
        help=f"JSON file to load/save annotations (default: {DEFAULT_EXAMPLES_PATH})",
    )
    parser.add_argument("--no-browser", action="store_true", help="Don't open a browser tab automatically")
    parser.add_argument(
        "--autoload", type=int, default=0, metavar="N",
        help="Auto-load N prompts per dataset on startup (0 = disabled)",
    )
    args = parser.parse_args()

    EXAMPLES_PATH = args.output.resolve()

    # Auto-load prompts from all datasets if requested
    if args.autoload > 0 and _REGISTRY_AVAILABLE:
        print(f"\nAuto-loading {args.autoload} prompts per dataset…")
        _autoload_all(args.autoload)
        print()

    url = f"http://localhost:{args.port}"
    print(f"Annotation tool → {url}")
    print(f"Annotations file → {EXAMPLES_PATH}")
    print("Press Ctrl+C to stop.\n")

    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()

    server = HTTPServer(("localhost", args.port), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()

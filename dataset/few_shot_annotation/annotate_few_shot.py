#!/usr/bin/env python3
"""
Interactive helper for building dataset-specific few-shot examples for
the extraction pipeline in construct_dataset.py.

For each source dataset, this tool:
  - Streams random prompts from that dataset (always in streaming mode)
  - Lets you interactively annotate which spans are templates (with {slot}
    placeholders) and which are option values (slot fillers), along with
    task type and compatibility metadata
  - Saves the annotated examples to a JSON file keyed by dataset name,
    which construct_dataset.py uses instead of the generic _EXTRACTION_FEW_SHOT
    when processing prompts from that dataset

Output format (few_shot_examples.json)
───────────────────────────────────────
{
  "IFEval": [
    {
      "prompt": "original raw prompt text",
      "annotation": {
        "templates": [{"text": "...", "slots": [...], "task_type": "...", "level": "..."}, ...],
        "options":   [{"value": "...", "slot": "...", "compatible_task_types": [...]}, ...]
      }
    },
    ...  ← examples_per_dataset entries
  ],
  "DollyV2": [...],
  ...
}

Usage
─────
  # Annotate all datasets (2 examples each)
  python dataset/annotate_few_shot.py

  # Annotate specific datasets only
  python dataset/annotate_few_shot.py --datasets IFEval DollyV2

  # Custom number of examples per dataset
  python dataset/annotate_few_shot.py --examples-per-dataset 3

  # Custom output file path
  python dataset/annotate_few_shot.py --output dataset/output/few_shot_examples.json

  # Resume / add to existing file
  python dataset/annotate_few_shot.py --datasets GSM8K
  (Already-annotated datasets are skipped automatically.)
"""

from __future__ import annotations

import argparse
import json
import random
import readline
import re
import sys
import textwrap
from pathlib import Path
from typing import Any

# ── Path setup ────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

FEW_SHOT_EXAMPLES_PATH = OUTPUT_DIR / "few_shot_examples.json"
TAXONOMY_PATH = OUTPUT_DIR / "taxonomy" / "taxonomy.json"

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

# ── Terminal colours ──────────────────────────────────────────────────────

BOLD   = "\033[1m"
CYAN   = "\033[36m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
RED    = "\033[31m"
RESET  = "\033[0m"
DIM    = "\033[2m"

LEVELS = ("content_task", "format", "style", "content_constraint", "meta")


# ═══════════════════════════════════════════════════════════════════════════
# Terminal UI helpers
# ═══════════════════════════════════════════════════════════════════════════

def _banner(text: str, colour: str = CYAN) -> None:
    width = max(60, len(text) + 6)
    bar = "─" * width
    print(f"\n{colour}{bar}{RESET}")
    print(f"{colour}{BOLD}  {text}{RESET}")
    print(f"{colour}{bar}{RESET}")


def _show_prompt(prompt: str, max_lines: int = 35) -> None:
    lines = prompt.splitlines()
    if len(lines) > max_lines:
        print(f"  {DIM}(showing first {max_lines} of {len(lines)} lines — use --max-lines to change){RESET}")
        lines = lines[:max_lines]
    for line in lines:
        wrapped = textwrap.fill(line, width=82, subsequent_indent="    ")
        print(f"  {wrapped}")


def _make_completer(choices: list[str]):
    def _complete(text: str, state: int) -> str | None:
        matches = [c for c in choices if c.startswith(text)]
        return matches[state] if state < len(matches) else None
    return _complete


def _input_with_tab(prompt_text: str, choices: list[str]) -> str:
    """Input with Tab completion from *choices*."""
    readline.set_completer(_make_completer(choices))
    readline.parse_and_bind("tab: complete")
    try:
        return input(prompt_text)
    finally:
        readline.set_completer(None)


def _required(prompt_text: str, choices: list[str] | None = None) -> str:
    """Input that loops until the user provides a non-empty value."""
    while True:
        val = (
            _input_with_tab(prompt_text, choices)
            if choices
            else input(prompt_text)
        ).strip()
        if val:
            return val
        print(f"  {RED}(a value is required — try again){RESET}")


def _ask_yn(prompt_text: str, default: str = "y") -> bool:
    hint = "[Y/n]" if default == "y" else "[y/N]"
    val = input(f"{prompt_text} {hint} ").strip().lower() or default
    return val.startswith("y")


# ═══════════════════════════════════════════════════════════════════════════
# Annotation flow
# ═══════════════════════════════════════════════════════════════════════════

_SLOT_RE = re.compile(r"\{(\w+)\}")


def _suggest_task_types(
    text: str,
    taxonomy: Taxonomy,
    prompt: str = "",
) -> list[dict]:
    """Return ranked suggestions [{name, level, score, source}, ...] for *text*."""
    combined = f"{text} {prompt}" if prompt else text
    scored: dict[str, dict] = {}

    # Pattern-based detection
    detected = taxonomy.detect(combined)
    for name, hit in detected.items():
        if hit:
            entry = taxonomy[name]
            scored[name] = {"name": name, "level": entry.level, "score": 2, "source": "pattern"}

    # Verb-phrase extraction
    try:
        for m in _VERB_PHRASE_RE.finditer(combined):
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
                    scored[task_name] = {"name": task_name, "level": lvl, "score": 1, "source": "verb"}
    except Exception:
        pass

    return sorted(scored.values(), key=lambda s: -s["score"])


def _show_suggestions(suggestions: list[dict]) -> tuple[str, str] | None:
    """Display suggestions and return (task_type, level) if user picks one, else None."""
    if not suggestions:
        return None
    print(f"\n  {CYAN}{BOLD}Suggestions:{RESET}")
    for i, s in enumerate(suggestions[:6], 1):
        print(f"    {CYAN}[{i}]{RESET} {s['name']}  {DIM}({s['level']}){RESET}")
    pick = input(f"  {CYAN}Pick a number (or Enter to type manually): {RESET}").strip()
    if pick.isdigit() and 1 <= int(pick) <= min(6, len(suggestions)):
        chosen = suggestions[int(pick) - 1]
        return (chosen["name"], chosen["level"])
    return None


def _annotate_templates(
    taxonomy_labels: list[str],
    taxonomy: Taxonomy | None = None,
    prompt: str = "",
) -> list[dict]:
    """Interactively collect template entries from the user."""
    templates: list[dict] = []

    print(f"\n{BOLD}── Templates ────────────────────────────────────────────────{RESET}")
    print(f"{DIM}Write template text replacing specific values with {{slot_name}} placeholders.")
    print(f"Examples:  'Write {{description}}'  |  'in exactly {{n}} sentences'")
    print(f"Press Enter on an empty template line when you are done.{RESET}")

    while True:
        print()
        text = input(f"  Template text (empty = done): ").strip()
        if not text:
            break

        # Auto-detect slot names from the template text
        auto_slots = _SLOT_RE.findall(text)
        if auto_slots:
            print(f"  {DIM}Detected slots: {auto_slots}{RESET}")
        else:
            print(f"  {YELLOW}No {{slot}} placeholders detected in this template.{RESET}")

        # Suggest task type and level
        picked = None
        if taxonomy is not None:
            suggestions = _suggest_task_types(text, taxonomy, prompt)
            picked = _show_suggestions(suggestions)

        if picked:
            task_type, level = picked
            print(f"  {GREEN}→ task_type: {task_type},  level: {level}{RESET}")
            if not _ask_yn(f"  Accept suggestion?"):
                picked = None

        if not picked:
            task_type = _required(
                "  task_type (Tab to complete): ",
                choices=taxonomy_labels,
            )
            level = _required(
                "  level (Tab to complete): ",
                choices=list(LEVELS),
            )

        templates.append({
            "text":      text,
            "slots":     auto_slots,
            "task_type": task_type,
            "level":     level,
        })
        print(f"  {GREEN}✓ Template added.{RESET}")

    return templates


def _annotate_options(
    templates: list[dict],
    taxonomy_labels: list[str],
) -> list[dict]:
    """Interactively collect option entries from the user."""
    options: list[dict] = []

    # Gather all slot names for Tab completion
    all_slots = sorted({s for t in templates for s in t.get("slots", [])})

    # Auto-suggest compatible task types from the templates
    template_types = sorted({t["task_type"] for t in templates if t.get("task_type")})

    print(f"\n{BOLD}── Options ──────────────────────────────────────────────────{RESET}")
    print(f"{DIM}Enter the concrete values that fill each slot (copied from the prompt).")
    print(f"Available slots: {all_slots}")
    print(f"Press Enter on an empty value line when you are done.{RESET}")

    while True:
        print()
        value = input(f"  Option value (empty = done): ").strip()
        if not value:
            break

        slot = _required(
            "  Slot name (Tab to complete): ",
            choices=all_slots,
        )

        # Suggest compatible task types based on templates
        if template_types:
            suggested = ", ".join(template_types)
            print(f"  {CYAN}Suggested compatible types (from templates): {suggested}{RESET}")
            if _ask_yn(f"  Accept suggested types?"):
                compat = list(template_types)
            else:
                print(f"  {DIM}Hint: type comma-separated task types, Tab completes one at a time.{RESET}")
                compat_raw = _required(
                    "  compatible_task_types (comma-separated, Tab = first match): ",
                    choices=taxonomy_labels,
                )
                compat = [x.strip() for x in compat_raw.split(",") if x.strip()]
        else:
            print(f"  {DIM}Hint: type comma-separated task types, Tab completes one at a time.{RESET}")
            compat_raw = _required(
                "  compatible_task_types (comma-separated, Tab = first match): ",
                choices=taxonomy_labels,
            )
            compat = [x.strip() for x in compat_raw.split(",") if x.strip()]

        options.append({
            "value":                value,
            "slot":                 slot,
            "compatible_task_types": compat,
        })
        print(f"  {GREEN}✓ Option added  ({slot} = {value!r}).{RESET}")

    return options


def annotate_one(
    prompt: str,
    taxonomy_labels: list[str],
    taxonomy: Taxonomy | None = None,
) -> dict | None:
    """
    Interactively annotate one prompt.

    Returns:
      - {"templates": [...], "options": [...]}  — completed annotation
      - {}                                       — user chose to skip
      - None                                     — user chose to quit
    """
    _banner("PROMPT")
    _show_prompt(prompt)

    action = input(
        f"\n{YELLOW}[a]nnotate  [s]kip  [q]uit{RESET}  → "
    ).strip().lower()

    if action == "q":
        return None
    if action != "a":
        return {}

    templates = _annotate_templates(taxonomy_labels, taxonomy=taxonomy, prompt=prompt)

    if not templates:
        print(f"  {YELLOW}No templates entered — skipping this prompt.{RESET}")
        return {}

    options = _annotate_options(templates, taxonomy_labels)

    annotation: dict[str, Any] = {"templates": templates, "options": options}

    _banner("ANNOTATION PREVIEW", colour=GREEN)
    print(json.dumps(annotation, indent=2, ensure_ascii=False))

    if not _ask_yn("\nConfirm and save this annotation?"):
        print(f"  {YELLOW}Discarded — skipping this prompt.{RESET}")
        return {}

    return annotation


# ═══════════════════════════════════════════════════════════════════════════
# Dataset streaming
# ═══════════════════════════════════════════════════════════════════════════

def _stream_prompts(ds_name: str, config: dict, n: int = 80) -> list[str]:
    """
    Stream up to *n* random prompt strings from *config*.

    Always uses the same streaming loaders as the main extraction pipeline
    so the prompts here are representative of what the model will process.
    """
    source = config.get("source", "")
    print(f"  {DIM}Streaming from {ds_name} ({source}) …{RESET}", flush=True)

    try:
        if source == "huggingface":
            # Request more than needed so we have variety after filtering
            cfg_with_larger_sample = {**config, "sample_size": n * 4}
            samples = load_hf_dataset(cfg_with_larger_sample, max_samples=n * 4)
        elif source == "github_raw":
            samples = load_github_data(config)
        elif source == "kcif_github":
            samples = load_kcif_data(config)
        else:
            print(f"  {RED}Unknown source '{source}' for {ds_name} — skipping.{RESET}")
            return []
    except Exception as exc:
        print(f"  {RED}Could not load {ds_name}: {exc}{RESET}")
        return []

    fields = config.get("instruction_fields", [])
    texts: list[str] = []
    for sample in samples:
        t = extract_instruction_text(sample, fields).strip()
        if t:
            texts.append(t)

    random.shuffle(texts)
    return texts[:n]


# ═══════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════

def _load_existing(path: Path) -> dict[str, list[dict]]:
    if not path.exists():
        return {}
    try:
        with open(path) as fh:
            data = json.load(fh)
        # Validate top-level structure
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError):
        pass
    print(f"{YELLOW}WARNING: {path} could not be parsed as JSON — starting fresh.{RESET}")
    return {}


def _save(data: dict[str, list[dict]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
# Main annotation loop
# ═══════════════════════════════════════════════════════════════════════════

def run(
    dataset_names: list[str],
    examples_per_dataset: int,
    output_path: Path,
    seed: int,
) -> None:
    random.seed(seed)

    # Load taxonomy labels for Tab-completion
    taxonomy_labels: list[str] = []
    taxonomy: Taxonomy | None = None
    if TAXONOMY_PATH.exists():
        with open(TAXONOMY_PATH) as fh:
            taxonomy_labels = list(json.load(fh).keys())
        # Build a Taxonomy instance for suggestion engine
        taxonomy = Taxonomy()
        try:
            saved = json.loads(TAXONOMY_PATH.read_text())
            for name, info in saved.items():
                lvl = info.get("level", "content_task")
                if name not in taxonomy:
                    # Create with a name-derived pattern so detect() works
                    name_pat = name.replace("_", r"[\s_]")
                    pat = re.compile(rf"\b{name_pat}\b", re.I)
                    entry = taxonomy.get_or_create(name, lvl)
                    entry.patterns.append(pat)
                else:
                    taxonomy.get_or_create(name, lvl)
        except (json.JSONDecodeError, OSError):
            pass
    else:
        print(f"{YELLOW}WARNING: taxonomy not found at {TAXONOMY_PATH}{RESET}")
        print(f"{YELLOW}         task_type Tab completion will not be available.{RESET}")

    # Load any already-annotated examples so we can resume
    existing: dict[str, list[dict]] = _load_existing(output_path)
    if existing:
        total_have = sum(len(v) for v in existing.values())
        ds_have = len([k for k, v in existing.items() if v])
        print(f"\n{DIM}Loaded {total_have} existing examples across {ds_have} datasets.{RESET}")

    # Determine which datasets still need examples
    work: list[tuple[str, int]] = []
    for ds_name in dataset_names:
        have = len(existing.get(ds_name, []))
        need = examples_per_dataset - have
        if need > 0:
            work.append((ds_name, need))
        else:
            print(f"  {DIM}{ds_name}: already has {have}/{examples_per_dataset} examples — skipping.{RESET}")

    if not work:
        print(f"\n{GREEN}All datasets already have {examples_per_dataset} examples. Nothing to do.{RESET}")
        return

    total_needed = sum(n for _, n in work)
    print(f"\n{BOLD}Will annotate {total_needed} example(s) across {len(work)} dataset(s).{RESET}")
    print(f"\nControls during annotation:")
    print(f"  {YELLOW}[a]{RESET} annotate this prompt")
    print(f"  {YELLOW}[s]{RESET} skip to next prompt (same dataset)")
    print(f"  {YELLOW}[q]{RESET} quit and save progress")

    for ds_name, need in work:
        config = DATASET_REGISTRY.get(ds_name)
        if config is None:
            print(f"\n{RED}'{ds_name}' not found in DATASET_REGISTRY — skipping.{RESET}")
            continue

        _banner(f"Dataset: {ds_name}  ({need} example(s) needed)")

        # Stream enough prompts to allow the user to skip some
        prompts = _stream_prompts(ds_name, config, n=max(need * 15, 60))
        if not prompts:
            print(f"  {RED}No usable prompts for {ds_name} — skipping.{RESET}")
            continue
        print(f"  {DIM}{len(prompts)} prompts available for annotation.{RESET}")

        collected: list[dict] = list(existing.get(ds_name, []))
        prompt_idx = 0

        while len(collected) < examples_per_dataset and prompt_idx < len(prompts):
            progress = f"{len(collected)+1}/{examples_per_dataset}"
            pos = f"prompt {prompt_idx+1}/{len(prompts)}"
            print(f"\n{DIM}[{ds_name}  example {progress}  {pos}]{RESET}")

            annotation = annotate_one(prompts[prompt_idx], taxonomy_labels, taxonomy=taxonomy)
            prompt_idx += 1

            if annotation is None:
                # User quit — save and exit
                print(f"\n{YELLOW}Quit.  Progress saved to {output_path}{RESET}")
                _save(existing, output_path)
                return

            if not annotation:
                # Skipped prompt
                continue

            collected.append({
                "prompt":     prompts[prompt_idx - 1],
                "annotation": annotation,
            })
            existing[ds_name] = collected
            _save(existing, output_path)
            print(f"  {GREEN}Saved ({len(collected)}/{examples_per_dataset} for {ds_name}).{RESET}")

        if len(collected) < examples_per_dataset:
            short = examples_per_dataset - len(collected)
            print(
                f"\n  {YELLOW}Only {len(collected)}/{examples_per_dataset} examples collected for "
                f"{ds_name} (ran out of prompts — try running again with a different --seed).{RESET}"
            )

    # ── Summary ───────────────────────────────────────────────────────
    total_saved = sum(len(v) for v in existing.values())
    ds_done = [k for k, v in existing.items() if len(v) >= examples_per_dataset]
    print(f"\n{GREEN}{BOLD}Done!{RESET}  Saved to {output_path}")
    print(f"  Total examples : {total_saved}")
    print(f"  Datasets complete ({examples_per_dataset}+ examples) : {len(ds_done)}")
    if ds_done:
        for ds in ds_done:
            print(f"    {GREEN}✓{RESET}  {ds}  ({len(existing[ds])} examples)")


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annotate_few_shot.py",
        description=(
            "Interactively build dataset-specific few-shot examples for "
            "the extraction pipeline in construct_dataset.py."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(__doc__.split("\n")[1:]),
    )
    p.add_argument(
        "--datasets", nargs="+", metavar="DATASET",
        help=(
            "Dataset names to annotate (default: all in DATASET_REGISTRY). "
            "Use the exact key names, e.g. IFEval DollyV2 GSM8K"
        ),
    )
    p.add_argument(
        "--examples-per-dataset", type=int, default=2, metavar="N",
        help="Target number of annotated examples per dataset (default: 2)",
    )
    p.add_argument(
        "--output", type=Path, default=FEW_SHOT_EXAMPLES_PATH, metavar="PATH",
        help=f"Output JSON file path (default: {FEW_SHOT_EXAMPLES_PATH})",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed used when shuffling streamed prompts (default: 42)",
    )
    p.add_argument(
        "--list-datasets", action="store_true",
        help="Print all available dataset names and exit",
    )
    return p


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.list_datasets:
        print("Available datasets:")
        for name in DATASET_REGISTRY:
            print(f"  {name}")
        return

    dataset_names = args.datasets or list(DATASET_REGISTRY.keys())

    # Warn about any requested names not in the registry
    unknown = [d for d in dataset_names if d not in DATASET_REGISTRY]
    if unknown:
        print(f"{YELLOW}WARNING: dataset(s) not found in registry: {unknown}{RESET}")
        dataset_names = [d for d in dataset_names if d in DATASET_REGISTRY]
        if not dataset_names:
            print(f"{RED}No valid datasets specified. Use --list-datasets to see options.{RESET}")
            sys.exit(1)

    run(
        dataset_names=dataset_names,
        examples_per_dataset=args.examples_per_dataset,
        output_path=args.output,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

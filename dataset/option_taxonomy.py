"""
option_taxonomy.py — LLM-constructed taxonomy of option types.

Replaces the previous static OPTION_TYPES dict with a data-driven taxonomy
built by an LLM from accumulated extraction results.

The taxonomy groups options by semantic type, assigns a canonical slot name,
and identifies cross-slot compatibility — for example, recognising that "essay"
is a text_type value that can also substitute into a {topic} slot.

Lifecycle
---------
Stage B (extraction): taxonomy loaded from disk (if available) and injected
    as context into each extraction batch.
Stage C (taxonomy build): after all options are extracted, the LLM analyses
    the full option pool and writes option_taxonomy.json.
Next run with --skip-segmentation: Stage B uses the richer taxonomy built in
    the previous Stage C, improving slot naming consistency.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from dataset.schema import Option, OPTIONS_PATH

logger = logging.getLogger(__name__)

TAXONOMY_PATH = OPTIONS_PATH.parent / "option_taxonomy.json"

_MAX_OPTIONS_PER_SLOT = 12    # sent to LLM per slot
_MAX_VALUE_LEN        = 120   # truncate long values before sending
_MIN_EXAMPLES_TO_TYPE = 2     # skip slots with fewer values

# ── WordNet helpers ─────────────────────────────────────────────────────

def _ensure_wordnet() -> bool:
    """Download WordNet data if needed.  Returns True when available."""
    try:
        import nltk
        from nltk.corpus import wordnet as wn          # noqa: F401
        # Trigger lazy load to check availability
        wn.synsets("test")
        return True
    except LookupError:
        try:
            import nltk
            nltk.download("wordnet", quiet=True)
            nltk.download("omw-1.4", quiet=True)
            return True
        except Exception:
            return False
    except ImportError:
        return False


# Stop-words at the top of the WordNet hierarchy that are too generic
# to be useful as category/type names.
_WN_STOP_HYPERNYMS = frozenset({
    "entity.n.01", "physical_entity.n.01", "abstraction.n.06",
    "object.n.01", "whole.n.02", "thing.n.12", "causal_agent.n.01",
    "matter.n.03", "attribute.n.02", "group.n.01", "relation.n.01",
    "state.n.02", "event.n.01", "act.n.02", "psychological_feature.n.01",
})


def _best_synset(value: str):
    """Return the most likely noun synset for *value*, or None."""
    from nltk.corpus import wordnet as wn
    # Try the full phrase first, then individual words (rightmost first,
    # since English heads are typically on the right).
    candidates = [value.replace(" ", "_")]
    words = value.split()
    if len(words) > 1:
        candidates.extend(reversed(words))
    for term in candidates:
        syns = wn.synsets(term, pos=wn.NOUN)
        if syns:
            return syns[0]
    return None


def _hypernym_chain(synset) -> list[str]:
    """Return the hypernym path from *synset* up to the root, as lemma names."""
    names: list[str] = []
    paths = synset.hypernym_paths()
    if not paths:
        return names
    # Pick the longest path (most informative)
    path = max(paths, key=len)
    for s in reversed(path):
        if s.name() in _WN_STOP_HYPERNYMS:
            continue
        names.append(s.lemmas()[0].name().replace("_", " "))
    return names


def _lowest_common_hypernym(synsets: list) -> str | None:
    """Find the most specific shared hypernym for a list of synsets.

    Skips over the very generic top of the hierarchy (entity, object, …).
    """
    if not synsets:
        return None
    if len(synsets) == 1:
        chain = _hypernym_chain(synsets[0])
        # Return the synset itself (most specific)
        return chain[0] if chain else None

    # Pairwise LCH — accumulate the common ancestor set
    from functools import reduce
    common = synsets[0]
    for s in synsets[1:]:
        lchs = common.lowest_common_hypernyms(s)
        if lchs:
            # Pick the deepest (most specific) LCH
            common = max(lchs, key=lambda h: h.min_depth())
        else:
            return None

    name = common.name()
    if name in _WN_STOP_HYPERNYMS:
        return None
    return common.lemmas()[0].name().replace("_", " ")


def _wordnet_hypernym_hints(slot_values: dict[str, list[str]]) -> dict[str, dict]:
    """Build WordNet-based semantic hints for each slot.

    Returns ``{slot: {"synsets": {value: hypernym_chain}, "common_ancestor": str}}``.
    Only includes slots where at least one value has a WordNet match.
    """
    if not _ensure_wordnet():
        return {}

    hints: dict[str, dict] = {}
    for slot, values in slot_values.items():
        slot_synsets = []
        value_chains: dict[str, list[str]] = {}
        for v in values:
            syn = _best_synset(v)
            if syn is None:
                continue
            slot_synsets.append(syn)
            chain = _hypernym_chain(syn)
            if chain:
                # Keep at most 4 levels: specific → general
                value_chains[v] = chain[:4]
        if not value_chains:
            continue
        ancestor = _lowest_common_hypernym(slot_synsets) if len(slot_synsets) >= 2 else None
        hints[slot] = {
            "value_hypernyms": value_chains,
            "common_ancestor": ancestor,
        }
    return hints


def _format_wordnet_context(hints: dict[str, dict]) -> str:
    """Format WordNet hints as a compact string for the LLM prompt."""
    if not hints:
        return ""
    lines = [
        "WordNet hypernym analysis (use this to inform category/type assignments; "
        "chain goes from specific → general):"
    ]
    for slot, info in sorted(hints.items()):
        ancestor = info.get("common_ancestor")
        anc_str = f" [shared ancestor: {ancestor}]" if ancestor else ""
        lines.append(f"  {slot}{anc_str}:")
        for val, chain in info["value_hypernyms"].items():
            lines.append(f"    {val} → {' → '.join(chain)}")
    return "\n".join(lines)


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class OptionType:
    """One semantic type in the LLM-constructed taxonomy.

    Part of a two-level hierarchy: each type belongs to a broad *category*
    (e.g. ``content_subject``) and has a specific *name* (e.g. ``geography``).
    """
    name: str                              # specific slot name (e.g. "geography")
    category: str                          # broad grouping   (e.g. "content_subject")
    description: str                       # one-line semantic description
    compatible_slots: list[str]            # all slot names that accept this type
    example_values: list[str]             # representative values from the data

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "compatible_slots": self.compatible_slots,
            "example_values": self.example_values,
        }

    @classmethod
    def from_dict(cls, d: dict, category_fallback: str = "") -> "OptionType":
        return cls(
            name=d["name"],
            category=d.get("category", category_fallback) or d["name"],
            description=d["description"],
            compatible_slots=d.get("compatible_slots", []),
            example_values=d.get("example_values", []),
        )


@dataclass
class OptionTaxonomy:
    """Full taxonomy of option types, keyed by canonical slot name.

    Supports a two-level hierarchy: *categories* group semantically related
    *types*.  For backward compatibility the flat ``types`` dict is still the
    primary lookup; ``categories`` provides the grouping overlay.
    """
    types: dict[str, OptionType] = field(default_factory=dict)

    # ── Derived helpers ───────────────────────────────────────────────

    @property
    def categories(self) -> dict[str, list[OptionType]]:
        """Return types grouped by their category."""
        from collections import defaultdict
        groups: dict[str, list[OptionType]] = defaultdict(list)
        for ot in self.types.values():
            groups[ot.category].append(ot)
        return dict(groups)

    # ── Serialisation ─────────────────────────────────────────────────

    def save(self, path: Path = TAXONOMY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save in hierarchical format: categories → types
        cats: dict[str, list[dict]] = {}
        for ot in self.types.values():
            cats.setdefault(ot.category, []).append(ot.to_dict())
        out = {
            "categories": [
                {"name": cat, "types": entries}
                for cat, entries in sorted(cats.items())
            ]
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2, ensure_ascii=False)
        logger.info(
            f"  Option taxonomy saved → {path} "
            f"({len(cats)} categories, {len(self.types)} types)"
        )

    @classmethod
    def load(cls, path: Path = TAXONOMY_PATH) -> "OptionTaxonomy":
        if not path.exists():
            return cls()
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            types: dict[str, OptionType] = {}
            # New hierarchical format
            if "categories" in data:
                for cat_entry in data["categories"]:
                    cat_name = cat_entry.get("name", "")
                    for entry in cat_entry.get("types", []):
                        ot = OptionType.from_dict(entry, category_fallback=cat_name)
                        types[ot.name] = ot
            # Legacy flat format
            elif "types" in data:
                for entry in data["types"]:
                    ot = OptionType.from_dict(entry)
                    types[ot.name] = ot
            logger.info(f"  Option taxonomy loaded from {path} ({len(types)} types)")
            return cls(types=types)
        except Exception as exc:
            logger.warning(f"  Could not load option taxonomy: {exc}")
            return cls()

    # ── Prompt injection ──────────────────────────────────────────────

    def update_from_options(self, new_options: list[Option]) -> None:
        """Incrementally add newly extracted options into the in-memory taxonomy.

        Called after each extraction batch so the taxonomy grows in real time
        and provides updated context for the next batch. No LLM required —
        values are simply accumulated under their slot name. A richer LLM
        consolidation pass (build_option_taxonomy) is run once at the end,
        which assigns proper categories and specific subcategory names.
        """
        for o in new_options:
            slot = o.slot.strip()
            value = o.value.strip()
            if not slot or not value or o.token_length > 50:
                continue
            if slot in self.types:
                if value not in self.types[slot].example_values:
                    self.types[slot].example_values.append(value)
            else:
                self.types[slot] = OptionType(
                    name=slot,
                    category="_uncategorized",  # filled in by LLM consolidation
                    description="",             # filled in by LLM consolidation
                    compatible_slots=[slot],
                    example_values=[value],
                )

    def to_prompt_context(self, *, max_types: int = 40) -> str:
        """Format the taxonomy as a compact hierarchical string for extraction prompts.

        The LLM sees this to:
        - Prefer existing canonical slot names at the right specificity level
        - Recognise that concrete words (e.g. "essay") belong to a specific type
          within a broader category
        - Know which slots are interchangeable for cross-template compatibility
        """
        if not self.types:
            return ""

        cats = self.categories
        lines = [
            "Option taxonomy (hierarchical — use specific slot names from each "
            "category; recognise listed values as belonging to that type):"
        ]
        shown = 0
        for cat_name, cat_types in sorted(cats.items()):
            if cat_name == "_uncategorized":
                continue
            lines.append(f"  [{cat_name}]")
            for ot in cat_types:
                if shown >= max_types:
                    break
                examples = ot.example_values[:6]
                if not examples:
                    continue
                compat_extra = [s for s in ot.compatible_slots if s != ot.name]
                compat_str = f" [also fits: {', '.join(compat_extra)}]" if compat_extra else ""
                desc_str = f" — {ot.description}" if ot.description else ""
                lines.append(f"    {ot.name}{desc_str}{compat_str}: {', '.join(examples)}")
                shown += 1
            if shown >= max_types:
                break
        return "\n".join(lines) if len(lines) > 1 else ""


# ── LLM taxonomy builder ─────────────────────────────────────────────────

_BUILD_SYSTEM_PROMPT = """\
You are constructing a TWO-LEVEL taxonomy of option types for a prompt-template \
extraction system.

You will receive a JSON object mapping slot_name → list_of_values, derived from \
real instruction-tuning prompts.

Your task: organise all slot/value groups into a hierarchy of CATEGORIES and TYPES.

LEVEL 1 — CATEGORIES (broad semantic groups). Use exactly these where applicable:
  - content_subject: what the text is about (topics, entities, domains)
  - text_genre: what kind of text is produced (essay, email, poem, report…)
  - formatting: how the output is structured (markdown, bullet points, table…)
  - quantitative: numbers, counts, limits, thresholds
  - linguistic_unit: units of text measurement (word, sentence, character, paragraph…)
  - stylistic: tone, voice, register, literary style
  - identity: people, roles, personas, authors
  - reference: titles, quotes, citations, external sources
  - constraint_detail: short descriptive constraints (e.g. "short, concise, and unique")
  - context_input: longer text passages that serve as input/context for the task
  You may create additional categories if the data clearly requires it, but \
  prefer the above list.

LEVEL 2 — TYPES (specific within each category). Each type must have:
- "name": a specific canonical slot name — be PRECISE, not generic. \
  Instead of "topic" use "geography", "academic_subject", "product", \
  "entity_name", "data_structure_element", etc. \
  Instead of "format" use "output_layout", "text_casing", "notation_system", etc. \
  The name should be specific enough that you can predict what values fit.
- "category": one of the category names above.
- "description": one sentence describing what values belong here.
- "compatible_slots": ALL slot names (from the input AND inferred) where these \
  values could meaningfully appear — the key cross-template compatibility signal.
- "example_values": up to 8 representative values from the input.

Rules:
- SPLIT overly broad groupings. "topic" values like "Japan" (geography), \
  "study skills" (academic_subject), "home security product" (product) MUST go \
  into different specific types, not one catch-all.
- Similarly, "format" values like "markdown" (output_layout), "uppercase" \
  (text_casing), "ABC notation" (notation_system) MUST be separate types.
- Merge slot names that describe the EXACT SAME kind of value \
  (e.g. "forbidden_words", "forbidden_element" → forbidden_keyword).
- A text-genre word (essay, poem, letter, report) should be a type under \
  text_genre, also compatible with topic/description/subject slots.
- CRITICAL: "description" is a CATCH-ALL slot that must be split. Values tagged \
  with [role] prefixes (e.g. [problem_statement], [output_specification], \
  [factual_context], [reference_passage], [example_explanation]) indicate the \
  functional role of long text. Group these into specific types under \
  context_input category: problem_statement, output_specification, \
  factual_context, reference_passage, example_explanation, etc. \
  Short descriptive constraints ("short, concise, and unique") go under \
  constraint_detail instead.
- Do NOT create a type for fewer than 2 distinct values.
- Output only types where you are confident in the grouping.
- You will also receive a WORDNET HYPERNYM ANALYSIS for values that have \
  WordNet entries. Use this to inform your type/category assignments: \
  values sharing a common ancestor likely belong to the same type; \
  the ancestor name often suggests a good type or category name.

Output a single JSON object:
{"categories": [{"name": "...", "types": [\
{"name": "...", "category": "...", "description": "...", \
"compatible_slots": ["...", ...], "example_values": ["...", ...]}, ...]}, ...]}\
"""


_ROLE_CLASSIFICATION_PROMPT = """\
You are classifying long text passages by their FUNCTIONAL ROLE in an instruction-tuning prompt.

For each passage, assign exactly one role from this list:
- problem_statement: describes a task, puzzle, or problem to solve
- input_specification: describes input format, constraints, or parameters
- output_specification: describes expected output format or what to produce
- example_explanation: walks through a worked example or test case
- factual_context: encyclopedic or factual background information
- reference_passage: a passage to read/analyse/summarise (the "input text")
- quality_constraint: describes desired qualities (concise, formal, detailed…)

Input: a JSON list of {"id": N, "value": "..."} objects.
Output: a JSON list of {"id": N, "role": "..."} objects, one per input.
Output ONLY the JSON list, no explanation.\
"""


def _classify_long_values_llm(
    long_values: list[tuple[int, str]],
    *,
    model: str,
    device: str | None,
) -> dict[int, str]:
    """Use the LLM to classify long text values by functional role.

    Parameters
    ----------
    long_values
        List of ``(index, value_text)`` tuples.

    Returns
    -------
    dict mapping index → role string.
    """
    from dataset.local_llm import generate_text as _generate

    if not long_values:
        return {}

    # Prepare input — truncate to keep prompt manageable
    items = [{"id": idx, "value": v[:200]} for idx, v in long_values]
    messages = [
        {"role": "system", "content": _ROLE_CLASSIFICATION_PROMPT},
        {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
    ]

    try:
        raw = _generate(
            model, messages,
            temperature=0.0, device=device,
            enable_thinking=False,
        )
    except Exception as exc:
        logger.warning(f"  Long-value classification failed: {exc}")
        return {}

    # Parse response
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
    try:
        results = json.loads(raw)
        return {r["id"]: r["role"] for r in results if "id" in r and "role" in r}
    except (json.JSONDecodeError, KeyError):
        m = re.search(r'\[.*\]', raw, re.DOTALL)
        if m:
            try:
                results = json.loads(m.group(0))
                return {r["id"]: r["role"] for r in results if "id" in r and "role" in r}
            except (json.JSONDecodeError, KeyError):
                pass
    logger.warning("  Could not parse long-value classification response")
    return {}


def _prepare_input(
    options: list[Option],
    long_value_roles: dict[int, str] | None = None,
) -> dict[str, list[str]]:
    """Group option values by slot, filtering and deduplicating.

    Short values (≤50 tokens) are included verbatim.  Longer values are
    truncated and tagged with a functional-role prefix (from LLM classification)
    so the taxonomy builder can split the catch-all ``description`` slot.
    """
    from collections import defaultdict
    slot_to_values: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)

    if long_value_roles is None:
        long_value_roles = {}

    for i, o in enumerate(options):
        slot = o.slot.strip()
        value = o.value.strip()
        if not slot or not value:
            continue
        vl = value.lower()
        if vl in seen[slot]:
            continue
        seen[slot].add(vl)

        if o.token_length <= 50:
            slot_to_values[slot].append(value[:_MAX_VALUE_LEN])
        else:
            role = long_value_roles.get(i, "passage")
            preview = value[:80].rstrip() + "…"
            slot_to_values[slot].append(f"[{role}] {preview}")

    # filter slots with too few examples
    return {
        s: vs[:_MAX_OPTIONS_PER_SLOT]
        for s, vs in slot_to_values.items()
        if len(vs) >= _MIN_EXAMPLES_TO_TYPE
    }


def _collect_long_values(options: list[Option]) -> list[tuple[int, str]]:
    """Collect (index, value) pairs for options with token_length > 50."""
    return [
        (i, o.value.strip())
        for i, o in enumerate(options)
        if o.value.strip() and o.token_length > 50
    ]


def _parse_taxonomy_response(raw: str) -> list[dict]:
    """Extract and parse the JSON from the LLM response.

    Handles both the new hierarchical format (``categories → types``) and the
    legacy flat format (``types`` list).  Returns a flat list of type dicts,
    each guaranteed to have a ``category`` key.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)

    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if data is None:
        logger.warning("  Could not parse taxonomy response")
        return []

    # New hierarchical format
    if "categories" in data:
        all_types: list[dict] = []
        for cat in data["categories"]:
            cat_name = cat.get("name", "")
            for t in cat.get("types", []):
                t.setdefault("category", cat_name)
                all_types.append(t)
        return all_types

    # Legacy flat format
    types = data.get("types", [])
    for t in types:
        t.setdefault("category", t.get("name", ""))
    return types


def build_option_taxonomy(
    options: list[Option],
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str | None = None,
    save: bool = True,
) -> OptionTaxonomy:
    """Run an LLM pass over all extracted options to build the option taxonomy.

    This is Stage C of the pipeline. It takes the full option pool, groups
    values by slot, and asks the LLM to produce a semantic taxonomy with
    canonical names and cross-slot compatibility annotations.

    The result is saved to option_taxonomy.json and returned.
    """
    from dataset.local_llm import generate_text as _generate

    # Step 1: classify long values (>50 tokens) by functional role using LLM
    long_vals = _collect_long_values(options)
    long_value_roles: dict[int, str] = {}
    if long_vals:
        logger.info(f"  Classifying {len(long_vals)} long values by functional role…")
        long_value_roles = _classify_long_values_llm(
            long_vals, model=model, device=device,
        )
        role_counts: dict[str, int] = {}
        for r in long_value_roles.values():
            role_counts[r] = role_counts.get(r, 0) + 1
        logger.info(f"  Long-value roles: {role_counts}")

    input_data = _prepare_input(options, long_value_roles)
    if not input_data:
        logger.warning("  No options to build taxonomy from.")
        return OptionTaxonomy()

    logger.info(
        f"  Building option taxonomy from {len(input_data)} slot groups "
        f"({sum(len(v) for v in input_data.values())} values)…"
    )

    # Build WordNet hypernym hints for the LLM
    wn_hints = _wordnet_hypernym_hints(input_data)
    wn_context = _format_wordnet_context(wn_hints)
    if wn_context:
        logger.info(
            f"  WordNet hints: {sum(len(h['value_hypernyms']) for h in wn_hints.values())} "
            f"values mapped across {len(wn_hints)} slots"
        )

    user_content = json.dumps(input_data, ensure_ascii=False)
    if wn_context:
        user_content += "\n\n" + wn_context

    messages = [
        {"role": "system", "content": _BUILD_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]

    try:
        raw = _generate(
            model, messages,
            temperature=0.0, device=device,
            enable_thinking=False,
        )
    except Exception as exc:
        logger.warning(f"  Taxonomy LLM call failed: {exc}")
        return OptionTaxonomy()

    entries = _parse_taxonomy_response(raw)
    types: dict[str, OptionType] = {}
    for entry in entries:
        name = entry.get("name", "").strip().lower()
        category = entry.get("category", "").strip().lower() or name
        desc = entry.get("description", "").strip()
        compat = [s.strip() for s in entry.get("compatible_slots", []) if s.strip()]
        # Strip functional-role tags (e.g. "[factual_context] …") from examples
        examples = [
            re.sub(r"^\[[\w]+\]\s*", "", v) for v in entry.get("example_values", []) if v
        ][:8]
        if not name or not desc:
            continue
        # ensure name is in its own compatible_slots
        if name not in compat:
            compat.insert(0, name)
        types[name] = OptionType(
            name=name,
            category=category,
            description=desc,
            compatible_slots=compat,
            example_values=examples,
        )

    taxonomy = OptionTaxonomy(types=types)
    logger.info(f"  Taxonomy built: {len(types)} types")

    # Post-hoc: enrich with WordNet validation
    _refine_taxonomy_with_wordnet(taxonomy, wn_hints)

    if save:
        taxonomy.save()

    return taxonomy


def _refine_taxonomy_with_wordnet(
    taxonomy: OptionTaxonomy,
    wn_hints: dict[str, dict],
) -> None:
    """Post-hoc refinement: use WordNet to validate and enrich the taxonomy.

    For each type whose example values have WordNet entries, compute the lowest
    common hypernym.  If it's more specific than the current type name and the
    LLM picked a generic name, log a suggestion.  Also enrich compatible_slots
    by finding other types that share a common ancestor.
    """
    if not wn_hints or not _ensure_wordnet():
        return

    # Build a synset map for each type's example values
    for ot in taxonomy.types.values():
        synsets = []
        for v in ot.example_values:
            syn = _best_synset(v)
            if syn:
                synsets.append(syn)
        if len(synsets) < 2:
            continue

        ancestor = _lowest_common_hypernym(synsets)
        if ancestor and ancestor.lower() != ot.name and ancestor.lower() != ot.category:
            logger.info(
                f"  WordNet: type '{ot.name}' (category={ot.category}) "
                f"— values share ancestor '{ancestor}'"
            )

    # Cross-type: find types whose values share a WordNet ancestor,
    # and enrich compatible_slots between them.
    type_synsets: dict[str, list] = {}
    for ot in taxonomy.types.values():
        syns = []
        for v in ot.example_values:
            s = _best_synset(v)
            if s:
                syns.append(s)
        if syns:
            type_synsets[ot.name] = syns

    type_names = list(type_synsets.keys())
    for i, t1 in enumerate(type_names):
        for t2 in type_names[i + 1:]:
            # Check if any synset pair across types shares a non-trivial ancestor
            for s1 in type_synsets[t1][:3]:
                for s2 in type_synsets[t2][:3]:
                    lchs = s1.lowest_common_hypernyms(s2)
                    if lchs:
                        best = max(lchs, key=lambda h: h.min_depth())
                        if best.min_depth() >= 4 and best.name() not in _WN_STOP_HYPERNYMS:
                            # These types are semantically close — cross-link
                            ot1 = taxonomy.types[t1]
                            ot2 = taxonomy.types[t2]
                            if t2 not in ot1.compatible_slots:
                                ot1.compatible_slots.append(t2)
                            if t1 not in ot2.compatible_slots:
                                ot2.compatible_slots.append(t1)
                            break
                else:
                    continue
                break

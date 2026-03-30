"""
option_taxonomy.py — Embedding-clustered, LLM-named taxonomy of option types.

Groups options by semantic type using sentence-transformer embeddings for
clustering and a local LLM for naming/describing the resulting types.

Lifecycle
---------
Stage B (extraction): taxonomy loaded from disk (if available) and injected
    as context into each extraction batch via retrieve_relevant_types().
Stage C (taxonomy build): after all options are extracted, embedding-based
    clustering groups slots, then the LLM names each cluster.
Next run with --skip-segmentation: Stage B uses the richer taxonomy built in
    the previous Stage C, improving slot naming consistency.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dataset.schema import Option, OPTIONS_PATH

logger = logging.getLogger(__name__)

TAXONOMY_PATH = OPTIONS_PATH.parent / "option_taxonomy.json"

_MAX_OPTIONS_PER_SLOT = 12    # sent to LLM per slot
_MAX_VALUE_LEN        = 120   # truncate long values before sending
_MIN_EXAMPLES_TO_TYPE = 2     # skip slots with fewer values

# ── Embedding helpers ─────────────────────────────────────────────────────

_EMBED_MODEL_CACHE: dict[str, object] = {}
_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


def _load_embed_model(model_name: str = _EMBED_MODEL_NAME):
    """Load and cache a sentence-transformer model."""
    if model_name in _EMBED_MODEL_CACHE:
        return _EMBED_MODEL_CACHE[model_name]
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    _EMBED_MODEL_CACHE[model_name] = model
    return model


def _embed_texts(texts: list[str], model_name: str = _EMBED_MODEL_NAME) -> np.ndarray:
    """Embed a list of texts, returning an (N, D) normalised array."""
    model = _load_embed_model(model_name)
    return model.encode(texts, batch_size=256, show_progress_bar=False,
                        normalize_embeddings=True)


# ── Data structures ──────────────────────────────────────────────────────

@dataclass
class OptionType:
    """One semantic type in the taxonomy.

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
    """Full taxonomy of option types, keyed by type name.

    Supports a two-level hierarchy: *categories* group semantically related
    *types*.
    """
    types: dict[str, OptionType] = field(default_factory=dict)

    # Cached embeddings for retrieval — lazily computed.
    _type_embeddings: np.ndarray | None = field(default=None, repr=False)
    _type_names_order: list[str] = field(default_factory=list, repr=False)

    # ── Derived helpers ───────────────────────────────────────────────

    @property
    def categories(self) -> dict[str, list[OptionType]]:
        """Return types grouped by their category."""
        groups: dict[str, list[OptionType]] = defaultdict(list)
        for ot in self.types.values():
            groups[ot.category].append(ot)
        return dict(groups)

    # ── Serialisation ─────────────────────────────────────────────────

    def save(self, path: Path = TAXONOMY_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
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
            if "categories" in data:
                for cat_entry in data["categories"]:
                    cat_name = cat_entry.get("name", "")
                    for entry in cat_entry.get("types", []):
                        ot = OptionType.from_dict(entry, category_fallback=cat_name)
                        types[ot.name] = ot
            elif "types" in data:
                for entry in data["types"]:
                    ot = OptionType.from_dict(entry)
                    types[ot.name] = ot
            logger.info(f"  Option taxonomy loaded from {path} ({len(types)} types)")
            return cls(types=types)
        except Exception as exc:
            logger.warning(f"  Could not load option taxonomy: {exc}")
            return cls()

    # ── Incremental update ────────────────────────────────────────────

    def update_from_options(self, new_options: list[Option]) -> None:
        """Incrementally add newly extracted options into the in-memory taxonomy.

        Called after each extraction batch so the taxonomy grows in real time
        and provides updated context for the next batch.
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
                    category="_uncategorized",
                    description="",
                    compatible_slots=[slot],
                    example_values=[value],
                )

    # ── Taxonomy retrieval (embedding-based) ──────────────────────────

    def _ensure_type_embeddings(self) -> None:
        """Lazily compute and cache embeddings for each type."""
        if self._type_embeddings is not None:
            return
        categorised = [
            ot for ot in self.types.values()
            if ot.category != "_uncategorized" and ot.description
        ]
        if not categorised:
            self._type_embeddings = np.empty((0, 1))
            self._type_names_order = []
            return

        texts = []
        names = []
        for ot in categorised:
            examples = ", ".join(ot.example_values[:5])
            texts.append(f"{ot.name}: {ot.description} Examples: {examples}")
            names.append(ot.name)

        self._type_embeddings = _embed_texts(texts)
        self._type_names_order = names

    def retrieve_relevant_types(
        self, query: str, *, max_types: int = 20,
    ) -> list[OptionType]:
        """Return the most relevant types for a given query text.

        Uses cosine similarity between the query embedding and pre-computed
        type representative embeddings.
        """
        self._ensure_type_embeddings()
        if self._type_embeddings is None or len(self._type_names_order) == 0:
            return []

        query_emb = _embed_texts([query])  # (1, D)
        sims = (query_emb @ self._type_embeddings.T).flatten()  # (N,)
        top_k = min(max_types, len(sims))
        top_indices = np.argpartition(-sims, top_k)[:top_k]
        top_indices = top_indices[np.argsort(-sims[top_indices])]

        result = []
        for idx in top_indices:
            name = self._type_names_order[idx]
            if name in self.types:
                result.append(self.types[name])
        return result

    # ── Prompt injection ──────────────────────────────────────────────

    def to_prompt_context(
        self, *, max_types: int = 40, query: str | None = None,
    ) -> str:
        """Format the taxonomy as a compact string for extraction prompts.

        When *query* is provided, uses embedding retrieval to select the most
        relevant types instead of a hard cutoff.
        """
        if not self.types:
            return ""

        if query is not None:
            selected = self.retrieve_relevant_types(query, max_types=max_types)
        else:
            selected = [
                ot for ot in self.types.values()
                if ot.category != "_uncategorized" and ot.description
            ][:max_types]

        if not selected:
            return ""

        # Group by category for display
        by_cat: dict[str, list[OptionType]] = defaultdict(list)
        for ot in selected:
            by_cat[ot.category].append(ot)

        lines = [
            "Option taxonomy (use specific slot names from each category; "
            "prefer these names when they fit the semantics):"
        ]
        for cat_name, cat_types in sorted(by_cat.items()):
            lines.append(f"  [{cat_name}]")
            for ot in cat_types:
                examples = ot.example_values[:6]
                if not examples:
                    continue
                compat_extra = [s for s in ot.compatible_slots if s != ot.name][:5]
                compat_str = f" [also fits: {', '.join(compat_extra)}]" if compat_extra else ""
                desc_str = f" — {ot.description}" if ot.description else ""
                lines.append(f"    {ot.name}{desc_str}{compat_str}: {', '.join(examples)}")
        return "\n".join(lines) if len(lines) > 1 else ""


# ── Input preparation ─────────────────────────────────────────────────────

def _prepare_input(options: list[Option]) -> dict[str, list[str]]:
    """Group option values by slot, filtering and deduplicating.

    Short values (≤50 tokens) are included verbatim.  Longer values are
    truncated with a preview — their functional role is already captured
    by the slot name (source_passage, problem_statement, etc.).
    """
    slot_to_values: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)

    for o in options:
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
            preview = value[:80].rstrip() + "…"
            slot_to_values[slot].append(preview)

    return {
        s: vs[:_MAX_OPTIONS_PER_SLOT]
        for s, vs in slot_to_values.items()
        if len(vs) >= _MIN_EXAMPLES_TO_TYPE
    }


# ── Embedding-based slot clustering ──────────────────────────────────────

def _cluster_slots_by_embedding(
    slot_values: dict[str, list[str]],
    *,
    distance_threshold: float = 0.35,
) -> list[list[str]]:
    """Cluster slots by semantic similarity of their name + example values.

    Returns a list of clusters, where each cluster is a list of slot names.
    """
    from sklearn.cluster import AgglomerativeClustering

    slot_names = list(slot_values.keys())
    if len(slot_names) <= 1:
        return [slot_names] if slot_names else []

    # Build representative text for each slot
    texts = []
    for slot in slot_names:
        values = slot_values[slot][:8]
        texts.append(f"{slot}: {', '.join(values)}")

    embeddings = _embed_texts(texts)

    # Cosine distance = 1 - cosine_similarity (embeddings are normalised)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = clustering.fit_predict(embeddings)

    clusters: dict[int, list[str]] = defaultdict(list)
    for slot, label in zip(slot_names, labels):
        clusters[label].append(slot)

    result = list(clusters.values())
    logger.info(
        f"  Embedding clustering: {len(slot_names)} slots → {len(result)} clusters "
        f"(threshold={distance_threshold})"
    )
    return result


# ── Token estimation ──────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Rough token count (chars / 4)."""
    return len(text) // 4


# ── LLM taxonomy builder ─────────────────────────────────────────────────

_BUILD_SYSTEM_PROMPT = """\
You are naming and categorising pre-clustered groups of option slots for a \
prompt-template extraction system.

Each group contains semantically similar slot names with example values.
Your job: assign a type name, category, description, and compatible_slots.

LEVEL 1 — CATEGORIES (broad semantic groups). Use exactly these where applicable:
  - content_subject: what the text is about (topics, entities, domains)
  - text_genre: what kind of text is produced (essay, email, poem, report…)
  - formatting: how the output is structured (markdown, bullet points, table…)
  - quantitative: numbers, counts, limits, thresholds
  - linguistic_unit: units of text measurement (word, sentence, character, paragraph…)
  - stylistic: tone, voice, register, literary style
  - identity: people, roles, personas, authors
  - reference: titles, quotes, citations, external sources
  - constraint_detail: short descriptive constraints
  - context_input: longer text passages that serve as input/context for the task
  You may create additional categories if the data clearly requires it.

LEVEL 2 — TYPES. For EACH cluster, output:
- "name": a specific type name (e.g. "academic_subject", not "topic").
- "category": one of the above categories.
- "description": one sentence describing what values belong here.
- "compatible_slots": list of ALL slot names from the cluster (and any \
  other slot names where these values could meaningfully appear).
- "example_values": up to 8 representative values from the input.

Rules:
- If a cluster contains clearly different sub-groups, SPLIT into multiple types.
- The type name should be specific enough to predict what values fit.
- Do NOT create a type for fewer than 2 distinct values.

Output a single JSON object:
{"categories": [{"name": "...", "types": [\
{"name": "...", "category": "...", "description": "...", \
"compatible_slots": ["...", ...], "example_values": ["...", ...]}, ...]}, ...]}\
"""


def _parse_taxonomy_response(raw: str) -> list[dict]:
    """Extract and parse the JSON from the LLM response.

    Returns a flat list of type dicts, each guaranteed to have a ``category`` key.
    """
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)

    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        # Try to find the outermost JSON object
        m = re.search(r'\{.*\}', raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if data is None:
        logger.warning(
            f"  Could not parse taxonomy response. "
            f"First 500 chars: {raw[:500]!r}"
        )
        return []

    if "categories" in data:
        all_types: list[dict] = []
        for cat in data["categories"]:
            cat_name = cat.get("name", "")
            for t in cat.get("types", []):
                t.setdefault("category", cat_name)
                all_types.append(t)
        return all_types

    types = data.get("types", [])
    for t in types:
        t.setdefault("category", t.get("name", ""))
    return types


def _cross_link_compatible_slots(
    taxonomy: OptionTaxonomy,
    slot_values: dict[str, list[str]],
    *,
    similarity_threshold: float = 0.6,
) -> None:
    """Enrich compatible_slots via embedding cosine similarity between types.

    Computes a mean embedding for each type from its member slot embeddings,
    then cross-links types whose similarity exceeds the threshold.
    """
    # Build slot → embedding mapping
    slot_names = list(slot_values.keys())
    if not slot_names:
        return

    texts = [
        f"{slot}: {', '.join(slot_values[slot][:8])}"
        for slot in slot_names
    ]
    slot_embeddings = _embed_texts(texts)
    slot_to_emb = dict(zip(slot_names, slot_embeddings))

    # Compute mean embedding per type
    type_names = []
    type_embeddings = []
    for ot in taxonomy.types.values():
        member_embs = [
            slot_to_emb[s] for s in ot.compatible_slots
            if s in slot_to_emb
        ]
        if not member_embs:
            continue
        mean_emb = np.mean(member_embs, axis=0)
        mean_emb = mean_emb / (np.linalg.norm(mean_emb) + 1e-8)
        type_names.append(ot.name)
        type_embeddings.append(mean_emb)

    if len(type_names) < 2:
        return

    type_emb_matrix = np.array(type_embeddings)
    sim_matrix = type_emb_matrix @ type_emb_matrix.T

    cross_links = 0
    for i in range(len(type_names)):
        for j in range(i + 1, len(type_names)):
            if sim_matrix[i, j] >= similarity_threshold:
                t1, t2 = type_names[i], type_names[j]
                ot1, ot2 = taxonomy.types[t1], taxonomy.types[t2]
                if t2 not in ot1.compatible_slots:
                    ot1.compatible_slots.append(t2)
                    cross_links += 1
                if t1 not in ot2.compatible_slots:
                    ot2.compatible_slots.append(t1)
                    cross_links += 1

    if cross_links:
        logger.info(
            f"  Cross-type compatible_slots: {cross_links} links added "
            f"(threshold={similarity_threshold})"
        )


# ── Main taxonomy builder ────────────────────────────────────────────────

_MAX_BATCH_TOKENS = 6000   # leave room for system prompt + output


def build_option_taxonomy(
    options: list[Option],
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str | None = None,
    save: bool = True,
    gpu_memory_utilization: float = 0.95,  # accepted for pipeline compat
) -> OptionTaxonomy:
    """Build the option taxonomy from extracted options.

    Stage C of the pipeline:
    1. Group values by slot name
    2. Cluster slots by embedding similarity
    3. LLM names and categorises each cluster
    4. Cross-link compatible_slots via embedding similarity
    """
    from dataset.local_llm import generate_text as _generate

    # Step 1: prepare input (group by slot, deduplicate, filter)
    input_data = _prepare_input(options)
    if not input_data:
        logger.warning("  No options to build taxonomy from.")
        return OptionTaxonomy()

    logger.info(
        f"  Building option taxonomy from {len(input_data)} slot groups "
        f"({sum(len(v) for v in input_data.values())} values)…"
    )

    # Step 2: cluster slots by embedding similarity
    clusters = _cluster_slots_by_embedding(input_data)

    # Step 3: LLM naming pass — batch multiple clusters per call
    types: dict[str, OptionType] = {}
    batch_clusters: list[list[str]] = []
    batch_data: dict[str, list[str]] = {}
    batch_token_est = 0
    llm_calls = 0

    def _flush_batch():
        nonlocal batch_clusters, batch_data, batch_token_est, llm_calls
        if not batch_data:
            return

        user_content = json.dumps(batch_data, ensure_ascii=False)
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
            llm_calls += 1
        except Exception as exc:
            logger.warning(f"  Taxonomy LLM call failed: {exc}")
            batch_clusters = []
            batch_data = {}
            batch_token_est = 0
            return

        entries = _parse_taxonomy_response(raw)
        for entry in entries:
            name = entry.get("name", "").strip().lower()
            category = entry.get("category", "").strip().lower() or name
            desc = entry.get("description", "").strip()
            compat = [s.strip() for s in entry.get("compatible_slots", []) if s.strip()]
            examples = [
                re.sub(r"^\[[\w]+\]\s*", "", v)
                for v in entry.get("example_values", []) if v
            ][:8]
            if not name or not desc:
                continue
            if name not in compat:
                compat.insert(0, name)
            if name in types:
                existing = types[name]
                for s in compat:
                    if s not in existing.compatible_slots:
                        existing.compatible_slots.append(s)
                for ex in examples:
                    if ex not in existing.example_values and len(existing.example_values) < 8:
                        existing.example_values.append(ex)
            else:
                types[name] = OptionType(
                    name=name,
                    category=category,
                    description=desc,
                    compatible_slots=compat,
                    example_values=examples,
                )

        batch_clusters = []
        batch_data = {}
        batch_token_est = 0

    for cluster in clusters:
        cluster_data = {s: input_data[s] for s in cluster if s in input_data}
        cluster_json = json.dumps(cluster_data, ensure_ascii=False)
        cluster_tokens = _estimate_tokens(cluster_json)

        # If adding this cluster would overflow, flush first
        if batch_data and batch_token_est + cluster_tokens > _MAX_BATCH_TOKENS:
            _flush_batch()

        batch_clusters.append(cluster)
        batch_data.update(cluster_data)
        batch_token_est += cluster_tokens

    _flush_batch()  # flush remaining

    taxonomy = OptionTaxonomy(types=types)
    logger.info(f"  Taxonomy built: {len(types)} types from {llm_calls} LLM calls")

    # Step 4: cross-link compatible_slots via embedding similarity
    _cross_link_compatible_slots(taxonomy, input_data)

    if save:
        taxonomy.save()

    return taxonomy

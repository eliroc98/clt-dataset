"""
embeddings.py — Embedding-based semantic compatibility for options and slots.

Computes embeddings for option values and template slot contexts using a
sentence-transformer model, then scores compatibility via cosine similarity.
Embeddings are cached to disk for reuse.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from dataset.schema import TaskTemplate, Option, OUTPUT_DIR

logger = logging.getLogger(__name__)

OPTION_EMBEDDINGS_PATH = OUTPUT_DIR / "option_embeddings.npz"
SLOT_EMBEDDINGS_PATH = OUTPUT_DIR / "slot_embeddings.npz"

# ── Embedding model ──────────────────────────────────────────────────────

_MODEL_CACHE: dict[str, Any] = {}


def _load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> Any:
    """Load and cache a sentence-transformer model."""
    if model_name in _MODEL_CACHE:
        return _MODEL_CACHE[model_name]

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required for embedding computation. "
            "Install it: pip install sentence-transformers"
        )

    logger.info(f"Loading embedding model '{model_name}'…")
    model = SentenceTransformer(model_name)
    _MODEL_CACHE[model_name] = model
    return model


# ── Embedding computation ────────────────────────────────────────────────

def compute_option_embeddings(
    options: list[Option],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    cache_path: Path = OPTION_EMBEDDINGS_PATH,
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """Compute embeddings for option values.

    Returns a dict mapping option_id → embedding vector.
    Results are cached to disk.
    """
    if not force_recompute and cache_path.exists():
        logger.info(f"Loading cached option embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return dict(zip(data["ids"].tolist(), data["embeddings"]))

    model = _load_embedding_model(model_name)

    # Deduplicate by value to avoid redundant computation
    value_to_ids: dict[str, list[str]] = {}
    for opt in options:
        value_to_ids.setdefault(opt.value, []).append(opt.id)

    unique_values = list(value_to_ids.keys())
    logger.info(
        f"Computing embeddings for {len(unique_values)} unique option values "
        f"({len(options)} total options)…"
    )

    all_embeddings = model.encode(
        unique_values,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    # Map back to option IDs
    result: dict[str, np.ndarray] = {}
    for value, embedding in zip(unique_values, all_embeddings):
        for opt_id in value_to_ids[value]:
            result[opt_id] = embedding

    # Cache to disk
    ids = list(result.keys())
    embeddings = np.array([result[oid] for oid in ids])
    np.savez_compressed(cache_path, ids=np.array(ids), embeddings=embeddings)
    logger.info(f"Cached {len(ids)} option embeddings → {cache_path}")

    return result


def compute_slot_context_embeddings(
    templates: list[TaskTemplate],
    *,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    cache_path: Path = SLOT_EMBEDDINGS_PATH,
    force_recompute: bool = False,
) -> dict[str, np.ndarray]:
    """Compute embeddings for template slot contexts.

    For each (template_id, slot) pair, creates a context string by replacing
    the slot placeholder with "[SOMETHING]" and embeds the full template.

    Returns a dict mapping "{template_id}:{slot}" → embedding vector.
    """
    if not force_recompute and cache_path.exists():
        logger.info(f"Loading cached slot context embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return dict(zip(data["ids"].tolist(), data["embeddings"]))

    model = _load_embedding_model(model_name)

    # Build context strings
    contexts: list[tuple[str, str]] = []  # (key, context_text)
    seen: set[str] = set()

    for tmpl in templates:
        for slot in tmpl.slots:
            key = f"{tmpl.id}:{slot}"
            if key in seen:
                continue
            seen.add(key)
            context_text = tmpl.text.replace(f"{{{slot}}}", "[SOMETHING]")
            contexts.append((key, context_text))

    if not contexts:
        return {}

    keys, texts = zip(*contexts)
    logger.info(f"Computing embeddings for {len(texts)} slot contexts…")

    all_embeddings = model.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )

    result = dict(zip(keys, all_embeddings))

    # Cache
    ids = list(result.keys())
    embeddings = np.array([result[k] for k in ids])
    np.savez_compressed(cache_path, ids=np.array(ids), embeddings=embeddings)
    logger.info(f"Cached {len(ids)} slot context embeddings → {cache_path}")

    return result


# ── Compatibility scoring ────────────────────────────────────────────────

def find_compatible_options(
    template: TaskTemplate,
    slot: str,
    option_embeddings: dict[str, np.ndarray],
    slot_context_embeddings: dict[str, np.ndarray],
    options_by_id: dict[str, Option],
    *,
    threshold: float = 0.7,
    max_results: int = 50,
) -> list[tuple[str, float]]:
    """Find options semantically compatible with a template slot.

    Returns list of (option_id, similarity_score) sorted by score descending.
    """
    slot_key = f"{template.id}:{slot}"
    slot_emb = slot_context_embeddings.get(slot_key)
    if slot_emb is None:
        return []

    scored: list[tuple[str, float]] = []
    for opt_id, opt_emb in option_embeddings.items():
        score = float(np.dot(slot_emb, opt_emb))
        if score >= threshold:
            scored.append((opt_id, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_results]


def build_compatibility_index(
    templates: list[TaskTemplate],
    options: list[Option],
    option_embeddings: dict[str, np.ndarray],
    slot_context_embeddings: dict[str, np.ndarray],
    *,
    threshold: float = 0.7,
    max_per_slot: int = 50,
) -> dict[str, list[str]]:
    """Build an index mapping "template_id:slot" → [compatible option_ids].

    This precomputes semantic compatibility for all (template, slot) pairs
    and can be used to extend the TemplateStore's compatibility logic.
    """
    options_by_id = {o.id: o for o in options}
    index: dict[str, list[str]] = {}

    for tmpl in templates:
        for slot in tmpl.slots:
            key = f"{tmpl.id}:{slot}"
            matches = find_compatible_options(
                tmpl, slot,
                option_embeddings, slot_context_embeddings, options_by_id,
                threshold=threshold, max_results=max_per_slot,
            )
            if matches:
                index[key] = [opt_id for opt_id, _score in matches]

    logger.info(
        f"Built compatibility index: {len(index)} slot contexts "
        f"with semantic matches"
    )
    return index

# CLT — Dataset Construction Pipeline

Semi-synthetic dataset construction from minimally-changing prompts. Decomposes real instruction-tuning prompts into reusable **task templates** (with `{slot}` placeholders) and **options** (concrete slot fillers), then recombines them to generate controlled synthetic prompts.

---

## Development Log

Reverse-chronological log of attempts and strategies for constructing the database.

### Phase 6: Specific Slot Naming (Mar 30)

**`6a3dca3` — replace generic canonical slots with specific descriptive slot naming (Mar 30)**
- **Design pivot:** replaced the small fixed vocabulary of generic canonical slots (`topic`, `description`, `text`, `passage`, …) with a "specific is better" naming philosophy — the slot name should predict what values fit.
  - LLM extraction prompt now instructs the model to use the most descriptive `snake_case` name for each slot (e.g. `academic_subject` over `topic`, `writing_tone` over `tone`, `interviewee` over `person`).
  - **Functional-role slots** introduced for paragraph/multi-sentence content: `source_passage`, `problem_statement`, `code_snippet`, `factual_background`, `input_data`, `example_context`.
  - `CANONICAL_PREFERRED_SLOTS` reduced from ~30 entries to 3 universals (`text_type`, `unit`, `number`) plus the 6 functional-role slots.
- **Compound-slot canonicalization simplified:** `_canonicalize_compound_slot()` no longer collapses qualified names to bare generics; only meaningless collection suffixes (`_list`, `_array`, …) are stripped. Removed `_COMPOUND_SLOT_REMAP` and `_SUFFIX_CANONICAL`.
- **`reclassify_exotic_slots()` removed** — with specific naming, post-hoc LLM remapping to generics is counterproductive; the option taxonomy handles slot grouping.
- **`_SLOT_TO_LEVEL` replaced by `_LEVEL_KEYWORDS`:** slot→taxonomy-level inference now uses lightweight keyword hints instead of a hardcoded slot-name lookup table.
- **Per-segment taxonomy retrieval:** extraction batches now pass `query=seg.span_text` to `to_prompt_context()` for relevant context instead of a static snapshot.
- `_TASK_TYPE_TO_DEFAULT_SLOT` and `_SELF_REP_VALUE_TO_SLOT` updated to use specific names (`subject_area`, `writing_tone`, `target_language`, etc.).
- Files: `extractor.py`, `fix_slots.py`, `option_taxonomy.py`, `pipeline.py`

---

### Phase 5: Option Refinement & Production Hardening (Mar 20–21)

**`454edd6` — fix validation crash, orphaned options, and taxonomy overflow (Mar 21)**
- Production bug fixes for running at scale:
  - **Validator OOM:** sub-batched 32K prompts into chunks of 500 to avoid vLLM out-of-memory; added JSON schema constraint for classification output.
  - **Orphaned options:** added `store.relink()` to rebuild option↔template links bidirectionally after normalization.
  - **Taxonomy overflow:** batched long-value classification (50/batch) and taxonomy build (15 slot groups/batch) to stay within `max_model_len`.
  - **Slot canonicalization:** added stripping of collection suffixes (`_list`, `_array`, `_set`, `_values`, `_names`, etc.) in `fix_slots.py`.
- Files: `fix_slots.py`, `option_taxonomy.py`, `pipeline.py`, `store.py`, `validator.py`

**`0d9d389` — working on option refinement (Mar 20)**
- **Major strategy pivot:** replaced the static `OPTION_TYPES` dict with a **data-driven, LLM-constructed taxonomy** in `option_taxonomy.py` (247 → 938 lines).
  - New lifecycle: Stage B loads taxonomy from disk and injects it as context; Stage C builds the taxonomy from the full option pool via LLM. Each run's taxonomy feeds the next run's extraction.
  - Uses WordNet for semantic hints, LLM to classify long text values by functional role.
- Extended `extractor.py` with slot vocabulary injection, text-type word detection rules, `_SLOT_TO_LEVEL` fallback mapping.
- Added `_filter_undercovered_templates()` to remove templates where all slots have only 1 option value.
- Expanded `segmenter.py`, `validator.py`, `fix_slots.py` with additional rules.
- Files: `extractor.py`, `fix_slots.py`, `option_taxonomy.py`, `pipeline.py`, `segmenter.py`, `validator.py`

---

### Phase 4: Two-Stage Pipeline & Validation (Mar 19)

**`e91e5c2` — template extraction is ok (Mar 19)**
- Declared template extraction satisfactory; shifted focus to options.
- Created `option_taxonomy.py` (247 lines) — a static taxonomy of option types (entity_reference, topical_subject, textual_content, quantitative, linguistic, structural, etc.) with slot compatibility mappings.
- Created `embeddings.py` (225 lines) — **embedding-based semantic compatibility** using sentence-transformers (`all-MiniLM-L6-v2`) + cosine similarity to score option↔slot compatibility.
- Created `augment_llm.py` (336 lines) — LLM-based option augmentation (shortened, alternative, generalized variations).
- Simplified `extractor.py` by moving responsibilities to new modules. Extended `store.py`.
- Files: `augment_llm.py`, `embeddings.py`, `extractor.py`, `generator.py`, `option_taxonomy.py`, `pipeline.py`, `schema.py`, `segmenter.py`, `store.py`

**`5084c57` — improving task extraction (Mar 19)**
- **Strategy shift to a two-stage pipeline:**
  - Created `segmenter.py` (543 lines) — "Stage A": splits a prompt into clauses and classifies each with a taxonomy label using a single LLM call. Uses **dynamic Pydantic models with enum constraints** so vLLM's guided decoding restricts output to valid labels only.
  - Created `validator.py` (352 lines) — post-extraction validation: tests random option substitutions and checks that filled prompts still match their task type via LLM classification.
- Extended `extractor.py` (+328 lines) — "Stage B" now receives segmentation hints.
- Added `_flatten_taxonomy_labels()` for the nested/hierarchical taxonomy format.
- Added `_filter_non_reusable()` to remove bare `{slot}` templates and overly long fixed text.
- Added spaCy dependency for NER/parsing-based ablation.
- Files: `extractor.py`, `fix_slots.py`, `local_llm.py`, `pipeline.py`, `schema.py`, `segmenter.py`, `validator.py`

---

### Phase 3: Major Refactor — Modular Pipeline (Mar 13)

**`26ffd52` — improving option extraction (Mar 13)**
- Created `fix_slots.py` (885 lines) — a dedicated slot canonicalization module with:
  - `CANONICAL_PREFERRED_SLOTS`: the definitive set of allowed slot names.
  - `_canonicalize_compound_slot()`: maps compound names (e.g., `research_topic` → `topic`) via suffix matching.
  - `_expand_list_options()`, `_drop_duplicate_slot_templates()`, full `normalize_existing()` function.
- Added LLM-based **slot reclassification** (`reclassify_exotic_slots()`): sends non-canonical slot names to the LLM for remapping.
- Rewrote extraction system prompt to enforce singular slot names and one-option-per-item.
- Expanded `few_shot_examples.json` significantly (+927 lines).
- Files: `extractor.py`, `few_shot_examples.json`, `fix_slots.py`, `pipeline.py`

**`91282e8` — improving extraction (Mar 13)**
- **Complete architectural rewrite:** deleted the monolithic `construct_dataset.py` (2,482 lines) and split into 7 focused modules:
  - `schema.py` (114 lines) — data structures (`TaskTemplate`, `Option`, `ExtractionResult`)
  - `extractor.py` (824 lines) — LLM extraction logic
  - `store.py` (188 lines) — `TemplateStore` persistence
  - `augmentor.py` (207 lines) — programmatic option augmentation
  - `generator.py` (226 lines) — synthetic prompt generation
  - `pipeline.py` (358 lines) — CLI orchestration
  - `token_counter.py` (102 lines) — tokenizer-based length computation
- Switched to **vLLM** for inference (replacing HuggingFace `model.generate()`), dramatically faster batched inference.
- Files: 13 files changed (+2,422/−2,833)

---

### Phase 2: Few-Shot Examples & Taxonomy Redesign (Mar 12)

**`a516ec7` — new taxonomy (Mar 12)**
- Wrote `explanation.md` (215 lines) documenting a **compositional taxonomy redesign**.
- Key insight: moved from a flat list to `instruction = task_type × [format_constraints] × [content_style_constraints] × [process_directives]`.
- Identified 3 problems with the old taxonomy: category conflation, redundancy from induction, flat structure where hierarchy is needed.
- Defined 5 task clusters (Information, Reasoning, Generative, Structured Output, Action/Planning) with ~25 task types and 3 orthogonal constraint axes.

**`db51ced` — adding scipy (Mar 12)**
- Added scipy dependency for statistical analysis of extraction quality.

**`2c19873` — improved taxonomy and examples (Mar 12)**
- **Major overhaul** (+2,532/−4,507):
  - Removed all per-dataset analysis JSONs and old flat taxonomy summary.
  - **Simplified taxonomy.json** from ~1,092 lines to ~467 lines.
  - Removed hardcoded few-shot examples from code; now relies on `few_shot_examples.json` or falls back to **zero-shot**.
  - **Massive prompt engineering**: added "Slot Naming Guidelines" (use `n` for all numbers, `options` as single slot not `option_a`/`option_b`, `text_type` for document types), "Common Mistakes to Avoid" with BAD/GOOD examples.
  - Built a **post-processing normalization layer** (`SLOT_CANONICAL_MAP`): maps bad slot names to canonical forms (e.g., `n1`→`n`, `option_a`→`option`, single letters like `f`→`function_code`).
  - Added `_detect_self_replicating_slots()`, `_merge_numbered_slots()`, `_sync_template_slots()`, `_remove_broken_options()`, `_remove_orphaned_options()`.
- Files: `construct_dataset.py`, `few_shot_examples.json`, taxonomy files

**`3284d34` — organizing repo (Mar 12)**
- Moved files into subdirectories: `analysis/`, `existing_taxonomy/`, `few_shot_annotation/`.

**`e7a948d` — few shot example and annotation (Mar 12)**
- Created `annotate_few_shot.py` (666 lines) — terminal-based interactive annotation tool for building dataset-specific few-shot examples.
- Created `annotation_app.py` (1,873 lines) — web-based annotation UI (localhost:8765).
- Saved `few_shot_examples.json` with hand-annotated examples (2,309 lines).
- Updated extraction to use **dataset-specific few-shot examples** — each dataset gets tailored examples instead of generic ones.
- Files: `annotate_few_shot.py`, `annotation_app.py`, `construct_dataset.py`, `few_shot_examples.json`

---

### Phase 1: Initial Setup & Getting It Running (Mar 7)

**`94d8372` — optimising (Mar 7)**
- Cached the `outlines.Generator` (FSM compilation against model vocabulary is expensive). Added `_GENERATOR_CACHE` keyed by `(model_name, schema_key)`.
- Added hint to extraction prompt about multiple-choice question formats.

**`870c874` — forcing json outputs (Mar 7)**
- **Strategy shift:** introduced Pydantic schemas (`_ExtractionResult`, `_TemplateItem`, `_OptionItem`) and passed them as `json_schema` to the LLM via **`outlines` library** for constrained/guided decoding — guaranteed valid JSON output.
- Added `generate_text_batch()` for batched inference (batch_size=8).
- Added `<think>...</think>` stripping for Qwen3-style models and `enable_thinking=False`.
- Added `_merge_into_store()` helper for deduplication.
- Files: `construct_dataset.py`, `local_llm.py`, `pyproject.toml`

**`902b465` — tqdms and json parsing fixes (Mar 7)**
- Added tqdm progress bars. Built robust `_repair_json()` function: strips markdown fences, extracts first JSON object via brace-depth counting, applies fixup chain (single-quoted strings, trailing commas, unquoted property names, truncation repair).
- Problem: LLM was returning malformed JSON that crashed the pipeline.

**`884fbe4` — fixing device issues (Mar 7)**
- Fixed GPU/CPU device handling in `local_llm.py` — model wasn't landing on the right device.

**`7495480` — avoiding poetry creating virtualenvs (Mar 7)**
- Set `[virtualenvs] create = false` in `poetry.toml`.

**`66264e3` — removed local python version (Mar 7)**
- Added `.python-version` to `.gitignore`.

**`a471d3a` — test mode (Mar 7)**
- Added `--test` CLI flag to run on a small subset for quick iteration.

**`2a5b39e` — initial commit (Mar 7)**
- Full project scaffold: `construct_dataset.py` (1,636 lines), `collect_task_types.py` (1,680 lines), `local_llm.py`, `auth.py`, `analyse_prompts.py`.
- Architecture: 5-stage pipeline (Extract → Store → Store compatibility → Augment options → Generate).
- Flat taxonomy with ~40+ task types auto-discovered from 10+ datasets (ARC, Alpaca, BoolQ, DROP, DollyV2, FLAN, etc.).
- LLM extraction using local HuggingFace model (Llama-3.1-8B-Instruct).

---

## Strategy Evolution Summary

| Phase | Dates | Core Strategy | Key Problem Solved |
|---|---|---|---|
| 1 | Mar 7 | Monolithic script + free-text LLM → structured output via `outlines` | Getting anything to run; JSON reliability |
| 2 | Mar 12 | Few-shot annotation + taxonomy redesign + slot normalization | LLM output quality & consistency |
| 3 | Mar 13 | Modular refactor (7 modules) + dedicated slot fixing + vLLM | Maintainability & inference speed |
| 4 | Mar 19 | Two-stage pipeline (segment→extract) + validation + embeddings | Template extraction accuracy |
| 5 | Mar 20–21 | LLM-built option taxonomy + production hardening | Option compatibility at scale |
| 6 | Mar 30 | Specific descriptive slot naming + functional-role slots | Slot precision & option taxonomy coherence |

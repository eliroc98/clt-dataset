"""
pipeline.py — Full dataset construction pipeline and CLI entry-point.

Implements the pipeline:
  1. Extract: decompose real prompts into templates + options via LLM
  2. Augment: expand the option pool programmatically
  3. Generate: synthetically produce prompts from templates + options

Usage
─────
  # Full pipeline
  python -m dataset.pipeline --n 5000 --density 3 --seed 42

  # Extract only (requires a local model)
  python -m dataset.pipeline --extract-only --model meta-llama/Llama-3.1-8B-Instruct

  # Extract from specific datasets
  python -m dataset.pipeline --extract-only --datasets Alpaca DollyV2

  # Generate from previously extracted templates
  python -m dataset.pipeline --generate-only --n 10000 --density 4

  # Normalize existing templates/options without re-running extraction
  python -m dataset.pipeline --normalize-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

from tqdm import tqdm

from dataset.schema import (
    TAXONOMY_PATH, TEMPLATES_PATH, OPTIONS_PATH, GENERATED_PATH,
    GeneratedPrompt,
)
from dataset.token_counter import set_token_counter, get_token_counter
from dataset.extractor import extract_templates_from_dataset, normalize_existing
from dataset.augmentor import augment_options
from dataset.store import TemplateStore
from dataset.generator import SyntheticGenerator

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dataset.taxonomy.collect_task_types import (
    DATASET_REGISTRY,
    load_hf_dataset,
    load_github_data,
    load_kcif_data,
    extract_instruction_text,
)
from dataset.auth import resolve_hf_token


# ── Dataset loading ───────────────────────────────────────────────────────

def _load_dataset_prompts(
    datasets: list[str] | None = None,
    max_per_dataset: int | None = None,
) -> list[dict]:
    """
    Stream prompts from official instruction-tuning datasets.

    Returns a list of {"prompt": ..., "source": ...} dicts.
    """
    registry = DATASET_REGISTRY
    if datasets:
        registry = {k: v for k, v in registry.items() if k in datasets}

    prompts: list[dict] = []
    pbar = tqdm(registry.items(), desc="Loading datasets", unit="dataset")
    for ds_name, config in pbar:
        pbar.set_description(f"Loading {ds_name}")
        if config.get("source") == "skip":
            continue
        try:
            if config["source"] == "huggingface":
                samples = load_hf_dataset(config)
            elif config["source"] == "github_raw":
                samples = load_github_data(config)
            elif config["source"] == "kcif_github":
                samples = load_kcif_data(config)
            else:
                tqdm.write(f"WARNING: Unknown source type for {ds_name}: {config['source']}")
                continue
        except Exception as e:
            tqdm.write(f"ERROR: Failed to load {ds_name}: {e}")
            continue

        if not samples:
            tqdm.write(f"WARNING: No samples for {ds_name}")
            continue

        instruction_fields = config.get("instruction_fields", [])
        ds_count = 0
        for sample in samples:
            text = extract_instruction_text(sample, instruction_fields)
            if text.strip():
                prompts.append({"prompt": text, "source": ds_name})
                ds_count += 1
                if max_per_dataset is not None and ds_count >= max_per_dataset:
                    break

        tqdm.write(f"INFO:   {ds_name}: {len(samples)} samples → {ds_count} prompts")
        pbar.set_postfix(prompts=len(prompts))

    logger.info(f"  Total prompts loaded: {len(prompts)}")
    return prompts


# ── Pipeline stages ───────────────────────────────────────────────────────

def run_extraction(
    taxonomy: dict,
    *,
    model: str = "meta-llama/Llama-3.1-8B-Instruct",
    device: str | None = None,
    datasets: list[str] | None = None,
    test: bool = False,
    batch_size: int = 32,
) -> TemplateStore:
    """
    Extract templates from real dataset prompts via a local model.

    Parameters
    ----------
    test
        When True, load only 2 prompts per dataset for a quick smoke-test.
    batch_size
        Number of prompts to process in a single batched forward pass.
        vLLM processes all requests concurrently so large batches are efficient.
    """
    store = TemplateStore()

    if test:
        logger.info("TEST MODE: loading 2 prompts per dataset.")

    logger.info(f"Streaming prompts for extraction (model={model}, batch_size={batch_size})…")
    prompts = _load_dataset_prompts(datasets, max_per_dataset=2 if test else None)

    if prompts:
        ext_tmpls, ext_opts = extract_templates_from_dataset(
            prompts, taxonomy,
            model=model, device=device, batch_size=batch_size,
        )
        store.add_templates(ext_tmpls)
        store.add_options(ext_opts)
        logger.info(f"  Extracted: {len(ext_tmpls)} templates, {len(ext_opts)} options")
    else:
        logger.warning("No prompts found from any dataset; store will be empty.")

    logger.info(f"  Total store: {len(store.templates)} templates, {len(store.options)} options")
    return store


def run_augmentation(store: TemplateStore, seed: int = 42) -> None:
    """Expand the option pool with programmatically generated variations."""
    logger.info("Augmenting options…")
    rng = random.Random(seed)
    augmented = augment_options(
        list(store.options.values()),
        list(store.templates.values()),
        rng,
    )
    store.add_options(augmented)
    logger.info(
        f"  Store after augmentation: {len(store.templates)} templates, "
        f"{len(store.options)} options"
    )


def run_generation(
    store: TemplateStore,
    taxonomy: dict,
    n: int = 5000,
    density: int = 3,
    min_tokens: int = 0,
    max_tokens: int = 0,
    seed: int = 42,
    output_path: Path = GENERATED_PATH,
) -> list[GeneratedPrompt]:
    """Generate synthetic prompts and save to JSONL."""
    logger.info(f"Generating {n} prompts (density={density})…")

    gen = SyntheticGenerator(
        store=store, taxonomy=taxonomy,
        density=density, min_tokens=min_tokens, max_tokens=max_tokens, seed=seed,
    )
    results = gen.generate_batch(n)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "id": r.id, "prompt": r.prompt,
                "task_template_id": r.task_template_id,
                "constraint_template_ids": r.constraint_template_ids,
                "option_ids": r.option_ids,
                "active_labels": r.active_labels,
                "task": r.task, "constraints": r.constraints,
                "density": r.density, "token_length": r.token_length,
                "combo_id": r.combo_id,
            }, ensure_ascii=False) + "\n")
    logger.info(f"  Saved {len(results)} prompts → {output_path}")

    cov = gen.coverage_report()
    logger.info(f"  Unique combos: {cov['unique_combos']}")
    with open(output_path.with_suffix(".coverage.json"), "w") as f:
        json.dump(cov, f, indent=2)

    return results


# ── CLI ───────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Semi-synthetic dataset construction from minimally-changing prompts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # Full pipeline
  python -m dataset.pipeline --n 5000 --density 3

  # Extract only
  python -m dataset.pipeline --extract-only --model meta-llama/Llama-3.1-8B-Instruct

  # Generate from existing templates
  python -m dataset.pipeline --generate-only --n 10000 --density 4

  # With token-length constraints
  python -m dataset.pipeline --n 10000 --density 4 --min-tokens 20 --max-tokens 200
""",
    )
    parser.add_argument("--n", type=int, default=5000,
                        help="Number of prompts to generate")
    parser.add_argument("--density", type=int, default=3,
                        help="Templates per prompt (1=task only, 3=task+2 constraints)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tokens", type=int, default=0,
                        help="Minimum prompt length in tokens (0=no limit)")
    parser.add_argument("--max-tokens", type=int, default=0,
                        help="Maximum prompt length in tokens (0=no limit)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSONL path")
    parser.add_argument("--tokenizer", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Tokenizer: tiktoken encoding, OpenAI model name, or HF model ID")
    parser.add_argument("--model", type=str,
                        default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID for extraction")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token override (default: read from .env.huggingface)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (ignored by vLLM; vLLM auto-selects GPUs)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Prompts per vLLM batch during extraction (default: 32)")
    parser.add_argument("--datasets", nargs="*", default=None,
                        help="Dataset names to use (default: all in DATASET_REGISTRY)")
    parser.add_argument("--extract-only", action="store_true",
                        help="Only extract templates; skip generation")
    parser.add_argument("--generate-only", action="store_true",
                        help="Generate from previously saved templates")
    parser.add_argument("--test", action="store_true",
                        help="Smoke-test: 2 prompts/dataset, 5 generated prompts")
    parser.add_argument("--merge-numbered-slots", action="store_true",
                        help="Merge numbered slot variants (n1, n2 → number_list) during extraction")
    parser.add_argument("--normalize-existing", action="store_true",
                        help="Run normalization on existing templates.json / options.json and exit")

    args = parser.parse_args()

    set_token_counter(args.tokenizer)
    tc = get_token_counter()
    logger.info(f"Tokenizer: {tc.name}")

    if not TAXONOMY_PATH.exists():
        logger.error(f"Taxonomy not found at {TAXONOMY_PATH}. Run collect_task_types.py first.")
        sys.exit(1)
    with open(TAXONOMY_PATH) as f:
        taxonomy = json.load(f)

    output_path = Path(args.output) if args.output else GENERATED_PATH

    # ── Normalize existing ────────────────────────────────────────────
    if args.normalize_existing:
        logger.info("Normalizing existing templates and options…")
        if not TEMPLATES_PATH.exists() or not OPTIONS_PATH.exists():
            logger.error("templates.json or options.json not found.")
            sys.exit(1)
        with open(TEMPLATES_PATH) as f:
            raw_templates = json.load(f)
        with open(OPTIONS_PATH) as f:
            raw_options = json.load(f)
        n_t, n_o = len(raw_templates), len(raw_options)
        raw_templates, raw_options = normalize_existing(raw_templates, raw_options)
        with open(TEMPLATES_PATH, "w") as f:
            json.dump(raw_templates, f, indent=2, ensure_ascii=False)
        with open(OPTIONS_PATH, "w") as f:
            json.dump(raw_options, f, indent=2, ensure_ascii=False)
        logger.info(
            f"  {n_t} templates, {n_o} options → "
            f"{len(raw_templates)} templates, {len(raw_options)} options"
        )
        return

    if args.test:
        logger.info("*** TEST MODE — 2 prompts per dataset, 5 generated prompts ***")

    # ── Extract ───────────────────────────────────────────────────────
    if not args.generate_only:
        if args.hf_token:
            os.environ["HF_TOKEN"] = args.hf_token

        store = run_extraction(
            taxonomy,
            model=args.model,
            device=args.device,
            datasets=args.datasets,
            test=args.test,
            batch_size=args.batch_size,
        )
        run_augmentation(store, seed=args.seed)
        store.save()
    else:
        store = TemplateStore.load()
        if not store.templates:
            logger.error("No templates found. Run without --generate-only first.")
            sys.exit(1)

    if args.extract_only:
        logger.info(f"Extraction complete: {json.dumps(store.summary(), indent=2)}")
        return

    # ── Generate ──────────────────────────────────────────────────────
    run_generation(
        store, taxonomy,
        n=5 if args.test else args.n,
        density=args.density,
        min_tokens=args.min_tokens,
        max_tokens=args.max_tokens,
        seed=args.seed,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()

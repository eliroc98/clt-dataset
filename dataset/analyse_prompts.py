#!/usr/bin/env python3
"""
Analyse constructed prompts for SAE training suitability.

Produces a comprehensive report covering:
  1. Density distribution
  2. Prompt length statistics
  3. Feature frequency & balance (per-label, per-level)
  4. Co-occurrence matrix & mutual information
  5. Combinatorial coverage estimation
  6. SAE-readiness diagnostics

Usage:
  python dataset/analyse_prompts.py                               # defaults
  python dataset/analyse_prompts.py --input path/to/prompts.jsonl
  python dataset/analyse_prompts.py --plot                        # save figures
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = BASE_DIR / "output" / "constructed_prompts.jsonl"
TAXONOMY_PATH = BASE_DIR / "output" / "taxonomy" /"taxonomy.json"
OUTPUT_DIR = BASE_DIR / "output" / "analysis_plots"
OUTPUT_DIR.mkdir(exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_prompts(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_taxonomy(path: Path = TAXONOMY_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def build_binary_matrix(records: list[dict], all_labels: list[str]) -> np.ndarray:
    """Build an (N, L) binary matrix where M[i, j] = 1 if label j is active in prompt i."""
    label_idx = {lab: j for j, lab in enumerate(all_labels)}
    M = np.zeros((len(records), len(all_labels)), dtype=np.float32)
    for i, rec in enumerate(records):
        for lab in rec["active_labels"]:
            if lab in label_idx:
                M[i, label_idx[lab]] = 1.0
    return M


# ═══════════════════════════════════════════════════════════════════════════
# 2. ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def density_stats(records: list[dict]) -> dict:
    densities = [r.get("density", len(r.get("active_labels", []))) for r in records]
    arr = np.array(densities)
    dist = Counter(densities)
    return {
        "n_prompts": len(records),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "median": float(np.median(arr)),
        "distribution": dict(sorted(dist.items())),
    }


def length_stats(records: list[dict]) -> dict:
    """Prompt length statistics (uses token_length if available, else word count)."""
    if not records:
        return {}
    if "token_length" in records[0]:
        lengths = [r["token_length"] for r in records]
        unit = "tokens"
    else:
        lengths = [len(r.get("prompt", "").split()) for r in records]
        unit = "words"
    arr = np.array(lengths)
    return {
        "unit": unit,
        "n_prompts": len(records),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "min": int(np.min(arr)),
        "max": int(np.max(arr)),
        "median": float(np.median(arr)),
    }


def feature_frequency(records: list[dict], taxonomy: dict) -> dict:
    """How often each label appears, grouped by level."""
    label_counts: Counter = Counter()
    for rec in records:
        for lab in rec["active_labels"]:
            label_counts[lab] += 1

    level_map = {name: info["level"] for name, info in taxonomy.items()}
    by_level: dict[str, dict[str, int]] = defaultdict(dict)
    for lab, cnt in label_counts.most_common():
        lv = level_map.get(lab, "unknown")
        by_level[lv][lab] = cnt

    n = len(records)
    # Balance metrics per level
    level_balance = {}
    for lv, counts in by_level.items():
        vals = np.array(list(counts.values()), dtype=float)
        if len(vals) > 1:
            # Coefficient of variation (lower = more balanced)
            cv = float(np.std(vals) / np.mean(vals)) if np.mean(vals) > 0 else 0
            # Normalised entropy (1.0 = perfectly uniform)
            probs = vals / vals.sum()
            ent = -np.sum(probs * np.log2(probs + 1e-12))
            max_ent = np.log2(len(vals))
            norm_ent = float(ent / max_ent) if max_ent > 0 else 0
        else:
            cv = 0.0
            norm_ent = 1.0
        level_balance[lv] = {
            "n_labels": len(counts),
            "total_activations": int(vals.sum()),
            "mean_freq": float(np.mean(vals)),
            "cv": round(cv, 3),
            "normalised_entropy": round(norm_ent, 3),
        }

    return {
        "by_level": {lv: dict(sorted(cnts.items(), key=lambda x: -x[1]))
                     for lv, cnts in sorted(by_level.items())},
        "level_balance": level_balance,
        "never_sampled": [name for name in taxonomy if label_counts[name] == 0],
    }


def cooccurrence_analysis(M: np.ndarray, all_labels: list[str], top_k: int = 30) -> dict:
    """Pairwise co-occurrence counts and pointwise mutual information."""
    N, L = M.shape
    # Co-occurrence matrix: C[i,j] = number of prompts where both i and j are active
    C = (M.T @ M).astype(int)
    np.fill_diagonal(C, 0)

    # Marginal probabilities
    p = M.mean(axis=0)  # shape (L,)

    # PMI matrix
    # PMI(i,j) = log2(P(i,j) / (P(i)*P(j)))
    P_joint = C.astype(float) / N
    P_outer = np.outer(p, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        PMI = np.log2(P_joint / (P_outer + 1e-12) + 1e-12)
    PMI[np.isnan(PMI)] = 0
    PMI[np.isinf(PMI)] = 0
    np.fill_diagonal(PMI, 0)

    # Top co-occurring pairs
    top_pairs = []
    indices = np.argsort(C.ravel())[::-1]
    seen = set()
    for flat_idx in indices:
        i, j = divmod(int(flat_idx), L)
        if i >= j:
            continue
        pair = (all_labels[i], all_labels[j])
        if pair in seen:
            continue
        seen.add(pair)
        top_pairs.append({
            "pair": pair,
            "count": int(C[i, j]),
            "pmi": round(float(PMI[i, j]), 3),
        })
        if len(top_pairs) >= top_k:
            break

    # Top PMI pairs (strongest associations)
    top_pmi_pairs = []
    pmi_flat = PMI.copy()
    np.fill_diagonal(pmi_flat, -999)
    pmi_indices = np.argsort(pmi_flat.ravel())[::-1]
    seen2 = set()
    for flat_idx in pmi_indices:
        i, j = divmod(int(flat_idx), L)
        if i >= j:
            continue
        pair = (all_labels[i], all_labels[j])
        if pair in seen2:
            continue
        seen2.add(pair)
        if C[i, j] < 3:  # skip very rare pairs for meaningful PMI
            continue
        top_pmi_pairs.append({
            "pair": pair,
            "pmi": round(float(PMI[i, j]), 3),
            "count": int(C[i, j]),
        })
        if len(top_pmi_pairs) >= top_k:
            break

    return {
        "top_cooccurring_pairs": top_pairs,
        "top_pmi_pairs": top_pmi_pairs,
    }


def coverage_analysis(records: list[dict], taxonomy: dict) -> dict:
    """Estimate how much of the combinatorial space is explored."""
    unique_combos = set()
    unique_label_sets = set()
    for rec in records:
        unique_combos.add(rec.get("combo_id", ""))
        unique_label_sets.add(tuple(sorted(rec["active_labels"])))

    # Theoretical space (very rough upper bound)
    level_map = {name: info["level"] for name, info in taxonomy.items()}
    tasks = [n for n, lv in level_map.items() if lv == "content_task"]
    constraints = [n for n, lv in level_map.items() if lv != "content_task"]
    n_tasks = len(tasks)
    n_constraints = len(constraints)

    # For density d, theoretical combos = n_tasks * C(n_constraints, d-1)
    densities = [r.get("density", len(r.get("active_labels", []))) for r in records]
    modal_density = Counter(densities).most_common(1)[0][0]
    k = max(modal_density - 1, 0)
    if k > 0 and n_constraints >= k:
        theoretical = n_tasks * math.comb(n_constraints, k)
    else:
        theoretical = n_tasks

    return {
        "unique_combo_ids": len(unique_combos),
        "unique_label_sets": len(unique_label_sets),
        "theoretical_upper_bound": theoretical,
        "coverage_ratio": round(len(unique_label_sets) / theoretical, 4) if theoretical > 0 else 1.0,
        "modal_density": modal_density,
        "n_tasks": n_tasks,
        "n_constraints": n_constraints,
    }


def sae_diagnostics(M: np.ndarray, all_labels: list[str], taxonomy: dict) -> dict:
    """
    SAE-specific diagnostics:
    - Feature sparsity: what fraction of features are active per prompt (should be low)
    - Feature dead-ness: features that are very rare (SAE may not learn them)
    - Correlation structure: eigenvalue spectrum of the correlation matrix
    - Effective rank: how many independent dimensions the data spans
    """
    N, L = M.shape

    # Sparsity
    activations_per_prompt = M.sum(axis=1)
    sparsity_ratio = float(np.mean(activations_per_prompt)) / L

    # Feature frequencies
    freq = M.mean(axis=0)
    dead_threshold = 5.0 / N  # fewer than 5 activations
    rare_threshold = 0.01     # < 1% of prompts
    dead_features = [all_labels[j] for j in range(L) if freq[j] < dead_threshold]
    rare_features = [all_labels[j] for j in range(L) if dead_threshold <= freq[j] < rare_threshold]

    # Correlation matrix & effective rank
    # Centre the data
    M_centred = M - M.mean(axis=0, keepdims=True)
    # Covariance
    cov = (M_centred.T @ M_centred) / N
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.sort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues, 0, None)  # numerical cleanup

    # Effective rank (exponential of spectral entropy)
    eig_sum = eigenvalues.sum()
    if eig_sum > 0:
        p_eig = eigenvalues / eig_sum
        p_eig = p_eig[p_eig > 1e-12]
        spectral_entropy = -np.sum(p_eig * np.log(p_eig))
        effective_rank = float(np.exp(spectral_entropy))
    else:
        effective_rank = 0.0

    # Variance explained by top-k components
    cumvar = np.cumsum(eigenvalues) / (eig_sum + 1e-12)
    var_explained = {}
    for k in [5, 10, 20, 30, 50]:
        if k <= len(cumvar):
            var_explained[f"top_{k}"] = round(float(cumvar[k-1]), 4)

    # Feature correlation distribution
    with np.errstate(divide="ignore", invalid="ignore"):
        std = np.sqrt(np.diag(cov))
        std[std == 0] = 1e-12
        corr = cov / np.outer(std, std)
    np.fill_diagonal(corr, 0)
    upper_tri = corr[np.triu_indices(L, k=1)]

    return {
        "n_prompts": N,
        "n_features": L,
        "sparsity": {
            "mean_active_per_prompt": round(float(np.mean(activations_per_prompt)), 2),
            "fraction_active": round(sparsity_ratio, 4),
            "description": (
                "Good for SAE" if sparsity_ratio < 0.1 else
                "Moderate — consider increasing feature count or reducing density"
                if sparsity_ratio < 0.2 else
                "High — SAE may struggle to find sparse codes"
            ),
        },
        "dead_features": {
            "count": len(dead_features),
            "labels": dead_features,
            "note": "Features with <5 activations. SAE won't learn these reliably.",
        },
        "rare_features": {
            "count": len(rare_features),
            "labels": rare_features,
            "note": "Features in <1% of prompts. May need upsampling.",
        },
        "effective_rank": {
            "value": round(effective_rank, 1),
            "total_features": L,
            "ratio": round(effective_rank / L, 3),
            "description": (
                "Excellent — data spans many independent directions"
                if effective_rank / L > 0.5 else
                "Good — reasonable dimensionality"
                if effective_rank / L > 0.3 else
                "Low — features are highly correlated, SAE may collapse dimensions"
            ),
        },
        "variance_explained": var_explained,
        "correlation_distribution": {
            "mean_abs_corr": round(float(np.mean(np.abs(upper_tri))), 4),
            "max_abs_corr": round(float(np.max(np.abs(upper_tri))), 4),
            "fraction_above_0.3": round(float(np.mean(np.abs(upper_tri) > 0.3)), 4),
            "description": (
                "Low inter-feature correlation — good for SAE"
                if np.mean(np.abs(upper_tri)) < 0.05 else
                "Moderate correlation — some feature entanglement expected"
                if np.mean(np.abs(upper_tri)) < 0.15 else
                "High correlation — SAE may have difficulty disentangling"
            ),
        },
    }


# ═══════════════════════════════════════════════════════════════════════════
# 3. PLOTTING (optional, requires matplotlib)
# ═══════════════════════════════════════════════════════════════════════════

def save_plots(
    records: list[dict],
    M: np.ndarray,
    all_labels: list[str],
    taxonomy: dict,
    output_dir: Path,
):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError:
        print("⚠  matplotlib not installed — skipping plots. Install with: pip install matplotlib")
        return

    level_map = {name: info["level"] for name, info in taxonomy.items()}
    level_colours = {
        "content_task": "#4C72B0",
        "format": "#DD8452",
        "style": "#55A868",
        "content_constraint": "#C44E52",
        "meta": "#8172B3",
    }

    fig_dir = output_dir / "analysis_plots"
    fig_dir.mkdir(exist_ok=True)

    # ── 1. Density distribution ───────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    densities = [r["density"] for r in records]
    dist = Counter(densities)
    xs = sorted(dist.keys())
    ax.bar(xs, [dist[x] for x in xs], color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Density (active features per prompt)")
    ax.set_ylabel("Count")
    ax.set_title("Density Distribution")
    for x in xs:
        ax.text(x, dist[x] + 10, str(dist[x]), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(fig_dir / "density_distribution.png", dpi=150)
    plt.close(fig)

    # ── 2. Feature frequency by level ─────────────────────────────────
    freq = M.mean(axis=0)
    order = np.argsort(freq)[::-1]
    fig, ax = plt.subplots(figsize=(16, 5))
    colours = [level_colours.get(level_map.get(all_labels[j], ""), "#999") for j in order]
    ax.bar(range(len(order)), freq[order], color=colours, edgecolor="none")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([all_labels[j] for j in order], rotation=90, fontsize=6)
    ax.set_ylabel("Frequency (fraction of prompts)")
    ax.set_title("Feature Frequency (coloured by level)")
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=lv) for lv, c in level_colours.items()]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "feature_frequency.png", dpi=150)
    plt.close(fig)

    # ── 3. Co-occurrence heatmap ──────────────────────────────────────
    C = (M.T @ M).astype(float)
    np.fill_diagonal(C, 0)
    # Only keep features with some occurrence
    active_mask = freq > 0
    active_idx = np.where(active_mask)[0]
    C_sub = C[np.ix_(active_idx, active_idx)]
    labels_sub = [all_labels[j] for j in active_idx]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(C_sub, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(labels_sub)))
    ax.set_xticklabels(labels_sub, rotation=90, fontsize=5)
    ax.set_yticks(range(len(labels_sub)))
    ax.set_yticklabels(labels_sub, fontsize=5)
    ax.set_title("Feature Co-occurrence Matrix")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Co-occurrence count")
    fig.tight_layout()
    fig.savefig(fig_dir / "cooccurrence_heatmap.png", dpi=150)
    plt.close(fig)

    # ── 4. Eigenvalue spectrum ────────────────────────────────────────
    M_c = M - M.mean(axis=0, keepdims=True)
    cov = (M_c.T @ M_c) / len(records)
    eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
    eigvals = np.clip(eigvals, 0, None)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(eigvals, "o-", markersize=3, color="#4C72B0")
    ax1.set_xlabel("Component index")
    ax1.set_ylabel("Eigenvalue")
    ax1.set_title("Eigenvalue Spectrum")
    ax1.set_yscale("log")

    cumvar = np.cumsum(eigvals) / (eigvals.sum() + 1e-12)
    ax2.plot(cumvar, "-", color="#DD8452")
    ax2.axhline(0.9, ls="--", color="gray", alpha=0.5, label="90% variance")
    ax2.axhline(0.95, ls=":", color="gray", alpha=0.5, label="95% variance")
    k90 = int(np.searchsorted(cumvar, 0.9)) + 1
    ax2.axvline(k90, ls="--", color="#C44E52", alpha=0.5, label=f"k={k90} for 90%")
    ax2.set_xlabel("Number of components")
    ax2.set_ylabel("Cumulative variance explained")
    ax2.set_title("Cumulative Variance Explained")
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "eigenvalue_spectrum.png", dpi=150)
    plt.close(fig)

    # ── 5. Level-level co-occurrence breakdown ────────────────────────
    level_names = ["content_task", "format", "style", "content_constraint", "meta"]
    level_co = np.zeros((len(level_names), len(level_names)))
    label_to_level_idx = {}
    for j, lab in enumerate(all_labels):
        lv = level_map.get(lab, "")
        if lv in level_names:
            label_to_level_idx[j] = level_names.index(lv)

    for rec in records:
        labs = rec["active_labels"]
        for a, b in combinations(labs, 2):
            ai = next((j for j, l in enumerate(all_labels) if l == a), None)
            bi = next((j for j, l in enumerate(all_labels) if l == b), None)
            if ai is not None and bi is not None:
                li = label_to_level_idx.get(ai)
                lj = label_to_level_idx.get(bi)
                if li is not None and lj is not None:
                    level_co[li, lj] += 1
                    level_co[lj, li] += 1

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(level_co, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(level_names)))
    ax.set_xticklabels(level_names, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(level_names)))
    ax.set_yticklabels(level_names, fontsize=9)
    ax.set_title("Level × Level Co-occurrence")
    for i in range(len(level_names)):
        for j in range(len(level_names)):
            ax.text(j, i, f"{int(level_co[i,j])}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(fig_dir / "level_cooccurrence.png", dpi=150)
    plt.close(fig)

    print(f"  Plots saved to {fig_dir}/")


# ═══════════════════════════════════════════════════════════════════════════
# 4. REPORT PRINTER
# ═══════════════════════════════════════════════════════════════════════════

def print_report(report: dict) -> None:
    """Pretty-print the analysis report to stdout."""
    W = 72

    def header(title: str):
        print(f"\n{'═' * W}")
        print(f"  {title}")
        print(f"{'═' * W}")

    def subheader(title: str):
        print(f"\n  ── {title} {'─' * max(0, W - len(title) - 6)}")

    # 1. Density
    header("DENSITY DISTRIBUTION")
    d = report["density"]
    print(f"  Prompts: {d['n_prompts']}")
    print(f"  Mean: {d['mean']:.2f}  Std: {d['std']:.2f}  Median: {d['median']}")
    print(f"  Range: [{d['min']}, {d['max']}]")
    print(f"  Distribution:")
    for k, v in sorted(d["distribution"].items()):
        bar = "█" * max(1, v // (d["n_prompts"] // 40 + 1))
        print(f"    density={k}: {v:>5}  {bar}")

    # 1b. Prompt length
    if "prompt_length" in report:
        header("PROMPT LENGTH")
        pl = report["prompt_length"]
        print(f"  Unit: {pl['unit']}")
        print(f"  Mean: {pl['mean']:.1f}  Std: {pl['std']:.1f}  Median: {pl['median']:.0f}")
        print(f"  Range: [{pl['min']}, {pl['max']}]")

    # 2. Feature frequency
    header("FEATURE FREQUENCY & BALANCE")
    ff = report["feature_frequency"]
    for lv, balance in sorted(ff["level_balance"].items()):
        subheader(f"{lv.upper()}")
        print(f"    Labels: {balance['n_labels']}  "
              f"Total activations: {balance['total_activations']}  "
              f"Mean freq: {balance['mean_freq']:.1f}")
        print(f"    CV: {balance['cv']}  "
              f"Normalised entropy: {balance['normalised_entropy']} (1.0=uniform)")

        # Top/bottom features in this level
        level_feats = ff["by_level"].get(lv, {})
        items = list(level_feats.items())
        if items:
            top3 = items[:3]
            bot3 = items[-3:] if len(items) > 3 else []
            print(f"    Most frequent:  {', '.join(f'{n}({c})' for n, c in top3)}")
            if bot3:
                print(f"    Least frequent: {', '.join(f'{n}({c})' for n, c in bot3)}")

    if ff["never_sampled"]:
        subheader("NEVER SAMPLED")
        print(f"    {', '.join(ff['never_sampled'])}")

    # 3. Coverage
    header("COMBINATORIAL COVERAGE")
    cov = report["coverage"]
    print(f"  Unique label sets: {cov['unique_label_sets']}")
    print(f"  Theoretical upper bound: {cov['theoretical_upper_bound']:,}")
    print(f"  Coverage ratio: {cov['coverage_ratio']:.4f} ({cov['coverage_ratio']*100:.2f}%)")
    print(f"  Modal density: {cov['modal_density']}  "
          f"Tasks: {cov['n_tasks']}  Constraints: {cov['n_constraints']}")

    # 4. Co-occurrence
    header("CO-OCCURRENCE ANALYSIS")
    co = report["cooccurrence"]
    subheader("Top co-occurring pairs (by count)")
    for p in co["top_cooccurring_pairs"][:10]:
        print(f"    {p['pair'][0]:30s} × {p['pair'][1]:30s}  count={p['count']:>4}  PMI={p['pmi']:+.3f}")
    subheader("Top associated pairs (by PMI)")
    for p in co["top_pmi_pairs"][:10]:
        print(f"    {p['pair'][0]:30s} × {p['pair'][1]:30s}  PMI={p['pmi']:+.3f}  count={p['count']:>4}")

    # 5. SAE diagnostics
    header("SAE TRAINING DIAGNOSTICS")
    sae = report["sae_diagnostics"]

    subheader("Sparsity")
    sp = sae["sparsity"]
    print(f"    Mean active features/prompt: {sp['mean_active_per_prompt']} / {sae['n_features']}")
    print(f"    Fraction active: {sp['fraction_active']:.4f}")
    print(f"    → {sp['description']}")

    subheader("Dead features")
    df = sae["dead_features"]
    print(f"    Count: {df['count']}")
    if df["labels"]:
        print(f"    Labels: {', '.join(df['labels'][:10])}")
    print(f"    {df['note']}")

    subheader("Rare features (<1% of prompts)")
    rf = sae["rare_features"]
    print(f"    Count: {rf['count']}")
    if rf["labels"]:
        print(f"    Labels: {', '.join(rf['labels'][:10])}")

    subheader("Effective rank")
    er = sae["effective_rank"]
    print(f"    Effective rank: {er['value']} / {er['total_features']} (ratio={er['ratio']})")
    print(f"    → {er['description']}")

    subheader("Variance explained")
    for k, v in sae["variance_explained"].items():
        print(f"    {k}: {v:.4f} ({v*100:.1f}%)")

    subheader("Feature correlation")
    fc = sae["correlation_distribution"]
    print(f"    Mean |corr|: {fc['mean_abs_corr']:.4f}")
    print(f"    Max  |corr|: {fc['max_abs_corr']:.4f}")
    print(f"    Fraction |corr| > 0.3: {fc['fraction_above_0.3']:.4f}")
    print(f"    → {fc['description']}")

    print(f"\n{'═' * W}")
    print("  ANALYSIS COMPLETE")
    print(f"{'═' * W}\n")


# ═══════════════════════════════════════════════════════════════════════════
# 5. MAIN
# ═══════════════════════════════════════════════════════════════════════════

def analyse(input_path: Path, save_json: bool = True, plot: bool = False) -> dict:
    records = load_prompts(input_path)
    taxonomy = load_taxonomy()

    all_labels = sorted(taxonomy.keys())
    M = build_binary_matrix(records, all_labels)

    report = {
        "density": density_stats(records),
        "prompt_length": length_stats(records),
        "feature_frequency": feature_frequency(records, taxonomy),
        "coverage": coverage_analysis(records, taxonomy),
        "cooccurrence": cooccurrence_analysis(M, all_labels),
        "sae_diagnostics": sae_diagnostics(M, all_labels, taxonomy),
    }

    print_report(report)

    if save_json:
        out_path = input_path.with_suffix(".analysis.json")
        # Convert numpy types for JSON serialisation
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, tuple):
                return list(obj)
            return obj

        import copy
        def deep_convert(d):
            if isinstance(d, dict):
                return {k: deep_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [deep_convert(i) for i in d]
            elif isinstance(d, tuple):
                return [deep_convert(i) for i in d]
            else:
                return convert(d)

        with open(out_path, "w") as f:
            json.dump(deep_convert(report), f, indent=2, ensure_ascii=False)
        print(f"Report saved to {out_path}")

    if plot:
        save_plots(records, M, all_labels, taxonomy, input_path.parent)

    return report


def main():
    parser = argparse.ArgumentParser(description="Analyse constructed prompts for SAE suitability")
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Path to constructed_prompts.jsonl")
    parser.add_argument("--plot", action="store_true", help="Save analysis plots")
    parser.add_argument("--no-json", action="store_true", help="Skip saving JSON report")
    args = parser.parse_args()

    analyse(
        input_path=Path(args.input),
        save_json=not args.no_json,
        plot=args.plot,
    )


if __name__ == "__main__":
    main()

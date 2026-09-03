"""
Confidence intervals and paired significance tests for the detection results.

Why this exists
---------------
Every headline F1 in this repo is a point estimate measured on one sample of
555 test documents. Several claims of the form "X and Y are within noise" were
made without ever testing them. This script measures the noise.

Method
------
*Cluster bootstrap over documents.* Resample WITH REPLACEMENT `--iters` times,
recompute micro-F1 on each resample, and take the 2.5th and 97.5th percentiles
as a 95% interval.

The resampling unit is the **unique document**, and getting this right matters
twice over.

First, spans inside a document are strongly correlated — a judgment full of
unusual foreign names is hard for *all* of its spans at once — so resampling
spans would produce intervals that are far too narrow.

Second, and less obviously: TAB's "555 test documents" are 555
*annotator-annotation pairs over only 127 unique documents*. One document
appears up to ten times, each with a different annotator's gold spans and
identical model predictions. Resampling those 555 rows treats ten views of one
judgment as ten independent draws. Clustering by `doc_id` — sampling documents
and taking all of their annotator rows together — widens the intervals by about
1.7x, and that width is the honest one.

*Paired comparison.* Two models are compared on the SAME resampled documents,
recording the difference each time. Because both models face identical
documents, "this resample happened to be an easy batch" cancels out, which
makes the paired test much more sensitive than checking whether two separate
intervals overlap. (Overlapping intervals do NOT imply "no difference" — that
is a common and costly misreading.)

*Permutation test.* A second, assumption-light check on the same pairing: under
the null hypothesis that the two models are equivalent, swapping their labels
on any document changes nothing. Randomly swap per document, recompute the
difference, and see how often the shuffled gap reaches the observed one.

Micro-F1 is computed from summed counts as 2·TP / (2·TP + FP + FN), matching
the `_ALL` row the rest of the project reports.

Run from the repo root:
    python scripts/bootstrap_ci.py
    python scripts/bootstrap_ci.py --iters 20000 --mode exact --seed 7
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from anonymisation.evaluation import score_spans
from anonymisation.mapping import TAB_TO_SPACY

CACHE = REPO / "results/predictions_cache.json"
OUT_CI = REPO / "results/bootstrap_ci.csv"
OUT_PAIRS = REPO / "results/paired_comparisons.csv"

# Any of these found in the cache get scored. RoBERTa is absent until
# scripts/cache_roberta_predictions.py has been run; the script adapts.
KNOWN_MODELS = ["spacy", "presidio", "legalbert", "roberta"]
LABELS = list(TAB_TO_SPACY.keys())
Span = Tuple[int, int, str, str]


def as_spans(raw: List[list]) -> List[Span]:
    return [(int(s), int(e), t, txt) for s, e, t, txt in raw]


def micro_f1(tp: np.ndarray, fp: np.ndarray, fn: np.ndarray) -> np.ndarray:
    """F1 from summed counts. Vectorised; 0.0 where the denominator vanishes."""
    denom = 2.0 * tp + fp + fn
    return np.divide(2.0 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def per_document_counts(cache: List[dict], model: str, mode: str, label: str = "_ALL"):
    """
    (tp, fp, fn) arrays with ONE ENTRY PER UNIQUE DOCUMENT.

    Rows sharing a doc_id are different annotators' gold for the same text, so
    their counts are summed into a single cluster. Summing first is exactly
    equivalent to concatenating each resample's rows, and it lets the bootstrap
    stay vectorised.
    """
    acc: Dict[str, List[int]] = {}
    order: List[str] = []
    for entry in cache:
        doc_id = entry["doc_id"]
        if doc_id not in acc:
            acc[doc_id] = [0, 0, 0]
            order.append(doc_id)
        r = score_spans(as_spans(entry[model]), as_spans(entry["gold"]), mode)[label]
        acc[doc_id][0] += r.tp
        acc[doc_id][1] += r.fp
        acc[doc_id][2] += r.fn
    a = np.array([acc[d] for d in order], dtype=np.int64)
    return a[:, 0], a[:, 1], a[:, 2]


def bootstrap_f1(counts, idx: np.ndarray) -> np.ndarray:
    """Micro-F1 for every resample. `idx` is (n_iters, n_clusters) of doc indices."""
    tp, fp, fn = counts
    return micro_f1(tp[idx].sum(axis=1), fp[idx].sum(axis=1), fn[idx].sum(axis=1))


def permutation_p(counts_a, counts_b, observed: float, iters: int, rng) -> float:
    """
    Two-sided paired permutation test.

    Under H0 the two models are interchangeable, so on each document we may
    swap which model's counts belong to which arm. Swap independently per
    document, recompute the F1 gap, and ask how often |gap| under the null
    reaches the observed |gap|.
    """
    tp_a, fp_a, fn_a = counts_a
    tp_b, fp_b, fn_b = counts_b
    n = len(tp_a)
    swap = rng.random((iters, n)) < 0.5

    # Where swap is True, the arms exchange that document's counts.
    a_tp = np.where(swap, tp_b, tp_a).sum(axis=1)
    a_fp = np.where(swap, fp_b, fp_a).sum(axis=1)
    a_fn = np.where(swap, fn_b, fn_a).sum(axis=1)
    b_tp = np.where(swap, tp_a, tp_b).sum(axis=1)
    b_fp = np.where(swap, fp_a, fp_b).sum(axis=1)
    b_fn = np.where(swap, fn_a, fn_b).sum(axis=1)

    null = micro_f1(a_tp, a_fp, a_fn) - micro_f1(b_tp, b_fp, b_fn)
    # +1 correction: the observed arrangement is itself one valid permutation.
    return float((np.abs(null) >= abs(observed)).sum() + 1) / (iters + 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=10000, help="bootstrap / permutation resamples (default 10000)")
    ap.add_argument("--mode", default="partial", choices=["partial", "exact", "both"])
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--by-label", action="store_true", help="also compute per-entity-type intervals")
    args = ap.parse_args()

    if not CACHE.exists():
        sys.exit(f"No prediction cache at {CACHE}. Run scripts/cache_predictions.py first.")

    cache = json.loads(CACHE.read_text())
    n_unique = len({e["doc_id"] for e in cache})
    models = [m for m in KNOWN_MODELS if m in cache[0]]
    missing = [m for m in KNOWN_MODELS if m not in cache[0]]
    modes = ["partial", "exact"] if args.mode == "both" else [args.mode]

    print(f"Cache: {len(cache)} annotator rows over {n_unique} unique documents"
          f" · models present: {', '.join(models)}")
    if missing:
        print(f"        not cached yet: {', '.join(missing)}"
              f"  (run scripts/cache_roberta_predictions.py)")
    print(f"Cluster bootstrap over {n_unique} unique documents — "
          f"{args.iters:,} iterations, seed {args.seed}\n")

    rng = np.random.default_rng(args.seed)
    # ONE index matrix, shared by every model and every pair. This is what makes
    # the comparisons paired: each resample is the same set of documents for all.
    idx = rng.integers(0, n_unique, size=(args.iters, n_unique))

    ci_rows: List[dict] = []
    pair_rows: List[dict] = []

    for mode in modes:
        labels = ["_ALL"] + (LABELS if args.by_label else [])
        counts_cache: Dict[Tuple[str, str], tuple] = {}

        for label in labels:
            for m in models:
                counts_cache[(m, label)] = per_document_counts(cache, m, mode, label)

        for label in labels:
            for m in models:
                c = counts_cache[(m, label)]
                point = float(micro_f1(c[0].sum(), c[1].sum(), c[2].sum()))
                dist = bootstrap_f1(c, idx)
                lo, hi = np.percentile(dist, [2.5, 97.5])
                ci_rows.append({
                    "mode": mode, "entity_type": label, "model": m,
                    "f1": point, "ci_low": lo, "ci_high": hi,
                    "ci_width": hi - lo, "boot_se": dist.std(ddof=1),
                    "tp": int(c[0].sum()), "fp": int(c[1].sum()), "fn": int(c[2].sum()),
                })
                if label == "_ALL":
                    print(f"  [{mode:7s}] {m:10s}  F1 = {point:.4f}   95% CI [{lo:.4f}, {hi:.4f}]"
                          f"   ±{(hi - lo) / 2:.4f}")

        # ── paired comparisons on _ALL ────────────────────────────────────
        print()
        for i, a in enumerate(models):
            for b in models[i + 1:]:
                ca, cb = counts_cache[(a, "_ALL")], counts_cache[(b, "_ALL")]
                fa = float(micro_f1(ca[0].sum(), ca[1].sum(), ca[2].sum()))
                fb = float(micro_f1(cb[0].sum(), cb[1].sum(), cb[2].sum()))
                observed = fa - fb
                # Same idx for both arms => the difference is paired.
                diff = bootstrap_f1(ca, idx) - bootstrap_f1(cb, idx)
                lo, hi = np.percentile(diff, [2.5, 97.5])
                p_perm = permutation_p(ca, cb, observed, args.iters, rng)
                distinguishable = not (lo <= 0.0 <= hi)
                pair_rows.append({
                    "mode": mode, "model_a": a, "model_b": b,
                    "f1_a": fa, "f1_b": fb, "diff": observed,
                    "diff_ci_low": lo, "diff_ci_high": hi,
                    "p_permutation": p_perm,
                    "distinguishable_at_95": distinguishable,
                })
                verdict = "DISTINGUISHABLE" if distinguishable else "within noise"
                print(f"  [{mode:7s}] {a:10s} - {b:10s}  Δ = {observed:+.4f}"
                      f"   95% CI [{lo:+.4f}, {hi:+.4f}]   p = {p_perm:.4f}   → {verdict}")
        print()

    OUT_CI.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(ci_rows).to_csv(OUT_CI, index=False)
    pd.DataFrame(pair_rows).to_csv(OUT_PAIRS, index=False)
    print(f"→ {OUT_CI.relative_to(REPO)}")
    print(f"→ {OUT_PAIRS.relative_to(REPO)}")


if __name__ == "__main__":
    main()

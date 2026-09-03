"""
Ablation: does the evaluation harness measure what the product actually does?

The mismatch
------------
Two different de-duplication rules are in play, and only one of them is in the
reported numbers.

  `predictors.make_finetuned_predictor`  drops exact duplicates only —
      spans identical in (start, end, type). This is what every headline F1
      in the project was computed on.

  `pipeline.base.Pipeline._dedupe_overlapping`  drops *overlapping* spans,
      keeping the longest at each overlap. This is what a user's document
      actually goes through.

Long documents are processed as overlapping sliding windows, and the same
entity seen in two windows can come back with slightly different boundaries.
Those near-duplicates are not exact matches, so the evaluation path keeps both
and scores one as a false positive — while the product would have collapsed
them into one.

If that gap is material, the reported F1 *understates* the shipped system, and
the inference-window ablation's stride effect has a mechanism rather than just
a shape.

Free to run: replays `results/predictions_cache.json`, no model needed.

    python scripts/ablate_postprocessing.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np
import pandas as pd

from anonymisation.evaluation import score_spans

CACHE = REPO / "results/predictions_cache.json"
OUT = REPO / "results/ablation_postprocessing.csv"
Span = Tuple[int, int, str, str]


def as_spans(raw: List[list]) -> List[Span]:
    return [(int(s), int(e), t, txt) for s, e, t, txt in raw]


def dedupe_overlapping(spans: List[Span]) -> List[Span]:
    """
    The pipeline's rule, applied to bare tuples: longest span wins at each
    overlap. Mirrors Pipeline._dedupe_overlapping minus the source-priority
    tie-break, which is irrelevant here (one source).
    """
    ordered = sorted(spans, key=lambda s: (s[0], -(s[1] - s[0])))
    kept: List[Span] = []
    for s in ordered:
        if any(s[0] < k[1] and s[1] > k[0] for k in kept):
            continue
        kept.append(s)
    return sorted(kept, key=lambda s: s[0])


def micro_f1(tp, fp, fn):
    d = 2.0 * tp + fp + fn
    return np.divide(2.0 * tp, d, out=np.zeros_like(d, dtype=float), where=d > 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--mode", default="partial", choices=["partial", "exact"])
    args = ap.parse_args()

    cache = json.loads(CACHE.read_text())
    models = [m for m in ("spacy", "presidio", "legalbert", "roberta") if m in cache[0]]

    # Cluster by doc_id — the cache holds several annotators per document.
    order, groups = [], {}
    for i, e in enumerate(cache):
        if e["doc_id"] not in groups:
            groups[e["doc_id"]] = []
            order.append(e["doc_id"])
        groups[e["doc_id"]].append(i)
    clusters = [groups[d] for d in order]
    n_clusters = len(clusters)

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n_clusters, size=(args.iters, n_clusters))

    print(f"Cache: {len(cache)} rows over {n_clusters} unique documents"
          f" · models: {', '.join(models)} · match: {args.mode}\n")
    print(f"{'model':11} {'as evaluated':>13} {'overlap-deduped':>16} {'Δ':>9} {'95% CI':>19} {'real?':>7}")
    print("─" * 82)

    rows = []
    for m in models:
        counts = {"raw": [], "deduped": []}
        for e in cache:
            gold = as_spans(e["gold"])
            pred = as_spans(e[m])
            for name, p in (("raw", pred), ("deduped", dedupe_overlapping(pred))):
                r = score_spans(p, gold, args.mode)["_ALL"]
                counts[name].append([r.tp, r.fp, r.fn])

        arrs = {k: np.array(v) for k, v in counts.items()}
        f1 = {k: float(micro_f1(a[:, 0].sum(), a[:, 1].sum(), a[:, 2].sum())) for k, a in arrs.items()}
        cl = {k: np.array([a[rows].sum(axis=0) for rows in clusters]) for k, a in arrs.items()}
        dists = {k: micro_f1(c[:, 0][idx].sum(1), c[:, 1][idx].sum(1), c[:, 2][idx].sum(1))
                 for k, c in cl.items()}
        diff = dists["deduped"] - dists["raw"]
        lo, hi = np.percentile(diff, [2.5, 97.5])
        real = not (lo <= 0.0 <= hi)
        delta = f1["deduped"] - f1["raw"]

        print(f"{m:11} {f1['raw']:13.4f} {f1['deduped']:16.4f} {delta:+9.4f}"
              f"  [{lo:+.4f},{hi:+.4f}] {'yes' if real else 'no':>7}")
        rows.append({
            "model": m, "mode": args.mode,
            "f1_as_evaluated": f1["raw"], "f1_overlap_deduped": f1["deduped"],
            "delta": delta, "delta_ci_low": lo, "delta_ci_high": hi,
            "distinguishable": real,
            "spans_removed": int(arrs["raw"][:, :2].sum() - arrs["deduped"][:, :2].sum()),
            "fp_removed": int(arrs["raw"][:, 1].sum() - arrs["deduped"][:, 1].sum()),
            "tp_removed": int(arrs["raw"][:, 0].sum() - arrs["deduped"][:, 0].sum()),
        })

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n  spans dropped by overlap-dedup, per model:")
    for r in rows:
        print(f"    {r['model']:11} {r['spans_removed']:6,} total  "
              f"({r['fp_removed']:,} false positives, {r['tp_removed']:,} true positives)")
    print(f"\n→ {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

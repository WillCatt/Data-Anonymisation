"""
Phase 6.2 — replay the prediction cache to score combiner strategies.

Loads `predictions_cache.json` (built by cache_predictions.py) and scores,
using the *same* evaluate_document matcher the rest of the project uses:

  - each model alone (spaCy / LegalBERT / Presidio)
  - union          (min_votes=1)  — reproduces the Phase-6 backfire
  - consensus-2    (min_votes=2)
  - consensus-3    (min_votes=3)
  - routed         (each label handled by its best-F1 model, data-driven)

The routing table is derived from the per-model per-label F1 measured on the
cache itself, so it is not hand-tuned. Saves combiner_comparison.csv.

Run from the repo root:
    PYTHONPATH=src legal-anon-env/bin/python \
        phase6_advanced_training/scripts/eval_combiners.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import pandas as pd

from anonymisation.ensemble import EnsemblePredictor
from anonymisation.evaluation import (
    EvalResult, _empty_results, merge_results, spans_overlap,
)
from anonymisation.mapping import TAB_TO_SPACY

CACHE = REPO / "phase6_advanced_training/results/predictions_cache.json"
OUT = REPO / "phase6_advanced_training/results/combiner_comparison.csv"
OUT_PERLABEL = REPO / "phase6_advanced_training/results/per_model_per_label_f1.csv"
MODELS = ["spacy", "legalbert", "presidio"]
LABELS = list(TAB_TO_SPACY.keys())
Span = Tuple[int, int, str, str]


def as_spans(raw: List[list]) -> List[Span]:
    return [(int(s), int(e), t, txt) for s, e, t, txt in raw]


def score(pred: List[Span], gold: List[Span], mode: str = "partial") -> Dict[str, EvalResult]:
    """Score one document's predicted spans against gold — mirrors evaluate_document."""
    pred = [(s, e, t, txt) for (s, e, t, txt) in pred if t in TAB_TO_SPACY]
    results = _empty_results()
    gold_matched: set = set()
    for ps, pe, pt, _ in pred:
        matched = False
        for gi, (gs, ge, gt, _) in enumerate(gold):
            if gi in gold_matched:
                continue
            if pt == gt and spans_overlap(ps, pe, gs, ge, mode):
                results[gt].tp += 1
                results["_ALL"].tp += 1
                gold_matched.add(gi)
                matched = True
                break
        if not matched:
            results[pt].fp += 1
            results["_ALL"].fp += 1
    for gi, (_, _, gt, _) in enumerate(gold):
        if gi not in gold_matched:
            results[gt].fn += 1
            results["_ALL"].fn += 1
    return results


def eval_strategy(cache: List[dict], combine, mode: str = "partial") -> Dict[str, EvalResult]:
    """`combine(entry) -> List[Span]` produces the combined prediction for a doc."""
    per_doc = [score(combine(e), as_spans(e["gold"]), mode) for e in cache]
    return merge_results(per_doc)


def per_label_f1(cache: List[dict], model: str) -> Dict[str, float]:
    merged = eval_strategy(cache, lambda e: as_spans(e[model]))
    return {lab: merged[lab].f1 for lab in LABELS}


def make_ensemble_combine(min_votes: int):
    def combine(entry):
        per_model = {m: (lambda txt, sp=as_spans(entry[m]): sp) for m in MODELS}
        ens = EnsemblePredictor(per_model, min_votes=min_votes, tie_break="longest")
        return ens.predict("x")  # text arg unused; closures ignore it
    return combine


def make_routed_combine(route: Dict[str, str]):
    """route: label -> owning model. Keep each model's spans only for labels it owns."""
    def combine(entry):
        out: List[Span] = []
        for m in MODELS:
            for s in as_spans(entry[m]):
                if route.get(s[2]) == m:
                    out.append(s)
        return sorted(out, key=lambda s: s[0])
    return combine


def fmt(r: EvalResult) -> str:
    return f"P={r.precision:6.1%}  R={r.recall:6.1%}  F1={r.f1:6.1%}  (tp={r.tp} fp={r.fp} fn={r.fn})"


def main():
    cache = json.loads(CACHE.read_text())
    print(f"Loaded cache: {len(cache)} docs\n")

    # --- per-model per-label F1 → routing table -------------------------
    f1_by_model = {m: per_label_f1(cache, m) for m in MODELS}
    f1_df = pd.DataFrame(f1_by_model)
    print("Per-label partial-F1 by model:")
    print(f1_df.map(lambda v: f"{v:.1%}").to_string(), "\n")
    f1_df.to_csv(OUT_PERLABEL)
    print(f"Saved → {OUT_PERLABEL}\n")

    route = {lab: max(MODELS, key=lambda m: f1_by_model[m][lab]) for lab in LABELS}
    print("Data-driven routing (label → best model):")
    for lab in LABELS:
        print(f"  {lab:10s} → {route[lab]:10s} (F1 {f1_by_model[route[lab]][lab]:.1%})")
    print()

    # --- strategies -----------------------------------------------------
    strategies = {
        "spacy_alone":      lambda e: as_spans(e["spacy"]),
        "legalbert_alone":  lambda e: as_spans(e["legalbert"]),
        "presidio_alone":   lambda e: as_spans(e["presidio"]),
        "union(min1)":      make_ensemble_combine(1),
        "consensus(min2)":  make_ensemble_combine(2),
        "consensus(min3)":  make_ensemble_combine(3),
        "routed":           make_routed_combine(route),
    }

    rows = []
    print("=" * 72)
    for name, combine in strategies.items():
        for mode in ("partial", "exact"):
            merged = eval_strategy(cache, combine, mode)
            allr = merged["_ALL"]
            rows.append({
                "strategy": name, "mode": mode,
                "precision": allr.precision, "recall": allr.recall, "f1": allr.f1,
                "tp": allr.tp, "fp": allr.fp, "fn": allr.fn,
            })
            if mode == "partial":
                print(f"{name:18s} {fmt(allr)}")
    print("=" * 72)

    df = pd.DataFrame(rows)
    OUT.write_text(df.to_csv(index=False))
    print(f"\nSaved → {OUT}")

    best = df[(df["mode"] == "partial")].sort_values("f1", ascending=False).iloc[0]
    lb = df[(df.strategy == "legalbert_alone") & (df["mode"] == "partial")].iloc[0]
    print(f"\nBest partial-F1: {best.strategy} = {best.f1:.1%}")
    print(f"LegalBERT alone: {lb.f1:.1%}  →  best beats single model by {best.f1 - lb.f1:+.1%}")


if __name__ == "__main__":
    main()

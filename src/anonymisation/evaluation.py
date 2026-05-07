"""
Span-level evaluation framework.

A predicted span counts as a True Positive when:
  1. its predicted TAB type matches the gold TAB type, AND
  2. its character offsets overlap (or exactly match, depending on `mode`)
     a previously-unmatched gold span.

Predicted spans that do not match any gold span are False Positives.
Gold spans that no prediction matched are False Negatives.

Only gold mentions tagged DIRECT or QUASI count — these are the entities
TAB requires us to mask. NO_MASK mentions are excluded entirely.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

from .mapping import SPACY_TO_TAB, TAB_TO_SPACY


@dataclass
class EvalResult:
    """Counts and derived metrics for one entity type."""

    entity_type: str
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def spans_overlap(ps: int, pe: int, gs: int, ge: int, mode: str = "partial") -> bool:
    """Whether a predicted span (ps, pe) hits a gold span (gs, ge)."""
    if mode == "exact":
        return ps == gs and pe == ge
    return ps < ge and pe > gs


def extract_gold_entities(doc: dict) -> List[Tuple[int, int, str, str]]:
    """Pull (start, end, tab_type, text) for every DIRECT/QUASI mention in a TAB doc."""
    return [
        (em["start_offset"], em["end_offset"], em["entity_type"], em["span_text"])
        for em in doc["entity_mentions"]
        if em["identifier_type"] in ("DIRECT", "QUASI")
    ]


def predict_entities(nlp, text: str) -> List[Tuple[int, int, str, str]]:
    """Run a spaCy pipeline and project each entity onto its TAB type."""
    doc = nlp(text)
    return [
        (ent.start_char, ent.end_char, SPACY_TO_TAB[ent.label_], ent.text)
        for ent in doc.ents
        if ent.label_ in SPACY_TO_TAB
    ]


def _empty_results() -> Dict[str, EvalResult]:
    types = list(TAB_TO_SPACY.keys()) + ["_ALL"]
    return {t: EvalResult(entity_type=t) for t in types}


def _is_spacy_pipeline(obj) -> bool:
    """Heuristic: spaCy Language objects expose a `pipe_names` attribute."""
    return hasattr(obj, "pipe_names") and hasattr(obj, "__call__")


def evaluate_document(predictor, doc: dict, mode: str = "partial") -> Dict[str, EvalResult]:
    """
    Score one TAB document.

    `predictor` may be either:
      * a spaCy Language object — for backward compatibility with Phase 1; we
        run `predict_entities(nlp, text)` to get spans, OR
      * any callable mapping `text -> [(start, end, tab_type, span_text), ...]` —
        which is how Phase 2's HuggingFace, Presidio and fine-tuned models plug in.

    Returns {entity_type: EvalResult}.
    """
    gold = extract_gold_entities(doc)
    if _is_spacy_pipeline(predictor):
        pred = predict_entities(predictor, doc["text"])
    else:
        pred = predictor(doc["text"])

    # Defensive: predictors must produce only TAB types we know how to score.
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


def merge_results(per_doc: List[Dict[str, EvalResult]]) -> Dict[str, EvalResult]:
    """Sum per-document EvalResults across the corpus."""
    merged = _empty_results()
    for doc_results in per_doc:
        for t, r in doc_results.items():
            merged[t].tp += r.tp
            merged[t].fp += r.fp
            merged[t].fn += r.fn
    return merged


def results_to_dataframe(merged_by_mode: Dict[str, Dict[str, EvalResult]]) -> pd.DataFrame:
    """Flatten {mode: {entity_type: EvalResult}} into a tidy DataFrame."""
    rows = []
    for mode, merged in merged_by_mode.items():
        for et in list(TAB_TO_SPACY.keys()) + ["_ALL"]:
            r = merged[et]
            rows.append({
                "mode": mode,
                "entity_type": et,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "tp": r.tp,
                "fp": r.fp,
                "fn": r.fn,
            })
    return pd.DataFrame(rows)

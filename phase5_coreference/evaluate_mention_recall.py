"""
Phase 5 evaluation — mention-recall on TAB.

Span-level F1 (Phase 1/2) treats every entity mention as an independent
prediction. That hides a specific failure mode: a model can score
respectably on "first mentions" while quietly leaking every shorthand
reference further down the document. For a redaction pipeline that's
a real liability — the second mention of a name is no less identifying
than the first.

This script measures *mention recall per entity*:

    For every gold entity (group of TAB mentions sharing an `entity_id`),
    what fraction of its mentions did the pipeline catch?

We then average across entities (macro) and across mentions (micro) and
compare the baseline (NER + regex) against the same pipeline with the
Phase-5 CorefExtender enabled.

Run with:

    python phase5_coreference/evaluate_mention_recall.py \\
        --sample 100 \\
        --out phase5_coreference/results/mention_recall.csv

Without --sample, it runs on the full TAB test split (555 documents,
about 8 minutes on `en_core_web_trf`).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

# Make `from anonymisation.pipeline import …` work regardless of cwd.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anonymisation.data import load_tab                                       # noqa: E402
from anonymisation.mapping import SPACY_TO_TAB                                # noqa: E402
from anonymisation.pipeline import LitePipeline                               # noqa: E402


# ---------------------------------------------------------------------------
# Predictor adapter — same shape as the Phase 3 walkthrough notebook
# ---------------------------------------------------------------------------
def make_spacy_predictor(model_name: str = "en_core_web_trf"):
    import spacy
    nlp = spacy.load(model_name)

    def predict(text: str):
        doc = nlp(text)
        return [
            (ent.start_char, ent.end_char, SPACY_TO_TAB[ent.label_], ent.text)
            for ent in doc.ents
            if ent.label_ in SPACY_TO_TAB
        ]
    return predict


# ---------------------------------------------------------------------------
# Per-document mention recall calculation
# ---------------------------------------------------------------------------
def _spans_overlap(ps: int, pe: int, gs: int, ge: int) -> bool:
    return ps < ge and pe > gs


def per_entity_recall(doc, predicted_spans):
    """
    For one TAB doc, return [(entity_id, entity_type, mentions_found, mentions_total), ...].

    Only DIRECT + QUASI mentions count — same scoping as Phase 1/2.
    """
    by_entity = defaultdict(list)
    for em in doc["entity_mentions"]:
        if em["identifier_type"] not in ("DIRECT", "QUASI"):
            continue
        by_entity[em["entity_id"]].append(em)

    rows = []
    for entity_id, mentions in by_entity.items():
        etype = mentions[0]["entity_type"]
        found = 0
        for em in mentions:
            gs, ge = em["start_offset"], em["end_offset"]
            if any(
                ps_pe_pt[2] == etype and _spans_overlap(ps_pe_pt[0], ps_pe_pt[1], gs, ge)
                for ps_pe_pt in predicted_spans
            ):
                found += 1
        rows.append((entity_id, etype, found, len(mentions)))
    return rows


def evaluate(predict_fn, dataset_test, label: str):
    """
    Walk every TAB test doc, score per-entity recall, aggregate.

    Returns a DataFrame with one row per gold entity and a summary dict.
    """
    rows = []
    for i, doc in enumerate(dataset_test, 1):
        if i % 50 == 0:
            print(f"  [{label}] {i}/{len(dataset_test)}")
        predicted = predict_fn(doc["text"])
        # Predicted is whatever the pipeline returns — its job is to give us
        # spans suitable for matching gold mentions.
        for entity_id, etype, found, total in per_entity_recall(doc, predicted):
            rows.append({
                "label": label,
                "doc_id": doc["doc_id"],
                "entity_id": entity_id,
                "entity_type": etype,
                "mentions_found": found,
                "mentions_total": total,
                "entity_recall": found / total if total else 0.0,
            })
    df = pd.DataFrame(rows)
    summary = {
        "label": label,
        "n_entities": len(df),
        "n_mentions": int(df["mentions_total"].sum()),
        "macro_recall": float(df["entity_recall"].mean()),
        "micro_recall": float(df["mentions_found"].sum() / max(1, df["mentions_total"].sum())),
    }
    return df, summary


# ---------------------------------------------------------------------------
# Pipeline-driven predictors that we'll compare
# ---------------------------------------------------------------------------
def make_pipeline_predictor(ner_predictor, *, coref_extend: bool):
    """
    Build a *prediction function* that runs the same span-detection pipeline
    as the redaction code paths, with or without coref. We use LitePipeline
    purely as a thin wrapper around `detect_spans` — we never call its
    `_redact()`, so the redacted text and audit log don't get built.
    """
    pipeline = LitePipeline(
        ner_provider=ner_predictor,
        coref_extend=coref_extend,
        run_regex=True,
    )

    def predict(text: str):
        spans = pipeline.detect_spans(text)
        return [(s.start, s.end, s.entity_type, s.text) for s in spans]
    return predict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sample", type=int, default=0,
                        help="If > 0, evaluate on the first N test docs only (debug runs).")
    parser.add_argument("--spacy-model", default="en_core_web_trf",
                        help="spaCy model for the NER backbone.")
    parser.add_argument("--out", default="phase5_coreference/results/mention_recall.csv",
                        help="Path for the per-entity CSV.")
    parser.add_argument("--summary", default="phase5_coreference/results/mention_recall_summary.json",
                        help="Path for the aggregate summary JSON.")
    args = parser.parse_args()

    print(f"Loading spaCy: {args.spacy_model}")
    ner = make_spacy_predictor(args.spacy_model)

    print("Loading TAB...")
    ds = load_tab()
    test = list(ds["test"])
    if args.sample > 0:
        test = test[: args.sample]
    print(f"Evaluating on {len(test)} TAB test documents.")

    predict_baseline = make_pipeline_predictor(ner, coref_extend=False)
    predict_coref    = make_pipeline_predictor(ner, coref_extend=True)

    print("\n— Baseline (NER + regex, no coref) —")
    df_b, sum_b = evaluate(predict_baseline, test, "baseline")
    print(f"  macro recall = {sum_b['macro_recall']:.3f}  micro recall = {sum_b['micro_recall']:.3f}"
          f"  ({sum_b['n_entities']:,} entities, {sum_b['n_mentions']:,} mentions)")

    print("\n— NER + regex + CorefExtender —")
    df_c, sum_c = evaluate(predict_coref, test, "with_coref")
    print(f"  macro recall = {sum_c['macro_recall']:.3f}  micro recall = {sum_c['micro_recall']:.3f}")

    # Per-entity-type breakdown
    print("\nPer-entity-type micro recall:")
    print(f"  {'type':<10s} {'baseline':>10s} {'+coref':>10s} {'lift':>10s}")
    print(" " + "-" * 44)
    types = sorted(set(df_b["entity_type"]).union(df_c["entity_type"]))
    type_rows = []
    for et in types:
        b_sub = df_b[df_b["entity_type"] == et]
        c_sub = df_c[df_c["entity_type"] == et]
        b_micro = b_sub["mentions_found"].sum() / max(1, b_sub["mentions_total"].sum())
        c_micro = c_sub["mentions_found"].sum() / max(1, c_sub["mentions_total"].sum())
        print(f"  {et:<10s} {b_micro:>10.3f} {c_micro:>10.3f} {c_micro - b_micro:>+10.3f}")
        type_rows.append({"entity_type": et, "baseline": b_micro, "with_coref": c_micro})

    out = pd.concat([df_b, df_c], ignore_index=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nSaved per-entity rows → {args.out}")

    Path(args.summary).write_text(json.dumps({
        "baseline": sum_b,
        "with_coref": sum_c,
        "per_type": type_rows,
    }, indent=2))
    print(f"Saved summary       → {args.summary}")


if __name__ == "__main__":
    main()

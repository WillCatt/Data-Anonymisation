"""Span-level evaluation: TP/FP/FN bookkeeping, partial vs exact, NO_MASK."""
from anonymisation.evaluation import evaluate_document


def _doc(text, mentions):
    return {
        "text": text,
        "entity_mentions": [
            {
                "start_offset": s,
                "end_offset": e,
                "entity_type": t,
                "span_text": text[s:e],
                "identifier_type": role,
            }
            for (s, e, t, role) in mentions
        ],
    }


def _predictor(spans):
    """A callable predictor returning a fixed span list, TAB-shaped."""
    return lambda text: [(s, e, t, text[s:e]) for (s, e, t) in spans]


DOC = _doc(
    "Maria lives in Sofia.",
    [(0, 5, "PERSON", "DIRECT"), (15, 20, "LOC", "QUASI")],
)


def test_perfect_prediction_all_true_positives():
    res = evaluate_document(_predictor([(0, 5, "PERSON"), (15, 20, "LOC")]), DOC)
    assert (res["_ALL"].tp, res["_ALL"].fp, res["_ALL"].fn) == (2, 0, 0)
    assert res["PERSON"].tp == 1 and res["LOC"].tp == 1


def test_type_mismatch_is_fp_plus_fn():
    # Right offsets, wrong type: the gold PERSON goes unmatched (FN) and the
    # wrong-typed prediction is a FP.
    res = evaluate_document(_predictor([(0, 5, "LOC")]), DOC)
    assert res["LOC"].fp == 1
    assert res["PERSON"].fn == 1
    assert res["PERSON"].tp == 0


def test_partial_overlap_matches_but_exact_does_not():
    pred = _predictor([(0, 4, "PERSON")])  # "Mari" — overlaps gold (0,5)
    partial = evaluate_document(pred, DOC, mode="partial")
    exact = evaluate_document(pred, DOC, mode="exact")
    assert partial["PERSON"].tp == 1
    assert exact["PERSON"].tp == 0 and exact["PERSON"].fp == 1 and exact["PERSON"].fn == 1


def test_no_mask_gold_is_ignored_so_overlap_is_a_false_positive():
    doc = _doc("The weather was fine.", [(4, 11, "MISC", "NO_MASK")])
    res = evaluate_document(_predictor([(4, 11, "MISC")]), doc)
    # NO_MASK is excluded from gold, so the prediction matches nothing -> FP.
    assert res["MISC"].fp == 1
    assert res["MISC"].tp == 0

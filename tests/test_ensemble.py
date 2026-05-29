"""EnsemblePredictor: overlap grouping, type voting, min_votes gating."""
import pytest

from anonymisation.ensemble import EnsemblePredictor


def _const(spans):
    return lambda text: list(spans)


def test_union_vote_keeps_agreed_span():
    ens = EnsemblePredictor(
        {"a": _const([(0, 5, "PERSON", "Maria")]),
         "b": _const([(0, 5, "PERSON", "Maria")])},
        min_votes=2,
    )
    assert ens.predict("Maria here") == [(0, 5, "PERSON", "Maria")]


def test_min_votes_drops_under_supported_span():
    ens = EnsemblePredictor(
        {"a": _const([(0, 5, "PERSON", "Maria")]),
         "b": _const([])},
        min_votes=2,
    )
    assert ens.predict("Maria here") == []


def test_type_vote_majority_wins_and_longest_kept():
    # Two predictors say PERSON, one says LOC, over overlapping offsets.
    ens = EnsemblePredictor(
        {"a": _const([(0, 5, "PERSON", "Maria")]),
         "b": _const([(0, 5, "PERSON", "Maria")]),
         "c": _const([(0, 6, "LOC", "Marias")])},
        min_votes=1,
    )
    result = ens.predict("Marias")
    assert result == [(0, 5, "PERSON", "Maria")]


def test_non_overlapping_spans_stay_separate():
    ens = EnsemblePredictor(
        {"a": _const([(0, 5, "PERSON", "Maria"), (15, 20, "LOC", "Sofia")])},
        min_votes=1,
    )
    result = ens.predict("Maria lives in Sofia")
    assert result == [(0, 5, "PERSON", "Maria"), (15, 20, "LOC", "Sofia")]


def test_min_votes_above_predictor_count_is_rejected():
    with pytest.raises(ValueError):
        EnsemblePredictor({"a": _const([])}, min_votes=2)

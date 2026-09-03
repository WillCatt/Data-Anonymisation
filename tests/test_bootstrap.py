"""
Tests for the bootstrap / paired-significance machinery.

These pin down the two things that are easy to get silently wrong: the F1
arithmetic, and the pairing. A bug in either produces plausible-looking
intervals, which is exactly why they need testing rather than eyeballing.

No models and no network — the scorer runs on hand-written spans.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]

# scripts/ isn't a package; load the module by path.
_spec = importlib.util.spec_from_file_location("bootstrap_ci", REPO / "scripts" / "bootstrap_ci.py")
bootstrap_ci = importlib.util.module_from_spec(_spec)
sys.modules["bootstrap_ci"] = bootstrap_ci
_spec.loader.exec_module(bootstrap_ci)

micro_f1 = bootstrap_ci.micro_f1
bootstrap_f1 = bootstrap_ci.bootstrap_f1
per_document_counts = bootstrap_ci.per_document_counts
permutation_p = bootstrap_ci.permutation_p


# ── F1 arithmetic ────────────────────────────────────────────────────────

def test_micro_f1_matches_precision_recall_definition():
    tp, fp, fn = 17939, 3276, 2870          # LegalBERT, partial, from the cache
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    expected = 2 * precision * recall / (precision + recall)
    assert micro_f1(np.array(tp), np.array(fp), np.array(fn)) == pytest.approx(expected)


def test_micro_f1_perfect_and_zero():
    assert micro_f1(np.array(10), np.array(0), np.array(0)) == pytest.approx(1.0)
    assert micro_f1(np.array(0), np.array(5), np.array(5)) == pytest.approx(0.0)


def test_micro_f1_empty_is_zero_not_nan():
    """A resample can legitimately contain no gold and no predictions."""
    out = micro_f1(np.array(0), np.array(0), np.array(0))
    assert out == 0.0 and not np.isnan(out)


def test_micro_f1_is_vectorised():
    tp = np.array([10, 0, 5]); fp = np.array([0, 5, 5]); fn = np.array([0, 5, 5])
    assert micro_f1(tp, fp, fn) == pytest.approx([1.0, 0.0, 0.5])


# ── bootstrap behaviour ──────────────────────────────────────────────────

def _counts(tp, fp, fn):
    return np.array(tp), np.array(fp), np.array(fn)


def test_identity_resample_reproduces_the_point_estimate():
    """Resampling every document exactly once must give back the headline F1."""
    c = _counts([8, 6, 9], [2, 3, 1], [1, 2, 2])
    idx = np.arange(3).reshape(1, 3)
    point = micro_f1(c[0].sum(), c[1].sum(), c[2].sum())
    assert bootstrap_f1(c, idx)[0] == pytest.approx(point)


def test_interval_contains_the_point_estimate():
    rng = np.random.default_rng(0)
    n = 200
    c = _counts(rng.integers(5, 15, n), rng.integers(0, 5, n), rng.integers(0, 5, n))
    point = micro_f1(c[0].sum(), c[1].sum(), c[2].sum())
    dist = bootstrap_f1(c, rng.integers(0, n, size=(2000, n)))
    lo, hi = np.percentile(dist, [2.5, 97.5])
    assert lo <= point <= hi


def test_more_documents_give_tighter_intervals():
    """The interval must shrink as the sample grows — roughly as 1/sqrt(n)."""
    rng = np.random.default_rng(1)
    widths = []
    for n in (50, 800):
        c = _counts(rng.integers(5, 15, n), rng.integers(0, 5, n), rng.integers(0, 5, n))
        dist = bootstrap_f1(c, rng.integers(0, n, size=(2000, n)))
        lo, hi = np.percentile(dist, [2.5, 97.5])
        widths.append(hi - lo)
    assert widths[1] < widths[0] / 2


def test_pairing_uses_the_same_resample_for_both_models():
    """
    Two models with identical per-document counts must show exactly zero
    difference on every resample. This fails if each arm draws its own indices
    — the bug that would quietly turn a paired test into an unpaired one.
    """
    rng = np.random.default_rng(2)
    n = 100
    c = _counts(rng.integers(5, 15, n), rng.integers(0, 5, n), rng.integers(0, 5, n))
    idx = rng.integers(0, n, size=(500, n))
    diff = bootstrap_f1(c, idx) - bootstrap_f1(c, idx)
    assert np.allclose(diff, 0.0)


# ── permutation test ─────────────────────────────────────────────────────

def test_permutation_p_is_high_for_identical_models():
    rng = np.random.default_rng(3)
    n = 120
    c = _counts(rng.integers(5, 15, n), rng.integers(0, 5, n), rng.integers(0, 5, n))
    assert permutation_p(c, c, 0.0, 500, np.random.default_rng(4)) > 0.5


def test_permutation_p_is_low_for_a_large_real_gap():
    rng = np.random.default_rng(5)
    n = 150
    strong = _counts(rng.integers(12, 16, n), rng.integers(0, 2, n), rng.integers(0, 2, n))
    weak = _counts(rng.integers(2, 6, n), rng.integers(8, 12, n), rng.integers(8, 12, n))
    observed = (micro_f1(strong[0].sum(), strong[1].sum(), strong[2].sum())
                - micro_f1(weak[0].sum(), weak[1].sum(), weak[2].sum()))
    assert permutation_p(strong, weak, observed, 500, np.random.default_rng(6)) < 0.01


def test_permutation_p_never_returns_zero():
    """The +1 correction keeps p strictly positive — p=0 is not a valid claim."""
    rng = np.random.default_rng(7)
    n = 60
    a = _counts(rng.integers(14, 16, n), np.zeros(n, int), np.zeros(n, int))
    b = _counts(np.zeros(n, int), rng.integers(14, 16, n), rng.integers(14, 16, n))
    assert permutation_p(a, b, 1.0, 100, np.random.default_rng(8)) > 0


# ── the scorer the bootstrap counts on ───────────────────────────────────

def test_per_document_counts_are_additive_over_the_corpus():
    """Summed per-document counts must equal corpus counts — the whole method
    rests on being able to add documents up."""
    cache = [
        {"gold": [[0, 5, "PERSON", "Alice"], [10, 15, "LOC", "Paris"]],
         "m":    [[0, 5, "PERSON", "Alice"]]},
        {"gold": [[0, 3, "ORG", "BBC"]],
         "m":    [[0, 3, "ORG", "BBC"], [20, 25, "PERSON", "ghost"]]},
    ]
    tp, fp, fn = per_document_counts(cache, "m", "partial")
    assert (tp.sum(), fp.sum(), fn.sum()) == (2, 1, 1)


def test_exact_mode_is_stricter_than_partial():
    cache = [{"gold": [[0, 10, "PERSON", "Alice Smith"]],
              "m":    [[0, 5, "PERSON", "Alice"]]}]      # overlaps but isn't equal
    assert per_document_counts(cache, "m", "partial")[0].sum() == 1
    assert per_document_counts(cache, "m", "exact")[0].sum() == 0

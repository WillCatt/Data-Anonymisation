"""MosaicScorer: k-anonymity counts over a haystack of fingerprints."""
from anonymisation.pipeline.scorer import MosaicScorer


def _sig(*pairs):
    """Build a fingerprint tuple, the way the pipeline does."""
    return tuple(sorted(pairs))


def test_empty_scorer_reports_unique():
    scorer = MosaicScorer.empty()
    # Any non-empty fingerprint is treated as unique (k=1) against an empty
    # haystack — the document itself is the only one with this fingerprint.
    assert scorer.k_for(_sig(("LOC", "sofia"))) == 1


def test_counts_shared_fingerprints():
    sofia = _sig(("LOC", "sofia"))
    plovdiv = _sig(("LOC", "plovdiv"))
    scorer = MosaicScorer([sofia, sofia, plovdiv])
    assert scorer.k_for(sofia) == 2
    assert scorer.k_for(plovdiv) == 1          # max(count=1, floor=1)
    assert scorer.k_for(_sig(("LOC", "rome"))) == 1  # unseen -> floor of 1


def test_empty_signature_leaks_nothing():
    scorer = MosaicScorer([_sig(("LOC", "sofia")), _sig(("LOC", "plovdiv"))])
    # A fully-suppressed document has no fingerprint; it can't be singled out,
    # so k is the whole haystack.
    assert scorer.k_for(()) == scorer.haystack_size == 2


def test_is_safe_threshold():
    sofia = _sig(("LOC", "sofia"))
    scorer = MosaicScorer([sofia] * 5)
    assert scorer.is_safe(sofia, k_target=5)
    assert not scorer.is_safe(sofia, k_target=6)

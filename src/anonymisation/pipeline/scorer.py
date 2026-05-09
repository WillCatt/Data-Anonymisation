"""
Mosaic risk scorer.

Wraps a "haystack" of QUASI fingerprints (in our demo, derived from TAB)
and answers two questions for the iterate-until-safe loop in Pipeline Pro:

  * `k_for(signature)` — how many haystack documents share this signature?
  * `is_safe(signature, k_target)` — does k meet the threshold?

In a real production deployment the haystack would be the firm's own
matter corpus, not TAB. But TAB is a reasonable methodological stand-in:
if a document's QUASI fingerprint is unique within TAB, it's at least
plausibly unique in the firm's corpus too — and that's the question the
buyer cares about.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Tuple

from ..mosaic import quasi_identifier_signature


class MosaicScorer:
    """
    Holds a precomputed Counter of haystack fingerprints.

    Construction is O(corpus); subsequent `k_for()` calls are O(1).
    """

    def __init__(self, haystack_signatures: Iterable[Tuple]):
        self._counts: Counter = Counter(haystack_signatures)
        self._haystack_size: int = sum(self._counts.values())

    @classmethod
    def from_tab(cls, tab_split: Iterable[dict]) -> "MosaicScorer":
        """Build a scorer from raw TAB documents."""
        signatures = [quasi_identifier_signature(d) for d in tab_split]
        signatures = [s for s in signatures if len(s) >= 2]  # skip empty/near-empty
        return cls(signatures)

    @classmethod
    def empty(cls) -> "MosaicScorer":
        """A no-op scorer (every fingerprint reports k=0). Useful for tests."""
        return cls([])

    def k_for(self, signature: Tuple) -> int:
        """
        Return how many haystack docs share this fingerprint, treating an
        unseen fingerprint as k=1 (the input *itself* is the only known
        document with this fingerprint, so it's effectively unique).
        """
        if not signature:
            return self._haystack_size or 1  # empty fingerprint = no information leaked
        return max(self._counts.get(signature, 0), 1)

    def is_safe(self, signature: Tuple, k_target: int = 5) -> bool:
        return self.k_for(signature) >= k_target

    @property
    def haystack_size(self) -> int:
        return self._haystack_size

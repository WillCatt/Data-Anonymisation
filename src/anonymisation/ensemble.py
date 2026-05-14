"""
Ensemble predictor — combine multiple NER predictors with voting.

Given a dict of {name: predictor}, where each predictor maps
    text -> [(start, end, tab_type, span_text), ...]
the ensemble:

  1. Runs every predictor on the input text.
  2. Groups the resulting spans by character-offset overlap (transitive).
  3. For each overlap group, votes on the entity type — the type backed by
     the most predictors wins.
  4. Among spans of the winning type, the longest one is kept.
  5. Groups that don't reach `min_votes` distinct predictors are dropped.

The output has the same shape as a single predictor — so the ensemble is a
drop-in replacement in `evaluate_document`, `LitePipeline`, `ProPipeline`,
etc.

Voting policy
-------------
- `min_votes=1` (default): keep any span supported by ≥1 predictor. Union
  of predictions. Maximises recall.
- `min_votes=2`: keep only spans supported by ≥2 predictors. Intersection-
  like. Maximises precision.
- `min_votes=K` where K = total predictors: strict consensus.

For redaction, recall is usually more valuable than precision (a missed
entity is a leak; a false positive is one extra `[TYPE]` tag). Default
ships at min_votes=1; the audit log makes false positives easy to spot.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Dict, List, Tuple

Span = Tuple[int, int, str, str]   # (start, end, tab_type, span_text)
Predictor = Callable[[str], List[Span]]


class EnsemblePredictor:
    """Vote-based ensemble over independent NER predictors."""

    def __init__(
        self,
        predictors: Dict[str, Predictor],
        *,
        min_votes: int = 1,
        tie_break: str = "longest",
    ) -> None:
        if not predictors:
            raise ValueError("predictors must contain at least one entry")
        if min_votes < 1:
            raise ValueError("min_votes must be >= 1")
        if min_votes > len(predictors):
            raise ValueError(
                f"min_votes={min_votes} > number of predictors={len(predictors)}"
            )
        if tie_break not in ("longest", "shortest"):
            raise ValueError("tie_break must be 'longest' or 'shortest'")

        self.predictors = predictors
        self.min_votes = min_votes
        self.tie_break = tie_break

    # ------------------------------------------------------------------ #
    def __call__(self, text: str) -> List[Span]:
        return self.predict(text)

    def predict(self, text: str) -> List[Span]:
        if not text:
            return []

        # Step 1 — collect every prediction, tagged with which predictor made it
        all_preds: List[Tuple[str, Span]] = []
        for name, predict in self.predictors.items():
            try:
                spans = predict(text)
            except Exception as exc:  # pragma: no cover — defensive
                # One broken predictor shouldn't kill the ensemble
                spans = []
                print(f"[ensemble] predictor {name!r} raised {exc}; skipping")
            for span in spans:
                all_preds.append((name, span))

        if not all_preds:
            return []

        # Step 2 — group by transitive overlap
        groups = self._group_overlapping(all_preds)

        # Step 3-5 — vote, pick winning span per group, gate on min_votes
        final: List[Span] = []
        for group in groups:
            voters = {name for name, _ in group}
            if len(voters) < self.min_votes:
                continue
            type_votes = Counter(span[2] for _, span in group)
            winning_type, _ = type_votes.most_common(1)[0]
            spans_of_winning = [span for _, span in group if span[2] == winning_type]
            picked = self._pick(spans_of_winning)
            final.append(picked)
        return sorted(final, key=lambda s: s[0])

    # ------------------------------------------------------------------ #
    @staticmethod
    def _overlaps(a: Span, b: Span) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    @classmethod
    def _group_overlapping(cls, preds: List[Tuple[str, Span]]) -> List[List[Tuple[str, Span]]]:
        """
        Union-find-ish grouping: two predictions are in the same group iff
        their spans overlap, transitively.
        """
        # Sort by start so each prediction only has to compare with the
        # most-recently-touched group's max end.
        ordered = sorted(preds, key=lambda p: (p[1][0], p[1][1]))
        groups: List[List[Tuple[str, Span]]] = []
        for entry in ordered:
            start, end = entry[1][0], entry[1][1]
            # Find the FIRST group whose span range touches (start, end);
            # because spans are sorted by start, we can also merge subsequent
            # groups if this entry bridges them.
            attached = False
            for g in groups:
                gmin = min(p[1][0] for p in g)
                gmax = max(p[1][1] for p in g)
                if start < gmax and end > gmin:
                    g.append(entry)
                    attached = True
                    break
            if not attached:
                groups.append([entry])
        # Possible follow-up merge — if an entry bridged two groups, merge.
        # In practice with sorted input this is rare but cheap to handle.
        return cls._merge_overlapping_groups(groups)

    @classmethod
    def _merge_overlapping_groups(
        cls, groups: List[List[Tuple[str, Span]]]
    ) -> List[List[Tuple[str, Span]]]:
        changed = True
        while changed:
            changed = False
            merged: List[List[Tuple[str, Span]]] = []
            for g in groups:
                gmin = min(p[1][0] for p in g)
                gmax = max(p[1][1] for p in g)
                placed = False
                for m in merged:
                    mmin = min(p[1][0] for p in m)
                    mmax = max(p[1][1] for p in m)
                    if gmin < mmax and gmax > mmin:
                        m.extend(g)
                        placed = True
                        changed = True
                        break
                if not placed:
                    merged.append(list(g))
            groups = merged
        return groups

    # ------------------------------------------------------------------ #
    def _pick(self, spans: List[Span]) -> Span:
        """Choose the canonical span from a group of same-type spans."""
        if self.tie_break == "longest":
            return max(spans, key=lambda s: (s[1] - s[0], s[3]))
        return min(spans, key=lambda s: (s[1] - s[0], s[3]))

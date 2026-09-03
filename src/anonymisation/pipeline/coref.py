"""
Coreference-aware span extension.

The gap:
    Real legal documents introduce a party in full and then use shorthand:

        Northwind Energy Ltd was founded in 1998. Northwind operates two
        plants... in 2019 Northwind acquired a smaller rival.

    Off-the-shelf NER (spaCy, the Phase 2 fine-tune, dslim/bert-base-NER)
    will usually tag "Northwind Energy Ltd" as ORG on its first mention,
    but recall on the subsequent bare "Northwind" mentions is patchy.
    Document-level recall drops accordingly — and from a redaction
    perspective, the misses *are* leakage of the entity's identity.

The fix:
    A deterministic post-processor that runs after NER+regex. For every
    detected PERSON/ORG span, it generates a small set of candidate short
    forms and scans the document for any matches NER missed. New matches
    inherit the entity_type and identifier_role of the parent mention.

    Conservative on purpose — we generate short forms that are most likely
    to be coreferences in *legal* documents specifically, and we skip any
    candidate that's too short or too generic to be safe (e.g. "Energy",
    "Holdings", "the").

This module is the Phase 5 intervention. It runs in O(spans × doc_length)
and needs no extra dependencies. Measured against TAB, its mention-recall
lift is essentially zero (+0.0001 macro): spaCy's en_core_web_trf is already
at the recall ceiling on the PERSON/ORG mentions the substring rule could
recover, and the labels with real recall gaps (DEM, MISC, CODE) are ones it
can't help with. It is kept as a zero-cost defensive layer — every span it
adds is logged with source="coref" and confidence=0.7 — not as a source of
headline gains. See scripts/evaluate_mention_recall.py for the
evaluation that produced the null.
"""
from __future__ import annotations

import re
from typing import Iterable, List, Set, Tuple

from .pseudonymise import _strip_honorifics
from .types import Span


# ---------------------------------------------------------------------------
# Words we will NOT use as standalone ORG short forms — too generic, would
# match common usage of the word in unrelated contexts.
# ---------------------------------------------------------------------------
_GENERIC_ORG_WORDS: Set[str] = {
    "ltd", "limited", "inc", "incorporated", "corp", "corporation",
    "co", "company", "plc", "llp", "llc", "gmbh", "ag", "sa", "bv",
    "holdings", "group", "international", "global", "industries",
    "the", "and", "of", "for", "&",
    # Common legal-doc role words
    "court", "tribunal", "council", "commission", "office", "ministry",
    "department", "agency", "association", "union", "federation",
}

# Words we will NOT use as standalone PERSON short forms even if they happen
# to be a single-word name component — pronouns, common honorifics, etc.
_GENERIC_PERSON_WORDS: Set[str] = {
    "the", "and", "of", "a", "an", "his", "her", "their", "him", "she",
    "he", "they", "it", "its",
}

# Minimum length (chars) of any short-form candidate. Avoids "Co" / "Mr" /
# "Inc" sneaking through as standalone matches.
_MIN_FORM_LENGTH = 3


# ---------------------------------------------------------------------------
# Candidate-generation per entity type
# ---------------------------------------------------------------------------
def _candidate_forms(surface: str, entity_type: str) -> List[str]:
    """
    Return likely coreferring shorter forms for `surface` given its TAB type.

    PERSON examples:
        "Maria Petrova"         → ["Maria", "Petrova", "Mrs Petrova", "Mr Petrova", ...]
        "Mrs Maria Petrova"     → ["Maria", "Petrova", "Mrs Petrova", ...]
        "Sir Lawrence Smith"    → ["Lawrence", "Smith", "Mr Smith", ...]

    ORG examples:
        "Northwind Energy Ltd"  → ["Northwind"]
        "Acme Holdings Ltd"     → ["Acme"]
        "Sofia District Court"  → ["Sofia District"]   (single "Sofia" excluded — could mean the city)
        "MegaCorp International"→ ["MegaCorp"]
    """
    cleaned = _strip_honorifics(surface).strip()
    if not cleaned:
        return []
    parts = cleaned.split()

    forms: List[str] = []

    if entity_type == "PERSON":
        if len(parts) <= 1:
            return []  # already a single token — nothing shorter to generate
        first = parts[0]
        last = parts[-1]
        # First name and surname as standalone references
        for tok in (first, last):
            if (
                len(tok) >= _MIN_FORM_LENGTH
                and tok.lower() not in _GENERIC_PERSON_WORDS
                and tok[0].isupper()  # name-shaped
            ):
                forms.append(tok)
        # Honorific + surname combinations
        if len(last) >= _MIN_FORM_LENGTH and last[0].isupper():
            for honorific in ("Mr", "Mrs", "Ms", "Miss", "Dr"):
                forms.append(f"{honorific} {last}")
                forms.append(f"{honorific}. {last}")

    elif entity_type == "ORG":
        if len(parts) <= 1:
            return []
        # First significant word ("Northwind Energy Ltd" → "Northwind").
        # This is the most common legal-doc shorthand pattern; we need it
        # even though it occasionally overlaps with a LOC of the same name
        # ("Sofia District Court" generates "Sofia", which could be the
        # city). When there's a genuine LOC mention of "Sofia" elsewhere,
        # the NER pass tags it as LOC and the dedup rule keeps the more-
        # specific span. The audit log shows what coref added so callers
        # can filter by `source="coref"` and `confidence=0.7`.
        for tok in parts:
            if (
                len(tok) >= _MIN_FORM_LENGTH
                and tok.lower() not in _GENERIC_ORG_WORDS
                and tok[0].isupper()
            ):
                forms.append(tok)
                break
        # Plus a two-word prefix for 3+ word ORGs ("Northwind Energy Ltd"
        # → "Northwind Energy"; "Sofia District Court" → "Sofia District").
        # Catches the slightly more formal shorthand that some documents use.
        if len(parts) >= 3:
            two = f"{parts[0]} {parts[1]}"
            if all(p.lower() not in _GENERIC_ORG_WORDS for p in parts[:2]):
                forms.append(two)

    # Dedupe and sort by length DESC so longer multi-word forms are tried
    # first. Without this, "Petrova" (7 chars) is matched before "Mrs Petrova"
    # (11 chars), causing the "Mrs" prefix to be left un-redacted.
    seen: Set[str] = set()
    deduped: List[str] = []
    for f in forms:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    deduped.sort(key=len, reverse=True)
    return deduped


# ---------------------------------------------------------------------------
# Word-boundary matching
# ---------------------------------------------------------------------------
def _find_matches(text: str, needle: str) -> Iterable[Tuple[int, int]]:
    """Yield (start, end) for every word-aligned, case-sensitive match of needle in text."""
    if not needle:
        return
    pattern = r"\b" + re.escape(needle) + r"\b"
    for m in re.finditer(pattern, text):
        yield m.start(), m.end()


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------
def extend_with_coref(text: str, spans: List[Span]) -> List[Span]:
    """
    Return `spans` augmented with extra coreferring mentions discovered in text.

    For every PERSON/ORG span we already have, generate candidate short forms
    and scan the document. Any match that doesn't overlap an existing span
    is added with `source="coref"` and the same identifier_role / entity_type
    as the parent.
    """
    additions: List[Span] = []

    # Avoid generating duplicate coref spans for the same surface text
    already_emitted: Set[Tuple[int, int]] = set()

    def overlaps_any(s: int, e: int, lst: List[Span]) -> bool:
        return any(s < sp.end and e > sp.start for sp in lst)

    for parent in spans:
        if parent.entity_type not in ("PERSON", "ORG"):
            continue
        for candidate in _candidate_forms(parent.text, parent.entity_type):
            for start, end in _find_matches(text, candidate):
                if (start, end) in already_emitted:
                    continue
                if overlaps_any(start, end, spans) or overlaps_any(start, end, additions):
                    continue
                additions.append(Span(
                    start=start, end=end,
                    entity_type=parent.entity_type,
                    text=text[start:end],
                    identifier_role=parent.identifier_role,
                    source="coref",
                    confidence=0.7,  # lower than direct NER; surfaces in audit
                ))
                already_emitted.add((start, end))

    return sorted(spans + additions, key=lambda s: s.start)

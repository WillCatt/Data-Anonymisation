"""
Regex post-pass.

NER models are bad at structured identifiers — case file numbers,
IBANs, NHS numbers, phone numbers, email addresses. Regex is much better
at these. The regex pass runs after the NER pass and adds spans the NER
might have missed.

We deliberately keep the recogniser set small and conservative; false
positives cost more in a redaction pipeline than misses (a missed regex
is a missed regex; a false positive over-redacts and damages the document).

Each recogniser maps to a TAB entity type so the downstream classifier
and renderer can treat it consistently.
"""
from __future__ import annotations

import re
from typing import List, Tuple

from .types import Span


# (regex, tab_type) — order matters; we prefer the first match at any position
RECOGNIZERS: List[Tuple[re.Pattern, str]] = [
    # ECHR / common-law application numbers
    (re.compile(r"\b(?:Application|App\.?|Case)\s+[Nn]o\.?\s*\d{1,7}/\d{2,4}\b"), "CODE"),
    # UK NHS number — three groups, total 10 digits (with optional spaces/dashes)
    (re.compile(r"\b\d{3}[\s\-]?\d{3}[\s\-]?\d{4}\b"), "CODE"),
    # IBAN — 15-34 chars, country code prefix
    (re.compile(r"\b[A-Z]{2}\d{2}[\s\-]?(?:[A-Z0-9]{4}[\s\-]?){2,7}[A-Z0-9]{1,4}\b"), "CODE"),
    # Email
    (re.compile(r"\b[\w._%+\-]+@[\w.\-]+\.[A-Za-z]{2,}\b"), "MISC"),
    # International phone numbers (lenient)
    (re.compile(r"(?:\+\d{1,3}\s?)?(?:\(\d{1,4}\)\s?|\d{1,4}[\s\-]?)\d{3,4}[\s\-]?\d{3,4}\b"), "MISC"),
]


def regex_pass(text: str) -> List[Span]:
    """Run every regex recogniser and return non-overlapping spans."""
    spans: List[Span] = []
    for pattern, tab_type in RECOGNIZERS:
        for m in pattern.finditer(text):
            start, end = m.start(), m.end()
            # Skip if we already have a span here (earlier pattern won)
            if any(s.start < end and s.end > start for s in spans):
                continue
            spans.append(Span(
                start=start, end=end, entity_type=tab_type,
                text=text[start:end], identifier_role="DIRECT",
                source="regex", confidence=1.0,
            ))
    return sorted(spans, key=lambda s: s.start)


def merge_with_ner(ner_spans: List[Span], regex_spans: List[Span]) -> List[Span]:
    """
    Merge two span lists, preferring regex on collision (regex matches are
    higher-precision than NER on structured identifiers like case numbers).
    """
    keep = list(regex_spans)
    for ner_span in ner_spans:
        if not any(ner_span.overlaps(rs) for rs in regex_spans):
            keep.append(ner_span)
    return sorted(keep, key=lambda s: s.start)

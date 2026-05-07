"""
Mosaic effect / re-identification helpers.

The "mosaic effect" is the observation that many small, individually-
unremarkable facts can combine into a unique fingerprint. A document with
"a 34-year-old Bulgarian woman, mother of two, employed as a nurse in
Plovdiv" contains no name, no DOB, no ID — but the combination is unique
enough to pin down the person.

This module provides simple tooling to demonstrate the effect against
the TAB corpus:

  * `quasi_identifier_signature` builds a bag of QUASI-tagged span values
    for a document — the tuple of demographic facts an attacker might have.
  * `k_anonymity_table` computes how often each signature is shared across
    the corpus. A signature with k=1 means a unique fingerprint —
    a document that is *re-identifiable* even after every DIRECT identifier
    is masked.

It is deliberately small and educational: real k-anonymisation is a
generalisation problem (binning ages into ranges, postcodes into prefixes)
that we do not attempt here. The point is to show the *risk*, not solve it.
"""
from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List, Tuple

import pandas as pd


def quasi_identifier_signature(
    doc: dict,
    entity_types: Iterable[str] = ("DEM", "DATETIME", "LOC", "QUANTITY"),
) -> Tuple[Tuple[str, str], ...]:
    """
    Build a sorted tuple of (entity_type, span_text.lower()) pairs from
    QUASI-tagged mentions in a TAB document.

    Returns a hashable signature that can be counted across documents.
    """
    parts: List[Tuple[str, str]] = []
    for em in doc["entity_mentions"]:
        if em["identifier_type"] != "QUASI":
            continue
        if em["entity_type"] not in entity_types:
            continue
        parts.append((em["entity_type"], em["span_text"].strip().lower()))
    # Deduplicate within the same document, then sort for stable hashing
    return tuple(sorted(set(parts)))


def k_anonymity_table(
    docs: Iterable[dict],
    entity_types: Iterable[str] = ("DEM", "DATETIME", "LOC", "QUANTITY"),
    min_signature_size: int = 2,
) -> pd.DataFrame:
    """
    Compute a k-anonymity table over a corpus.

    For every document we build a quasi-identifier signature, then count
    how many documents share each signature. The result is a DataFrame
    with one row per (doc_id, signature) and a `k` column showing how
    many documents share that signature.

    A document with k = 1 is uniquely identifiable from its quasi-
    identifiers alone — even if every name, date, and ID is masked.
    """
    rows = []
    sig_by_doc: Dict[str, Tuple] = {}
    for doc in docs:
        sig = quasi_identifier_signature(doc, entity_types=entity_types)
        if len(sig) < min_signature_size:
            continue
        sig_by_doc[doc["doc_id"]] = sig

    sig_counts: Counter = Counter(sig_by_doc.values())

    for doc_id, sig in sig_by_doc.items():
        rows.append({
            "doc_id": doc_id,
            "signature_size": len(sig),
            "k": sig_counts[sig],
            "signature": sig,
        })

    return pd.DataFrame(rows).sort_values("k", ascending=True).reset_index(drop=True)

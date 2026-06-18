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

import re
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Fact normalisation — fold surface variants of the *same* quasi-identifier
# ---------------------------------------------------------------------------
# Matching fingerprints on raw surface form overstates uniqueness: "47-year-old",
# "aged 47" and "forty-seven" are the same identifying fact but three different
# strings, so three documents that share it look distinct. A real attacker reads
# past the surface form. Normalisation folds genuinely-equal facts together
# *without* merging different ones (we keep the head noun on counts, so
# "three children" and "three convictions" stay separate). It only ever makes
# the corpus *less* unique — the conservative direction.

# Spelled-out cardinals occasionally appear in QUANTITY spans.
_ONES = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
    "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TENS = {
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
    "seventy": 70, "eighty": 80, "ninety": 90,
}

_AGE_RE = re.compile(
    r"\b(\d{1,3})\s*[- ]?\s*year[s]?[- ]?old\b"   # 47-year-old / 47 years old
    r"|\baged\s+(\d{1,3})\b"                       # aged 47
)


def _split_leading_spelled(text: str) -> Tuple[Optional[int], str]:
    """
    Parse a leading run of cardinal words into an int and return the remainder.

    "forty-seven"     -> (47, "")
    "three children"  -> (3, "children")
    "nurse"           -> (None, "nurse")
    """
    words = re.split(r"[\s-]+", text.strip().lower())
    total: Optional[int] = None
    i = 0
    while i < len(words) and (words[i] in _TENS or words[i] in _ONES):
        total = (total or 0) + (_TENS.get(words[i]) or _ONES[words[i]])
        i += 1
    return total, " ".join(words[i:]).strip()


def normalise_quasi_value(entity_type: str, text: str) -> str:
    """
    Canonicalise one quasi-identifier value so semantically-equal facts collide.

    Conservative by design — only folds variants that mean the same thing:
      * DATETIME → the 4-digit year ("12 March 2018", "in 2018" → "2018").
      * QUANTITY ages → "age:N" ("47-year-old", "aged 47", "forty-seven" → "age:47"),
        kept distinct from unrelated counts that happen to share the number.
      * QUANTITY counts with a spelled number → digit + noun ("three children" →
        "3 children"), so the head noun still separates different facts.
      * LOC / DEM / other → lower-cased, punctuation stripped, whitespace collapsed.
    """
    t = text.strip().lower()

    if entity_type == "DATETIME":
        year = re.search(r"\b(1[5-9]\d{2}|20\d{2})\b", t)
        if year:
            return year.group(1)
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", t)).strip()

    if entity_type == "QUANTITY":
        age = _AGE_RE.search(t)
        if age:
            n = next(g for g in age.groups() if g)
            return f"age:{int(n)}"
        # "47", "EUR 4.2 million" → keep digits + units, just drop separators
        if re.search(r"\d", t):
            return re.sub(r"\s+", " ", re.sub(r"[^\w\s.]+", " ", t)).strip()
        # spelled count: fold the leading number, keep the noun ("three children")
        spelled, tail = _split_leading_spelled(t)
        if spelled is not None:
            return f"{spelled} {tail}".strip()
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", t)).strip()

    # LOC, DEM, and anything else: drop punctuation, collapse whitespace.
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]+", " ", t)).strip()


def quasi_identifier_signature(
    doc: dict,
    entity_types: Iterable[str] = ("DEM", "DATETIME", "LOC", "QUANTITY"),
    normalise: bool = False,
) -> Tuple[Tuple[str, str], ...]:
    """
    Build a sorted tuple of (entity_type, value) pairs from QUASI-tagged
    mentions in a TAB document.

    With ``normalise=False`` (default) the value is just ``span_text.lower()`` —
    the historical behaviour the scorer and existing fixtures rely on. With
    ``normalise=True`` each value is passed through :func:`normalise_quasi_value`
    so surface variants of the same fact collide.

    Returns a hashable signature that can be counted across documents.
    """
    parts: List[Tuple[str, str]] = []
    for em in doc["entity_mentions"]:
        if em["identifier_type"] != "QUASI":
            continue
        if em["entity_type"] not in entity_types:
            continue
        etype = em["entity_type"]
        value = (
            normalise_quasi_value(etype, em["span_text"])
            if normalise
            else em["span_text"].strip().lower()
        )
        if value:
            parts.append((etype, value))
    # Deduplicate within the same document, then sort for stable hashing
    return tuple(sorted(set(parts)))


def min_facts_to_identify(
    target_facts: Iterable,
    other_docs_facts: Iterable[Iterable],
) -> Optional[int]:
    """
    Greedy smart-attacker estimate of re-identification difficulty.

    Models an attacker who, instead of learning a document's quasi-identifiers
    in the order they happen to appear, learns the *most discriminating* ones
    first. Returns the fewest facts that attacker must learn about ``target``
    before no other document in the corpus shares all of them — i.e. before the
    target is uniquely identified.

    The attacker reveals the globally-rarest relevant fact at each step (rarest
    facts shrink the candidate pool fastest), and we count reveals until the
    pool of still-matching other documents is empty. Returns ``None`` if even
    the full fingerprint is shared by some other document (target never unique).

    This is a greedy upper bound on the true minimum (exact minimum is set-cover,
    NP-hard); it is always ≤ the document-order count, so it makes
    re-identification look *at least* as easy as the naive curve — the realistic
    direction.
    """
    target = set(target_facts)
    if not target:
        return None
    others = [set(d) for d in other_docs_facts]

    # Frequency of each *target-relevant* fact across the rest of the corpus.
    freq: Counter = Counter()
    for d in others:
        freq.update(d & target)

    ordered = sorted(target, key=lambda f: freq.get(f, 0))  # rarest first
    candidates = others
    for i, fact in enumerate(ordered, start=1):
        candidates = [d for d in candidates if fact in d]
        if not candidates:
            return i
    return None  # never singled out — some other doc has a superset fingerprint


def k_anonymity_table(
    docs: Iterable[dict],
    entity_types: Iterable[str] = ("DEM", "DATETIME", "LOC", "QUANTITY"),
    min_signature_size: int = 2,
    normalise: bool = False,
) -> pd.DataFrame:
    """
    Compute a k-anonymity table over a corpus.

    For every document we build a quasi-identifier signature, then count
    how many documents share each signature. The result is a DataFrame
    with one row per (doc_id, signature) and a `k` column showing how
    many documents share that signature.

    A document with k = 1 is uniquely identifiable from its quasi-
    identifiers alone — even if every name, date, and ID is masked.

    Note on multi-annotator corpora: TAB (and similar) ships multiple
    annotator entries per logical document, distinguished by annotator_id
    but sharing a doc_id. We aggregate by *taking the union* of QUASI
    mentions across all annotators for each doc_id — the canonical
    "what anyone might have considered identifying" fingerprint per doc.
    """
    # Step 1 — aggregate QUASI mentions per unique doc_id across annotators
    mentions_by_doc: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    for doc in docs:
        sig = quasi_identifier_signature(doc, entity_types=entity_types, normalise=normalise)
        if sig:
            mentions_by_doc[doc["doc_id"]].update(sig)

    # Step 2 — canonicalise to sorted tuples and apply min_signature_size
    sig_by_doc: Dict[str, Tuple[Tuple[str, str], ...]] = {}
    for doc_id, mentions in mentions_by_doc.items():
        if len(mentions) < min_signature_size:
            continue
        sig_by_doc[doc_id] = tuple(sorted(mentions))

    # Step 3 — k = how many docs share this signature
    sig_counts: Counter = Counter(sig_by_doc.values())

    rows = []
    for doc_id, sig in sig_by_doc.items():
        rows.append({
            "doc_id": doc_id,
            "signature_size": len(sig),
            "k": sig_counts[sig],
            "signature": sig,
        })

    return pd.DataFrame(rows).sort_values("k", ascending=True).reset_index(drop=True)

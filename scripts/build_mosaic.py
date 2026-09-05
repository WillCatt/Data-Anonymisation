"""
Build the mosaic / re-identification figure for Act I of the writeup.

This is the *hardened* version of the analysis. The first cut matched
fingerprints on raw surface form and revealed a document's quasi-identifiers
in the order they happen to appear — both of which a real attacker ignores.
This version closes that gap and measures how much it matters:

  (a) Re-identification curves — fraction of documents uniquely identifiable
      as an attacker learns more quasi-identifiers, under two strategies:
        * document order  — facts revealed as they appear (the naive curve);
        * smart attacker  — the most-discriminating (globally rarest) facts
          first, which is what someone actually trying to re-identify does.
      Both run on *normalised* facts, so surface variants of the same fact
      ("47-year-old" / "aged 47", "12 March 2018" / "in 2018") collide — the
      conservative choice, since it can only make documents look less unique.

  (b) Minimum-facts histogram — for each document, the fewest facts the smart
      attacker needs to single it out. The mass sits at one or two facts.

The uniqueness test is the honest one: a document is identified once *no other
document contains all the revealed facts*. Under that test a handful of
documents are never unique (their whole fingerprint is a subset of another's) —
reported rather than hidden.

Run with:
    legal-anon-env/bin/python scripts/build_mosaic.py

Output:
    figures/mosaic_reidentification.png

Needs the TAB data — loads from the HuggingFace cache (downloads ~50 MB on
first run).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # package isn't installed; mirror the notebooks

from anonymisation.data import load_tab  # noqa: E402
from anonymisation.mosaic import (  # noqa: E402
    min_facts_to_identify,
    normalise_quasi_value,
)

sns.set_theme(style="whitegrid", font_scale=0.95)

# The QUASI types that make up the re-identification fingerprint — same set
# the rest of the project uses (anonymisation.mosaic.quasi_identifier_signature).
QUASI_TYPES = ("DEM", "DATETIME", "LOC", "QUANTITY")

# The conservative subset: the "ordinary details" the write-up actually claims
# — a nationality, an occupation, a place. It drops DATETIME and QUANTITY,
# which is where the original 100% artefact came from (63% of the fingerprint
# was DATETIME and 85% of those a bare year, which identifies a *case* rather
# than a person). Both curves are plotted so the two numbers the project
# quotes — 81% and 77% — appear together instead of in separate documents.
HEADLINE_TYPES = ("DEM", "LOC")

ACCENT = "#d73a49"   # the project's "mosaic / risk" red — the smart attacker
MUTED = "#e8a3a3"    # the naive document-order curve
GREY = "#95a5a6"

Fact = Tuple[str, str]


# ---------------------------------------------------------------------------
# Per document: normalised quasi-identifiers, ordered by first appearance
# ---------------------------------------------------------------------------
def ordered_quasi_by_doc(docs, types=QUASI_TYPES) -> Dict[str, List[Fact]]:
    """
    For each doc_id, return its normalised QUASI facts as (entity_type, value)
    pairs, deduplicated and ordered by first appearance in the document.

    Mentions are unioned across annotators (TAB ships multiple annotator entries
    per doc_id); first-appearance offset breaks ties for ordering.
    """
    first_seen: Dict[str, Dict[Fact, int]] = {}
    for doc in docs:
        bucket = first_seen.setdefault(doc["doc_id"], {})
        for em in doc["entity_mentions"]:
            if em["identifier_type"] != "QUASI":
                continue
            if em["entity_type"] not in types:
                continue
            value = normalise_quasi_value(em["entity_type"], em["span_text"])
            if not value:
                continue
            key = (em["entity_type"], value)
            start = em["start_offset"]
            if key not in bucket or start < bucket[key]:
                bucket[key] = start

    ordered: Dict[str, List[Fact]] = {}
    for doc_id, keys in first_seen.items():
        ordered[doc_id] = [k for k, _ in sorted(keys.items(), key=lambda kv: kv[1])]
    return {d: q for d, q in ordered.items() if q}  # need ≥1 fact to fingerprint


# ---------------------------------------------------------------------------
# Re-identification: facts needed to single a document out
# ---------------------------------------------------------------------------
def _facts_needed_in_order(ordered_facts: List[Fact], others: List[set]) -> Optional[int]:
    """How many facts, revealed in the given order, until no other doc holds them all."""
    candidates = others
    for i, fact in enumerate(ordered_facts, start=1):
        candidates = [d for d in candidates if fact in d]
        if not candidates:
            return i
    return None


def reidentification_counts(
    quasi_by_doc: Dict[str, List[Fact]],
) -> Tuple[List[Optional[int]], List[Optional[int]]]:
    """
    For every document return (facts_needed_document_order, facts_needed_smart).

    Smart order reuses the tested library greedy (rarest relevant fact first).
    """
    items = list(quasi_by_doc.items())
    sets = [set(facts) for _, facts in items]

    doc_order: List[Optional[int]] = []
    smart: List[Optional[int]] = []
    for i, (_doc_id, facts) in enumerate(items):
        others = sets[:i] + sets[i + 1:]
        doc_order.append(_facts_needed_in_order(facts, others))
        smart.append(min_facts_to_identify(facts, others))
    return doc_order, smart


def cdf(counts: List[Optional[int]], n_docs: int, max_n: int) -> Tuple[List[int], List[float]]:
    """Fraction of all documents identified within ≤ n facts (None = never)."""
    ns = list(range(1, max_n + 1))
    fracs = [
        100.0 * sum(1 for c in counts if c is not None and c <= n) / n_docs
        for n in ns
    ]
    return ns, fracs


def _first_crossing(ns: List[int], fracs: List[float], threshold: float) -> Optional[int]:
    for n, f in zip(ns, fracs):
        if f >= threshold:
            return n
    return None


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_curves(ns, doc_fracs, smart_fracs, headline_fracs, n_docs, ax: plt.Axes) -> None:
    ax.plot(ns, doc_fracs, marker="o", markersize=3.5, color=MUTED, linewidth=1.8,
            label="Facts in document order")
    ax.plot(ns, smart_fracs, marker="o", markersize=4, color=ACCENT, linewidth=2.4,
            label="Rarest facts first — all quasi-identifiers")
    ax.fill_between(ns, smart_fracs, color=ACCENT, alpha=0.07)
    ax.plot(ns, headline_fracs, marker="o", markersize=3.5, color="#8a6d12",
            linewidth=2.0, linestyle="--",
            label="Rarest facts first — demographics and places only")

    one = smart_fracs[0]
    ax.annotate(
        f"{one:.0f}% unique\nfrom 1 fact",
        xy=(1, one), xytext=(2.4, one - 17),
        fontsize=9, color=ACCENT, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ACCENT, linewidth=1),
    )
    head_one = headline_fracs[0]
    ax.annotate(
        f"{head_one:.0f}% from one\nordinary detail",
        xy=(1, head_one), xytext=(3.4, head_one - 38),
        fontsize=9, color="#8a6d12", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#8a6d12", linewidth=1),
    )
    n95 = _first_crossing(ns, doc_fracs, 95)
    if n95 is not None:
        ax.annotate(
            f"document order needs\n{n95} facts for 95%",
            xy=(n95, 95), xytext=(n95 + 1.0, 60),
            fontsize=8.5, color="#586069",
            arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.8),
        )

    ax.set_xlim(1, max(ns))
    ax.set_ylim(0, 103)
    ax.set_xlabel("Quasi-identifiers known to the attacker")
    ax.set_ylabel("Documents uniquely identifiable (%)")
    ax.set_title(
        "How fast a document becomes unique\n"
        f"(normalised QUASI facts, after a perfect DIRECT redaction · n={n_docs:,} docs)",
        fontweight="bold", fontsize=12, pad=10,
    )
    ax.legend(loc="lower right", fontsize=8.5, frameon=True)
    ax.set_axisbelow(True)


def panel_min_facts(smart: List[Optional[int]], n_docs: int, ax: plt.Axes) -> None:
    unique = [c for c in smart if c is not None]
    never = sum(1 for c in smart if c is None)
    median = int(np.median(unique))
    top = max(unique)

    ax.hist(unique, bins=range(1, top + 2), color=ACCENT, edgecolor="white",
            alpha=0.85, align="left", rwidth=0.9)
    ax.axvline(median, color="#586069", linewidth=2, linestyle="--")
    ax.annotate(
        f"median = {median} fact",
        xy=(median, ax.get_ylim()[1] * 0.7),
        xytext=(median + 1.5, ax.get_ylim()[1] * 0.78),
        fontsize=9, color="#586069", fontweight="bold",
        arrowprops=dict(arrowstyle="->", color="#586069", linewidth=1),
    )
    ax.set_xlabel("Fewest facts a smart attacker needs to single out the document")
    ax.set_ylabel("Number of documents")
    ax.set_title(
        "Re-identification is usually a one-fact problem\n"
        f"({never} of {n_docs:,} docs are never unique — a superset fingerprint hides them)",
        fontweight="bold", fontsize=12, pad=10,
    )
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading TAB …")
    ds = load_tab()
    docs = [d for split in ds for d in ds[split]]
    print(f"  {len(docs):,} annotator rows across {len(ds)} splits")

    quasi_by_doc = ordered_quasi_by_doc(docs)
    n_docs = len(quasi_by_doc)
    print(f"  {n_docs:,} documents with ≥1 normalised quasi-identifier")

    doc_order, smart = reidentification_counts(quasi_by_doc)
    max_n = 20
    ns, doc_fracs = cdf(doc_order, n_docs, max_n)
    _, smart_fracs = cdf(smart, n_docs, max_n)

    # The same attacker restricted to the facts the write-up headlines.
    headline_by_doc = ordered_quasi_by_doc(docs, HEADLINE_TYPES)
    _, headline_smart = reidentification_counts(headline_by_doc)
    _, headline_fracs = cdf(headline_smart, len(headline_by_doc), max_n)
    print(f"  demographics + places only — unique from 1 fact: {headline_fracs[0]:.1f}%   "
          f"2 facts: {cdf(headline_smart, len(headline_by_doc), 2)[1][-1]:.1f}%")

    never = sum(1 for c in smart if c is None)
    uniq = [c for c in smart if c is not None]
    print(f"  smart attacker — unique from 1 fact: {smart_fracs[0]:.1f}%   "
          f"≤3 facts: {cdf(smart, n_docs, 3)[1][-1]:.1f}%")
    print(f"  smart attacker — median min-facts: {int(np.median(uniq))}   "
          f"never unique: {never} ({100*never/n_docs:.1f}%)")
    print(f"  document order — unique from 1 fact: {doc_fracs[0]:.1f}%   "
          f"95% crossing: n={_first_crossing(ns, doc_fracs, 95)}")

    fig, (ax_curve, ax_hist) = plt.subplots(1, 2, figsize=(15, 5.5))
    panel_curves(ns, doc_fracs, smart_fracs, headline_fracs, n_docs, ax_curve)
    panel_min_facts(smart, n_docs, ax_hist)
    fig.suptitle(
        "The mosaic effect — one or two ordinary facts are usually enough",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.text(
        0.005, -0.03,
        "An earlier version of this analysis reported 1,268 / 1,268 documents uniquely "
        "identifiable. That figure was an artefact of requiring a match on the *entire* "
        "fingerprint —\na median of 14 facts — and has been withdrawn: a control using "
        "meaningless tokens with the same set sizes also returns 100%. These curves ask "
        "the honest question instead,\nwhich is how few facts an attacker needs.",
        fontsize=8.5, color="#586069", va="top",
    )

    out = ROOT / "figures" / "mosaic_reidentification.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

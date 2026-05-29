"""
Build the mosaic / re-identification figure for Act I of the writeup.

The old `mosaic_k_distribution.png` plotted the distribution of k across
documents — but every TAB document is unique (k=1), so the "distribution"
was a single bar. It stated the conclusion without showing the mechanism.

This script replaces it with two panels that actually carry insight:

  (a) Re-identification curve — as an attacker learns more of a document's
      quasi-identifiers (the first n distinct QUASI facts in document order),
      what fraction of documents become uniquely identifiable? The curve
      rises from "lots of collisions" at n=1 to ~100% within a handful of
      facts. This is also the strictness/sensitivity analysis the project
      wanted: it shows *how fast* uniqueness emerges, not just that it does.

  (b) Signature-size histogram — how many distinct quasi-identifiers each
      document carries (median ≈ 25). This is *why* the full fingerprint is
      always unique: the joint distribution of ~25 demographic facts is
      effectively a hash.

Run with:
    legal-anon-env/bin/python figures/build_mosaic.py

Output:
    figures/mosaic_reidentification.png

Needs the TAB data — loads from the HuggingFace cache (downloads ~50 MB on
first run).
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # package isn't installed; mirror the notebooks

from anonymisation.data import load_tab  # noqa: E402

sns.set_theme(style="whitegrid", font_scale=0.95)

# The QUASI types that make up the re-identification fingerprint — same set
# the rest of the project uses (anonymisation.mosaic.quasi_identifier_signature).
QUASI_TYPES = ("DEM", "DATETIME", "LOC", "QUANTITY")

ACCENT = "#d73a49"   # the project's "mosaic / risk" red
GREY = "#95a5a6"


# ---------------------------------------------------------------------------
# Build, per document, the ordered list of distinct quasi-identifiers
# ---------------------------------------------------------------------------
def ordered_quasi_by_doc(docs) -> Dict[str, List[Tuple[str, str]]]:
    """
    For each doc_id, return its QUASI mentions as (entity_type, text.lower())
    pairs, deduplicated and ordered by first appearance in the document.

    Mentions are unioned across annotators (TAB ships multiple annotator
    entries per doc_id); first-appearance offset breaks ties for ordering.
    """
    # doc_id -> {(type, text): earliest_start_offset}
    first_seen: Dict[str, Dict[Tuple[str, str], int]] = {}
    for doc in docs:
        bucket = first_seen.setdefault(doc["doc_id"], {})
        for em in doc["entity_mentions"]:
            if em["identifier_type"] != "QUASI":
                continue
            if em["entity_type"] not in QUASI_TYPES:
                continue
            key = (em["entity_type"], em["span_text"].strip().lower())
            start = em["start_offset"]
            if key not in bucket or start < bucket[key]:
                bucket[key] = start

    ordered: Dict[str, List[Tuple[str, str]]] = {}
    for doc_id, keys in first_seen.items():
        ordered[doc_id] = [k for k, _ in sorted(keys.items(), key=lambda kv: kv[1])]
    return ordered


# ---------------------------------------------------------------------------
# Re-identification curve
# ---------------------------------------------------------------------------
def uniqueness_curve(
    quasi_by_doc: Dict[str, List[Tuple[str, str]]],
    max_n: int = 30,
) -> Tuple[List[int], List[float], int]:
    """
    For each n in 1..max_n, truncate every document's fingerprint to its
    first n distinct quasi-identifiers and report the fraction of documents
    that are unique (k=1) under that truncated fingerprint.
    """
    docs = [q for q in quasi_by_doc.values() if q]  # need ≥1 quasi to fingerprint
    n_docs = len(docs)
    ns, fracs = [], []
    for n in range(1, max_n + 1):
        sigs = [tuple(sorted(set(q[:n]))) for q in docs]
        counts = Counter(sigs)
        unique = sum(1 for s in sigs if counts[s] == 1)
        ns.append(n)
        fracs.append(100.0 * unique / n_docs)
    return ns, fracs, n_docs


def _first_crossing(ns: List[int], fracs: List[float], threshold: float) -> int | None:
    for n, f in zip(ns, fracs):
        if f >= threshold:
            return n
    return None


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def panel_curve(ns, fracs, n_docs, ax: plt.Axes) -> None:
    ax.plot(ns, fracs, marker="o", markersize=4, color=ACCENT, linewidth=2)
    ax.fill_between(ns, fracs, color=ACCENT, alpha=0.08)

    # Mark the 50% and 95% crossings — the "how fast" of the story.
    for thr, style in ((50, ":"), (95, "--")):
        n_cross = _first_crossing(ns, fracs, thr)
        if n_cross is not None:
            ax.axvline(n_cross, color=GREY, linestyle=style, linewidth=1)
            ax.annotate(
                f"{thr}% unique\nby {n_cross} fact{'s' if n_cross != 1 else ''}",
                xy=(n_cross, thr), xytext=(n_cross + 1.2, thr - 14),
                fontsize=8.5, color="#586069",
                arrowprops=dict(arrowstyle="-", color=GREY, linewidth=0.8),
            )

    ax.set_xlim(1, max(ns))
    ax.set_ylim(0, 103)
    ax.set_xlabel("Quasi-identifiers known to the attacker (first n distinct facts)")
    ax.set_ylabel("Documents uniquely identifiable (%)")
    ax.set_title(
        "How fast a document becomes unique\n"
        f"(QUASI fingerprint only, after a perfect DIRECT redaction · n={n_docs:,} docs)",
        fontweight="bold", fontsize=12, pad=10,
    )
    ax.set_axisbelow(True)


def panel_sizes(quasi_by_doc: Dict[str, List[Tuple[str, str]]], ax: plt.Axes) -> None:
    sizes = [len(q) for q in quasi_by_doc.values() if q]
    median = int(np.median(sizes))
    ax.hist(sizes, bins=range(0, max(sizes) + 3, 2), color=GREY,
            edgecolor="white", alpha=0.9)
    ax.axvline(median, color=ACCENT, linewidth=2)
    ax.annotate(
        f"median = {median} distinct\nquasi-identifiers / doc",
        xy=(median, ax.get_ylim()[1] * 0.6),
        xytext=(median + 4, ax.get_ylim()[1] * 0.7),
        fontsize=9, color=ACCENT, fontweight="bold",
        arrowprops=dict(arrowstyle="->", color=ACCENT, linewidth=1),
    )
    ax.set_xlabel("Distinct quasi-identifiers per document")
    ax.set_ylabel("Number of documents")
    ax.set_title(
        "Why the full fingerprint is always unique\n"
        "(the joint distribution of ~25 facts is effectively a hash)",
        fontweight="bold", fontsize=12, pad=10,
    )
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("Loading TAB …")
    ds = load_tab()
    # Use the whole corpus (all splits) — the headline is "1,268 / 1,268".
    docs = [d for split in ds for d in ds[split]]
    print(f"  {len(docs):,} annotator rows across {len(ds)} splits")

    quasi_by_doc = ordered_quasi_by_doc(docs)
    ns, fracs, n_docs = uniqueness_curve(quasi_by_doc)

    full_unique = fracs[-1]
    print(f"  {n_docs:,} documents with ≥1 quasi-identifier")
    print(f"  unique at 1 fact: {fracs[0]:.1f}%   "
          f"| at {ns[-1]} facts: {full_unique:.1f}%")
    print(f"  50% crossing: n={_first_crossing(ns, fracs, 50)}   "
          f"95% crossing: n={_first_crossing(ns, fracs, 95)}")

    fig, (ax_curve, ax_sizes) = plt.subplots(1, 2, figsize=(15, 5.5))
    panel_curve(ns, fracs, n_docs, ax_curve)
    panel_sizes(quasi_by_doc, ax_sizes)
    fig.suptitle(
        "The mosaic effect — quasi-identifiers re-identify documents NER can't touch",
        fontsize=15, fontweight="bold", y=1.02,
    )
    fig.tight_layout()

    out = ROOT / "figures" / "mosaic_reidentification.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\n→ {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

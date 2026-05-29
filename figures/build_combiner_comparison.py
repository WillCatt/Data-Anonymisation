"""
Build the Phase 6.2 combiner-comparison figure.

Reads phase6_advanced_training/results/combiner_comparison.csv (produced by
phase6_advanced_training/scripts/eval_combiners.py) and writes a single
bar chart of partial-match F1 across the combiner strategies, with the
LegalBERT-alone "ceiling" drawn as a reference line.

The story the chart tells: fixing the voting rule climbs the union backfire
back up (55 -> 66 -> 80), but no combiner clears the single fine-tune,
because the fine-tune Pareto-dominates every member per label.

Run with:
    python figures/build_combiner_comparison.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "phase6_advanced_training" / "results" / "combiner_comparison.csv"
OUT = ROOT / "figures" / "phase6_combiner_comparison.png"

sns.set_theme(style="whitegrid", font_scale=0.95)

# Display order + labels (chronological / argumentative order).
ORDER = [
    "spacy_alone", "presidio_alone", "union(min1)", "consensus(min2)",
    "consensus(min3)", "routed", "legalbert_alone",
]
LABELS = {
    "spacy_alone":     "spaCy-trf\nalone",
    "presidio_alone":  "Presidio\nalone",
    "union(min1)":     "Union\n(min_votes=1)",
    "consensus(min2)": "Consensus\n(min_votes=2)",
    "consensus(min3)": "Consensus\n(min_votes=3)",
    "routed":          "Routed\n(per-label best)",
    "legalbert_alone": "LegalBERT-FT\nalone",
}
COLORS = {
    "spacy_alone":     "#95a5a6",
    "presidio_alone":  "#8e44ad",
    "union(min1)":     "#e74c3c",   # the backfire — red
    "consensus(min2)": "#e67e22",
    "consensus(min3)": "#f1c40f",
    "routed":          "#16a085",
    "legalbert_alone": "#16a085",   # ceiling — same teal as the fine-tune
}


def main():
    df = pd.read_csv(CSV)
    df = df[df["mode"] == "partial"].set_index("strategy")

    ceiling = df.loc["legalbert_alone", "f1"]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    xs = list(range(len(ORDER)))
    f1s = [df.loc[s, "f1"] for s in ORDER]
    bars = ax.bar(xs, f1s, color=[COLORS[s] for s in ORDER],
                  edgecolor="white", width=0.72, zorder=3)

    # Ceiling reference line = single fine-tune.
    ax.axhline(ceiling, ls="--", lw=1.4, color="#16a085", alpha=0.7, zorder=2)
    ax.text(-0.35, ceiling + 0.015,
            f"single fine-tune ceiling = {ceiling:.1%}",
            ha="left", va="bottom", fontsize=9, color="#0e6b57", style="italic")

    for x, s, v in zip(xs, ORDER, f1s):
        ax.text(x, v + 0.008, f"{v:.1%}", ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[s] for s in ORDER], fontsize=8.5)
    ax.set_ylabel("Partial-match F1 (TAB test, 555 docs)")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "Phase 6.2 — no combiner beats the single fine-tune\n"
        "Fixing the vote rule recovers the union backfire (55%→80%), "
        "but the fine-tune dominates every label, so routing = the model alone",
        fontsize=11, loc="left",
    )
    ax.margins(x=0.02)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()

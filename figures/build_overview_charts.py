"""
Stakeholder-facing charts for the portfolio "Overview" tab.

These are the plain-language counterparts to the technical figures: no F1
axes, no entity-type jargon, big honest numbers. Two charts:

  1. overview_accuracy.png        — reliability score: off-the-shelf vs trained
  2. overview_reidentification.png — "removing names isn't enough" curve

Styled in the portfolio's warm palette so they sit natively on the site.

Numbers are not invented: the reliability scores are read from the project's
results CSVs, and the re-identification curve reuses the exact computation
behind figures/build_mosaic.py.

Run:
    legal-anon-env/bin/python figures/build_overview_charts.py
Outputs into figures/ (copied into Portfolio/assets/ for the site).
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "figures"))            # import the mosaic helpers
from build_mosaic import (  # noqa: E402
    cdf,
    load_tab,
    ordered_quasi_by_doc,
    reidentification_counts,
)

# ── Portfolio palette (matches project.css :root) ──────────────────────────
BG      = "#faf8f4"
INK     = "#1a1714"
MUTED   = "#7a6e63"
GREY    = "#cdbfae"                       # warm grey for the "before" bar
AMBER   = "#a86a2c"                       # accent
GREEN   = "#3a9d4e"                       # "improved / good"
RED     = "#d73a49"                       # "risk"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG,
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.edgecolor": "#e4ddd2", "font.size": 13, "axes.titlesize": 16,
})


def _strip(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def read_overall_f1() -> tuple[int, int]:
    """Overall partial-match F1 for the off-the-shelf baseline and the fine-tune."""
    base = pd.read_csv(ROOT / "results" / "phase1_results.csv")
    base_f1 = base[(base["mode"] == "partial") & (base["entity_type"] == "_ALL")]["f1"].iloc[0]
    ft = pd.read_csv(ROOT / "phase2_baseline_comparison" / "results" / "finetuned_results.csv")
    ft_f1 = ft[(ft["mode"] == "partial") & (ft["entity_type"] == "_ALL")]["f1"].iloc[0]
    return round(base_f1 * 100), round(ft_f1 * 100)


# ── Chart 1 — reliability bars ─────────────────────────────────────────────
def chart_accuracy(out: Path) -> None:
    base, trained = read_overall_f1()
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(
        ["Off-the-shelf AI tool", "After training on\nlegal documents"],
        [base, trained],
        color=[GREY, GREEN], width=0.6, edgecolor="none",
    )
    ax.tick_params(axis="x", length=0)   # no tick marks under the labels
    ax.grid(False)                         # kill the stray vertical gridline at bar centers
    for bar, val in zip(bars, (base, trained)):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 2, f"{val}",
                ha="center", va="bottom", fontsize=30, fontweight="bold",
                color=INK)
    ax.set_ylim(0, 100)
    ax.set_yticks([])
    ax.set_title("Training on legal data nearly doubled the tool's reliability",
                 fontweight="bold", pad=16, loc="left")
    ax.text(0, -0.16, "Overall reliability score (out of 100) at finding sensitive "
            "information across 555 real court documents.",
            transform=ax.transAxes, fontsize=11.5, color=MUTED)
    _strip(ax)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.relative_to(ROOT)}  ({base} → {trained})")


# ── Chart 2 — re-identification curve (plain language) ─────────────────────
def chart_reidentification(out: Path) -> None:
    ds = load_tab()
    docs = [d for split in ds for d in ds[split]]
    quasi = ordered_quasi_by_doc(docs)
    _, smart = reidentification_counts(quasi)            # realistic attacker
    ns, fracs = cdf(smart, len(quasi), max_n=8)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(ns, fracs, marker="o", markersize=7, color=RED, linewidth=3)
    ax.fill_between(ns, fracs, color=RED, alpha=0.07)

    # Highlight the "1 telling detail → 81%" point
    one = fracs[0]
    ax.scatter([1], [one], s=170, color=RED, zorder=5, edgecolor=BG, linewidth=2)
    ax.annotate(f"Just one distinctive detail\nidentifies {one:.0f}% of people",
                xy=(1, one), xytext=(1.6, one - 40),
                fontsize=12.5, fontweight="bold", color=RED,
                arrowprops=dict(arrowstyle="->", color=RED, linewidth=1.6))

    ax.set_xlim(1, 8); ax.set_ylim(0, 105)
    ax.set_xlabel("Number of ordinary personal details known\n"
                  "(age, town, job, nationality …)", fontsize=12)
    ax.set_ylabel("People who can be\nuniquely identified", fontsize=12)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("Removing names isn't enough",
                 fontweight="bold", pad=16, loc="left")
    ax.text(0, -0.30, "Even after deleting every name, date and ID number, the "
            "everyday details left behind\nstill single out almost everyone.",
            transform=ax.transAxes, fontsize=11.5, color=MUTED)
    _strip(ax)
    ax.grid(axis="y", color="#e4ddd2", linewidth=1)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out.relative_to(ROOT)}  (1 detail → {one:.0f}%)")


def main() -> None:
    fig_dir = ROOT / "figures"
    print("Building stakeholder charts:")
    chart_accuracy(fig_dir / "overview_accuracy.png")
    chart_reidentification(fig_dir / "overview_reidentification.png")
    print("Done.")


if __name__ == "__main__":
    main()

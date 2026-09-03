"""
Build the fine-tune results figure for the portfolio RESULTS tab.

Reads results/finetune_roberta.csv (the Phase-2
RoBERTa fine-tune, the project's production model) and plots per-entity F1 for
both partial- and exact-match, with the overall scores called out.

Run with:
    python figures/build_finetune_results.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "results" / "finetune_roberta.csv"
OUT = ROOT / "figures" / "finetune_results_by_entity.png"

sns.set_theme(style="whitegrid", font_scale=0.95)

ENTITY_ORDER = ["PERSON", "ORG", "LOC", "DATETIME", "QUANTITY", "CODE", "DEM", "MISC"]


def main():
    df = pd.read_csv(CSV)
    partial = df[df["mode"] == "partial"].set_index("entity_type")
    exact = df[df["mode"] == "exact"].set_index("entity_type")

    p_f1 = [partial.loc[e, "f1"] for e in ENTITY_ORDER]
    e_f1 = [exact.loc[e, "f1"] for e in ENTITY_ORDER]
    overall_p = partial.loc["_ALL", "f1"]
    overall_e = exact.loc["_ALL", "f1"]

    x = np.arange(len(ENTITY_ORDER))
    w = 0.38

    fig, ax = plt.subplots(figsize=(10, 5.2))
    b1 = ax.bar(x - w / 2, p_f1, w, label=f"Partial match  (overall {overall_p:.1%})",
                color="#27ae60", edgecolor="white", zorder=3)
    b2 = ax.bar(x + w / 2, e_f1, w, label=f"Exact match  (overall {overall_e:.1%})",
                color="#16a085", alpha=0.65, edgecolor="white", zorder=3)

    ax.axhline(overall_p, ls="--", lw=1.3, color="#27ae60", alpha=0.6, zorder=2)
    ax.text(len(ENTITY_ORDER) - 0.4, overall_p + 0.012,
            f"overall F1 = {overall_p:.1%}", ha="right", va="bottom",
            fontsize=9, color="#1e7e44", style="italic")

    for bars in (b1, b2):
        for r in bars:
            h = r.get_height()
            ax.text(r.get_x() + r.get_width() / 2, h + 0.008, f"{h:.0%}",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(ENTITY_ORDER, fontsize=9)
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Fine-tuned RoBERTa — per-entity F1 on the TAB test split (555 documents)\n"
        "Strong on the high-volume identifiers (PERSON, DATETIME, CODE); "
        "weakest on the sparse, fuzzy MISC class",
        fontsize=11, loc="left",
    )
    ax.legend(loc="upper right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()

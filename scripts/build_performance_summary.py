"""
Build the portfolio-headline performance figure.

Reads every results CSV the project has produced (gracefully skipping
phases that haven't been run yet) and writes a single multi-panel summary
PNG to figures/phase_summary.png — plus three individual panels for use
when embedding piece by piece into a portfolio site.

Run with:
    python figures/build_performance_summary.py

Outputs:
    figures/phase_summary.png            (multi-panel)
    figures/phase_overall_f1.png         (top-line bar chart)
    figures/phase_f1_by_entity.png       (per-entity-type grouped bars)
    figures/phase_precision_recall.png   (P/R scatter)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
sns.set_theme(style="whitegrid", font_scale=0.95)


# ---------------------------------------------------------------------------
# Load all available results
# ---------------------------------------------------------------------------
# Display order — matches the chronological story we tell in the writeup.
MODEL_ORDER = [
    "spacy_trf",
    "hf_bert_base_ner",
    "presidio_stock",
    "presidio_plus_case_number",
    "roberta_finetuned_tab",
    "legalbert_finetuned_tab",
    "ensemble_v1",
]
MODEL_LABELS = {
    "spacy_trf":                 "Phase 1 — spaCy trf",
    "hf_bert_base_ner":          "Phase 2 — HF bert-base-NER",
    "presidio_stock":            "Phase 2 — Presidio (stock)",
    "presidio_plus_case_number": "Phase 2 — Presidio + CASE_NUMBER",
    "roberta_finetuned_tab":     "Phase 2 — RoBERTa fine-tuned",
    "legalbert_finetuned_tab":   "Phase 6 — LegalBERT fine-tuned",
    "ensemble_v1":               "Phase 6 — Ensemble (3-way)",
}
MODEL_COLORS = {
    "spacy_trf":                 "#95a5a6",
    "hf_bert_base_ner":          "#3498db",
    "presidio_stock":            "#9b59b6",
    "presidio_plus_case_number": "#8e44ad",
    "roberta_finetuned_tab":     "#27ae60",
    "legalbert_finetuned_tab":   "#16a085",
    "ensemble_v1":               "#e74c3c",
}


def _safe_read(path: Path, default_model: Optional[str] = None) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"  (skipping — {path} not found)")
        return None
    df = pd.read_csv(path)
    if "model" not in df.columns and default_model:
        df.insert(0, "model", default_model)
    return df


def load_all() -> pd.DataFrame:
    sources = [
        (ROOT / "results" / "baseline_spacy.csv",                        "spacy_trf"),
        (ROOT / "results" / "baseline_huggingface.csv",         None),
        (ROOT / "results" / "baseline_presidio.csv",   None),
        (ROOT / "results" / "finetune_roberta.csv",  None),
        (ROOT / "results" / "finetune_legalbert.csv",    None),
        (ROOT / "results" / "null_ensemble.csv",     None),
    ]
    frames: List[pd.DataFrame] = []
    print("Loading results:")
    for path, default in sources:
        df = _safe_read(path, default)
        if df is not None:
            frames.append(df)
            print(f"  ✓ {path.relative_to(ROOT)}  ({len(df)} rows)")
    if not frames:
        raise SystemExit("No results CSVs found — run at least Phase 1 first.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Panels
# ---------------------------------------------------------------------------
def _models_available(df: pd.DataFrame) -> List[str]:
    """Return the models we have data for, in the canonical display order."""
    have = set(df["model"].unique())
    return [m for m in MODEL_ORDER if m in have]


def panel_overall_f1(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Headline bar chart — overall partial-match F1 across phases."""
    overall = df[(df["mode"] == "partial") & (df["entity_type"] == "_ALL")].copy()
    models = _models_available(overall)
    overall = overall.set_index("model").reindex(models).reset_index()

    bars = ax.barh(
        [MODEL_LABELS.get(m, m) for m in overall["model"]],
        overall["f1"] * 100,
        color=[MODEL_COLORS.get(m, "#7f8c8d") for m in overall["model"]],
        edgecolor="white",
    )
    for bar, f1 in zip(bars, overall["f1"]):
        ax.text(f1 * 100 + 0.7, bar.get_y() + bar.get_height() / 2,
                f"{f1:.1%}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlim(0, max(100, overall["f1"].max() * 100 + 8))
    ax.set_xlabel("Overall F1 (partial match, DIRECT + QUASI mentions on TAB test)")
    ax.set_title("Overall F1 across phases", fontweight="bold", fontsize=12, pad=10)
    ax.invert_yaxis()
    ax.set_axisbelow(True)


def panel_f1_by_entity(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Per-entity-type grouped bars — shows where each model wins or loses."""
    entity_order = ["PERSON", "ORG", "LOC", "DATETIME", "QUANTITY", "CODE", "DEM", "MISC"]
    per_type = df[(df["mode"] == "partial") & (df["entity_type"].isin(entity_order))]
    models = _models_available(per_type)

    pivot = (per_type
             .pivot_table(index="entity_type", columns="model", values="f1")
             .reindex(entity_order)
             .reindex(columns=models))

    x = np.arange(len(entity_order))
    width = 0.8 / max(1, len(models))
    for i, model in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        vals = (pivot[model].fillna(0).values * 100)
        ax.bar(x + offset, vals, width=width,
               color=MODEL_COLORS.get(model, "#7f8c8d"),
               label=MODEL_LABELS.get(model, model),
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(entity_order, rotation=0)
    ax.set_ylabel("F1 (%)")
    ax.set_title("F1 by entity type — where each model wins or loses",
                 fontweight="bold", fontsize=12, pad=10)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=8.5, ncols=1)
    ax.set_axisbelow(True)


# Pre-computed label offsets per model (in display pixels). Tuned to keep
# the legend points readable without overlapping in the typical run.
_PR_LABEL_OFFSETS = {
    "spacy_trf":                 (-90,  10),
    "hf_bert_base_ner":          ( 10, -14),
    "presidio_stock":            (-95, -20),
    "presidio_plus_case_number": ( 12,  10),
    "roberta_finetuned_tab":     ( 12,  14),
    "legalbert_finetuned_tab":   ( 12, -22),
    "ensemble_v1":               ( 12,  -8),
}


def panel_precision_recall(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Precision/recall scatter — every model's overall trade-off."""
    overall = df[(df["mode"] == "partial") & (df["entity_type"] == "_ALL")]
    models = _models_available(overall)

    for model in models:
        row = overall[overall["model"] == model].iloc[0]
        ax.scatter(
            row["recall"] * 100, row["precision"] * 100,
            s=200, c=MODEL_COLORS.get(model, "#7f8c8d"),
            edgecolors="white", linewidth=1.5, alpha=0.9, zorder=5,
        )
        # Compact label — strip "Phase X — " prefix for the scatter
        short_label = MODEL_LABELS.get(model, model).split(" — ", 1)[-1]
        # Trim "Phase 2 — " / "Phase 6 — " in case the split produced "Phase X"
        if short_label.lower().startswith("phase "):
            short_label = short_label.split(" — ", 1)[-1]
        offset = _PR_LABEL_OFFSETS.get(model, (8, 6))
        ax.annotate(
            short_label, (row["recall"] * 100, row["precision"] * 100),
            textcoords="offset points", xytext=offset,
            fontsize=8.5, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="none", alpha=0.85),
        )

    # F1 isocurves
    xx, yy = np.meshgrid(np.linspace(1, 100, 100), np.linspace(1, 100, 100))
    f1 = 2 * xx * yy / (xx + yy + 1e-9)
    cs = ax.contour(xx, yy, f1, levels=[30, 50, 70, 85],
                    colors="grey", alpha=0.35, linestyles=":")
    ax.clabel(cs, fmt="F1=%d", fontsize=7)

    ax.set_xlim(0, 110)
    ax.set_ylim(0, 110)
    ax.set_xlabel("Recall (% of gold entities caught)")
    ax.set_ylabel("Precision (% of predictions that were correct)")
    ax.set_title("Precision vs recall — overall (partial match)",
                 fontweight="bold", fontsize=12, pad=10)
    ax.set_axisbelow(True)


def panel_phase5_mention_recall(ax: plt.Axes) -> bool:
    """
    Render the Phase 5 mention-recall comparison.

    Reads results/null_coreference_summary.json. Returns
    True if the panel rendered (file existed); False otherwise so the
    caller can fall back to the progression callout.
    """
    summary_path = ROOT / "results" / "null_coreference_summary.json"
    if not summary_path.exists():
        return False

    summary = json.loads(summary_path.read_text())
    per_type = summary["per_type"]
    types = [r["entity_type"] for r in per_type]
    baseline = [r["baseline"] * 100 for r in per_type]
    coref    = [r["with_coref"] * 100 for r in per_type]

    x = np.arange(len(types))
    width = 0.4
    ax.bar(x - width/2, baseline, width=width,
           color="#95a5a6", edgecolor="white", label="Baseline (no coref)")
    ax.bar(x + width/2, coref, width=width,
           color="#27ae60", edgecolor="white", label="+ Phase 5 CorefExtender")

    ax.set_xticks(x)
    ax.set_xticklabels(types, rotation=0, fontsize=9)
    ax.set_ylabel("Mention recall (%)")
    ax.set_title(
        f"Phase 5 — mention recall (TAB test, "
        f"{summary['baseline']['n_entities']:,} entities)",
        fontweight="bold", fontsize=12, pad=10,
    )
    ax.set_ylim(0, 105)
    ax.legend(loc="upper center", fontsize=8.5, framealpha=0.95)

    # Annotate the bars where there's an actual delta worth seeing.
    for i, (b, c) in enumerate(zip(baseline, coref)):
        if abs(c - b) >= 0.05:  # only annotate non-zero deltas
            ax.annotate(
                f"+{c-b:.2f}pp", xy=(x[i], max(b, c) + 1),
                ha="center", fontsize=7.5, fontweight="bold",
                color="#27ae60",
            )

    # Headline finding caption
    macro_b = summary["baseline"]["macro_recall"]
    macro_c = summary["with_coref"]["macro_recall"]
    ax.text(
        0.5, -0.18,
        f"Macro recall: {macro_b:.4f} → {macro_c:.4f}   "
        f"({(macro_c - macro_b) * 100:+.2f} pp · essentially zero — "
        "spaCy already at the ceiling on PERSON/ORG)",
        transform=ax.transAxes, ha="center", fontsize=9, style="italic",
        color="#586069",
    )
    return True


def panel_progression_callout(df: pd.DataFrame, ax: plt.Axes) -> None:
    """Compact narrative callout — F1 lift + project phases at a glance."""
    overall = df[(df["mode"] == "partial") & (df["entity_type"] == "_ALL")]
    models = _models_available(overall)

    ax.axis("off")
    first = overall[overall["model"] == models[0]].iloc[0]
    best_row = overall.loc[overall["f1"].idxmax()]

    # Title
    ax.text(0.02, 0.98, "Project progression",
            fontsize=13, fontweight="bold", va="top", transform=ax.transAxes)

    # Headline metrics — two rows
    metrics = [
        f"Baseline (Phase 1 — spaCy en_core_web_trf):  F1 = {first['f1']:.1%}",
        f"Best:  {MODEL_LABELS[best_row['model']]}  →  F1 = {best_row['f1']:.1%}",
        f"Absolute lift over baseline: +{(best_row['f1'] - first['f1']) * 100:.1f} percentage points",
    ]
    ax.text(0.02, 0.86, "\n".join(metrics),
            fontsize=10, va="top", transform=ax.transAxes,
            family="monospace")

    # Phases — compact list
    phase_text = (
        "Phases shipped:\n"
        "  1. Proof of concept — measure off-the-shelf gap on TAB\n"
        "  2. Baseline comparison + RoBERTa fine-tune\n"
        "  3. Two-variant production pipeline (Lite + Pro)\n"
        "  4. Pseudonymisation + round-trip restore\n"
        "  5. Coreference-aware span extension\n"
        "  6. Domain backbone (LegalBERT) + ensemble"
    )
    ax.text(0.02, 0.62, phase_text,
            fontsize=9.5, va="top", transform=ax.transAxes,
            family="monospace")

    # Read the ensemble note dynamically — only show if the ensemble underperformed
    if "ensemble_v1" in models:
        ens = overall[overall["model"] == "ensemble_v1"].iloc[0]
        best_single = overall[overall["model"] == "roberta_finetuned_tab"]
        if len(best_single) and ens["f1"] < best_single["f1"].iloc[0]:
            note = (
                "Note: ensemble (min_votes=1) underperforms the best single\n"
                "model — Presidio's low-precision ORG predictions flood the\n"
                "union with false positives. Try min_votes=2 (intersection)\n"
                "or drop Presidio from the member set for a fairer test."
            )
            ax.text(0.02, 0.20, note,
                    fontsize=9, va="top", style="italic",
                    color="#735c0f", transform=ax.transAxes)

    # Footer
    ax.text(0.02, 0.02,
            "Eval: 555 ECHR docs · character-offset span overlap · DIRECT+QUASI · partial match",
            fontsize=8, va="bottom", style="italic",
            color="#586069", transform=ax.transAxes)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_all()
    out_dir = ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    # — Standalone panels for piecemeal embedding —
    print("\nWriting standalone panels:")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    panel_overall_f1(df, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_overall_f1.png", dpi=130, bbox_inches="tight")
    print(f"  → figures/phase_overall_f1.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    panel_f1_by_entity(df, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_f1_by_entity.png", dpi=130, bbox_inches="tight")
    print(f"  → figures/phase_f1_by_entity.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 6))
    panel_precision_recall(df, ax)
    fig.tight_layout()
    fig.savefig(out_dir / "phase_precision_recall.png", dpi=130, bbox_inches="tight")
    print(f"  → figures/phase_precision_recall.png")
    plt.close(fig)

    # Standalone Phase 5 mention-recall panel — only if results exist
    fig, ax = plt.subplots(figsize=(10, 5))
    if panel_phase5_mention_recall(ax):
        fig.tight_layout()
        fig.savefig(out_dir / "phase5_mention_recall.png", dpi=130, bbox_inches="tight")
        print(f"  → figures/phase5_mention_recall.png")
    plt.close(fig)

    # — Multi-panel headline figure —
    print("\nWriting multi-panel summary:")
    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22)

    ax_overall   = fig.add_subplot(gs[0, 0])
    ax_byentity  = fig.add_subplot(gs[0, 1])
    ax_pr        = fig.add_subplot(gs[1, 0])
    ax_bottom_right = fig.add_subplot(gs[1, 1])

    panel_overall_f1(df, ax_overall)
    panel_f1_by_entity(df, ax_byentity)
    panel_precision_recall(df, ax_pr)
    # Prefer the Phase 5 panel when we have data; otherwise show the
    # narrative callout.
    if not panel_phase5_mention_recall(ax_bottom_right):
        ax_bottom_right.clear()
        panel_progression_callout(df, ax_bottom_right)

    fig.suptitle(
        "Legal Text Anonymisation — Performance Across Phases",
        fontsize=16, fontweight="bold", y=0.995,
    )
    fig.savefig(out_dir / "phase_summary.png", dpi=130, bbox_inches="tight")
    print(f"  → figures/phase_summary.png")
    plt.close(fig)

    print("\nDone.")


if __name__ == "__main__":
    main()

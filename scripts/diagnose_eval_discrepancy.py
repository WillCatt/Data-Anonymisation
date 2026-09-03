"""
Reconcile the two LegalBERT numbers.

The repo reported two different partial-F1 values for the same fine-tuned
LegalBERT checkpoint on the same 555-document test split:

    results/finetune_legalbert.csv      0.8486   (notebook 13)
    results/combiner_comparison.csv     0.8538   (cache replay)

0.5 pp apart, and they straddle RoBERTa's 0.8510 — so which CSV you quoted
decided which model "won". That is not a rounding difference; it needed a
cause.

The cause is an inference-config mismatch, not a metric bug:

    notebook 13            make_finetuned_predictor(..., max_length=384, stride=64)
    cache_predictions.py   make_finetuned_predictor(...)   -> default max_length=512

Long documents are sliding-windowed. A smaller window means more chunks, more
window boundaries, and more boundary-fragment spans that the matcher counts as
false positives. Both paths see identical gold (20,809 mentions); only the
prediction count differs.

This script re-runs the checkpoint over a sample of the test split at both
window sizes and reports the gap, so the explanation is measured rather than
asserted.

Run from the repo root:
    python scripts/diagnose_eval_discrepancy.py --sample 80
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import warnings
warnings.filterwarnings("ignore")

LEGALBERT_DIR = REPO / "models/legalbert-tab/final"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", type=int, default=80, help="documents to score (0 = all 555)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-dir", default=str(LEGALBERT_DIR))
    ap.add_argument("--windows", type=int, nargs="+", default=[384, 512])
    ap.add_argument("--stride", type=int, default=64)
    args = ap.parse_args()

    from anonymisation.data import load_tab
    from anonymisation.evaluation import evaluate_document, merge_results
    from anonymisation.predictors import make_finetuned_predictor
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    docs = list(load_tab()["test"])
    if args.sample:
        docs = docs[: args.sample]

    print(f"Checkpoint : {args.model_dir}")
    print(f"Documents  : {len(docs)}   device: {args.device}   stride: {args.stride}")
    print(f"Windows    : {args.windows}\n")

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)

    rows = []
    for max_length in args.windows:
        predict = make_finetuned_predictor(
            model, tok, device=args.device, max_length=max_length, stride=args.stride
        )
        started = time.time()
        merged = merge_results([evaluate_document(predict, d, "partial") for d in docs])
        r = merged["_ALL"]
        rows.append((max_length, r))
        print(f"  max_length={max_length:4d}   F1 = {r.f1:.4f}   "
              f"P = {r.precision:.4f}  R = {r.recall:.4f}   "
              f"tp={r.tp} fp={r.fp} fn={r.fn}   preds={r.tp + r.fp}   "
              f"({time.time() - started:.0f}s)")

    if len(rows) >= 2:
        (w_a, a), (w_b, b) = rows[0], rows[1]
        print(f"\n  gap ({w_b} - {w_a}) = {b.f1 - a.f1:+.4f} F1")
        print(f"  predictions: {a.tp + a.fp:,} -> {b.tp + b.fp:,}  ({(b.tp + b.fp) - (a.tp + a.fp):+,})")
        print(f"  of which FP: {a.fp:,} -> {b.fp:,}  ({b.fp - a.fp:+,})")
        print(f"  gold is identical both ways: {a.tp + a.fn:,} vs {b.tp + b.fn:,}")
        print("\n  A smaller window fragments spans at chunk boundaries; the extra")
        print("  fragments are scored as false positives, which costs precision.")

    print("\nFix: pin inference config next to the checkpoint rather than letting a")
    print("shared helper's default silently disagree with the training notebook.")


if __name__ == "__main__":
    main()

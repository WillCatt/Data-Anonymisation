"""
Ablation: how much does inference-time data preparation move the score?

Motivation
----------
Reconciling the two LegalBERT numbers turned up something awkward: the same
checkpoint scored 0.8486 with `max_length=384` and 0.8538 with `512`. That is
0.5 pp from a tokenisation setting — **larger than the entire measured gap
between the two fine-tuned backbones** (0.0022, and not statistically
distinguishable).

That is worth a proper sweep rather than a footnote. Long documents are
processed as overlapping sliding windows, so two settings control how the text
is cut up before the model ever sees it:

  max_length  window size in tokens
  stride      tokens of overlap between consecutive windows

Neither requires retraining — they are pure inference-time data preparation.
So this sweep is free: it replays existing checkpoints over the test split.

What it produces
----------------
For every (max_length, stride) config: micro-F1 with a bootstrap 95% interval,
and a paired test against the baseline config the model was trained with. Per
document counts are cached so the bootstrap can be re-run without re-running
inference.

Run from the repo root:
    python scripts/ablate_inference_config.py                    # RoBERTa, 3x3 grid
    python scripts/ablate_inference_config.py --model legalbert
    python scripts/ablate_inference_config.py --windows 384 512 --strides 64
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

MODEL_DIRS = {
    "roberta": REPO / "models/roberta-tab/final",
    "legalbert": REPO / "models/legalbert-tab/final",
}
COUNTS_PATH = REPO / "results/ablation_inference_counts.json"
OUT_CSV = REPO / "results/ablation_inference.csv"

# What the fine-tune notebooks trained and evaluated with. Everything is
# reported relative to this so the sweep answers "does deviating help?".
BASELINE = (384, 64)


def micro_f1(tp, fp, fn):
    denom = 2.0 * tp + fp + fn
    return np.divide(2.0 * tp, denom, out=np.zeros_like(denom, dtype=float), where=denom > 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="roberta", choices=list(MODEL_DIRS))
    ap.add_argument("--windows", type=int, nargs="+", default=[256, 384, 512])
    ap.add_argument("--strides", type=int, nargs="+", default=[0, 64, 128])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=None, help="documents (default: all 555)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--mode", default="partial", choices=["partial", "exact"])
    args = ap.parse_args()

    from anonymisation.data import load_tab
    from anonymisation.evaluation import extract_gold_entities, score_spans
    from anonymisation.predictors import make_finetuned_predictor
    from transformers import AutoModelForTokenClassification, AutoTokenizer

    model_dir = MODEL_DIRS[args.model]
    if not model_dir.exists():
        sys.exit(f"No checkpoint at {model_dir}")

    docs = list(load_tab()["test"])
    if args.limit:
        docs = docs[: args.limit]
    gold = [extract_gold_entities(d) for d in docs]

    configs = [(w, s) for w in args.windows for s in args.strides if s < w]
    print(f"Model      : {args.model}  ({model_dir})")
    print(f"Documents  : {len(docs)} rows over {len({d['doc_id'] for d in docs})} unique"
          f"   device: {args.device}   match: {args.mode}")
    print(f"Configs    : {len(configs)}  = {args.windows} x {args.strides}")
    print(f"Baseline   : max_length={BASELINE[0]}, stride={BASELINE[1]} (what the notebooks trained with)\n")

    # Reuse any counts already computed for this model/mode/doc-count.
    store = json.loads(COUNTS_PATH.read_text()) if COUNTS_PATH.exists() else {}
    scope = f"{args.model}|{args.mode}|{len(docs)}"
    store.setdefault(scope, {})

    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))

    for max_length, stride in configs:
        key = f"{max_length}|{stride}"
        if key in store[scope]:
            print(f"  max_length={max_length:4d} stride={stride:4d}  (cached)")
            continue
        predict = make_finetuned_predictor(
            model, tok, device=args.device, max_length=max_length, stride=stride
        )
        started = time.time()
        counts = []
        for d, g in zip(docs, gold):
            r = score_spans(predict(d["text"]), g, args.mode)["_ALL"]
            counts.append([r.tp, r.fp, r.fn])
        store[scope][key] = counts
        COUNTS_PATH.write_text(json.dumps(store))
        arr = np.array(counts)
        f1 = float(micro_f1(arr[:, 0].sum(), arr[:, 1].sum(), arr[:, 2].sum()))
        print(f"  max_length={max_length:4d} stride={stride:4d}  F1 = {f1:.4f}"
              f"   ({time.time() - started:.0f}s)", flush=True)

    # ── bootstrap every config on ONE shared resample matrix ─────────────
    # Cluster by doc_id: TAB's test rows are annotator-annotation pairs over far
    # fewer unique documents (555 rows -> 127 docs), and resampling rows treats
    # several views of one judgment as independent draws.
    doc_ids = [d["doc_id"] for d in docs]
    order, groups = [], {}
    for i, did in enumerate(doc_ids):
        if did not in groups:
            groups[did] = []
            order.append(did)
        groups[did].append(i)
    clusters = [groups[d] for d in order]
    n_clusters = len(clusters)

    rng = np.random.default_rng(args.seed)
    idx = rng.integers(0, n_clusters, size=(args.iters, n_clusters))

    def boot(counts):
        a = np.array(counts)
        # collapse annotator rows into their document before resampling
        c = np.array([a[rows].sum(axis=0) for rows in clusters])
        tp, fp, fn = c[:, 0], c[:, 1], c[:, 2]
        return micro_f1(tp[idx].sum(1), fp[idx].sum(1), fn[idx].sum(1))

    base_key = f"{BASELINE[0]}|{BASELINE[1]}"
    base_dist = boot(store[scope][base_key]) if base_key in store[scope] else None

    rows = []
    for max_length, stride in configs:
        counts = store[scope][f"{max_length}|{stride}"]
        a = np.array(counts)
        point = float(micro_f1(a[:, 0].sum(), a[:, 1].sum(), a[:, 2].sum()))
        dist = boot(counts)
        lo, hi = np.percentile(dist, [2.5, 97.5])
        row = {
            "model": args.model, "mode": args.mode,
            "max_length": max_length, "stride": stride,
            "f1": point, "ci_low": lo, "ci_high": hi,
            "tp": int(a[:, 0].sum()), "fp": int(a[:, 1].sum()), "fn": int(a[:, 2].sum()),
            "n_predictions": int(a[:, 0].sum() + a[:, 1].sum()),
        }
        if base_dist is not None:
            diff = dist - base_dist
            d_lo, d_hi = np.percentile(diff, [2.5, 97.5])
            row.update({
                "delta_vs_baseline": point - float(micro_f1(
                    np.array(store[scope][base_key])[:, 0].sum(),
                    np.array(store[scope][base_key])[:, 1].sum(),
                    np.array(store[scope][base_key])[:, 2].sum())),
                "delta_ci_low": d_lo, "delta_ci_high": d_hi,
                "distinguishable_from_baseline": not (d_lo <= 0.0 <= d_hi),
            })
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)

    print(f"\n{'win':>5} {'stride':>7} {'F1':>8} {'95% CI':>18} {'Δ vs base':>11} {'real?':>7} {'preds':>8}")
    print("─" * 70)
    for _, r in df.iterrows():
        tag = " ←baseline" if (r.max_length, r.stride) == BASELINE else ""
        delta = f"{r.get('delta_vs_baseline', float('nan')):+.4f}" if "delta_vs_baseline" in r else "—"
        real = ("yes" if r.get("distinguishable_from_baseline") else "no") if "delta_vs_baseline" in r else "—"
        print(f"{int(r.max_length):5d} {int(r.stride):7d} {r.f1:8.4f}"
              f"  [{r.ci_low:.4f},{r.ci_high:.4f}] {delta:>11} {real:>7} {r.n_predictions:8,}{tag}")

    spread = df.f1.max() - df.f1.min()
    print(f"\n  spread across configs: {spread:.4f} F1")
    print(f"  for scale, roberta - legalbert was 0.0022 (not distinguishable)")
    print(f"\n→ {OUT_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()

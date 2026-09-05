"""
Score a fine-tuned checkpoint into the project's standard results CSV.

Why this exists
---------------
Every headline table and figure reads a CSV shaped
`model,mode,entity_type,precision,recall,f1,tp,fp,fn`, and until now the only
things that produced one were the fine-tuning notebooks — which also *retrain*
the model on the way past. That made scoring an existing checkpoint impossible
without either a notebook run or a fresh training run, and a fresh training run
is not the same model (the seed sweep in notebook 16 puts that spread at 0.0249
F1, larger than most differences this project reports).

So: point this at a checkpoint, get the same numbers the notebooks would have
produced, without touching the weights.

The inference window is an explicit argument and is recorded in the output, for
the reason documented in the README: the same checkpoint scores 0.8486 at
`max_length=384` and 0.8538 at 512, and forgetting which one produced a number
is how this project once ended up with two incompatible figures for one model.

Run from the repo root:
    python scripts/score_checkpoint.py \\
        --checkpoint models/longformer-tab/final \\
        --name longformer_finetuned_tab \\
        --max-length 4096 --stride 0 \\
        --out results/finetune_longformer.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, help="directory holding the fine-tuned model")
    ap.add_argument("--name", required=True, help="value for the CSV's `model` column")
    ap.add_argument("--out", required=True, help="where to write the results CSV")
    ap.add_argument("--tokenizer", default=None,
                    help="defaults to the checkpoint; give the base model if it has no tokenizer")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=None, help="fewer documents (smoke test)")
    ap.add_argument("--device", default=None, help="default: auto (cuda > mps > cpu)")
    args = ap.parse_args()

    from transformers import AutoModelForTokenClassification, AutoTokenizer

    from anonymisation.data import load_tab
    from anonymisation.device import best_device
    from anonymisation.evaluation import (
        evaluate_document, merge_results, results_to_dataframe,
    )
    from anonymisation.predictors import make_finetuned_predictor

    device = args.device or best_device()[0]
    tokenizer_src = args.tokenizer or args.checkpoint
    byte_bpe = any(k in str(tokenizer_src).lower() for k in ("roberta", "longformer"))
    tok_kwargs = {"add_prefix_space": True} if byte_bpe else {}

    print(f"Checkpoint : {args.checkpoint}")
    print(f"Tokenizer  : {tokenizer_src}")
    print(f"Device     : {device}   window: {args.max_length}/{args.stride}")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_src, **tok_kwargs)
    model = AutoModelForTokenClassification.from_pretrained(args.checkpoint)
    predict = make_finetuned_predictor(
        model, tokenizer, device=device,
        max_length=args.max_length, stride=args.stride,
    )

    docs = list(load_tab()[args.split])
    if args.limit:
        docs = docs[: args.limit]
    print(f"Scoring {len(docs)} {args.split} rows\n")

    merged = {}
    for mode in ("partial", "exact"):
        started = time.time()
        per_doc = []
        for i, doc in enumerate(docs, 1):
            per_doc.append(evaluate_document(predict, doc, mode=mode))
            if i % 50 == 0:
                rate = i / (time.time() - started)
                print(f"  {mode}: {i}/{len(docs)}  ({rate:.1f} docs/s)", flush=True)
        merged[mode] = merge_results(per_doc)
        overall = merged[mode]["_ALL"]
        print(f"  {mode}: F1 {overall.f1:.4f}  P {overall.precision:.4f}  "
              f"R {overall.recall:.4f}  in {time.time() - started:.0f}s\n")

    frame = results_to_dataframe(merged)
    frame.insert(0, "model", args.name)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    print(f"→ {out}  ({len(frame)} rows, window {args.max_length}/{args.stride})")


if __name__ == "__main__":
    main()

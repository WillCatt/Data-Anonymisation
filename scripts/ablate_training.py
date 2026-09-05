"""
Ablation: seed variance and hyperparameter sensitivity for the fine-tunes.

The question this answers
------------------------
The project reports RoBERTa 0.851 and LegalBERT 0.849 and calls it a tie. The
paired bootstrap agrees (delta = 0.0022, p = 0.43). But there is a prior
question nobody asked: **how much does the same recipe move if you only change
the random seed?**

If seed-to-seed spread is comparable to, or larger than, the gap between the
two backbones, then the backbone comparison was never measuring the backbone.
That would not be a disappointing result — it would be the most useful thing
in the whole model section, because it establishes the noise floor that every
other claim has to clear.

Method
------
Retrain from the same recipe as `notebooks/05_finetune_roberta.ipynb`, varying
one factor at a time, and score every run with the project's own span-level
metric (NOT the seqeval token-level metric the Trainer logs — that is a
different number and mixing them is how you end up with two incompatible
results for one model).

Per-document counts are written out so `bootstrap_ci.py`-style intervals can be
computed on any run without retraining.

Selection discipline
--------------------
Validation and test are both scored, and **validation is the one to select on**.
Picking a learning rate by test F1 and then reporting that same test F1 is how
you talk yourself into an improvement that isn't there. For a pure seed-variance
study there is no selection, so either split is honest — but the discipline is
built in so the LR sweep can't quietly break it.

Checkpoints are deleted after scoring unless --keep-checkpoints; five runs of
RoBERTa-base is several GB otherwise.

Run on the GPU box (~10-15 min per run):
    python scripts/ablate_training.py --seeds 42 43 44 45 46
    python scripts/ablate_training.py --seeds 42 --lrs 1e-5 2e-5 5e-5
    python scripts/ablate_training.py --base-model nlpaueb/legal-bert-base-uncased --seeds 42 43 44
    python scripts/ablate_training.py --limit-train 100 250 550 1112   # learning curve (one per run)

One deviation from notebook 05, stated so the numbers can be reconciled: the
notebook selects its best checkpoint by seqeval token-level F1, this selects by
validation loss. Loss avoids a network fetch for the seqeval metric on a
headless box, and it is applied identically to every run, so comparisons within
this sweep are clean. Absolute values may sit a hair off the notebook's 0.851.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

OUT_CSV = REPO / "results/ablation_training.csv"
COUNTS_PATH = REPO / "results/ablation_training_counts.json"


def micro_f1(tp, fp, fn) -> float:
    d = 2 * tp + fp + fn
    return (2 * tp / d) if d else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-model", default="roberta-base")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--lrs", type=float, nargs="+", default=[2e-5])
    ap.add_argument("--epochs", type=int, nargs="+", default=[3])
    ap.add_argument("--batch-size", type=int, default=None, help="default: 16 on CUDA, 8 otherwise")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="gradient accumulation steps. A long-window model may only fit a "
                         "batch of 1, and comparing it against a run at batch 8 would vary "
                         "two things at once; set this so batch x accum matches.")
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--device", default=None, help="default: auto (cuda > mps > cpu)")
    ap.add_argument("--keep-checkpoints", action="store_true")
    ap.add_argument("--limit-train", type=int, default=None,
                    help="train on only the first N docs — also the learning-curve axis")
    ap.add_argument("--limit-eval", type=int, default=None, help="fewer eval docs (smoke test)")
    args = ap.parse_args()

    import torch
    from datasets import Dataset
    from transformers import (
        AutoModelForTokenClassification, AutoTokenizer,
        DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed,
    )

    from anonymisation.data import load_tab
    from anonymisation.device import best_device
    from anonymisation.evaluation import extract_gold_entities, score_spans
    from anonymisation.iob import BIO_LABELS, ID_TO_LABEL, LABEL_TO_ID, gold_spans_for_training, offsets_to_bio
    from anonymisation.predictors import make_finetuned_predictor

    device = args.device or best_device()[0]
    batch_size = args.batch_size or (16 if device == "cuda" else 8)
    short_name = args.base_model.split("/")[-1]

    print(f"Base model : {args.base_model}")
    effective = batch_size * args.grad_accum
    print(f"Device     : {device}   batch: {batch_size}"
          f"{f' x {args.grad_accum} accum = {effective}' if args.grad_accum > 1 else ''}"
          f"   window: {args.max_length}/{args.stride}")
    print(f"Grid       : seeds={args.seeds}  lrs={args.lrs}  epochs={args.epochs}\n")

    dataset = load_tab()
    # RoBERTa is byte-BPE and needs add_prefix_space for word-aligned tokens;
    # BERT-family WordPiece tokenizers reject the argument. Longformer is a
    # RoBERTa tokenizer under a different name, so it belongs on this side of
    # the fence — without it its tokenisation would differ from the RoBERTa run
    # it is being compared against, for no reason anyone intended.
    _byte_bpe = ("roberta", "longformer")
    tok_kwargs = ({"add_prefix_space": True}
                  if any(k in args.base_model.lower() for k in _byte_bpe) else {})
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, **tok_kwargs)

    def tokenise_and_align(doc):
        spans = gold_spans_for_training(doc)
        enc = tokenizer(
            doc["text"], return_offsets_mapping=True, return_overflowing_tokens=True,
            truncation=True, max_length=args.max_length, stride=args.stride, padding=False,
        )
        return [{
            "input_ids": enc["input_ids"][i],
            "attention_mask": enc["attention_mask"][i],
            "labels": offsets_to_bio(enc["offset_mapping"][i], spans, word_ids=enc.word_ids(batch_index=i)),
        } for i in range(len(enc["input_ids"]))]

    def build(split, limit=None):
        docs = list(dataset[split])[: limit] if limit else list(dataset[split])
        return Dataset.from_list([ex for d in docs for ex in tokenise_and_align(d)])

    n_train_docs = args.limit_train or len(dataset["train"])
    print("Tokenising …", flush=True)
    train_ds = build("train", args.limit_train)
    val_ds = build("validation", args.limit_eval)
    print(f"  train chunks: {len(train_ds):,}   val chunks: {len(val_ds):,}\n")

    eval_docs = {
        "validation": list(dataset["validation"])[: args.limit_eval] if args.limit_eval else list(dataset["validation"]),
        "test": list(dataset["test"])[: args.limit_eval] if args.limit_eval else list(dataset["test"]),
    }
    eval_gold = {k: [extract_gold_entities(d) for d in v] for k, v in eval_docs.items()}

    done = set()
    if OUT_CSV.exists():
        prev = pd.read_csv(OUT_CSV)
        done = {(r.base_model, r.seed, r.lr, r.epochs, r.n_train_docs) for r in prev.itertuples()}
    counts_store = json.loads(COUNTS_PATH.read_text()) if COUNTS_PATH.exists() else {}
    rows = []

    for seed in args.seeds:
        for lr in args.lrs:
            for epochs in args.epochs:
                if (args.base_model, seed, lr, epochs, n_train_docs) in done:
                    print(f"skip (already done): seed={seed} lr={lr} epochs={epochs}")
                    continue
                print(f"── seed={seed}  lr={lr}  epochs={epochs} " + "─" * 30, flush=True)
                set_seed(seed)
                model = AutoModelForTokenClassification.from_pretrained(
                    args.base_model, num_labels=len(BIO_LABELS),
                    id2label=ID_TO_LABEL, label2id=LABEL_TO_ID,
                )
                tmp = Path(tempfile.mkdtemp(prefix=f"abl_{short_name}_{seed}_"))
                targs = TrainingArguments(
                    output_dir=str(tmp), num_train_epochs=epochs, learning_rate=lr,
                    per_device_train_batch_size=batch_size, per_device_eval_batch_size=batch_size,
                    gradient_accumulation_steps=args.grad_accum,
                    weight_decay=0.01, eval_strategy="epoch", save_strategy="epoch",
                    logging_steps=50, load_best_model_at_end=True, metric_for_best_model="loss",
                    save_total_limit=1, seed=seed, report_to="none", fp16=(device == "cuda"),
                )
                trainer = Trainer(
                    model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
                    tokenizer=tokenizer, data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
                )
                started = time.time()
                trainer.train()
                train_mins = (time.time() - started) / 60

                # Score with the PROJECT's span metric, at the same inference
                # config, so runs differ only by what we varied.
                predict = make_finetuned_predictor(
                    trainer.model, tokenizer, device=device,
                    max_length=args.max_length, stride=args.stride,
                )
                row = {"base_model": args.base_model, "seed": seed, "lr": lr, "epochs": epochs,
                       "batch_size": batch_size, "grad_accum": args.grad_accum,
                       "max_length": args.max_length, "stride": args.stride,
                       "n_train_docs": n_train_docs, "n_train_chunks": len(train_ds),
                       "train_minutes": round(train_mins, 1)}
                for split in ("validation", "test"):
                    counts = []
                    for d, g in zip(eval_docs[split], eval_gold[split]):
                        r = score_spans(predict(d["text"]), g, "partial")["_ALL"]
                        counts.append([r.tp, r.fp, r.fn])
                    a = np.array(counts)
                    f1 = micro_f1(int(a[:, 0].sum()), int(a[:, 1].sum()), int(a[:, 2].sum()))
                    row[f"f1_{split}"] = f1
                    counts_store[f"{args.base_model}|{seed}|{lr}|{epochs}|{n_train_docs}|{split}"] = counts
                    print(f"   {split:10s} partial F1 = {f1:.4f}")
                print(f"   trained in {train_mins:.1f} min")

                rows.append(row)
                pd.DataFrame(rows if not OUT_CSV.exists() else
                             pd.concat([pd.read_csv(OUT_CSV), pd.DataFrame([row])], ignore_index=True).to_dict("records")
                             ).to_csv(OUT_CSV, index=False)
                COUNTS_PATH.write_text(json.dumps(counts_store))

                if args.keep_checkpoints:
                    print(f"   checkpoints kept at {tmp}")
                else:
                    del trainer, model
                    if device == "cuda":
                        torch.cuda.empty_cache()
                    shutil.rmtree(tmp, ignore_errors=True)

    if OUT_CSV.exists():
        df = pd.read_csv(OUT_CSV)
        print("\n" + "=" * 60)
        print(df.to_string(index=False))
        for model_name, g in df.groupby("base_model"):
            seeds_only = g[(g.lr == 2e-5) & (g.epochs == 3) & (g.n_train_docs == g.n_train_docs.max())]
            if len(seeds_only) > 1:
                s = seeds_only.f1_test
                print(f"\n{model_name} — seed variance over {len(s)} runs (lr=2e-5, 3 epochs):")
                print(f"  test F1: mean {s.mean():.4f}  sd {s.std(ddof=1):.4f}"
                      f"  min {s.min():.4f}  max {s.max():.4f}  spread {s.max() - s.min():.4f}")
                print(f"  for scale: roberta - legalbert = 0.0022, and NOT distinguishable.")
                if s.max() - s.min() > 0.0022:
                    print("  >> Seed spread EXCEEDS the backbone gap. The backbone comparison")
                    print("     was inside the noise of the training procedure itself.")
        print(f"\n→ {OUT_CSV.relative_to(REPO)}")


if __name__ == "__main__":
    main()

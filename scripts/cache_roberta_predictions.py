"""
Add the RoBERTa fine-tune's predictions to the existing prediction cache.

`cache_predictions.py` cached spaCy / LegalBERT / Presidio but not RoBERTa —
which is the project's headline model, so no paired test could include it.
This script backfills a "roberta" key onto every record in
`results/predictions_cache.json`, leaving the existing keys untouched.

Once it finishes, `scripts/bootstrap_ci.py` picks RoBERTa up automatically and
the RoBERTa-vs-LegalBERT paired comparison becomes available.

Resumable: writes every `--save-every` documents and skips records that already
carry a "roberta" key, so it is safe to interrupt and re-run.

Run from the repo root (~4 min on CPU for 555 documents):
    python scripts/cache_roberta_predictions.py
    python scripts/cache_roberta_predictions.py --device mps --limit 50
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

ROBERTA_DIR = REPO / "models/roberta-tab/final"
CACHE_PATH = REPO / "results/predictions_cache.json"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # CPU by default, matching cache_predictions.py: MPS leaked memory badly over
    # the 555-document loop there (1.7 -> 0.04 docs/s). Also keeps this backend
    # identical to the one the other cached models used, so the paired tests
    # compare models rather than hardware.
    ap.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    ap.add_argument("--key", default="roberta", help="cache key to write (default: roberta)")
    ap.add_argument("--model-dir", default=str(ROBERTA_DIR))
    ap.add_argument("--limit", type=int, default=None, help="only process the first N docs (smoke test)")
    ap.add_argument("--save-every", type=int, default=25)
    # Explicit, and matched to the other cached models. The fine-tune notebooks
    # use 384; this path uses 512. Mixing the two inside one cache would make
    # the paired tests compare window sizes rather than models.
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true", help="recompute even where the key already exists")
    args = ap.parse_args()

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        sys.exit(f"No checkpoint at {model_dir}")
    if not CACHE_PATH.exists():
        sys.exit(f"No cache at {CACHE_PATH}. Run scripts/cache_predictions.py first.")

    from anonymisation.data import load_tab
    from anonymisation.predictors import make_finetuned_predictor

    cache = json.loads(CACHE_PATH.read_text())
    test_docs = list(load_tab()["test"])
    if len(cache) != len(test_docs):
        sys.exit(f"Cache has {len(cache)} records but TAB test has {len(test_docs)} docs — refusing to guess.")

    # The cache is index-aligned to the test split; verify rather than trust it.
    for i, (entry, doc) in enumerate(zip(cache, test_docs)):
        if entry["doc_id"] != doc["doc_id"]:
            sys.exit(f"Cache/TAB misalignment at index {i}: {entry['doc_id']} != {doc['doc_id']}")
    print(f"Cache aligned with TAB test split: {len(cache)} documents ✓")

    todo = [i for i, e in enumerate(cache) if args.overwrite or args.key not in e]
    if args.limit:
        todo = todo[: args.limit]
    if not todo:
        print(f"Nothing to do — every record already has a '{args.key}' key.")
        return
    print(f"To predict: {len(todo)} documents (key '{args.key}', device {args.device})")

    from transformers import AutoModelForTokenClassification, AutoTokenizer
    print(f"Loading {model_dir} …", flush=True)
    tok = AutoTokenizer.from_pretrained(str(model_dir))
    model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    predict = make_finetuned_predictor(
        model, tok, device=args.device, max_length=args.max_length, stride=args.stride
    )

    started = time.time()
    for n, i in enumerate(todo, 1):
        cache[i][args.key] = [list(s) for s in predict(test_docs[i]["text"])]
        if n % args.save_every == 0 or n == len(todo):
            CACHE_PATH.write_text(json.dumps(cache))
            rate = n / (time.time() - started)
            eta = (len(todo) - n) / rate if rate else 0
            print(f"  {n}/{len(todo)}  {rate:.2f} docs/s  ETA {eta / 60:.1f} min", flush=True)

    spans = sum(len(e.get(args.key, [])) for e in cache)
    print(f"\nDone in {(time.time() - started) / 60:.1f} min — {spans:,} spans under '{args.key}'.")
    print("Now run:  python scripts/bootstrap_ci.py --mode both")


if __name__ == "__main__":
    main()

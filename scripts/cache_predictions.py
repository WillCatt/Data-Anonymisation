"""
Phase 6.2 — cache raw per-model predictions over the TAB test set.

Runs the three Phase-6 predictors (spaCy-trf, LegalBERT fine-tune, Presidio)
over every test document ONCE and dumps each model's raw spans plus the gold
spans to a JSON cache. This is the only expensive pass; `eval_combiners.py`
then replays the cache to score any number of combiner strategies instantly.

Run from the repo root:
    PYTHONPATH=src legal-anon-env/bin/python \
        scripts/cache_predictions.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import warnings
warnings.filterwarnings("ignore")

from anonymisation.data import load_tab
from anonymisation.mapping import SPACY_TO_TAB
from anonymisation.evaluation import extract_gold_entities
from anonymisation.predictors import make_finetuned_predictor, build_presidio_analyzer, make_presidio_predictor
from anonymisation.device import best_device

LEGALBERT_DIR = REPO / "models/legalbert-tab/final"

# Inference window, stated explicitly rather than inherited from
# make_finetuned_predictor's defaults. Leaving it implicit is what caused the
# 0.5 pp LegalBERT discrepancy: the fine-tune notebooks pass max_length=384
# while this path silently took the 512 default, so the same checkpoint scored
# 0.8486 one way and 0.8538 the other. Sliding-window size changes how many
# boundary fragments become false positives. See
# scripts/diagnose_eval_discrepancy.py.
MAX_LENGTH = 512
STRIDE = 64
OUT_PATH = REPO / "results/predictions_cache.json"


def build_predictors():
    import spacy
    print("Loading spaCy en_core_web_trf …", flush=True)
    nlp = spacy.load("en_core_web_trf")

    def spacy_predict(text):
        return [
            (e.start_char, e.end_char, SPACY_TO_TAB[e.label_], e.text)
            for e in nlp(text).ents if e.label_ in SPACY_TO_TAB
        ]

    print(f"Loading LegalBERT fine-tune from {LEGALBERT_DIR} …", flush=True)
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    # Force CPU: MPS leaks memory across the 555-doc loop (throughput death-
    # spiralled from 1.7 to 0.04 docs/s). LegalBERT-base on CPU is ~steady.
    device = "cpu"
    print(f"  device = {device} (forced; MPS leaks over the long loop)", flush=True)
    tok = AutoTokenizer.from_pretrained(str(LEGALBERT_DIR))
    model = AutoModelForTokenClassification.from_pretrained(str(LEGALBERT_DIR))
    legalbert_predict = make_finetuned_predictor(
        model, tok, device=device, max_length=MAX_LENGTH, stride=STRIDE
    )

    print("Building Presidio analyzer (en_core_web_lg backbone + CASE_NUMBER) …", flush=True)
    analyzer = build_presidio_analyzer(add_case_number_recognizer=True, spacy_model="en_core_web_lg")
    presidio_predict = make_presidio_predictor(analyzer)

    return {"spacy": spacy_predict, "legalbert": legalbert_predict, "presidio": presidio_predict}


def main():
    test_docs = list(load_tab()["test"])
    print(f"TAB test docs: {len(test_docs)}", flush=True)

    # Resume: reload any docs already cached (cache is index-aligned to test_docs).
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = []
    if OUT_PATH.exists():
        try:
            cache = json.loads(OUT_PATH.read_text())
            print(f"Resuming — {len(cache)} docs already cached.", flush=True)
        except Exception:
            cache = []
    start_idx = len(cache)
    if start_idx >= len(test_docs):
        print("Cache already complete.", flush=True)
        return

    predictors = build_predictors()

    t0 = time.time()
    for i in range(start_idx, len(test_docs)):
        doc = test_docs[i]
        text = doc["text"]
        entry = {
            "doc_id": doc.get("doc_id", i),
            "gold": [list(g) for g in extract_gold_entities(doc)],
        }
        for name, predict in predictors.items():
            try:
                entry[name] = [list(s) for s in predict(text)]
            except Exception as exc:
                print(f"  ! {name} failed on doc {i}: {exc}", flush=True)
                entry[name] = []
        cache.append(entry)
        done = i - start_idx + 1
        if done % 25 == 0:
            rate = done / (time.time() - t0)
            eta = (len(test_docs) - i - 1) / rate if rate else 0
            print(f"  {i + 1}/{len(test_docs)}  ({rate:.2f} docs/s, ETA {eta/60:.1f} min)", flush=True)
            OUT_PATH.write_text(json.dumps(cache, ensure_ascii=False))  # checkpoint

    OUT_PATH.write_text(json.dumps(cache, ensure_ascii=False))
    print(f"\nDone in {(time.time() - t0)/60:.1f} min → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()

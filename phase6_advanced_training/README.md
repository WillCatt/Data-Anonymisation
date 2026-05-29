# Phase 6 — Advanced Training: Domain Backbone + Ensemble

**Status:** Run complete — **two measured nulls.** LegalBERT fine-tune = F1 0.849 (within noise of RoBERTa-FT). 3-way union ensemble = F1 0.553 (backfired). Phase 6.2 then tested whether a smarter combiner rescues the ensemble — see [Phase 6.2 below](#phase-62--does-a-smarter-combiner-rescue-the-ensemble-measured). It does not: **no combiner beats the single fine-tune**, because the fine-tune Pareto-dominates every member on every label.

Phase 2 fine-tuned `roberta-base` on TAB train. Phase 6 asks two follow-up questions that have well-known answers in the NER literature, neither of which we'd tested in this project yet:

1. **Does a *domain-pretrained* backbone beat a general-purpose one?** Swap `roberta-base` for [`nlpaueb/legal-bert-base-uncased`](https://huggingface.co/nlpaueb/legal-bert-base-uncased) — same training recipe, same evaluation. Typical reported lift on legal-domain NER is 1–3 F1.
2. **Does an ensemble of independent NER predictors beat any single one of them?** Combine spaCy + LegalBERT-fine-tuned + Microsoft Presidio, vote on each span. Typical reported lift over the best single member is 2–5 F1 — the cheapest single intervention with the biggest reliable improvement.

The interesting comparison is not "is each individual improvement big" but "do they stack?" — does the ensemble of LegalBERT + spaCy + Presidio beat LegalBERT alone, or do the predictors all make the same mistakes? TAB's per-entity-type breakdown will tell us.

## Folder structure

```
phase6_advanced_training/
├── README.md                           (this file)
├── notebooks/
│   ├── 01_legalbert_finetune.ipynb     LegalBERT swap-in for Phase 2 recipe
│   ├── 02_ensemble.ipynb               vote across spaCy + LegalBERT + Presidio
│   └── 03_head_to_head.ipynb           compare all six models on TAB
├── scripts/                            Phase 6.2 — combiner investigation
│   ├── cache_predictions.py            run the 3 models once, cache raw spans
│   └── eval_combiners.py               replay cache → score every combiner
└── results/
    ├── legalbert_results.csv           ← from 01
    ├── ensemble_results.csv            ← from 02 (union, F1 0.553)
    ├── predictions_cache.json          ← from cache_predictions.py
    ├── per_model_per_label_f1.csv      ← from eval_combiners.py
    └── combiner_comparison.csv         ← from eval_combiners.py
```

## How to run

Notebooks 01 and 02 are independent. Notebook 03 reads the CSVs the first two produce.

```bash
# Activate the main repo venv (NOT the gradio venv)
source legal-anon-env/bin/activate

# 1. LegalBERT fine-tune (~30 min on T4 / Colab; multi-hour on CPU)
jupyter notebook phase6_advanced_training/notebooks/01_legalbert_finetune.ipynb

# 2. Ensemble evaluation (fast — reads the trained model + spaCy + Presidio)
jupyter notebook phase6_advanced_training/notebooks/02_ensemble.ipynb

# 3. Head-to-head plot, no extra training
jupyter notebook phase6_advanced_training/notebooks/03_head_to_head.ipynb
```

Notebook 01 needs a GPU realistically. The Colab T4 free tier is the path of least resistance; an M-series Mac with `mps` works but slower; CPU is a 4–6 hour run.

## What the ensemble does

`src/anonymisation/ensemble.py` defines `EnsemblePredictor`. Construction:

```python
from anonymisation.ensemble import EnsemblePredictor

ensemble = EnsemblePredictor(
    predictors={
        "spacy":     spacy_predictor,
        "legalbert": legalbert_predictor,
        "presidio":  presidio_predictor,
    },
    min_votes=1,         # accept any span with ≥1 supporting predictor
    tie_break="longest", # on overlapping spans, prefer the longest
)
```

Per text:

1. Run each predictor independently. Collect every (start, end, tab_type) span they emit.
2. Group spans by character-offset overlap (transitive: A overlaps B, B overlaps C → all in one group).
3. Within each group, vote on entity type. Pick the type with the most votes.
4. Among spans of the winning type, keep the longest. (Ties resolved alphabetically — irrelevant for the metrics but makes output deterministic.)
5. Drop groups that didn't reach `min_votes` distinct predictors.

`min_votes=1` is the recall-maximising setting (union of predictions). `min_votes=2` is the precision-maximising setting (intersection of any two predictors). The default is 1 because for redaction, recall is usually more valuable than precision.

## Expected pattern in the results

A few outcomes worth predicting up front so the post-run write-up has structure:

- **LegalBERT vs RoBERTa, overall F1.** LegalBERT is pre-trained on legal text and should beat general-purpose RoBERTa on most TAB labels. The lift is typically modest (1–3 F1 macro-averaged) but real.
- **LegalBERT vs RoBERTa, per label.** Where LegalBERT is reliably ahead: PERSON, ORG (legal-specific naming conventions), LOC (court / jurisdiction language), MISC (case-name-like patterns). Where it's a wash: DATETIME and QUANTITY are domain-agnostic.
- **Ensemble vs any single model.** The ensemble should beat the best individual member on most labels — that's the point of ensembling. The interesting question is whether it beats by 2 F1 (predictors are diverse) or by < 0.5 F1 (predictors are making correlated errors).
- **CODE recall.** Off-the-shelf NER (spaCy, dslim/bert-base-NER) had 0% on this in Phase 1. Presidio's regex recogniser fixes it. The ensemble should inherit that — verify in the breakdown.

## Phase 6.2 — does a smarter combiner rescue the ensemble? (measured)

The Phase 6 union ensemble backfired (F1 0.553, *below* every individual model).
The obvious next question: was that the *voting rule*, not the idea of ensembling?
`min_votes=1` is a union — a span survives if **any** model emits it — so the
output inherits the worst member's false positives (Presidio's ORG precision is
0.14). Phase 6.2 tests whether requiring agreement, or routing each label to its
best model, recovers the loss.

Method: run all three predictors over the TAB test set **once**, cache their raw
spans (`cache_predictions.py`), then replay the cache through every combiner
against the *same* `evaluate_document` matcher (`eval_combiners.py`). The replay
reproduces the original union result bit-for-bit (F1 0.553, tp/fp/fn
17922/26086/2887 = `ensemble_results.csv`), which validates the harness.

**Result (partial-match F1, TAB test, 555 docs):**

| Strategy | Precision | Recall | F1 |
|---|---|---|---|
| spaCy-trf alone | 42.8% | 83.3% | 56.6% |
| Presidio alone | 49.4% | 77.5% | 60.3% |
| Union (`min_votes=1`) | 40.7% | 86.1% | 55.3% ← the backfire |
| Consensus (`min_votes=2`) | 54.8% | 83.0% | 66.1% |
| Consensus (`min_votes=3`) | 87.8% | 72.6% | 79.5% |
| Routed (per-label best model) | 84.6% | 86.2% | 85.4% |
| **LegalBERT-FT alone** | **84.6%** | **86.2%** | **85.4%** |

![Combiner comparison](../figures/phase6_combiner_comparison.png)

**Two findings:**

1. **The combiner *was* the cause of the backfire.** Requiring agreement climbs it
   straight back: union 55.3 → consensus-2 66.1 → consensus-3 79.5, as precision
   recovers from 41% to 88%. So the failure was union-voting specifically, exactly
   as hypothesised.

2. **But no combiner beats the single fine-tune — and the reason is decisive.**
   The per-label F1 table (`per_model_per_label_f1.csv`) shows the LegalBERT
   fine-tune is the **best model on every one of the 8 labels**:

   | | spaCy | LegalBERT | Presidio |
   |---|---|---|---|
   | PERSON | 89.9% | **90.6%** | 83.5% |
   | ORG | 29.7% | **63.3%** | 21.9% |
   | LOC | 46.8% | **74.3%** | 43.3% |
   | DATETIME | 90.6% | **94.0%** | 87.7% |
   | QUANTITY | 21.1% | **72.3%** | 0.0% |
   | CODE | 0.0% | **96.1%** | 3.3% |
   | DEM | 34.2% | **45.8%** | 35.1% |
   | MISC | 1.5% | **9.1%** | 0.0% |

   Because the fine-tune **Pareto-dominates**, "route each label to its best model"
   picks LegalBERT all 8 times — so *routed ≡ the fine-tune alone (85.4%)*. This
   also refutes the original 6.2 hypothesis (*route CODE → Presidio*): Presidio's
   CODE F1 is 3.3%, while the fine-tune learned application numbers to 96.1%.

**Conclusion.** Ensembling can only help when members are *complementary* — when
one wins where another loses. Here one member dominates everywhere, so there is no
combiner (and no second same-level fine-tune, which would be correlated and still
non-complementary) that can beat it. The single fine-tune is the ceiling. This is
a stronger, more useful negative result than "voting failed": it says *why*
ensembling is the wrong tool for this particular model lineup. A routed
`EnsemblePredictor` was deliberately **not** added to the package — it would be a
no-op equal to the fine-tune.

## What this phase does NOT do

- **No CRF head.** Adding a Conditional Random Field on top of the token-classification output would fix exact-match boundary errors. Listed as Phase 6.1.
- **No coref-augmented fine-tuning.** Phase 5's CorefExtender is a post-processor; a deeper fix is to retrain with denser supervision so the model itself catches shorthand. Listed as Phase 6.2.
- **No hyperparameter sweep.** LR, batch size, number of epochs — all use the Phase 2 defaults. A small sweep is the next obvious tuning step.

If the LegalBERT + Ensemble run lands well, those three are the natural next investments.

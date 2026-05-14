# Phase 6 — Advanced Training: Domain Backbone + Ensemble

**Status:** Code complete, runs pending. The training notebook + ensemble evaluator are written and verified to import; the actual GPU run is your job.

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
└── results/
    ├── legalbert_results.csv           ← from 01
    └── ensemble_results.csv            ← from 02
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

## What this phase does NOT do

- **No CRF head.** Adding a Conditional Random Field on top of the token-classification output would fix exact-match boundary errors. Listed as Phase 6.1.
- **No coref-augmented fine-tuning.** Phase 5's CorefExtender is a post-processor; a deeper fix is to retrain with denser supervision so the model itself catches shorthand. Listed as Phase 6.2.
- **No hyperparameter sweep.** LR, batch size, number of epochs — all use the Phase 2 defaults. A small sweep is the next obvious tuning step.

If the LegalBERT + Ensemble run lands well, those three are the natural next investments.

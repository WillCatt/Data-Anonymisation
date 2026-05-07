# Phase 2 — Baseline Comparison & Fine-Tuning

**Status:** Code complete, awaiting runs. The four notebooks in this folder are written, lint clean, and verified to import end-to-end. Final results land in `results/` once you execute them.

Phase 1 established that off-the-shelf spaCy NER is not enough to safely anonymise legal text. Phase 2 quantifies *how much room there is to improve* by comparing several baselines and then fine-tuning a transformer on the TAB train split.

## Models in the bake-off

| Model | Why it's here |
|---|---|
| `spacy/en_core_web_trf` (Phase 1 baseline) | The strongest off-the-shelf English NER. Our anchor. |
| `dslim/bert-base-NER` | A widely used HuggingFace transformer NER. Different label set; see `mapping_hf.py`. |
| `Microsoft Presidio` | Rule-based + NER hybrid that is the de-facto industry tool for PII redaction. Includes regex for case numbers — should help with TAB's `CODE` blind spot. |
| **Fine-tuned** `roberta-base` on TAB train | The contender. Same architecture class as `bert-base`, but trained on legal entity types directly. |

## Folder structure

```
phase2_baseline_comparison/
├── README.md                       (this file)
├── 01_hf_baseline.ipynb            evaluate dslim/bert-base-NER on TAB test
├── 02_presidio_baseline.ipynb      Presidio (stock + custom CASE_NUMBER recogniser)
├── 03_finetune_roberta.ipynb       train roberta-base on TAB train, eval on test
├── 04_head_to_head.ipynb           side-by-side comparison + figures
└── results/
    ├── hf_results.csv              ← created by 01
    ├── presidio_results.csv        ← created by 02
    └── finetuned_results.csv       ← created by 03
```

## How to run

> Notebooks 01, 02, 03 are independent — run them in any order. Notebook 04 reads the CSVs the first three produce.

1. Make sure your venv has Phase 2 deps installed (`pip install -r ../requirements.txt` from the repo root).
2. Download Presidio's recommended spaCy model (only needed for notebook 02):
   ```
   python -m spacy download en_core_web_lg
   ```
3. Open notebooks 01–04 in turn. The fine-tune in 03 is the slow one — see compute notes below.

If you're running on Colab, every notebook has a top-of-file install cell (commented out) ready to uncomment.

## What I expect to find (hypothesis, to be tested)

1. `dslim/bert-base-NER` will beat spaCy on `PERSON` and `ORG` (recent training data) but still miss `CODE` and `DEM`.
2. Presidio will close the `CODE` gap with its regex recogniser, but will be noisy on `ORG`/`MISC` like spaCy.
3. The fine-tuned RoBERTa will dominate on every TAB-specific label (`CODE`, `DEM`, `MISC`) because those are exactly the labels TAB was annotated with.
4. **None** of these will be sufficient on their own to make the mosaic effect go away — that motivates Phase 3.

## Why this matters for the law-firm framing

A buyer at a law firm needs to see two things:

1. The off-the-shelf gap is real (Phase 1 ✓)
2. The gap is closeable with reasonable effort (this phase)

Without Phase 2 the project reads as "a problem statement". With it, it reads as "a problem statement and a credible path to a solution".

## Compute notes

Fine-tuning RoBERTa on TAB train (1,014 docs, ~50K entity mentions) fits comfortably on a single T4 / Colab free tier. Budget ~30–60 minutes for 3 epochs.

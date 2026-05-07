# Legal Text Anonymisation

> Can a law firm safely use AI on its own client matter data? Building a measured answer, in three phases, with NLP.

[![Phase 1](https://img.shields.io/badge/Phase_1-Proof_of_Concept-2ecc71?style=flat-square)](notebooks/)
[![Phase 2](https://img.shields.io/badge/Phase_2-Code_Complete,_Runs_Pending-f39c12?style=flat-square)](phase2_baseline_comparison/)
[![Phase 3](https://img.shields.io/badge/Phase_3-Production_Pipeline-grey?style=flat-square)](phase3_pipeline/)

A portfolio project on PII redaction for legal text, anchored in the [Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark) (TAB). Tests whether off-the-shelf NER is enough to anonymise court documents, and quantifies the residual mosaic / re-identification risk that NER alone cannot fix.

**[→ Read the full writeup](writeup/README.md)**

---

## TL;DR

- spaCy's strongest off-the-shelf English NER scores **~0.57 partial-match F1 on TAB** — meaning it misses or misidentifies ≈ half of all sensitive entities.
- It has **0% recall** on case file numbers (TAB's `CODE` type), because that label doesn't exist in the model's training vocabulary.
- Even with a hypothetical *perfect* NER on direct identifiers, a non-trivial share of TAB documents remains uniquely identifiable from their **quasi-identifier fingerprint** alone — the mosaic effect.
- Implication: anonymisation is not an NER problem. The pipeline needs a re-identification check too.

---

## Repo layout

```
.
├── notebooks/                         Phase 1 — proof of concept
│   ├── 01_problem_setup.ipynb         framing, EDA, mosaic preview
│   ├── 02_baseline_evaluation.ipynb   spaCy vs TAB — main experiment
│   └── 03_mosaic_effect.ipynb         k-anonymity over QUASI fingerprints
├── phase2_baseline_comparison/        Phase 2 — model bake-off + fine-tune
│   ├── 01_hf_baseline.ipynb           dslim/bert-base-NER on TAB
│   ├── 02_presidio_baseline.ipynb     Presidio (stock + custom CODE recogniser)
│   ├── 03_finetune_roberta.ipynb      RoBERTa fine-tuned on TAB train
│   └── 04_head_to_head.ipynb          comparison plots
├── phase3_pipeline/                   Phase 3 — scoped, not yet built
├── src/anonymisation/                 reusable Python package
│   ├── data.py mapping.py             TAB loader + spaCy mapping
│   ├── evaluation.py                  span-level P/R/F1
│   ├── mosaic.py                      k-anonymity over QUASI fingerprints
│   ├── predictors.py                  HF / Presidio / fine-tuned adapters
│   ├── iob.py                         BIO tagging utilities for fine-tuning
│   ├── device.py                      CUDA / MPS / CPU detection
│   └── demo.py                        try-it-yourself helper
├── figures/                           plots for the writeup
├── results/                           Phase 1 per-entity metrics CSV
├── writeup/                           portfolio narrative (markdown)
└── demo/                              demo plan
```

---

## Setup

```bash
git clone <this repo>
cd "Data Anonymisation"

python -m venv legal-anon-env
source legal-anon-env/bin/activate          # (Windows: legal-anon-env\Scripts\activate)

pip install -r requirements.txt
python -m spacy download en_core_web_trf

jupyter notebook notebooks/
```

First run of the notebooks will download TAB from HuggingFace (~50 MB, cached afterwards) and the spaCy transformer model (~440 MB). After that it's offline.

If you only want to read the experiment, the writeup at `writeup/README.md` covers everything without needing to run the code.

### Lighter-weight alternative

If you don't want to pull the transformer model:

```python
# In notebooks/02_baseline_evaluation.ipynb, change:
SPACY_MODEL = "en_core_web_trf"
# to:
SPACY_MODEL = "en_core_web_sm"   # ~12 MB, faster, lower scores
```

The qualitative conclusions hold — the smaller model is just worse on PERSON and ORG.

---

## How to use this repo

| If you want to... | Start here |
|---|---|
| Read the story | [`writeup/README.md`](writeup/README.md) |
| See the experimental setup and the EDA | [`notebooks/01_problem_setup.ipynb`](notebooks/01_problem_setup.ipynb) |
| Reproduce the F1 numbers | [`notebooks/02_baseline_evaluation.ipynb`](notebooks/02_baseline_evaluation.ipynb) |
| See the mosaic-effect analysis | [`notebooks/03_mosaic_effect.ipynb`](notebooks/03_mosaic_effect.ipynb) |
| Reuse the code in your own project | `from anonymisation import ...` (see `src/anonymisation/__init__.py`) |
| Understand what's planned for later phases | [`phase2_baseline_comparison/README.md`](phase2_baseline_comparison/README.md), [`phase3_pipeline/README.md`](phase3_pipeline/README.md) |

---

## Roadmap

- [x] **Phase 1 — Proof of concept.** Baseline measurement + mosaic-effect quantification.
- [~] **Phase 2 — Baseline comparison + fine-tune.** Code complete (4 notebooks under `phase2_baseline_comparison/`), runs pending. spaCy vs `dslim/bert-base-NER` vs Microsoft Presidio vs RoBERTa fine-tuned on TAB train.
- [ ] **Phase 3 — Production pipeline.** NER + regex + mosaic-risk scorer + audit log + human-in-the-loop fallback. Deployable as a Docker image for on-prem use.
- [ ] **Live demo.** Gradio app on HuggingFace Spaces, embedded in the portfolio site.

---

## License + acknowledgments

TAB is the work of Pilán et al. at the Norwegian Computing Center, distributed under its own license — see the [original repo](https://github.com/NorskRegnesentral/text-anonymization-benchmark) for terms.

This project is for portfolio / educational use; it is not a substitute for legal or compliance advice.

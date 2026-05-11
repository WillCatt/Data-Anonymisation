# Legal Text Anonymisation

> Can a law firm safely use AI on its own client matter data? Building a measured answer, in three phases, with NLP.

[![Phase 1](https://img.shields.io/badge/Phase_1-Proof_of_Concept-2ecc71?style=flat-square)](notebooks/)
[![Phase 2](https://img.shields.io/badge/Phase_2-Code_Complete,_Runs_Pending-f39c12?style=flat-square)](phase2_baseline_comparison/)
[![Phase 3](https://img.shields.io/badge/Phase_3-Lite_+_Pro_Pipelines-2ecc71?style=flat-square)](phase3_pipeline/)
[![Phase 4](https://img.shields.io/badge/Phase_4-Pseudonymisation_+_Round--Trip-2ecc71?style=flat-square)](phase4_pseudonymisation/)
[![Phase 5](https://img.shields.io/badge/Phase_5-Coreference--Aware-2ecc71?style=flat-square)](phase5_coreference/)
[![Demo](https://img.shields.io/badge/Demo-Static_Showcase-3498db?style=flat-square)](demo/index.html)

A portfolio project on PII redaction for legal text, anchored in the [Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark) (TAB). Tests whether off-the-shelf NER is enough to anonymise court documents, and quantifies the residual mosaic / re-identification risk that NER alone cannot fix.

**[→ Read the full writeup](writeup/README.md)**

---

## TL;DR

- spaCy's strongest off-the-shelf English NER scores **~0.57 partial-match F1 on TAB** — meaning it misses or misidentifies ≈ half of all sensitive entities.
- It has **0% recall** on case file numbers (TAB's `CODE` type), because that label doesn't exist in the model's training vocabulary.
- Even with a hypothetical *perfect* NER on direct identifiers, a non-trivial share of TAB documents remains uniquely identifiable from their **quasi-identifier fingerprint** alone — the mosaic effect.
- Implication: anonymisation is not an NER problem. The pipeline needs a re-identification check too.
- **Phase 4** — `pseudonymise=True` makes the redacted document still *useful*: each distinct entity gets a stable referential token (`[PERSON_A]`, `[PERSON_B]`), and the firm can `restore()` an LLM's answer locally without ever exposing real names.

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
├── phase3_pipeline/                   Phase 3 — Lite vs Pro pipelines
│   └── notebooks/
│       ├── 01_lite_walkthrough.ipynb
│       ├── 02_pro_walkthrough.ipynb
│       └── 03_lite_vs_pro.ipynb
├── phase4_pseudonymisation/           Phase 4 — referential tokens + restore
│   └── notebooks/
│       └── 01_pseudonymisation_walkthrough.ipynb
├── phase5_coreference/                Phase 5 — coref-aware span extension
│   ├── evaluate_mention_recall.py     TAB mention-recall evaluation
│   ├── notebooks/01_coref_walkthrough.ipynb
│   └── results/                       CSV + summary land here
├── src/anonymisation/                 reusable Python package
│   ├── data.py mapping.py             TAB loader + spaCy mapping
│   ├── evaluation.py                  span-level P/R/F1
│   ├── mosaic.py                      k-anonymity over QUASI fingerprints
│   ├── predictors.py                  HF / Presidio / fine-tuned adapters
│   ├── iob.py                         BIO tagging utilities for fine-tuning
│   ├── device.py                      CUDA / MPS / CPU detection
│   ├── demo.py                        Phase 1 try-it-yourself helper
│   ├── cli.py                         CLI entry point (python -m anonymisation.cli)
│   └── pipeline/                      Phase 3 + 4 redaction pipelines
│       ├── types.py                   Span, AuditEntry, RedactionResult (with vault)
│       ├── regex_pass.py              case nos, IBANs, phones, emails
│       ├── roles.py                   DIRECT/QUASI default classifier
│       ├── generalization.py          per-entity-type generalization rules
│       ├── scorer.py                  MosaicScorer (k-anonymity over a haystack)
│       ├── pseudonymise.py            Pseudonymiser + restore() (Phase 4)
│       ├── coref.py                   CorefExtender — shorthand mention recovery (Phase 5)
│       ├── base.py                    shared Pipeline base class
│       ├── lite.py                    LitePipeline — DIRECT only (+ pseudonymise / coref flags)
│       └── pro.py                     ProPipeline — DIRECT + iterate-until-safe (+ pseudonymise / coref flags)
├── figures/                           plots for the writeup
├── results/                           Phase 1 per-entity metrics CSV
├── writeup/                           portfolio narrative (markdown)
└── demo/                              Static side-by-side showcase
    ├── index.html                     the showcase (open in browser)
    ├── build_showcase.py               regenerator script
    ├── examples/                      input samples
    └── app.py                         (deferred — Gradio interactive version)
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
- [x] **Phase 3 — Production pipeline (two variants).** `LitePipeline` (DIRECT-only) and `ProPipeline` (DIRECT + mosaic-aware QUASI generalization). Library + CLI + walkthrough notebooks.
- [x] **Phase 4 — Pseudonymisation + round-trip.** `pseudonymise=True` flag on both pipelines (referential `[PERSON_A]`, `[PERSON_B]` tokens with substring-rule coreference). `restore()` helper for the LLM-answer round-trip. CLI `restore` subcommand + walkthrough notebook + showcase Sample 4.
- [x] **Phase 5 — Coreference-aware extension.** `coref_extend=True` flag on both pipelines (defaults on). Post-processor catches the shorthand mentions NER misses. TAB mention-recall evaluation script + walkthrough notebook.
- [x] **Static demo.** Self-contained `demo/index.html` — 12 examples across 4 categories (Law civil rights, Law corporate, Medical, Pseudonymisation).
- [ ] **Interactive demo.** Gradio version (`demo/app.py`) — deferred; revive once the Gradio/spaCy dep tree settles.
- [ ] **Phase 4.1 — Cross-document pseudonymisation.** Persistent registry so the same person stays the same token across documents. Real key-management questions to answer first.
- [ ] **Phase 3.1 — Productionisation.** Docker packaging, FastAPI service, layout-aware extraction (PDF/DOCX), human-in-the-loop UI around the audit log.

---

## License + acknowledgments

TAB is the work of Pilán et al. at the Norwegian Computing Center, distributed under its own license — see the [original repo](https://github.com/NorskRegnesentral/text-anonymization-benchmark) for terms.

This project is for portfolio / educational use; it is not a substitute for legal or compliance advice.

# Legal Text Anonymisation

> Can a law firm safely use AI on its own client matter data? An empirical answer in six phases — measurement, modelling, deployable pipeline, and honest negative results.

[![Phase 1](https://img.shields.io/badge/Phase_1-Proof_of_Concept-2ecc71?style=flat-square)](notebooks/)
[![Phase 2](https://img.shields.io/badge/Phase_2-RoBERTa_FT_85.1%25_F1-2ecc71?style=flat-square)](phase2_baseline_comparison/)
[![Phase 3](https://img.shields.io/badge/Phase_3-Lite_+_Pro_Pipelines-2ecc71?style=flat-square)](phase3_pipeline/)
[![Phase 4](https://img.shields.io/badge/Phase_4-Pseudonymisation_+_Round--Trip-2ecc71?style=flat-square)](phase4_pseudonymisation/)
[![Phase 5](https://img.shields.io/badge/Phase_5-Coref_(measured_null)-95a5a6?style=flat-square)](phase5_coreference/)
[![Phase 6](https://img.shields.io/badge/Phase_6-LegalBERT_84.9%25_·_Ensemble_55.3%25↓-95a5a6?style=flat-square)](phase6_advanced_training/)
[![Demo](https://img.shields.io/badge/Demo-Static_+_Gradio_+_Spaces-3498db?style=flat-square)](demo/)
[![Best F1](https://img.shields.io/badge/Best_F1-85.1%25_(RoBERTa)-27ae60?style=flat-square)](figures/phase_overall_f1.png)
[![Mosaic](https://img.shields.io/badge/Mosaic-1268/1268_documents_unique-d73a49?style=flat-square)](figures/mosaic_k_distribution.png)

A portfolio project on PII redaction for legal text, anchored in the [Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark) (TAB). Tests whether off-the-shelf NER is enough to anonymise court documents, and quantifies the residual mosaic / re-identification risk that NER alone cannot fix.

**[→ Read the full writeup](writeup/README.md)**

---

## TL;DR

- spaCy's strongest off-the-shelf English NER scores **~0.57 partial-match F1 on TAB** — meaning it misses or misidentifies ≈ half of all sensitive entities.
- It has **0% recall** on case file numbers (TAB's `CODE` type), because that label doesn't exist in the model's training vocabulary.
- Even with a hypothetical *perfect* NER on direct identifiers, **every TAB document (1,268 / 1,268)** remains uniquely identifiable from its **quasi-identifier fingerprint** alone — the mosaic effect, in its starkest form.
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
├── phase6_advanced_training/          Phase 6 — domain backbone + ensemble
│   ├── notebooks/
│   │   ├── 01_legalbert_finetune.ipynb
│   │   ├── 02_ensemble.ipynb
│   │   └── 03_head_to_head.ipynb
│   └── results/                       CSVs land here after running
├── src/anonymisation/                 reusable Python package
│   ├── data.py mapping.py             TAB loader + spaCy mapping
│   ├── evaluation.py                  span-level P/R/F1
│   ├── mosaic.py                      k-anonymity over QUASI fingerprints
│   ├── predictors.py                  HF / Presidio / fine-tuned adapters
│   ├── ensemble.py                    EnsemblePredictor — vote-based combiner (Phase 6)
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
├── results/                           Phase 1 per-entity metrics CSV
├── writeup/                           portfolio narrative (markdown)
│   ├── README.md                      full v1 writeup
│   └── SKELETON.md                    restructured v2 (3 acts + epilogue) — fill-in scaffold
├── demo/                              Static showcase + local Gradio app
│   ├── index.html                     static showcase (open in browser)
│   ├── build_showcase.py              regenerator script
│   ├── examples/                      input samples
│   ├── app.py                         Gradio interactive demo (auto-detects best NER backend)
│   ├── requirements-gradio.txt        pinned deps for the Gradio venv
│   └── run_gradio.sh                  bootstrap + launch script
├── spaces/                            HuggingFace Spaces deployment bundle
│   ├── README.md                      Spaces deploy walkthrough (+ YAML frontmatter)
│   ├── app.py                         Spaces-tailored entry point
│   ├── requirements.txt               Spaces deps
│   ├── pre-build.sh                   spaCy model download on first build
│   └── sync.sh                        refresh package + examples from main repo
└── figures/                           plots for the writeup + portfolio site
    ├── mosaic_k_distribution.png      100% of TAB docs uniquely identifiable (Act I centerpiece)
    ├── phase_overall_f1.png           trimmed 5-bar headline ladder
    ├── gap_closed_by_entity.png       per-entity F1 lift, P1 baseline → P2 fine-tune
    ├── ensemble_backfire.png          Phase 6 — per-entity precision collapse of 3-way ensemble
    ├── phase_f1_by_entity.png         per-entity-type grouped bars (all models)
    ├── phase_precision_recall.png     precision/recall scatter (all models)
    ├── phase_summary.png              4-panel cross-phase performance summary
    ├── phase5_mention_recall.png      Phase 5 coref extender — measured null
    ├── pipeline_architecture.svg/.png Lite + Pro + Pseudonymise system diagram (Act III)
    ├── round_trip_threat_model.svg/.png  firm ↔ LLM provider trust boundary (Phase 4)
    └── build_performance_summary.py   regenerator script
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
| Read the story | [`writeup/README.md`](writeup/README.md) (long v1) · [`writeup/SKELETON.md`](writeup/SKELETON.md) (restructured v2 outline) |
| See the headline result in one chart | [`figures/phase_overall_f1.png`](figures/phase_overall_f1.png) (5-bar ladder, ensemble falling-knife at the bottom) |
| See the mosaic-effect finding | [`figures/mosaic_k_distribution.png`](figures/mosaic_k_distribution.png) (1,268 / 1,268 TAB docs uniquely identifiable) |
| Understand the system architecture | [`figures/pipeline_architecture.png`](figures/pipeline_architecture.png) and [`figures/round_trip_threat_model.png`](figures/round_trip_threat_model.png) |
| See the experimental setup and the EDA | [`notebooks/01_problem_setup.ipynb`](notebooks/01_problem_setup.ipynb) |
| Reproduce the Phase 1 baseline | [`notebooks/02_baseline_evaluation.ipynb`](notebooks/02_baseline_evaluation.ipynb) |
| Reproduce the mosaic-effect analysis | [`notebooks/03_mosaic_effect.ipynb`](notebooks/03_mosaic_effect.ipynb) |
| Reproduce the RoBERTa fine-tune (best model) | [`phase2_baseline_comparison/03_finetune_roberta.ipynb`](phase2_baseline_comparison/03_finetune_roberta.ipynb) |
| Try the pipeline locally | `./demo/run_gradio.sh` (Gradio at `localhost:7860`) |
| Reuse the code in your own project | `from anonymisation import ...` (see `src/anonymisation/__init__.py`) |

---

## Roadmap

- [x] **Phase 1 — Proof of concept.** Baseline measurement (spaCy F1 = 0.566) + mosaic-effect quantification (1,268 / 1,268 TAB docs uniquely identifiable from QUASI fingerprint alone).
- [x] **Phase 2 — Baseline comparison + fine-tune.** Four models evaluated on TAB test. **RoBERTa fine-tuned on TAB wins at F1 = 0.851** (+28.5 pp over spaCy). Presidio + CASE_NUMBER = 0.603 (regex closes CODE gap but ORG noise dominates). `dslim/bert-base-NER` = 0.167 (CoNLL label set discards most of TAB's annotation — not a meaningful comparison).
- [x] **Phase 3 — Production pipeline (two variants).** `LitePipeline` (DIRECT-only) and `ProPipeline` (DIRECT + mosaic-aware QUASI generalization). Library + CLI + walkthrough notebooks + per-decision audit log.
- [x] **Phase 4 — Pseudonymisation + round-trip.** `pseudonymise=True` flag on both pipelines (referential `[PERSON_A]`, `[PERSON_B]` tokens with substring-rule coreference). `restore()` helper for the LLM-answer round-trip. CLI `restore` subcommand + walkthrough notebook + showcase Sample 4 + threat-model diagram.
- [x] **Phase 5 — Coreference-aware extension** *(measured null).* `coref_extend=True` post-processor; integrated as a defensive layer. **TAB mention-recall lift: +0.0001 macro** — spaCy is already at the recall ceiling on PERSON/ORG where the substring rule could help. Honest negative result, kept in the audit log.
- [x] **Phase 6 — Domain backbone + ensemble** *(two measured nulls).* LegalBERT fine-tune: F1 = 0.849 — **within noise of RoBERTa-FT** (domain pretraining helps with vocabulary, not labels). 3-way ensemble (spaCy + LegalBERT + Presidio, union-vote): F1 = 0.553 — **backfired**, because Presidio's ORG precision is 0.14 and union voting inherits its worst member's noise.
- [x] **Static demo.** Self-contained `demo/index.html` — 12 examples across 4 categories. *(Portfolio trim to ~4 examples is a TODO.)*
- [x] **Gradio interactive demo.** `./demo/run_gradio.sh` bootstraps a separate isolated venv (gitignored) and launches the app at localhost:7860.
- [ ] **Phase 4.1 — Cross-document pseudonymisation.** Persistent registry so the same person stays the same token across documents. Real key-management questions to answer first.
- [ ] **Phase 6.2 — Label-weighted ensemble.** Route per-label to the strongest predictor (CODE → Presidio, everything else → fine-tune) instead of union voting. Tests whether the failure mode was union-voting specifically.
- [ ] **Phase 6.1 — CRF head on the fine-tune.** Closes the 0.85 partial vs 0.77 exact-match F1 gap. Standard NER-literature move.
- [ ] **Phase 3.1 — Productionisation.** Docker packaging, FastAPI service, layout-aware extraction (PDF/DOCX), human-in-the-loop UI around the audit log.
- [ ] **Fuzzy fingerprint matching for the mosaic scorer.** Currently exact surface form. Lets us sweep "how does the 100% finding change as match strictness loosens?" — the sensitivity analysis the project most needs.

---

## License + acknowledgments

TAB is the work of Pilán et al. at the Norwegian Computing Center, distributed under its own license — see the [original repo](https://github.com/NorskRegnesentral/text-anonymization-benchmark) for terms.

This project is for portfolio / educational use; it is not a substitute for legal or compliance advice.

# Legal Text Anonymisation

> **Can a law firm safely put its own client matter data through an LLM?**
> An empirical answer: measure the detector, then measure what's *still* identifying after the detector is perfect.

[![tests](https://github.com/WillCatt/Data-Anonymisation/actions/workflows/tests.yml/badge.svg)](https://github.com/WillCatt/Data-Anonymisation/actions/workflows/tests.yml)
[![Best F1](https://img.shields.io/badge/detection-0.851_F1_(RoBERTa_FT)-27ae60?style=flat-square)](figures/phase_overall_f1.png)
[![Mosaic](https://img.shields.io/badge/re--identification-1,268_/_1,268_unique-d73a49?style=flat-square)](figures/mosaic_reidentification.png)
[![Demo](https://img.shields.io/badge/demo-Gradio_+_HF_Spaces-3498db?style=flat-square)](https://willxo-legal-text-anonymisation.hf.space)

Anchored in the [Text Anonymization Benchmark](https://github.com/NorskRegnesentral/text-anonymization-benchmark) (TAB) — 1,268 European Court of Human Rights judgments annotated with entity type **and** identifier role (`DIRECT` / `QUASI` / `NO_MASK`). That second axis is what makes the re-identification question answerable at all, and it is why this corpus and not a general NER one.

**[→ Read the full writeup](docs/writeup.md)**

---

## Three findings

1. **Off-the-shelf NER is not close.** spaCy's strongest English model scores **0.566 partial-match F1** on TAB, with **0% recall on case-file numbers** — that label doesn't exist in its vocabulary. The failure is categorical, not marginal.
2. **Fine-tuning closes the detection gap.** RoBERTa fine-tuned on TAB reaches **0.851 F1** (+28.5 pp), and the lift lands exactly where TAB's label set diverges from newswire NER.
3. **It doesn't matter.** Assume a *perfect* detector; mask every direct identifier. **All 1,268 documents remain uniquely identifiable** from their residual quasi-identifier fingerprint — age, nationality, occupation, location, dates. Median fingerprint: 25 facts. The joint distribution of 25 ordinary details behaves like a hash.

**The consequence that shaped the product:** anonymisation is not an entity-detection problem. It needs a re-identification check, and it needs a way to keep the text *useful* after redaction.

---

## How the project moved

The repo is organised by function, not chronology. The narrative runs like this — each stage links to the evidence.

### 1 · The problem
A legal team wants summaries and chronologies from an LLM over files that carry privileged names, case numbers and identifying context. The reflex answer is *"just strip the names out first."* This project treats that as a hypothesis to test rather than a solution to implement.

→ [`notebooks/01_problem_and_data.ipynb`](notebooks/01_problem_and_data.ipynb)

### 2 · Detection: how good is off-the-shelf, and what does training buy?
Four detectors on identical inputs and an identical metric. Presidio plus a custom case-number recogniser fixes `CODE` recall (0% → 92%) but barely moves the headline, because its ORG precision of 0.13 floods the output with false positives. Fine-tuning wins.

→ [`02_baseline_spacy`](notebooks/02_baseline_spacy.ipynb) · [`03_baseline_huggingface`](notebooks/03_baseline_huggingface.ipynb) · [`04_baseline_presidio`](notebooks/04_baseline_presidio.ipynb) · [`05_finetune_roberta`](notebooks/05_finetune_roberta.ipynb) · [`06_detection_head_to_head`](notebooks/06_detection_head_to_head.ipynb)

### 3 · Re-identification: the finding that changed the product ⭐
Mask every `DIRECT` span, fingerprint each document by what remains, count how many documents share a fingerprint. None reach k ≥ 5. Perfect detection would not have helped.

→ [`notebooks/07_reidentification_mosaic.ipynb`](notebooks/07_reidentification_mosaic.ipynb) · [`figures/mosaic_reidentification.png`](figures/mosaic_reidentification.png)

### 4 · The pipeline: three modes, chosen by threat model
- **Redact** (`LitePipeline`) — strip `DIRECT` identifiers. Fast, predictable, no mosaic handling.
- **Anonymise** (`ProPipeline`) — then iterate: generalise `QUASI` spans one level at a time (47 → ~50 → 40s → suppressed; Plovdiv → Bulgaria → Europe) until k ≥ target or the budget runs out.
- **Pseudonymise** — stable referential tokens (`[PERSON_A]`) with the token→name **vault held locally**, so an LLM can still follow who did what to whom and the answer is restored on-prem.

Flat redaction is *more private and less useful*. The vault is what makes the text worth sending at all — and it generalises well beyond law.

→ [`08_pipeline_lite`](notebooks/08_pipeline_lite.ipynb) · [`09_pipeline_pro`](notebooks/09_pipeline_pro.ipynb) · [`10_pipeline_lite_vs_pro`](notebooks/10_pipeline_lite_vs_pro.ipynb) · [`11_pseudonymisation`](notebooks/11_pseudonymisation.ipynb) · [`figures/round_trip_threat_model.png`](figures/round_trip_threat_model.png)

### 5 · Evaluation, and three things that didn't work
See [What didn't work](#what-didnt-work). All three are kept deliberately — the honest null is the finding.

→ [`12_null_coreference`](notebooks/12_null_coreference.ipynb) · [`13_finetune_legalbert`](notebooks/13_finetune_legalbert.ipynb) · [`14_null_ensemble`](notebooks/14_null_ensemble.ipynb) · [`15_final_head_to_head`](notebooks/15_final_head_to_head.ipynb)

### 6 · Where it goes next
The two controls this project arrived at — a minimum group size before anything is reportable, and generalising indirect details until a combination stops being unique — are k-anonymity and the mosaic effect. Employee-survey platforms already ship them as product settings. Free-text comments are where they're hardest to enforce, and exactly the text people now want to run through an LLM. See [Limits and next](#limits-and-next).

---

## Results

Partial-match span F1 on the 555-document TAB test split. Every model runs behind the same predictor contract and is scored by the same code, so the numbers are directly comparable.

| Model | Partial F1 | Exact F1 | Note |
|---|---|---|---|
| spaCy `en_core_web_trf` | 0.5656 | 0.3807 | baseline · CODE recall **0.0** |
| `dslim/bert-base-NER` | 0.1669 | 0.0546 | CoNLL label set discards most of TAB — not a meaningful comparison |
| Presidio (stock) | 0.6024 | 0.4252 | ORG precision 0.13 |
| Presidio + CASE_NUMBER | 0.6031 | 0.4249 | regex fixes CODE; ORG noise dominates |
| **RoBERTa fine-tuned** | **0.8510** | **0.7842** | **best** |
| LegalBERT fine-tuned | 0.8486 | 0.7847 | tie — see null #2 |
| Ensemble, union (min 1) | 0.5530 | 0.4793 | backfired — see null #3 |
| Ensemble, consensus (min 3) | 0.7949 | 0.7220 | recovers, still loses |
| Routed (per-label best) | 0.8538 | 0.7902 | collapses to a single model |

> **Known discrepancy.** LegalBERT reports 0.8486 via `results/finetune_legalbert.csv` and 0.8538 via `results/combiner_comparison.csv` — two evaluation paths, 0.5 pp apart, straddling RoBERTa's 0.8510. Reconciling this (and putting confidence intervals on every number above) is the top open item.

**Re-identification:** 1,268 / 1,268 TAB documents have k = 1 on the QUASI fingerprint; median signature 25 mentions. Exact surface-form matching — see [Limits](#limits-and-next).

---

## What didn't work

Three measured nulls, kept in the repo on purpose.

**1 · Coreference extension moved recall by 0.0001.** The hypothesis was that NER under-recalls shorthand mentions ("Maria" after "Maria Petrova"). Macro recall went 0.8399 → 0.8400. spaCy is already at ≥95% recall on PERSON and ORG, so there was nothing for the rule to find; the categories that *do* under-recall (DEM, MISC, CODE) are exactly the ones a substring rule can't help. Kept as a zero-cost defensive layer, kept out of the headline.

**2 · LegalBERT's domain pretraining bought nothing.** 12 GB of legal text, same fine-tune recipe: 0.849 vs 0.851. Domain pretraining helps when the task *vocabulary* differs from the pretraining corpus. TAB's labels are general-NER labels — knowing "estoppel" doesn't help you recognise a name.

**3 · The three-way ensemble scored worse than its own baseline.** Union voting over spaCy + LegalBERT + Presidio gave 0.553, against 0.851 for either fine-tune alone. The mechanism is clean: Presidio's ORG precision is 0.14, and under `min_votes=1` every one of its false positives lands in the output even when both fine-tunes correctly rejected it. **The ensemble inherited its worst member.** Fixing the vote rule confirms the diagnosis — consensus recovers to 0.795 — but no combiner beats the single fine-tune, because it is Pareto-dominant across all eight labels, so per-label routing collapses to it alone.

→ [`figures/ensemble_backfire.png`](figures/ensemble_backfire.png) · [`docs/notes/advanced-training.md`](docs/notes/advanced-training.md)

---

## Repo layout

```
.
├── src/anonymisation/        the package — everything below imports this
│   ├── predictors.py         every NER backend behind one contract:
│   │                           text -> [(start, end, tab_type, span_text)]
│   ├── pipeline/             Redact / Anonymise / Pseudonymise + audit log
│   │   ├── base.py           shared detect_spans() engine
│   │   ├── lite.py pro.py    the two redaction strategies
│   │   ├── scorer.py         MosaicScorer — k-anonymity over a haystack
│   │   ├── generalization.py QUASI broadening levels
│   │   ├── pseudonymise.py   referential tokens + vault + restore()
│   │   └── types.py          Span · AuditEntry · RedactionResult
│   ├── mosaic.py             the re-identification analysis
│   ├── evaluation.py         span P/R/F1, partial + exact
│   └── cli.py                `anonymise redact` / `anonymise restore`
├── notebooks/                01–15, in narrative order (see above)
├── scripts/                  runnable analysis + figure builders
├── results/                  every metric CSV/JSON, one place
├── figures/                  generated charts and diagrams
├── models/                   fine-tuned checkpoints (gitignored, ~8.8 GB)
├── tests/                    52 pure-logic tests — no models, no network
├── demo/                     Gradio app, worked example, static showcase
├── spaces/                   HuggingFace Spaces deployment bundle
└── docs/                     writeup + per-stage process notes
```

**The seam worth knowing about:** every detector — spaCy, HuggingFace, Presidio, both fine-tunes, the ensemble — is normalised to one callable, `text -> [(start_char, end_char, tab_type, span_text)]`. That is why five models could be compared on identical inputs, and why adding a sixth costs about sixty lines rather than a rewrite.

---

## Setup

```bash
git clone https://github.com/WillCatt/Data-Anonymisation.git
cd Data-Anonymisation

python -m venv legal-anon-env
source legal-anon-env/bin/activate        # Windows: legal-anon-env\Scripts\activate

pip install -e ".[research,dev]"          # or just `pip install -e .` for the library
python -m spacy download en_core_web_trf  # baseline / best off-the-shelf NER
python -m spacy download en_core_web_lg    # required by Presidio's NLP engine
```

First run downloads TAB (~50 MB) and the spaCy transformer (~440 MB), cached afterwards.

**Tests** need neither models nor network — every heavy import is lazy:

```bash
pytest            # 52 tests, ~0.5s
```

**Try it on one document:**

```bash
python demo/worked_example.py             # Redact · Anonymise · Pseudonymise + audit log
./demo/run_gradio.sh                      # interactive app at localhost:7860
anonymise redact --variant pro --pseudonymise --vault-out vault.json doc.txt
```

> The Gradio demo runs in its **own** isolated venv (`demo/venv-gradio/`, bootstrapped on first run) because Gradio 4.x conflicts with the research stack. Don't install it into `legal-anon-env`.

---

## Reproducing the results

| To reproduce | Run |
|---|---|
| The detection ladder | `notebooks/02` → `06` in order |
| The re-identification finding | `notebooks/07_reidentification_mosaic.ipynb` |
| Coreference mention-recall | `python scripts/evaluate_mention_recall.py --sample 50` |
| Cached predictions for combiner work | `python scripts/cache_predictions.py` |
| The combiner sweep | `python scripts/eval_combiners.py` |
| All figures | `python scripts/build_performance_summary.py` (and the other `build_*.py`) |

---

## Limits and next

**Limits — stated plainly.**
- **TAB is English-only and ECHR-only.** Findings are sharp on this corpus; cross-corpus validation is needed before claiming generality.
- **Mosaic fingerprints match on exact surface form.** A real attacker matches fuzzily. The 100% number is conservative in one direction (real risk is at least this bad) and optimistic in another (a real haystack is bigger and noisier than TAB).
- **No human ceiling established.** TAB has non-trivial annotator disagreement on identifier role. Without a human-vs-human F1, we don't know how close 0.851 is to the realistic maximum — which also means we don't know when to stop optimising.
- **Text in, text out.** No PDF/DOCX extraction, no layout handling, no cross-document linking, no service packaging.

**Next, in priority order.**
1. **Confidence intervals and paired significance tests** on every headline number. Several "within noise" claims in this repo have never actually been tested, and the LegalBERT discrepancy above needs resolving.
2. **Fuzzy fingerprint matching** — sweep how the 100% finding moves as match strictness loosens. The sensitivity analysis this project most needs.
3. **Human annotator ceiling** — establish the realistic upper bound before chasing more F1.
4. **Cross-document pseudonymisation** — a persistent registry so the same person keeps the same token across a matter. Real key-management questions first.
5. **CRF head on the fine-tune** — closes the 0.85 partial vs 0.78 exact-match gap.

---

## License and acknowledgments

TAB is the work of Pilán et al. at the Norwegian Computing Center, distributed under its own license — see the [original repo](https://github.com/NorskRegnesentral/text-anonymization-benchmark). k-anonymity framing follows Sweeney (2002). spaCy by Explosion AI; Presidio by Microsoft.

This is a portfolio and research project. It is not a substitute for legal or compliance advice, and it is not a production-hardened product.

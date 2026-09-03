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
2. **Fine-tuning closes the detection gap.** RoBERTa fine-tuned on TAB reaches **0.856 F1, 95% CI [0.848, 0.864]** (+29 pp), and the lift lands exactly where TAB's label set diverges from newswire NER. LegalBERT, despite 12 GB of legal pretraining, is statistically indistinguishable from it.
3. **It doesn't matter.** Assume a *perfect* detector; mask every direct identifier. **One well-chosen residual fact — a nationality, an occupation, a place — singles out 77% of documents within the corpus; two facts reach 88%.** The identifying information was never only in the names.

   ⚠️ *An earlier version of this README reported "1,268 / 1,268 uniquely identifiable". That figure was an artefact and has been withdrawn — see [the mosaic claim, corrected](#the-mosaic-claim-corrected).*

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
Mask every `DIRECT` span and ask how many of the ordinary details left behind an attacker needs before a document stands alone. Usually one. Perfect detection would not have helped — and the first version of this analysis was wrong in an instructive way ([below](#the-mosaic-claim-corrected)).

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

The table reports the **fine-tune notebooks'** runs (inference `max_length=384`). The confidence intervals below are measured from the **prediction cache** (`max_length=512`), which scores both fine-tunes ~0.5 pp higher for the reason documented under [the resolved discrepancy](#results). Same checkpoints, same gold, same matcher — different sliding-window size. Rankings are unaffected.

| Model | Partial F1 | Exact F1 | Note |
|---|---|---|---|
| spaCy `en_core_web_trf` | 0.5656 | 0.3807 | baseline · CODE recall **0.0** |
| `dslim/bert-base-NER` | 0.1669 | 0.0546 | CoNLL label set discards most of TAB — not a meaningful comparison |
| Presidio (stock) | 0.6024 | 0.4252 | ORG precision 0.13 |
| Presidio + CASE_NUMBER | 0.6031 | 0.4249 | regex fixes CODE; ORG noise dominates |
| **RoBERTa fine-tuned** | **0.8510** | **0.7842** | best point estimate — but see the paired test below |
| LegalBERT fine-tuned | 0.8486 | 0.7847 | tie — see null #2 |
| Ensemble, union (min 1) | 0.5530 | 0.4793 | backfired — see null #3 |
| Ensemble, consensus (min 3) | 0.7949 | 0.7220 | recovers, still loses |
| Routed (per-label best) | 0.8538 | 0.7902 | collapses to a single model |

**Confidence intervals** *(from the prediction cache, `max_length=512`)*. Cluster bootstrap, 10,000 resamples, **resampling the 127 unique documents rather than the 555 rows**.

That distinction is not cosmetic. TAB's "555 test documents" are 555 *annotator-annotation pairs over 127 unique judgments* — one document appears up to ten times with different gold spans and identical model predictions. Treating those rows as independent draws makes every interval about **1.7× too narrow**. Spans within a document are correlated for the same reason, one level down.

| Model | Partial F1 | 95% CI |
|---|---|---|
| spaCy `en_core_web_trf` | 0.5656 | [0.5424, 0.5884] |
| Presidio + CASE_NUMBER | 0.6031 | [0.5807, 0.6246] |
| LegalBERT fine-tuned | 0.8538 | [0.8362, 0.8706] |
| **RoBERTa fine-tuned** | **0.8559** | **[0.8413, 0.8702]** |

**Is RoBERTa actually better than LegalBERT?** No — and this is now tested rather than asserted. Paired bootstrap on identical resampled documents gives **Δ = −0.0022, 95% CI [−0.0114, +0.0061], permutation p = 0.64**. Zero sits comfortably inside the interval. Every other pairing *is* distinguishable (all p ≈ 0.0001), including Presidio over spaCy.

Note the paired interval (±0.009) is narrower than either model's own (±0.015 to ±0.017). That is the point of pairing: both models face the same documents each resample, so "this batch happened to be easy" cancels out. Overlapping individual intervals would not have settled the question either way.

> **Resolved: the 0.5 pp LegalBERT discrepancy.** The same checkpoint scored 0.8486 in `finetune_legalbert.csv` and 0.8538 in `combiner_comparison.csv` — straddling RoBERTa's reported 0.8510, so *which CSV you quoted decided which model won*. Cause: an inference-config mismatch, not a metric bug. The fine-tune notebooks pass `max_length=384`; the cache-replay path silently took `make_finetuned_predictor`'s `512` default. Smaller windows fragment spans at chunk boundaries and those fragments score as false positives — measured at **+0.0063 F1** for 512 over 384, with identical gold both ways ([`scripts/diagnose_eval_discrepancy.py`](scripts/diagnose_eval_discrepancy.py)). Both fine-tunes were affected equally, so the tie between them held either way — but only by luck. The window is now pinned explicitly in both cache scripts.

Reproduce: `python scripts/bootstrap_ci.py --mode both` → [`results/bootstrap_ci.csv`](results/bootstrap_ci.csv), [`results/paired_comparisons.csv`](results/paired_comparisons.csv).

**Re-identification:** see [the mosaic claim, corrected](#the-mosaic-claim-corrected).

---

## What didn't work

Three measured nulls, kept in the repo on purpose.

**1 · Coreference extension moved recall by 0.0001.** The hypothesis was that NER under-recalls shorthand mentions ("Maria" after "Maria Petrova"). Macro recall went 0.8399 → 0.8400. spaCy is already at ≥95% recall on PERSON and ORG, so there was nothing for the rule to find; the categories that *do* under-recall (DEM, MISC, CODE) are exactly the ones a substring rule can't help. Kept as a zero-cost defensive layer, kept out of the headline.

**2 · LegalBERT's domain pretraining bought nothing.** 12 GB of legal text, same fine-tune recipe: 0.849 vs 0.851. Domain pretraining helps when the task *vocabulary* differs from the pretraining corpus. TAB's labels are general-NER labels — knowing "estoppel" doesn't help you recognise a name.

**3 · The three-way ensemble scored worse than its own baseline.** Union voting over spaCy + LegalBERT + Presidio gave 0.553, against 0.851 for either fine-tune alone. The mechanism is clean: Presidio's ORG precision is 0.14, and under `min_votes=1` every one of its false positives lands in the output even when both fine-tunes correctly rejected it. **The ensemble inherited its worst member.** Fixing the vote rule confirms the diagnosis — consensus recovers to 0.795 — but no combiner beats the single fine-tune, because it is Pareto-dominant across all eight labels, so per-label routing collapses to it alone.

→ [`figures/ensemble_backfire.png`](figures/ensemble_backfire.png) · [`docs/notes/advanced-training.md`](docs/notes/advanced-training.md)

---

## The mosaic claim, corrected

This project's most-quoted number was **"1,268 / 1,268 TAB documents are uniquely identifiable"**. It is withdrawn. It was an artefact of how the fingerprint was matched, and the tell was visible in the output all along: *every* k threshold returned 100%, including k ≤ 10. A measurement with no spread in it is usually measuring its own construction.

### What was wrong

`k_anonymity_table` counted a document as re-identifiable when no other document shared its **entire** quasi-identifier set. The median set holds ~14 facts. Two distinct court judgments matching on all fourteen exactly is close to impossible, so k = 1 was very nearly guaranteed before any data was read.

The composition made it worse. **63% of the fingerprint is DATETIME, and 85% of those normalise to a bare year** — "1999", "2003". Procedural years identify a *case*, not a person, and they are shared widely: "1999" appears in 55 documents.

### The control that settles it

Replace every real fact with a meaningless token from a fixed pool, keeping each document's set **size** identical:

```
real facts                    k=1 for 1,268/1,268 (100.0%)
random facts, same set sizes  k=1 for 1,268/1,268 (100.0%)   ← control
```

Meaningless data scores identically. The metric was measuring set size, not identifiability.

### What survives, and it is still a real finding

The right question is not "is the whole fingerprint unique" but **"how many facts must an attacker learn, rarest first, before the target stands alone?"** That has an actual distribution:

| Attribute set | n | median facts | 1 fact | 2 facts | 3 facts | never unique |
|---|---|---|---|---|---|---|
| All (DEM + DATETIME + LOC + QUANTITY) | 1,268 | 14 | 81.2% | 93.3% | 96.5% | 0.9% |
| **Person-like only (DEM + LOC)** | **887** | **4** | **76.8%** | **87.5%** | **88.3%** | **11.3%** |
| Demographics only (DEM) | 378 | 3 | 80.2% | 88.4% | 89.2% | 10.6% |

Two things make this trustworthy where the old number wasn't. It has a **distribution** — 11% of documents are never unique, even given every fact. And it **survives deleting the dates and the money entirely**: restricting to nationality, ethnicity, occupation and location barely moves it, from 81% to 77% on one fact. The finding was never actually driven by the years; the old metric just made it impossible to tell.

The design consequence is unchanged, which is why the product was built the way it is: **one ordinary residual detail is usually enough, so anonymisation needs a re-identification check and not just entity masking.**

### The caveat that now leads

Uniqueness here is uniqueness **within a 1,268-document corpus**, not within a population. A "Bulgarian nurse" may be alone among these judgments and one of many thousands in the world. This measures how distinguishable the documents are from each other; it is an upper bound on real-world re-identification risk, not a re-identification rate. Any honest use of this result has to say so.

Reproduce: `python scripts/diagnose_mosaic_claim.py` → [`results/mosaic_diagnosis.csv`](results/mosaic_diagnosis.csv)

---

## Ablations: which knobs actually moved the number

Reconciling the LegalBERT discrepancy raised an uncomfortable question. If a tokenisation setting was worth 0.5 pp, and the gap between the two backbones was 0.2 pp and not statistically distinguishable, then **what was actually driving the score?** So I swept it properly.

### Inference window and overlap

Long documents are processed as overlapping sliding windows. `max_length` sets the window, `stride` the overlap. Neither needs retraining — this is pure inference-time data preparation.

| max_length | stride | Partial F1 | 95% CI | Δ vs trained config | real? | predictions |
|---|---|---|---|---|---|---|
| 512 | 0 | **0.8624** | [0.8472, 0.8771] | +0.0113 | yes | 20,913 |
| 384 | 0 | 0.8616 | [0.8475, 0.8752] | +0.0106 | yes | 20,912 |
| 256 | 0 | 0.8560 | [0.8422, 0.8690] | +0.0049 | no | 20,698 |
| 512 | 64 | 0.8559 | [0.8413, 0.8702] | +0.0049 | no | 21,336 |
| 512 | 128 | 0.8519 | [0.8368, 0.8670] | +0.0009 | no | 21,561 |
| 384 | 64 | 0.8510 | [0.8367, 0.8649] | — *(as trained)* | — | 21,460 |
| 256 | 64 | 0.8413 | [0.8267, 0.8556] | −0.0097 | yes | 22,071 |
| 384 | 128 | 0.8393 | [0.8236, 0.8550] | −0.0117 | yes | 22,121 |
| 256 | 128 | 0.8219 | [0.8064, 0.8378] | −0.0291 | yes | 23,161 |

**Spread across configs: 0.0405 F1 — eighteen times the 0.0022 gap between the two fine-tuned backbones.** The sweep also reproduces the notebook's headline 0.8510 exactly at the config it was trained with, which is what makes the rest of the table trustworthy rather than harness drift.

Overlap is monotonically harmful at every window size, and prediction count rises with it: 20,913 spans at stride 0 against 23,161 at the worst setting. More overlap, more spans, more false positives.

### The mechanism — and why the fix isn't "tune the window"

Overlap exists for a good reason: it stops entities being split at a window boundary. It backfires here because of a de-duplication mismatch between two parts of the codebase:

- `predictors.make_finetuned_predictor` drops **exact** duplicates — spans identical in (start, end, type). **Every reported F1 was computed on this.**
- `pipeline.Pipeline._dedupe_overlapping` drops **overlapping** spans, longest wins. **This is what a real document goes through.**

The same entity seen in two adjacent windows comes back with slightly different boundaries. Not an exact duplicate, so evaluation keeps both and scores one as a false positive — while the product would have collapsed them into one.

| Model | As evaluated | Overlap-deduped | Δ | 95% CI | real? |
|---|---|---|---|---|---|
| spaCy | 0.5656 | 0.5656 | +0.0000 | — | no |
| Presidio | 0.6031 | 0.6032 | +0.0001 | [−0.0000, +0.0002] | no |
| LegalBERT | 0.8538 | 0.8588 | +0.0051 | [+0.0037, +0.0066] | yes |
| RoBERTa | 0.8559 | **0.8636** | +0.0077 | [+0.0058, +0.0098] | yes |

382 of the 387 spans removed for RoBERTa were false positives. spaCy and Presidio are untouched because neither uses sliding windows — exactly the signature the explanation predicts.

Put the two tables together and the window setting stops being interesting:

```
384/64, as trained, exact-match dedup      0.8510
512/0,  no overlap, exact-match dedup      0.8624   +0.0113
512/64, overlap + overlap-aware dedup      0.8636   +0.0126
```

**Overlap was never the problem. The de-duplication rule was.** Fix the dedup and the hyperparameter stops mattering — which is a better outcome than a tuned window, because it removes a setting someone would otherwise have to get right.

### What this says about the project

The honest summary is uncomfortable and worth stating plainly: **the choices that moved the number were not the ones that got a phase of attention.** Backbone selection was inside the noise floor. Data preparation and prediction post-processing were worth 3–18× more, and one of them was a mismatch between the evaluation path and the shipped path that no amount of model comparison would have surfaced.

*Not yet applied.* Aligning the predictor's de-duplication with the pipeline's would change every reported number in this repo, so it is documented rather than silently patched — it is the top item in [Limits and next](#limits-and-next).

Reproduce: `python scripts/ablate_inference_config.py` · `python scripts/ablate_postprocessing.py` · `python scripts/ablate_training.py --seeds 42 43 44 45 46`

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
- **Re-identification is measured within the corpus, not a population.** Uniqueness among 1,268 judgments is not uniqueness among people. The curve above bounds how distinguishable these documents are; it is not a re-identification rate. Fingerprints also match on exact surface form after normalisation, where a real attacker would match fuzzily.
- **No human ceiling established.** TAB has non-trivial annotator disagreement on identifier role. Without a human-vs-human F1, we don't know how close 0.851 is to the realistic maximum — which also means we don't know when to stop optimising.
- **Text in, text out.** No PDF/DOCX extraction, no layout handling, no cross-document linking, no service packaging.

**Next, in priority order.**
1. ~~Confidence intervals and paired significance tests.~~ **Done** — see [Results](#results). The LegalBERT discrepancy is resolved and its cause fixed.
2. **Align the predictor's de-duplication with the pipeline's.** The evaluation path drops only exact-duplicate spans; the shipped pipeline drops overlapping ones. Worth +0.0077 F1 for RoBERTa and it makes the reported numbers describe the system that actually runs. Changes every figure in this repo, so it wants doing deliberately.
3. **Fuzzy fingerprint matching** — sweep how the 100% finding moves as match strictness loosens. The sensitivity analysis this project most needs.
4. **Human annotator ceiling** — establish the realistic upper bound before chasing more F1.
5. **Cross-document pseudonymisation** — a persistent registry so the same person keeps the same token across a matter. Real key-management questions first.
6. **CRF head on the fine-tune** — closes the 0.85 partial vs 0.78 exact-match gap.

---

## License and acknowledgments

TAB is the work of Pilán et al. at the Norwegian Computing Center, distributed under its own license — see the [original repo](https://github.com/NorskRegnesentral/text-anonymization-benchmark). k-anonymity framing follows Sweeney (2002). spaCy by Explosion AI; Presidio by Microsoft.

This is a portfolio and research project. It is not a substitute for legal or compliance advice, and it is not a production-hardened product.

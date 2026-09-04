# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A portfolio research project on PII redaction for legal text, anchored in the **Text Anonymization Benchmark (TAB)**. It is organised **by function, not chronology** — one reusable package (`src/anonymisation/`) with `notebooks/` (01–17, in narrative order), `scripts/`, `results/`, `models/`, `figures/`, `docs/` beside it. The thesis it argues empirically: off-the-shelf NER is *not* sufficient to anonymise legal documents, and even perfect NER leaves a residual re-identification ("mosaic") risk. Three results are deliberately kept as **measured negative results** — the `12_null_coreference` and `14_null_ensemble` notebooks and the LegalBERT tie in `13_finetune_legalbert`. Do not "fix" them or delete them; the honest null is the finding. Files prefixed `null_` in `notebooks/` and `results/` are load-bearing.

Note the British spelling **"Anonymisation"** throughout (directory, package name `anonymisation`, module names). Match it.

## Environment & running code

The package **is installed** (editable) via `pyproject.toml` — no `sys.path` juggling and no `PYTHONPATH` needed:

```bash
pip install -e ".[research,dev]"
anonymise redact --variant lite path/to/doc.txt   # console entry point
```

Notebooks still carry a `sys.path.insert(0, "../src")` line for anyone who hasn't installed the package; it is harmless once it is installed. Pytest config lives in `pyproject.toml` under `[tool.pytest.ini_options]` (there is no `pytest.ini`).

The main virtualenv is **`legal-anon-env/`** (per the README; it holds spaCy 3.7 + the transformers stack). A second `.venv/` also exists. Setup:
```bash
source legal-anon-env/bin/activate
pip install -e ".[research,dev]"
python -m spacy download en_core_web_trf   # spaCy baseline / best off-the-shelf NER
python -m spacy download en_core_web_lg    # required by Presidio's default NLP engine
```
First run downloads TAB from HuggingFace (~50 MB) and models; cached afterwards. Python is pinned to **3.11** (`.python-version`).

The **Gradio demo runs in its own isolated venv** (`demo/venv-gradio/`, gitignored) because Gradio 4.x conflicts with the main stack's deps. Never install Gradio into `legal-anon-env`. Launch with `./demo/run_gradio.sh` (bootstraps the venv on first run, serves at `localhost:7860`).

## Testing / validation

A `pytest` suite in `tests/` covers the pure-logic modules (pseudonymise round-trip, generalization levels, mosaic scorer, span evaluation, role classification, coref extender, ensemble voting). Run it with:
```bash
legal-anon-env/bin/python -m pytest        # config in pyproject.toml puts src/ on the path
```
It needs no models or network — fast and safe to run on every change. It does **not** cover the model-dependent paths (NER backends, fine-tuning, the notebooks). GitHub Actions runs this suite on every push to `main` (`.github/workflows/tests.yml`, deps `pytest + pandas` only — the package imports without `datasets`/`torch` because those imports are lazy).

For an end-to-end sanity check of the actual product (not just metrics), `demo/worked_example.py` runs one document through Lite / Pro / pseudonymise with the fine-tuned NER and prints the redactions + audit log.

End-to-end / metric validation is still manual:
- Re-run the relevant notebooks and check the reported metrics.
- The standalone evaluation script `scripts/evaluate_mention_recall.py` (run with `--sample N` for a fast subset; full TAB test split is 555 docs, ~8 min on `en_core_web_trf`).
- `scripts/evaluate_coref_links.py` scores the pseudonymiser's linking against TAB's gold `entity_id` clusters (`--profile` interrogates the annotation, `--fit` refits `LINK_CONFIDENCE`). No models, ~30 s.
- Every metric CSV/JSON lands in the single top-level `results/` directory.

When changing pipeline logic, run `pytest` and re-run the relevant walkthrough notebook to confirm the headline metric hasn't silently regressed.

## Architecture

### The predictor contract
Every NER backend (spaCy, HuggingFace, Presidio, fine-tuned RoBERTa/LegalBERT, ensemble) is normalised to one interface — a callable:
```
text -> List[(start_char, end_char, tab_type, span_text)]
```
This is the seam the whole package is built on. `src/anonymisation/predictors.py` holds the adapters; the CLI's `build_ner_provider()` constructs them lazily (so a spaCy-only install still works). `tab_type` is one of TAB's 8 entity types, mapped from each model's native label set via `mapping.py`.

### The pipeline (`src/anonymisation/pipeline/`)
`Pipeline` (base.py) is the shared engine; `LitePipeline` and `ProPipeline` override `_redact()`:

1. **`detect_spans()`** (shared): run NER provider → regex pass (`regex_pass.py`: case numbers, IBANs, phones, emails) → optional coref extension (`coref.py`) → dedupe overlapping spans (longest wins, source priority regex > ner > coref) → classify each span's **identifier role** (`roles.py`: `DIRECT` = must remove vs `QUASI` = quasi-identifier, may be generalized).
2. **Lite** redacts `DIRECT` spans only — cheap, no mosaic handling.
3. **Pro** redacts `DIRECT`, then runs an **iterate-until-safe loop** over `QUASI` spans: a `MosaicScorer` (`scorer.py`) measures k-anonymity against a haystack corpus (TAB test set by default), and `generalize()` (`generalization.py`) progressively broadens quasi-identifiers (level 0 = original → ≥3 = suppressed) until `k_target` is reached or `max_iterations` hits.

Replacements are applied in **reverse offset order** so earlier character offsets stay valid (`apply_replacements`).

### Pseudonymisation & round-trip
With `pseudonymise=True`, redacted entities become stable referential tokens (`[PERSON_A]`, `[PERSON_B]`, …) instead of flat `[TYPE]` tags, and a **pseudonym vault** (token → original surface form) is returned on the result.

Deciding *which* mentions share a token is a measured decision, not a heuristic (notebook 17). `classify_link_evidence` names the rule that fired — exact repeat, honorific-stripped equality, containment in either direction — and `LINK_CONFIDENCE` holds that rule's precision per entity type, fitted by `scripts/evaluate_coref_links.py` on TAB train+validation. `Pseudonymiser` merges only above `min_link_confidence` (0.5) and refuses a merge when two entities match equally well; every resolution is recorded as a `LinkDecision` on `result.pseudonym_links`. `link_policy="legacy"` reproduces the old first-match-wins behaviour for comparison. **Do not raise the threshold to maximise pairwise F1** — F1 peaks at 0.30 and the default is deliberately 0.50, because a false merge makes `restore()` write the wrong person's name into the output while a missed merge only costs context. The intended workflow: redact locally → send pseudonymised text to an external LLM → `restore()` the LLM's answer locally using the vault, so real names never leave the firm. CLI: `redact --pseudonymise --vault-out vault.json`, then `restore --vault vault.json`.

### Data model (`pipeline/types.py`)
`Span` (offset + entity_type + identifier_role + source + generalization_level + replacement), `AuditEntry` (every redact/generalize/leave decision + rationale, exposed for compliance review), and `RedactionResult` (redacted_text, spans, audit, mosaic risk before/after, pseudonym_vault, pseudonym_links, `to_dict()` for JSON). The **audit log is a first-class output**, not debug logging — preserve it through any refactor.

### Other key modules
- `mosaic.py` — k-anonymity over QUASI fingerprints; the "1,268/1,268 docs uniquely identifiable" finding.
- `evaluation.py` — span-level precision/recall/F1 (partial + exact match). Partial-match F1 is the headline metric reported throughout.
- `iob.py` — BIO tagging utilities for the fine-tuning notebooks.
- `device.py` — picks CUDA → MPS → CPU.

## Models

Fine-tuned checkpoints live in `models/` (gitignored, ~8.8 GB): `models/roberta-tab/final` (best detector, used by the demo) and `models/legalbert-tab/final`. `demo/app.py` falls back to spaCy when they're absent — which is why the hosted Space currently runs a weaker model than the reported numbers.

## Deployment / sync

`spaces/` is the HuggingFace Spaces deployment bundle; `Anonymiser/` is a **nested git clone of the Space** (gitignored, not part of this repo). Refresh the Space via `./spaces/sync.sh` then copy into `Anonymiser/` — don't edit `Anonymiser/` as if it were source.

Figures for the writeup are regenerated by scripts (`scripts/build_performance_summary.py`, `demo/build_showcase.py`) rather than hand-edited.

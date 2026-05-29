# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A portfolio research project on PII redaction for legal text, anchored in the **Text Anonymization Benchmark (TAB)**. It is structured as six sequential "phases" (notebooks + analysis) sitting on top of one reusable package (`src/anonymisation/`). The thesis it argues empirically: off-the-shelf NER is *not* sufficient to anonymise legal documents, and even perfect NER leaves a residual re-identification ("mosaic") risk. Several phases (5, 6) are deliberately kept as **measured negative results** — do not "fix" them or delete them; the honest null is the finding.

Note the British spelling **"Anonymisation"** throughout (directory, package name `anonymisation`, module names). Match it.

## Environment & running code

There is **no `pyproject.toml`/`setup.py`** — the package is never installed. It is imported by putting `src/` on the path:

- **Notebooks** do `sys.path.insert(0, "../src")` in their first cell.
- **CLI / scripts** need `PYTHONPATH=src`, e.g.:
  ```bash
  PYTHONPATH=src python -m anonymisation.cli redact --variant lite path/to/doc.txt
  ```

The main virtualenv is **`legal-anon-env/`** (per the README; it holds spaCy 3.7 + the Phase 2 transformers stack). A second `.venv/` also exists. Setup:
```bash
source legal-anon-env/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf   # spaCy baseline / best off-the-shelf NER
python -m spacy download en_core_web_lg    # required by Presidio's default NLP engine
```
First run downloads TAB from HuggingFace (~50 MB) and models; cached afterwards. Python is pinned to **3.11** (`.python-version`).

The **Gradio demo runs in its own isolated venv** (`demo/venv-gradio/`, gitignored) because Gradio 4.x conflicts with the main stack's deps. Never install Gradio into `legal-anon-env`. Launch with `./demo/run_gradio.sh` (bootstraps the venv on first run, serves at `localhost:7860`).

## Testing / validation

A `pytest` suite in `tests/` covers the pure-logic modules (pseudonymise round-trip, generalization levels, mosaic scorer, span evaluation, role classification, coref extender, ensemble voting). Run it with:
```bash
legal-anon-env/bin/python -m pytest        # config in pytest.ini puts src/ on the path
```
It needs no models or network — fast and safe to run on every change. It does **not** cover the model-dependent paths (NER backends, fine-tuning, the notebooks). GitHub Actions runs this suite on every push to `main` (`.github/workflows/tests.yml`, deps `pytest + pandas` only — the package imports without `datasets`/`torch` because those imports are lazy).

For an end-to-end sanity check of the actual product (not just metrics), `demo/worked_example.py` runs one document through Lite / Pro / pseudonymise with the fine-tuned NER and prints the redactions + audit log.

End-to-end / metric validation is still manual:
- Re-run the phase notebooks and check the reported metrics.
- The standalone evaluation script `phase5_coreference/evaluate_mention_recall.py` (run with `--sample N` for a fast subset; full TAB test split is 555 docs, ~8 min on `en_core_web_trf`).
- Phase result CSVs land in each phase's `results/` directory.

When changing pipeline logic, run `pytest` and re-run the relevant phase's walkthrough notebook to confirm the headline metric hasn't silently regressed.

## Architecture

### The predictor contract
Every NER backend (spaCy, HuggingFace, Presidio, fine-tuned RoBERTa/LegalBERT, ensemble) is normalised to one interface — a callable:
```
text -> List[(start_char, end_char, tab_type, span_text)]
```
This is the seam the whole package is built on. `src/anonymisation/predictors.py` holds the adapters; the CLI's `build_ner_provider()` constructs them lazily (so a Phase-1-only install with just spaCy still works). `tab_type` is one of TAB's 8 entity types, mapped from each model's native label set via `mapping.py`.

### The pipeline (`src/anonymisation/pipeline/`)
`Pipeline` (base.py) is the shared engine; `LitePipeline` and `ProPipeline` override `_redact()`:

1. **`detect_spans()`** (shared): run NER provider → regex pass (`regex_pass.py`: case numbers, IBANs, phones, emails) → optional coref extension (`coref.py`, Phase 5) → dedupe overlapping spans (longest wins, source priority regex > ner > coref) → classify each span's **identifier role** (`roles.py`: `DIRECT` = must remove vs `QUASI` = quasi-identifier, may be generalized).
2. **Lite** redacts `DIRECT` spans only — cheap, no mosaic handling.
3. **Pro** redacts `DIRECT`, then runs an **iterate-until-safe loop** over `QUASI` spans: a `MosaicScorer` (`scorer.py`) measures k-anonymity against a haystack corpus (TAB test set by default), and `generalize()` (`generalization.py`) progressively broadens quasi-identifiers (level 0 = original → ≥3 = suppressed) until `k_target` is reached or `max_iterations` hits.

Replacements are applied in **reverse offset order** so earlier character offsets stay valid (`apply_replacements`).

### Pseudonymisation & round-trip (Phase 4)
With `pseudonymise=True`, redacted entities become stable referential tokens (`[PERSON_A]`, `[PERSON_B]`, …) instead of flat `[TYPE]` tags, and a **pseudonym vault** (token → original surface form) is returned on the result. The intended workflow: redact locally → send pseudonymised text to an external LLM → `restore()` the LLM's answer locally using the vault, so real names never leave the firm. CLI: `redact --pseudonymise --vault-out vault.json`, then `restore --vault vault.json`.

### Data model (`pipeline/types.py`)
`Span` (offset + entity_type + identifier_role + source + generalization_level + replacement), `AuditEntry` (every redact/generalize/leave decision + rationale, exposed for compliance review), and `RedactionResult` (redacted_text, spans, audit, mosaic risk before/after, pseudonym_vault, `to_dict()` for JSON). The **audit log is a first-class output**, not debug logging — preserve it through any refactor.

### Other key modules
- `mosaic.py` — k-anonymity over QUASI fingerprints; the Phase 1 "1,268/1,268 docs uniquely identifiable" finding.
- `evaluation.py` — span-level precision/recall/F1 (partial + exact match). Partial-match F1 is the headline metric reported across phases.
- `iob.py` — BIO tagging utilities for the fine-tuning notebooks.
- `device.py` — picks CUDA → MPS → CPU.

## Deployment / sync

`spaces/` is the HuggingFace Spaces deployment bundle; `Anonymiser/` is a **nested git clone of the Space** (gitignored, not part of this repo). Refresh the Space via `./spaces/sync.sh` then copy into `Anonymiser/` — don't edit `Anonymiser/` as if it were source.

Figures for the writeup are regenerated by scripts (`figures/build_performance_summary.py`, `demo/build_showcase.py`) rather than hand-edited.

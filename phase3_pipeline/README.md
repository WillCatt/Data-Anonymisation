# Phase 3 — Production Pipeline (Lite + Pro)

**Status:** Code complete. Walkthrough notebooks ready to run; Gradio demo ready to launch locally; HuggingFace Spaces deployment pending.

Phase 3 turns the measurements from Phases 1 and 2 into something a firm could actually deploy. The headline design choice: **two variants, not one**, because real-world anonymisation needs are heterogeneous.

## The two pipelines

### `LitePipeline` — DIRECT-only redaction

Strips every span the role classifier marks as DIRECT (PERSON, ORG, CODE, MISC by default; structured identifiers from the regex pass) and replaces with `[TAB_TYPE]`. Quasi-identifiers (dates, locations, demographics, quantities) are left intact.

**Use case:** the firm's threat model is "an LLM provider could log our prompts" or "a curious vendor employee could see this matter file". Lite gets the obvious PII out of the document before it leaves the firm's network. Doesn't try to defend against mosaic re-identification.

**Cost:** O(text length). No corpus comparison, no iteration. Predictable latency.

### `ProPipeline` — DIRECT + iterate-until-safe QUASI generalization

Everything Lite does, then runs a mosaic-aware loop:

1. Build the residual QUASI fingerprint from the surviving (un-redacted) spans.
2. Ask the `MosaicScorer` how many haystack documents share that fingerprint (= *k*).
3. If *k* ≥ k_target (default 5), stop.
4. Otherwise, advance every QUASI span one generalization level deeper:
   - DATETIME: original → year → decade → suppressed
   - QUANTITY: original → round-10 → decade band → suppressed
   - LOC: original → country → continent → suppressed
   - DEM: original → broader category → suppressed
5. Recompute *k*. Repeat up to `max_iterations`. If we run out of room, fall back to full QUASI suppression.

**Use case:** the firm has a heavier privacy obligation — regulated industries, GDPR-sensitive jurisdictions, employee data, high-profile matters where redaction is a compliance artefact, not a hygiene step.

**Cost:** O(text length × iterations) plus a one-time O(haystack size) on construction. The scorer is cached, so multiple documents share the haystack scan.

## What's in this folder

```
phase3_pipeline/
├── README.md                       (this file)
├── notebooks/
│   ├── 01_lite_walkthrough.ipynb   Lite end-to-end on synthetic + TAB samples
│   ├── 02_pro_walkthrough.ipynb    Pro with the iteration trace exposed
│   └── 03_lite_vs_pro.ipynb        side-by-side on the same inputs
```

The actual pipeline code lives in `../src/anonymisation/pipeline/` — separated from the notebooks so it's reusable from the CLI and the Gradio demo.

## How to run

From the repo root, with your venv activated:

```bash
# Walkthrough notebooks
jupyter notebook phase3_pipeline/notebooks/

# CLI — quick redact
echo "Maria Petrova lives in Plovdiv." | python -m anonymisation.cli redact --variant lite -

# CLI — full Pro audit as JSON
python -m anonymisation.cli redact --variant pro --json some_file.txt

# Gradio demo (local)
pip install gradio
python demo/app.py
```

## Design choices worth flagging

A few decisions that came up while building this. Each is the kind of thing a hiring manager / engineering reviewer might ask about.

- **Conservative DIRECT/QUASI defaults.** When the NER model produces a span, we don't know whether it's a DIRECT or QUASI identifier — TAB has that annotation, real-world inputs don't. The defaults in `roles.py` over-assign to DIRECT (PERSON, ORG, CODE, MISC) on the principle that over-redacting is safer than under-redacting in this domain. Callers can override per-span via a callback.
- **Regex pass after NER, not before.** Structured identifiers (case numbers, IBANs, NHS numbers) are higher-precision via regex than via NER. We let regex matches take priority on collisions so the NER model's noisier ORG / MISC predictions don't displace a high-confidence regex hit.
- **"Advance every QUASI by one level" rather than "generalize the most-specific first".** The literature has cleverer strategies. The simple sweep converges in 2–3 iterations on most TAB documents and is much easier to reason about; a smarter strategy is the obvious Phase-3.1 improvement.
- **Suppression-only fallback when the loop fails to converge.** This is the safe option but it destroys document utility. A real deployment would surface the document to a human reviewer with the unsafe spans pre-highlighted instead. The audit log makes that follow-up workflow possible.

## What this phase does NOT do (deliberately)

- **No layout-aware extraction.** The pipeline is text-in, text-out. PDF/DOCX → text with offset preservation is a separate piece of plumbing that should sit upstream.
- **No human-in-the-loop UI.** The audit log makes review possible; the workflow tooling around it isn't built.
- **No FastAPI service.** The library + CLI surface is enough to integrate from any other process; a service wrapper is straightforward but adds a moving piece (auth, logging, deployment) that hasn't been done.
- **No on-prem packaging.** The pipeline is pure Python with no external API calls, so it *could* run on-prem. A Docker image and the supporting Helm chart aren't built.

These are all good Phase-3.1 work. Each one would be a few days; none of them changes the design of what's already here.

## Caveat about the mosaic haystack

The `MosaicScorer` answers "is this fingerprint unique within the haystack?". The haystack in our demo is TAB itself — 1,268 ECHR cases. In production it should be the firm's own matter database, because uniqueness within a public corpus is only a proxy for the question the firm cares about (uniqueness within their own data, where re-identification attacks would actually originate).

The TAB-as-haystack choice is a methodological stand-in. It's defensible because uniqueness in the smaller TAB corpus is a *lower* bound on uniqueness in the wider world, but a deployment would wire the scorer up to the firm's own data and re-tune k_target accordingly.

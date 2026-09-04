# Anonymiser Studio

The pipeline as an application: one document, every mode computed at once, and
every decision traceable back to the code that made it.

```bash
./demo/run_studio.sh          # http://127.0.0.1:7861
```

## Why this exists alongside `demo/app.py`

`demo/app.py` is a Gradio form — pick one variant, get one output. It cannot
answer the question people actually ask, which is *what is the difference
between the three modes on my document*. It also runs in `demo/venv-gradio`,
which **has no torch**, so it has never been able to load the fine-tuned model
and silently falls back to spaCy. The Studio runs in `legal-anon-env` and loads
the real fine-tune.

`app.py` is left untouched: it is what `spaces/sync.sh` deploys.

## What it shows

| Tab | What it answers |
|---|---|
| **Source** | What did the detector find, and which findings are direct identifiers vs quasi-identifiers |
| **Redact** | What `LitePipeline` removes, and what it deliberately leaves |
| **Anonymise** | What `ProPipeline`'s iterate-until-safe loop does, and whether it worked |
| **Pseudonymise** | Coded references, the vault, and the round trip back from an LLM answer |

Click any marked phrase for an inspector showing that entity's fate under all
three modes side by side, each with the pipeline's own audit rationale.

## Things it is careful about

**It states what it is running.** The badge shows the model, its F1, the device
and the inference window. If the fine-tune is missing it says the fallback is
weaker than the reported numbers rather than quietly serving worse results.

**The window is pinned to 384/64.** That is the config the reported F1
(0.8559, 95% CI [0.8475, 0.8644]) was measured at. `demo/app.py` never passed
these and ran at the 512 default — the same mismatch that produced the
two-numbers-for-one-model discrepancy in the evaluation path.

**The rendering is checked against the pipeline.** The server rebuilds each
mode's output from the source text plus the span replacements, and asserts the
result equals `RedactionResult.redacted_text`. If they ever disagree it renders
the pipeline's text and flags a mismatch rather than showing a plausible lie.

**It does not dress up a non-result.** When the mosaic loop exhausts its
iterations and falls back to full suppression, `k` jumps to the haystack size
only because the fingerprint is now empty. The panel reports that as a failure
to converge — which is what `ProPipeline` itself reports — instead of printing
a large safe-looking number. On the twelve bundled samples, **none converge**:
broadening runs `k = 1 → 1 → 1` and only moves when the last quasi-identifier is
deleted. That is the mosaic finding, visible in the product.

## A bug this surfaced

`AuditEntry` holds a **reference** to its `Span`, and `ProPipeline` keeps
mutating that span after appending each iteration's entry. So in
`result.to_dict()`, every iteration of a generalisation reports the *final*
replacement:

```
iter 1: replacement = '[LOC]'   rationale = "Generalized to level 1: 'Plovdiv' → 'Bulgaria'. Post-step k=1."
iter 2: replacement = '[LOC]'   rationale = "Generalized to level 2: 'Plovdiv' → 'Europe'.   Post-step k=1."
iter 3: replacement = '[LOC]'   rationale = "Generalized to level 3: 'Plovdiv' → '[LOC]'.    Post-step k=555."
```

The rationale string, formatted at the time, is the only place the intermediate
value survives. The audit log is a first-class compliance output, so it should
be able to evidence the steps of the loop it documents.

`src/` is unchanged — the fix is a snapshot of the span at append time
(`dataclasses.replace(s)`), which is one line plus a test. Until then this
server recomputes the trail by calling the same `generalize()` the pipeline
calls, which is exact rather than parsed out of prose.

## Environment

```bash
./demo/run_studio.sh                     # CPU — safe while a training run holds the GPU
ANON_DEVICE=mps ./demo/run_studio.sh     # once the GPU is free
ANON_BACKEND=legalbert ./demo/run_studio.sh
PORT=8000 ./demo/run_studio.sh
```

No new dependencies: `fastapi`, `uvicorn`, `torch` and `transformers` are
already in `legal-anon-env`. The front end is three static files and loads
nothing from a CDN, so it works offline.

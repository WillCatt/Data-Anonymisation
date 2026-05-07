# Phase 3 — Production Pipeline

**Status:** Planned. Not yet implemented.

Phase 2 will give us a fine-tuned model good enough to use. Phase 3 wraps that model in the kind of pipeline a law firm could actually plug into their document workflow.

## Design goals

A mid-sized firm has three concerns the model alone does not address:

1. **Mosaic / re-identification risk.** Even if every DIRECT identifier is masked, a document can still be uniquely identifiable from its QUASI fingerprint (see `notebooks/03_mosaic_effect.ipynb`). The pipeline needs a re-identification check, not just a NER pass.
2. **Auditability.** A redaction made by a model is a decision the firm is liable for. They need a per-entity log: what was masked, why, and at what confidence.
3. **Human-in-the-loop fallback.** When the model's confidence is low or the mosaic check flags residual risk, route the document to a human reviewer with the uncertain spans pre-highlighted.

## Planned architecture

```
        ┌──────────────────────────────────────────────────────────┐
        │  Document in (DOCX / PDF / TXT)                          │
        └─────────────────────┬────────────────────────────────────┘
                              ▼
                ┌───────────────────────────┐
                │  1. Text extraction       │   ← preserves offsets
                └────────────┬──────────────┘
                             ▼
                ┌───────────────────────────┐
                │  2. NER (Phase 2 model)   │   ← per-entity confidence
                └────────────┬──────────────┘
                             ▼
                ┌───────────────────────────┐
                │  3. Regex post-pass       │   ← case numbers, IBAN, NHS no.
                └────────────┬──────────────┘
                             ▼
                ┌───────────────────────────┐
                │  4. Mosaic risk scorer    │   ← k-anonymity over QUASI bag
                └────────────┬──────────────┘
                             ▼
                  ┌──────────┴──────────┐
                  │ k ≥ threshold?      │
                  ├──────────┬──────────┤
                 yes         │         no
                  │          │          │
                  ▼          │          ▼
         auto-redact         │   route to human reviewer
                  │          │          │
                  ▼          ▼          ▼
        ┌──────────────────────────────────────┐
        │  5. Audit log: every span, decision, │
        │     confidence, applied rule         │
        └──────────────────────────────────────┘
```

## Planned structure

```
phase3_pipeline/
├── README.md                       (this file)
├── api/
│   ├── main.py                     FastAPI service (POST /anonymise)
│   └── schemas.py                  Pydantic request/response
├── pipeline/
│   ├── extract.py                  PDF/DOCX → text (preserves char offsets)
│   ├── ner.py                      thin wrapper over the Phase 2 model
│   ├── regex_pass.py               UK case numbers, IBAN, NHS, etc.
│   ├── mosaic_scorer.py            re-uses src/anonymisation/mosaic.py
│   ├── redactor.py                 applies replacements; preserves layout
│   └── audit.py                    structured per-decision log
├── tests/
│   ├── fixtures/                   sample legal-style docs (synthetic)
│   └── test_pipeline.py
└── deploy/
    └── docker/Dockerfile
```

## Open questions for this phase (for the writeup, "what I'd do next")

* What's a defensible mosaic-risk threshold? k≥5 is the textbook answer, but legal docs are long and have many QUASI mentions, so k=5 may be unachievable without ruining the document. Some hybrid policy is probably needed.
* How do you measure "the firm trusts this enough to use it"? A user study with paralegals comparing time-to-redact under three workflows: manual / pipeline-only / pipeline-with-human-review.
* Hosted vs on-prem. Most law firms will not allow client matter data to leave their network. The pipeline needs to be deployable as a self-contained Docker image with no external API calls.

# Phase 5 — Coreference-Aware Span Extension

**Status:** Built, measured, honest negative result. The post-processor moves TAB mention-recall by essentially zero (+0.07pp on PERSON, +0.10pp on ORG, zero elsewhere — see the **measured result** section below). It remains a defensive layer worth keeping, but the recall lift this phase originally predicted didn't materialise.

## The gap this fills

Off-the-shelf NER (spaCy, dslim/bert-base-NER, even the Phase 2 fine-tuned RoBERTa) tags the *first full mention* of a named entity reliably but loses recall on the shorthand references that follow. From a redaction perspective, every miss is a leak of the entity's identity — the second mention of a name is no less identifying than the first.

A concrete example you can see in the showcase before/after this phase:

> *"Northwind Energy Ltd was founded in 1998. **Northwind** operates two plants in the UK. In 2019 **Northwind** acquired a smaller rival."*

Pre-Phase 5: only the first mention is redacted. Three later `Northwind`s leak.
Post-Phase 5: all four `Northwind` mentions redacted.

## How it works

`src/anonymisation/pipeline/coref.py` adds a post-processor that runs after NER + regex:

1. For every PERSON/ORG span the model already found, generate **candidate short forms**:
   - PERSON: surname alone, first name alone, honorific + surname.
   - ORG: first significant word (skipping `Ltd`, `Inc`, `Holdings`, `the`, etc.), or two-word prefix when both words are non-generic.
2. Scan the document for **word-boundary, case-sensitive** matches of each candidate.
3. Any match that doesn't overlap an existing span becomes a new `Span` with `source="coref"`, `confidence=0.7`, and the parent's `entity_type` / `identifier_role`.

It's an O(spans × doc_length) post-processor. No retraining, no extra model dependency.

## Defensive defaults — and an honest trade-off

The substring rule is fast but can misfire. We avoid the worst cases:

- Generic ORG-suffix words (`Ltd`, `Inc`, `Holdings`, `International`, `Group`, etc.) never become standalone candidates.
- Generic legal-doc words (`court`, `tribunal`, `commission`, `agency`) never become standalone candidates.
- Anything shorter than 3 characters is dropped (kills `Co`, `Mr`, `Inc` from ever firing standalone).
- Coref-derived spans are tagged `source="coref"` and `confidence=0.7`. A downstream consumer that wants to gate them can filter on confidence.

**The trade-off we accept:** `Sofia District Court` generates both `Sofia District` AND `Sofia` as candidates. The bare `Sofia` could match the city Sofia elsewhere in the document — a false positive. We accept this because in practice, when a document also references the city, the NER pass tags it as `LOC` independently and the pipeline's dedup rule keeps the LOC span. The remaining residual risk (NER missed both the city LOC and the ORG full mention) is rare. Erring toward higher recall is the right call for a redaction tool — a false positive over-redacts a single token, whereas a false negative leaks an identity.

For sharper coreference (pronouns, semantic equivalence, two `Smith`s referring to different people), the right next step is `fastcoref` or `spacy-coref`. Sketched in the walkthrough notebook as Phase 5.1.

## Evaluating against TAB

TAB's gold annotations include `entity_id` linking every mention of the same canonical entity. That gives us a clean way to measure mention-recall — for each gold entity, what fraction of its mentions did the pipeline catch?

```bash
# Quick smoke (50 docs, ~30 seconds)
python phase5_coreference/evaluate_mention_recall.py --sample 50

# Full TAB test split (555 docs, ~8 min on en_core_web_trf)
python phase5_coreference/evaluate_mention_recall.py
```

Output:
- `phase5_coreference/results/mention_recall.csv` — one row per gold entity per variant.
- `phase5_coreference/results/mention_recall_summary.json` — macro + micro recall, per-entity-type breakdown.

The script reports two numbers for both `baseline` (NER+regex, no coref) and `with_coref` (full pipeline):

- **Micro recall** — `total mentions caught / total gold mentions`. Weights every mention equally.
- **Macro recall** — average per-entity recall. Weights every entity equally regardless of how many times it's mentioned.

## The measured result — full TAB test, 17,448 entities

| | Baseline | + Coref extender | Lift |
|---|---|---|---|
| Macro recall | 0.8399 | 0.8400 | +0.0001 |
| Micro recall | 0.8360 | 0.8363 | +0.0003 |
| PERSON recall | 0.954 | 0.955 | +0.0007 |
| ORG recall    | 0.806 | 0.807 | +0.0010 |
| DATETIME / LOC / QUANTITY / DEM / MISC / CODE | unchanged | unchanged | 0 |

The heuristic does not measurably lift recall on TAB. Why:

- **TAB is densely human-annotated** — every coreferring mention of every entity is already tagged in the gold.
- **spaCy `en_core_web_trf` is already strong on PERSON / ORG / LOC.** Those labels hit ≥95% / 81% / 95% recall before the post-processor runs. There aren't many shorthand mentions left for the heuristic to discover; the model catches them directly.
- **The post-processor only generates candidates for PERSON and ORG** (by design — those are the labels where the substring rule is safest). Categories with genuine recall problems (DEM at 34%, MISC at 8%, CODE at 0.6%) get no lift from this intervention at all.

The post-processor is still defensible as a layer: it costs nothing at inference, the audit log shows exactly what it added, and on documents less densely annotated than TAB it would catch genuine misses. But it isn't the recall lift this phase originally predicted.

## How to use it

By default, `coref_extend=True` on both `LitePipeline` and `ProPipeline`. Disable explicitly if you need the older behaviour (e.g. for like-for-like comparison with the Phase 3 walkthrough numbers):

```python
lite = LitePipeline(ner_provider=ner, coref_extend=False)
```

CLI flag mirrors this:

```bash
python -m anonymisation.cli redact --variant lite --no-coref my_doc.txt
```

## Folder layout

```
phase5_coreference/
├── README.md                          (this file)
├── evaluate_mention_recall.py         TAB mention-recall evaluation
├── notebooks/
│   └── 01_coref_walkthrough.ipynb     walks through the candidate rules + lift
└── results/                           CSV + summary land here after running
```

The pipeline code lives in `../src/anonymisation/pipeline/coref.py` so the module is reusable from the CLI, the static showcase, and the consumer notebooks.

## What this phase does NOT do

- **No pronoun coref.** Substring rule doesn't catch "she", "he", "they". A real coref model does.
- **No semantic coref.** Two "Smith"s referring to different people will merge under the substring rule.
- **No retraining.** This is a post-processor. First-mention recall is unchanged; we only improve subsequent-mention recall.
- **No cross-document persistence.** Each document processed independently — same scope as Phase 4.

If the TAB lift is small in your run, the right next move is `fastcoref`. If it's large, you've added meaningful safety at zero training cost.

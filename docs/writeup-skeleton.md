# Anonymising Legal Text — Why Stripping Names Isn't Enough

*Portfolio writeup · restructured skeleton (3 acts + epilogue)*

> **Skeleton notes for the author.** Each section below has (a) the section's purpose in one line, (b) the key facts/numbers it must contain, (c) the figure(s) it references, and (d) target word count. Replace the bracketed prose seeds with finished writing. Aim ~3,000 words total for ~12 pages with figures.

---

## Hero / opening (1 page · 150 words)

**Purpose.** Commit a generalist reader to keep reading in 30 seconds.

**Must contain.** One sentence framing the problem, the three headline numbers, one CTA.

**Figure.** `figures/phase_overall_f1.png` (the trimmed 5-bar ladder) as the hero image.

**Prose seed:**

> Every law firm wants its paralegals using LLMs on client matter data. Every law firm also has a confidentiality obligation that makes pasting privileged documents into a third-party model into a regulatory breach, a contract breach, or a malpractice claim — usually all three. The cheap answer is "just strip the names before you send." This project is the empirical answer to whether that's enough.
>
> It isn't. Three headline numbers:
>
> - **Off-the-shelf NER misses ~43% of sensitive entities on TAB** (F1 = 0.566 partial-match, spaCy en_core_web_trf).
> - **Fine-tuning a transformer on TAB closes the gap** (F1 = 0.851, RoBERTa fine-tuned on TAB train, +28.5 pp).
> - **Even with perfect NER, 100% of TAB documents are still uniquely identifiable** from their residual demographic fingerprint alone — the mosaic effect.

---

# ACT I — THE PROBLEM

## 1. Why "strip the names" is the wrong question (1–1.5 pages · 350 words)

**Purpose.** Set up the law-firm framing and motivate the empirical investigation.

**Must contain.**
- The threat model: client matter data → LLM provider → logs/training.
- The cheap solution ("just NER it") and its appeal.
- Statement of what this project measures.
- Brief pointer to the dataset (TAB) and *why* TAB specifically — the DIRECT/QUASI/NO_MASK annotation is what makes the mosaic question answerable.

**Prose seed:** [lift from current writeup, sections "The premise" and "The dataset"]

## 2. The baseline gap (1.5 pages · 350 words)

**Purpose.** Show *quantitatively* that off-the-shelf NER doesn't meet the bar, and *categorically* where it leaks.

**Figure.** `figures/phase1_f1_scores.png` (per-entity F1 — partial vs exact match for spaCy baseline). *Drop later if duplicate of gap_closed chart.*

**Must contain.**
- The headline: F1 = 0.566 partial / 0.381 exact.
- Three concrete failure modes: **CODE 0% recall** (no label in the model's vocab), **DEM ~30% F1** ("asylum-seeker" is just a noun to spaCy), **QUANTITY high recall / terrible precision** (fires on every number).
- One sentence on why the failures *make sense* given how spaCy was trained — they cluster where TAB's labels diverge from newswire-trained vocabulary.

## 3. The mosaic effect (2 pages · 500 words)

**Purpose.** Deliver the project's most original finding.

**Figure.** `figures/mosaic_reidentification.png` (re-identification curve — 58% unique from 1 quasi-fact, 95% from 3, 1,268/1,268 at the full fingerprint — plus the signature-size histogram).

**Must contain.**
- The thought experiment: pretend the NER is perfect; every DIRECT identifier is masked. What's left?
- Definition of QUASI: nationality, age, occupation, location, dates — innocuous individually.
- k-anonymity intuition: a document with k=1 has a fingerprint unique within the corpus.
- **The number**: 1,268 of 1,268 TAB documents have k=1. None reach k≥5. Median signature size = 25 QUASI mentions.
- Why: the joint distribution of 25 demographic facts is effectively a hash.
- Caveat: exact surface-form matching — fuzzy matching would shift the number but not the qualitative finding.
- **Design consequence**: anonymisation is not an NER problem. The pipeline needs a re-identification check, not just entity masking.

**Prose seed:** [lift from current writeup section "Finding 3", update with corrected 100% number]

---

# ACT II — CLOSING THE MODEL GAP

## 4. The bake-off (2 pages · 500 words)

**Purpose.** Show that the model gap closes dramatically with fine-tuning, but that the obvious "more models is better" follow-ups don't pay off.

**Figures.**
- `figures/phase_overall_f1.png` (the trimmed 5-bar ladder — the headline)
- `figures/gap_closed_by_entity.png` (per-entity F1 lift, Phase 1 → Phase 2)

**Must contain.**

The five contenders, in order, with one sentence each:

1. **spaCy en_core_web_trf** (baseline) — 0.566 F1. The Phase 1 anchor.
2. **Presidio + CASE_NUMBER recogniser** — 0.603 F1. Regex closes the CODE recall gap (0% → 92%) but the overall F1 barely moves because Presidio's ORG precision is 0.14 (it fires on common nouns). Identical to stock Presidio at the headline level — the regex layer's contribution is localised, not global.
3. **RoBERTa fine-tuned on TAB** — **0.851 F1**. The win. +28.5 pp over baseline.
4. **LegalBERT fine-tuned on TAB** (Phase 6) — 0.849 F1. **Within noise of RoBERTa**, despite being pre-trained on 12 GB of legal text. The lesson: domain pretraining helps when task *vocabulary* differs, not when topic differs. TAB's labels (PERSON, ORG, …) are general-NER labels; the legal-domain advantage doesn't pay off.
5. **3-way ensemble (spaCy + LegalBERT + Presidio, union voting)** — 0.553 F1. **Worse than baseline.** This deserves its own section — see §6.

**Where the gap closed.** From `gap_closed_by_entity.png`:
- Biggest lifts: **CODE +92 pp** (0 → 92, from "no label" to fully recognised), **QUANTITY +51 pp**, **LOC +31 pp**, **ORG +35 pp**.
- Flat / regression: **PERSON −1 pp** (already at ceiling for both), **DATETIME +4 pp** (already 91%).
- The lift concentrates exactly where TAB has labels that newswire-trained NER doesn't — exactly where fine-tuning on a TAB-specific label set is supposed to help. The hypothesis the project entered with is confirmed.

**Footnote.** Also evaluated: `dslim/bert-base-NER`. Scored 0.17 F1. Its CoNLL2003 label set (PER/LOC/ORG/MISC) discards most of TAB's annotation by construction — not a meaningful comparison. Kept the notebook in the repo for completeness.

---

# ACT III — THE DEPLOYABLE PIPELINE

## 5. From measurement to system (3 pages · 750 words)

**Purpose.** Show the deployable engineering, not just the model. This is the chapter that proves "I can ship", not just "I can measure".

**Figures.**
- `figures/pipeline_architecture.png` (or .svg) — the system diagram
- `figures/round_trip_threat_model.png` (or .svg) — the pseudonymisation flow
- Optional: a screenshot of the Gradio demo with one example processed

### 5.1 Two threat models, two pipelines (200 words)

**Lite** — DIRECT-only redaction. Threat model: *"an LLM provider could log our prompts"*. Strips names, organisations, case numbers, structured identifiers. QUASI mentions intact. Fast (< 100ms) and predictable.

**Pro** — DIRECT + iterate-until-safe QUASI generalisation. Threat model: *"a determined adversary could re-identify the document"*. After DIRECT suppression, asks the MosaicScorer for k. If k < target, advance every QUASI span one generalisation level (47 → ~50 → 40s → [QUANTITY]; Plovdiv → Bulgaria → Europe → [LOC]) and recompute. Up to `max_iterations`; falls back to full suppression if it can't converge.

The point of two pipelines, not one: real-world anonymisation needs are heterogeneous. A firm can A/B them on the same matter notes and pick a default per matter type.

### 5.2 The pseudonymisation round-trip (250 words)

**The problem with plain redaction.** After Lite, `"Maria Petrova sued John Doe"` → `"[PERSON] sued [PERSON]"`. An LLM asked "who sued whom?" can no longer answer; three `[PERSON]`s look identical.

**Pseudonymise with `pseudonymise=True`.** Each distinct surface form gets a stable referential token: `[PERSON_A]`, `[PERSON_B]`, `[ORG_A]`. The vault (token → real name) stays at the firm. The LLM round-trips with tokens only, and a local `restore(answer, vault)` swaps them back.

This is the same pattern HIPAA Safe Harbor uses for clinical de-identification — and it's the pattern that makes "cloud LLMs on confidential data" defensible at all. The LLM provider's logs, retention policies, and training pipeline never see real client names.

[Insert `round_trip_threat_model.png` here.]

### 5.3 The audit log + Gradio UI (150 words)

Every span the pipeline touches is recorded in a per-decision JSON: source (NER/regex/coref), confidence, role (DIRECT/QUASI), generalisation level. This makes a paralegal-review workflow *possible* — they can see why every decision was made and correct mistakes.

The Gradio demo exposes Lite, Pro, and Pseudonymise as toggles; live mosaic-risk badge (`k_initial → k_final`) for Pro; vault tab when pseudonymise is on; full audit log tab.

[Insert demo screenshot here.]

### 5.4 What this phase does NOT do (150 words)

Explicit out-of-scopes — list five honestly:

1. PDF / DOCX → text extraction. Text-in, text-out.
2. Document layout (letterheads, footers, watermarks).
3. Cross-document pseudonymisation. Per-document scope.
4. Network-isolated deployment. Pure Python, runs on-prem in principle; no Docker / FastAPI service.
5. Active-learning loop from paralegal corrections. The audit log makes this possible; the workflow tooling isn't built.

---

# EPILOGUE — WHAT I TRIED THAT DIDN'T WORK

## 6. Three honest negatives (2 pages · 500 words)

**Purpose.** The credibility chapter. Most portfolio writeups skip this; yours leads with it.

**Figures.**
- `figures/ensemble_backfire.png` (the per-entity precision crater)
- Phase 5 mention-recall — *optionally include if narrative demands; chart's bars are visually equal-height so the prose has to carry it.*

### 6.1 The 3-way ensemble dropped overall F1 by 30 points (200 words)

The plan was textbook: combine three diverse predictors (spaCy + LegalBERT-FT + Presidio) and union-vote. Standard NER-literature move; standard reported lift of 2–5 F1.

**Result: 0.553 F1, vs 0.851 for either fine-tune alone.**

[Insert `ensemble_backfire.png` here.]

The mechanism is clean to diagnose. Presidio's ORG precision is **0.14** — it fires on common nouns, date fragments, generic verbs. Under union voting (`min_votes=1`), every Presidio false positive on ORG gets added to the ensemble's output, even though both fine-tunes correctly rejected it. The ensemble inherited its worst member's noise instead of its best member's accuracy.

**Two takeaways.**
- Ensembles only work when members are individually competent on every label they vote on. Presidio is competent on CODE/IBAN; dangerous on ORG.
- The fix is label-weighted ensembling (route per-label to the strongest predictor). Scoped as Phase 6.2; not built.

### 6.2 LegalBERT pretraining didn't beat RoBERTa (150 words)

Phase 6's other experiment: swap `roberta-base` for `nlpaueb/legal-bert-base-uncased` (BERT-base pre-trained on 12 GB of US court cases, EU legislation, contracts). Same fine-tune recipe, same evaluation.

**Result: 0.849 vs 0.851. A tie within noise.**

The widely-held assumption "domain pretraining always helps" is too coarse. Domain pretraining helps when the *task vocabulary* differs from the pretraining corpus. TAB's labels (PERSON, ORG, …) are general-NER labels; the legal-domain advantage of LegalBERT — knowing words like "estoppel" or "interlocutory" — doesn't translate to recognising people's names better. Different problem.

### 6.3 The coref post-processor moved recall by 0.0001 (150 words)

Hypothesis: NER under-recalls *shorthand* mentions ("Maria" after introducing "Maria Petrova"). Built a substring-rule post-processor; integrated it into the pipeline as `coref_extend=True`.

**Result on TAB: macro recall 0.8399 → 0.8400. Essentially zero.**

Why: spaCy's `en_core_web_trf` is already at ≥95% recall on PERSON and ORG. There aren't many shorthand mentions for the heuristic to discover; the model catches them directly. The categories with genuine recall problems (DEM at 34%, MISC at 8%, CODE at 0.6%) are exactly the ones the substring rule *can't* help with.

The post-processor stays in the pipeline as a defensive layer (costs nothing at inference, shows up in the audit log), but it doesn't earn a place in the headline metrics. **Measure before you celebrate.**

---

## 7. Limits, ceilings, and what I'd do next (1 page · 250 words)

**Purpose.** Position the project as a credible piece, not as a finished product.

**Must contain.** Three honest limits, three priority follow-ups.

**Limits.**
- **TAB is English-only, ECHR-only.** Findings sharp on this corpus; need cross-corpus validation (US commercial, UK contracts) before claiming generalisability.
- **Mosaic fingerprints match on exact surface form.** A real attacker matches fuzzily; the 100% finding is conservative in one direction (real risk is at least this bad) and optimistic in another (the attacker's haystack is bigger and noisier than TAB).
- **Annotator-disagreement ceiling.** TAB has non-trivial inter-annotator disagreement on identifier role; computing human-vs-human F1 would establish the realistic upper bound. Probably shows the fine-tune is closer to the human ceiling than to 1.0.

**Priority follow-ups.**
1. **Fuzzy fingerprint matching for the mosaic scorer.** Sweep "how does 100% change as match strictness loosens?" — exactly the kind of sensitivity analysis a thoughtful reviewer asks for.
2. **Label-weighted ensemble.** Tests whether the failure mode was union-voting specifically (and not ensembling in principle). Likely beats RoBERTa-FT by 1–3 F1.
3. **CRF head on the fine-tune.** Closes the 0.85 partial vs 0.77 exact-match F1 gap. Standard NER-literature move.

---

## 8. Things this project is NOT (½ page · 150 words)

**Purpose.** Self-aware framing — limits the reader's expectations explicitly.

**Lift verbatim from current writeup section "Things this project is NOT".** The three points (not a deployable product / not legal advice / not novel research) are well-pitched and worth keeping.

---

## Acknowledgments + sources (½ page)

- TAB: Pilán et al., Norwegian Computing Center.
- spaCy: Explosion AI.
- k-anonymity framing: Latanya Sweeney, 2002.
- GitHub repo: [link]
- Live demo (HuggingFace Spaces): [link]

---

# FIGURE INVENTORY — WHAT'S CITED

| § | Figure | Status |
|---|---|---|
| Hero | `phase_overall_f1.png` (trimmed 5-bar) | ✅ Built |
| §2 | `phase1_f1_scores.png` | Existing |
| §3 | `mosaic_reidentification.png` (re-id curve + signature sizes) | ✅ Built |
| §4 | `phase_overall_f1.png` + `gap_closed_by_entity.png` | ✅ Built |
| §5 | `pipeline_architecture.png` + `round_trip_threat_model.png` | ✅ Built |
| §5.3 | Demo screenshot | TODO — record + crop |
| §6.1 | `ensemble_backfire.png` | ✅ Built |
| §6.3 | (no figure — prose carries it) | n/a |

# WHAT TO CUT FROM THE FIGURES FOLDER

After this restructure, these become dead weight (kept in repo for reproducibility, removed from the writeup):

- `phase2_overall_f1.png`, `phase2_f1_by_entity.png` — subsumed by cross-phase charts.
- `phase6_overall_f1.png`, `phase6_f1_by_entity.png` — subsumed by cross-phase charts.
- `phase1_precision_recall.png` — duplicates `phase_precision_recall.png`.
- `phase1_gap_analysis.png` — likely predates the multi-phase view; assess.
- `phase5_mention_recall.png` — only if §6.3 stays prose-only.

# DRAFT-FILL ORDER

Recommended order to write the prose:

1. §3 (mosaic) — your most original claim, biggest payoff to nail.
2. §4 (bake-off) — straightforward; you have the numbers.
3. §6 (negatives) — the credibility chapter; write while the analysis is fresh.
4. §5 (pipeline) — engineering, has lots of material in the existing READMEs.
5. §1, §2 (premise + baseline) — lift from existing writeup with light editing.
6. Hero — write LAST. Once the rest exists, this is a one-shot distillation.

# WORD-COUNT BUDGET

| Section | Words | Pages (with figures) |
|---|---|---|
| Hero | 150 | 1 |
| §1 premise | 350 | 1.5 |
| §2 baseline | 350 | 1.5 |
| §3 mosaic | 500 | 2 |
| §4 bake-off | 500 | 2 |
| §5 pipeline | 750 | 3 |
| §6 negatives | 500 | 2 |
| §7 limits | 250 | 1 |
| §8 not | 150 | 0.5 |
| **Total** | **~3,500** | **~12** |

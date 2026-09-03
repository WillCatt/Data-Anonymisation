# Phase 4 — Pseudonymisation & Round-Trip Restore

**Status:** Code complete. Walkthrough notebook + static showcase Sample 4 demonstrate the full pattern.

Phase 3 redaction is good for "I don't want the LLM provider to see this person's name". Phase 4 is for "I also want the LLM to be able to *answer questions* about the document". The two are not the same.

## Why Phase 4 exists

Plain Phase 3 collapses every named entity to a generic tag:

> *"Maria Petrova sued John Doe for breach of the Acme contract"*

becomes

> *"[PERSON] sued [PERSON] over the [ORG] contract"*

A downstream LLM cannot answer "**who** sued **whom**?" from that. The referential structure is gone. What you needed wasn't redaction *as such* — it was redaction that **preserves identity-of-mention** while hiding the literal names.

That's pseudonymisation:

> *"[PERSON_A] sued [PERSON_B] over the [ORG_A] contract"*

Same surface form gets the same token. The mapping (`vault`) stays with the firm. The LLM sees only opaque tokens. When the answer comes back, the firm runs `restore()` locally to swap tokens for real names. The LLM provider never holds the secret.

## How it slots into the existing pipelines

A new `pseudonymise=True` flag on **both** `LitePipeline` and `ProPipeline`:

```python
from anonymisation.pipeline import LitePipeline, restore

# Forward — redact + pseudonymise
lite = LitePipeline(ner_provider=my_predictor, pseudonymise=True)
result = lite("Maria Petrova lives in Sofia. Petrova works for Acme.")

print(result.redacted_text)
# → "[PERSON_A] lives in Sofia. [PERSON_A] works for [ORG_A]."

print(result.pseudonym_vault)
# → {"[PERSON_A]": "Maria Petrova", "[ORG_A]": "Acme"}

# Reverse — round-trip an LLM answer
llm_answer = "[PERSON_A] is the plaintiff."
print(restore(llm_answer, result.pseudonym_vault))
# → "Maria Petrova is the plaintiff."
```

Pro mode supports the flag too — DIRECTs get pseudonymised; QUASI generalisation is unchanged.

## What's in this folder

```
phase4_pseudonymisation/
├── README.md                                  (this file)
└── notebooks/
    └── 01_pseudonymisation_walkthrough.ipynb  forward + round-trip + edge cases
```

The pipeline code itself lives in `../src/anonymisation/pipeline/pseudonymise.py` so it's reusable from the CLI, the static showcase, and any consumer of the package.

## CLI surface

```bash
# Redact with pseudonymisation, write the vault to disk
python -m anonymisation.cli redact --variant lite --pseudonymise \
    --vault-out vault.json my_doc.txt > redacted.txt

# Send redacted.txt to an LLM with a question. The LLM returns answer.txt.

# Round-trip the answer back to original surface forms
python -m anonymisation.cli restore --vault vault.json answer.txt > restored.txt
```

## Coreference — the substring rule

The hard part of pseudonymisation is recognising that "Maria Petrova", "Mrs Petrova", and "Petrova" all refer to the same person. The `Pseudonymiser` uses a deliberately simple rule:

> Two surface forms refer to the same entity if one appears, **as whole words**, inside the other — and they're the same entity type.

Concrete behaviour:

| First seen | Then seen | Token | Notes |
|---|---|---|---|
| Maria Petrova | Maria | PERSON_A | substring match |
| Maria Petrova | Petrova | PERSON_A | substring match |
| Maria Petrova | Mrs Petrova | PERSON_A | substring match |
| Maria Petrova | John Smith | PERSON_B | distinct |
| Acme Holdings Ltd | Acme | ORG_A | substring match |
| Plovdiv (LOC) | Plovdiv (DEM) | LOC_A + DEM_A | different types — distinct tokens |

This is fast and predictable but it has known limits:

- Two different "Smith"s in the same document would incorrectly merge.
- Initialism gaps ("International Business Machines" / "IBM") won't merge.
- Pronoun coref ("she", "he", "they") is out of scope — the NER step doesn't tag them as PERSON.

Drop in a proper coref model (e.g. spaCy-coref) if these matter. The Pseudonymiser's API is small enough that swapping the matcher is local.

## Why DIRECTs only?

Pseudonymisation in Pro mode is applied to DIRECT spans only — QUASIs continue down the generalisation path (level 1 → level 2 → level 3 suppression). Two reasons:

1. **Conceptual orthogonality.** Pseudonymisation is "this is a name we're hiding"; generalisation is "this is a place we're broadening". Conflating the two — generating `[LOC_A]`, `[LOC_B]` for individual cities — adds expressive richness that the iterate-until-safe loop can't easily reason about.
2. **Mosaic risk.** Pseudonymising QUASIs would *preserve* their identifying joint distribution under stable tokens — the very thing the mosaic loop is trying to break. The two features would fight each other.

If a deployment really wants pseudonymised QUASIs, that's a different design (probably involving k-anonymous *equivalence classes* rather than per-entity tokens). Out of scope here.

## What this phase does NOT do

- **Cross-document persistence.** Each `Pseudonymiser` instance is independent. "Maria Petrova" in document A and document B will both probably be `[PERSON_A]`, but those tokens are not interchangeable. A persistent registry is straightforward to build but raises real key-management questions (where does the registry live? who can read it? what's the rotation policy?).
- **Vault encryption.** The vault is the secret. The CLI writes it to plain JSON because that's appropriate for a portfolio demo; in production it should be encrypted at rest, audited, and treated as a session key.
- **Smart coref.** See above — substring rule only.

## See also

- `phase4_pseudonymisation/notebooks/01_pseudonymisation_walkthrough.ipynb` — full walkthrough with the substring rule explored explicitly and the round-trip step demonstrated end-to-end.
- `demo/index.html` Sample 4 — the static showcase panel that shows the LLM round-trip pattern for a multi-party contract dispute.
- `src/anonymisation/pipeline/pseudonymise.py` — the implementation; small enough to read top-to-bottom.

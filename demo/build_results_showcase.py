"""
Build the real-model results showcase for the portfolio RESULTS tab.

Runs a set of legal-style documents through the SAME redaction pipeline twice:

  1. with the lite demo's NER backbone (spaCy `en_core_web_sm`) — what a visitor
     sees in the hosted Space, and
  2. with the real fine-tuned RoBERTa (the project's strongest model, F1 0.851).

The point is to make the quality gap concrete: the same document, the same
pipeline, only the detector swapped — and to list, verbatim, the identifiers
the lite model leaves on the page that the fine-tune removes.

Outputs deterministic JSON to demo/results_showcase.json for transcription into
the static portfolio page (no runtime model dependency for site visitors).

Run with:
    legal-anon-env/bin/python demo/build_results_showcase.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anonymisation.cli import build_ner_provider          # noqa: E402
from anonymisation.pipeline import LitePipeline            # noqa: E402
from anonymisation.pipeline.pseudonymise import restore    # noqa: E402

FINETUNED = ROOT / "models" / "roberta-tab" / "final"
OUT = ROOT / "demo" / "results_showcase.json"

# Documents chosen to exercise names, organisations, dates and reference/case
# codes — the categories where a small general-purpose model and a domain
# fine-tune diverge most.
DOCS = [
    {
        "id": "grievance",
        "title": "HR grievance memorandum",
        "text": (
            "This memorandum concerns the grievance brought by Dr. Eleanor Whitcombe, "
            "a consultant cardiologist, against St. Andrew's Regional Hospital Trust. "
            "On 4 February 2021, Dr. Whitcombe lodged a formal complaint (Ref. "
            "HR-2021-0847) with the Trust's board, chaired by Mr. Harold Beecham. "
            "After internal mediation failed, the matter was referred to the "
            "Employment Tribunal in Manchester (Case no. 2402319/2021). Whitcombe was "
            "represented by Patel & Rosenthal LLP; the Trust instructed Carrow Chambers."
        ),
    },
    {
        "id": "settlement",
        "title": "Cross-border settlement note",
        "text": (
            "Anya Kowalski of Meridian Capital Partners met with Tomás Ferreira, "
            "general counsel of Halcyon Biotech GmbH, in Frankfurt on 12 March 2022 "
            "to finalise the settlement under Application no. 41872/19. Ms Kowalski "
            "confirmed that payment of EUR 4.2 million would be wired to the account "
            "held at Banco Nacional de Lisboa (IBAN PT50 0002 0123 1234 5678 9015 4)."
        ),
    },
]


def caught_direct(result):
    """The DIRECT identifiers the pipeline removed, as (type, surface form)."""
    seen, out = set(), []
    for s in result.spans:
        if s.identifier_role != "DIRECT":
            continue
        k = (s.entity_type, s.text)
        if k in seen:
            continue
        seen.add(k)
        out.append({"type": s.entity_type, "text": s.text})
    return out


def run_doc(doc, lite_ner, real_ner):
    lite = LitePipeline(ner_provider=lite_ner)(doc["text"])
    real = LitePipeline(ner_provider=real_ner)(doc["text"])

    return {
        "id": doc["id"],
        "title": doc["title"],
        "input": doc["text"],
        "lite_output": lite.redacted_text,
        "real_output": real.redacted_text,
        "lite_caught": caught_direct(lite),
        "real_caught": caught_direct(real),
    }


def run_pseudonymise(doc, real_ner):
    """Pseudonymise mode: referential [PERSON_A]/[ORG_A] tokens + vault + restore round-trip."""
    r = LitePipeline(ner_provider=real_ner, pseudonymise=True)(doc["text"])
    vault = dict(r.pseudonym_vault)

    # Build a plausible "external LLM answer" that only uses tokens we actually have,
    # then restore it locally — demonstrating real names never leave the building.
    persons = sorted(t for t in vault if t.startswith("[PERSON_"))
    orgs = sorted(t for t in vault if t.startswith("[ORG_"))
    pa, pb = (persons + [None, None])[:2]
    oa, ob = (orgs + [None, None])[:2]
    if pa and pb and oa and ob:
        llm_answer = (f"{pa} of {oa} is the acquiring party; {pb}, counsel to {ob}, "
                      f"must secure board approval before {pa} proceeds.")
    else:
        llm_answer = " ".join(f"{t} is a party." for t in (persons + orgs))

    return {
        "id": doc["id"],
        "title": doc["title"],
        "input": doc["text"],
        "pseudonymised_output": r.redacted_text,
        "vault": vault,
        "llm_answer_tokenised": llm_answer,
        "restored": restore(llm_answer, vault),
    }


def main():
    if not FINETUNED.exists():
        raise SystemExit(f"Fine-tuned checkpoint not found at {FINETUNED}")
    print("Loading lite NER (en_core_web_sm) …", flush=True)
    lite_ner = build_ner_provider("spacy", "en_core_web_sm")
    print("Loading real NER (fine-tuned RoBERTa) …", flush=True)
    real_ner = build_ner_provider("finetuned", str(FINETUNED))

    runs = []
    for doc in DOCS:
        print(f"  · {doc['id']}", flush=True)
        runs.append(run_doc(doc, lite_ner, real_ner))

    # Pseudonymise round-trip on the settlement doc (two people + two orgs → A/B tokens).
    pseudo = run_pseudonymise(DOCS[1], real_ner)

    payload = {"redact_runs": runs, "pseudonymise": pseudo}
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\nSaved → {OUT}\n")

    # Human-readable summary — the rendered outputs are the source of truth.
    for r in runs:
        print("=" * 78)
        print(f"{r['title']}")
        print(f"  LITE output: {r['lite_output']}")
        print(f"  REAL output: {r['real_output']}")
    print("=" * 78)
    print(f"PSEUDONYMISE — {pseudo['title']}")
    print(f"  output:   {pseudo['pseudonymised_output']}")
    print(f"  vault:    {pseudo['vault']}")
    print(f"  LLM ans:  {pseudo['llm_answer_tokenised']}")
    print(f"  restored: {pseudo['restored']}")


if __name__ == "__main__":
    main()

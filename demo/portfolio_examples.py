"""
Worked examples for the portfolio "Worked examples" tab.

Runs three representative documents through the *real* pipeline (the Phase-2
fine-tuned RoBERTa NER if its checkpoint is present) and prints clean,
transcribable before/after blocks for:

    1. REDACT      — direct identifiers only (the lighter mode)
    2. ANONYMISE   — direct + mosaic-aware quasi generalisation (iterate-until-safe)
    3. PSEUDONYMISE— stable referential tokens + a restore() round-trip

Regenerate with:

    legal-anon-env/bin/python demo/portfolio_examples.py

NB on example 2: against the full TAB corpus a single out-of-corpus document
is unique, so Anonymise correctly suppresses every quasi-identifier (the
project's headline finding). To *show the graduated generalisation* the mode
performs when comparable matters exist, this script seeds a small simulated
"firm's own corpus" haystack — the generalisation rules, signature logic and
k-anonymity loop are the unmodified pipeline; only the haystack is illustrative.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anonymisation.cli import build_ner_provider                       # noqa: E402
from anonymisation.pipeline import LitePipeline, ProPipeline           # noqa: E402
from anonymisation.pipeline.generalization import generalize           # noqa: E402
from anonymisation.pipeline.pro import ProPipeline as _Pro             # noqa: E402
from anonymisation.pipeline.scorer import MosaicScorer                 # noqa: E402
from anonymisation.pipeline.pseudonymise import restore                # noqa: E402

RULE = "─" * 78

DOC_REDACT = (
    "This memorandum concerns the grievance brought by Dr. Eleanor Whitcombe, "
    "a consultant cardiologist, against St. Andrew's Regional Hospital Trust. "
    "On 4 February 2021, Dr. Whitcombe lodged a formal complaint (Ref. "
    "HR-2021-0847) with the Trust's board, chaired by Mr. Harold Beecham. After "
    "internal mediation failed, the matter was referred to the Employment "
    "Tribunal in Manchester (Case no. 2402319/2021). Whitcombe was represented "
    "by Patel & Rosenthal LLP; the Trust instructed Carrow Chambers."
)

DOC_PSEUDO = (
    "Following preliminary talks, Anya Kowalski of Meridian Capital Partners met "
    "with Tomás Ferreira, general counsel of Halcyon Biotech, to negotiate terms. "
    "Kowalski proposed that Meridian Capital Partners acquire a controlling stake "
    "in Halcyon Biotech; Ferreira undertook to consult the board. A follow-up "
    "call between Ms Kowalski and Mr Ferreira was scheduled for the following week."
)


def build_best_ner():
    finetuned = ROOT / "phase2_baseline_comparison" / "checkpoints" / "roberta-tab" / "final"
    if finetuned.exists():
        print(f"NER backend: fine-tuned RoBERTa ({finetuned.relative_to(ROOT)})\n")
        return build_ner_provider("finetuned", str(finetuned))
    for model in ("en_core_web_trf", "en_core_web_sm"):
        try:
            ner = build_ner_provider("spacy", model)
            print(f"NER backend: spaCy {model} (fine-tuned checkpoint not found)\n")
            return ner
        except Exception:
            continue
    raise SystemExit("No NER backend available.")


def show(title, text):
    print(f"\n{title}\n{RULE}\n{text}")


def example_redact(ner):
    print(f"\n\n{'='*78}\nEXAMPLE 1 — REDACT (direct identifiers only)\n{'='*78}")
    show("INPUT", DOC_REDACT)
    r = LitePipeline(ner_provider=ner)(DOC_REDACT)
    show("OUTPUT — Redact", r.redacted_text)
    caught = [(s.entity_type, s.text) for s in r.spans if s.identifier_role == "DIRECT"]
    print(f"\nCaught {len(caught)} direct identifiers:")
    for et, txt in caught:
        print(f"  [{et:<8}] {txt}")


def example_pseudonymise(ner):
    print(f"\n\n{'='*78}\nEXAMPLE 3 — PSEUDONYMISE (re-linkable tokens + restore)\n{'='*78}")
    show("INPUT", DOC_PSEUDO)
    r = LitePipeline(ner_provider=ner, pseudonymise=True)(DOC_PSEUDO)
    show("OUTPUT — Pseudonymise (safe to send to an external LLM)", r.redacted_text)
    print(f"\nVault (kept locally, never leaves the firm):")
    for tok, original in r.pseudonym_vault.items():
        print(f"  {tok:<14} → {original}")
    # Simulate an LLM answer that uses the tokens, then restore locally.
    llm_answer = (
        "[PERSON_A] of [ORG_A] is the acquirer; [PERSON_B] represents the target "
        "[ORG_B] and must obtain board approval before [PERSON_A] proceeds."
    )
    show("LLM answer (tokenised — what the model sees and returns)", llm_answer)
    show("RESTORED locally via the vault (real names back, never sent out)",
         restore(llm_answer, r.pseudonym_vault))


def example_anonymise(ner):
    print(f"\n\n{'='*78}\nEXAMPLE 2 — ANONYMISE (mosaic-aware generalisation)\n{'='*78}")
    text = (ROOT / "demo" / "examples" / "sample_1_demographic_heavy.txt").read_text().strip()
    show("INPUT", text)

    k_target = 5
    # Detect spans once to learn the level-1 generalised fingerprint, then seed a
    # small simulated firm corpus that shares it (≥ k_target comparable matters).
    probe = ProPipeline(ner_provider=ner, scorer=MosaicScorer.empty(), k_target=k_target)
    spans = probe.detect_spans(text)
    quasi = [s for s in spans if s.identifier_role == "QUASI"]
    for s in quasi:
        s.replacement = generalize(s.entity_type, s.text, 1)
    level1_sig = _Pro._signature_from_quasi(quasi)
    haystack = MosaicScorer([level1_sig] * k_target)  # k_target similar matters

    pro = ProPipeline(ner_provider=ner, scorer=haystack, k_target=k_target, max_iterations=5)
    r = pro(text)
    show("OUTPUT — Anonymise (firm-corpus haystack: graduated generalisation)",
         r.redacted_text)
    print(f"\n  re-identification risk: k={r.mosaic_risk_initial} (unique) → "
          f"k={r.mosaic_risk_final} after {r.iterations_used} generalisation step(s); "
          f"converged={r.converged}")
    print("\n  Generalisation ladder actually applied (real generalize() output):")
    seen = set()
    for s in quasi:
        if s.text in seen:
            continue
        seen.add(s.text)
        ladder = " → ".join([s.text] + [generalize(s.entity_type, s.text, lv) for lv in (1, 2, 3)])
        print(f"    [{s.entity_type:<8}] {ladder}")

    # And the honest full-TAB behaviour, for contrast.
    print("\n  (Against the full TAB corpus this document is unique at every level,")
    print("   so Anonymise falls back to suppressing every quasi-identifier — the")
    print("   project's headline finding. The graduated output above is what happens")
    print("   once the firm has comparable matters to blend into.)")


def main():
    ner = build_best_ner()
    example_redact(ner)
    example_anonymise(ner)
    example_pseudonymise(ner)


if __name__ == "__main__":
    main()

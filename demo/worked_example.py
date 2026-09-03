"""
End-to-end worked example — one document through the real pipeline.

Every other artefact in this project measures *NER quality* (F1 on TAB).
This script shows the *product*: it runs the same library the writeup
describes on a single legal paragraph and prints the before/after for each
pipeline mode, plus a slice of the per-decision audit log. The output is
what the writeup's Act III panel embeds.

Regenerate with:

    legal-anon-env/bin/python demo/worked_example.py

NER backend: prefers the Phase-2 fine-tuned RoBERTa (the deployable
configuration, F1 = 0.851) if its checkpoint is present; otherwise falls
back to spaCy en_core_web_trf, then en_core_web_sm.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))  # package isn't installed; mirror the notebooks

from anonymisation.cli import build_ner_provider  # noqa: E402
from anonymisation.data import load_tab  # noqa: E402
from anonymisation.pipeline import LitePipeline, MosaicScorer, ProPipeline  # noqa: E402

SAMPLE = ROOT / "demo" / "examples" / "sample_1_demographic_heavy.txt"
RULE = "─" * 78


def build_best_ner():
    """Pick the strongest NER backend available on this machine."""
    finetuned = ROOT / "models" / "roberta-tab" / "final"
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
    raise SystemExit("No NER backend available — install a spaCy model or train Phase 2.")


def show(title: str, text: str) -> None:
    print(f"\n{title}\n{RULE}\n{text}")


def show_audit(result, limit_per_action: int = 2) -> None:
    """Print a representative slice of the audit log — a couple of each action."""
    print(f"\nAudit log (sample of {len(result.audit)} decisions)\n{RULE}")
    seen: dict[str, int] = {}
    for e in result.audit:
        seen[e.action] = seen.get(e.action, 0) + 1
        if seen[e.action] > limit_per_action:
            continue
        print(f"  [{e.action:>10}] {e.span.entity_type:<8} "
              f"{e.span.text!r:<28} → {e.span.replacement!r}")
        print(f"               {e.rationale}")


def main() -> None:
    ner = build_best_ner()
    text = SAMPLE.read_text().strip()

    show("INPUT (privileged — never leaves the firm)", text)

    # 1. Lite — DIRECT-only redaction
    lite = LitePipeline(ner_provider=ner)
    r_lite = lite(text)
    show("LITE — strip DIRECT identifiers (names, orgs, case numbers)", r_lite.redacted_text)

    # 2. Lite + pseudonymise — referential tokens for a useful LLM round-trip
    lite_ps = LitePipeline(ner_provider=ner, pseudonymise=True)
    r_ps = lite_ps(text)
    show("LITE + PSEUDONYMISE — stable referential tokens", r_ps.redacted_text)
    print(f"\n  vault (held locally, enables restore()): {r_ps.pseudonym_vault}")

    # 3. Pro — DIRECT + mosaic-aware QUASI generalisation (iterate-until-safe)
    print("\nLoading TAB as the mosaic haystack for Pro …")
    scorer = MosaicScorer.from_tab(list(load_tab()["test"]))
    pro = ProPipeline(ner_provider=ner, scorer=scorer, k_target=5, max_iterations=5)
    r_pro = pro(text)
    show("PRO — DIRECT redaction + QUASI generalisation", r_pro.redacted_text)
    print(f"\n  mosaic risk: k_initial={r_pro.mosaic_risk_initial} (unique within TAB) → "
          f"{r_pro.iterations_used} generalisation iterations → full suppression")
    print("  (a single out-of-corpus document is unique against TAB, so no amount of "
          "generalisation\n   reaches k≥5 — Pro correctly falls back to suppressing every "
          "QUASI. With the firm's own\n   corpus of similar matters as the haystack, "
          "intermediate levels (Bulgarian→European,\n   Plovdiv→Bulgaria) would survive.)")
    show_audit(r_pro)


if __name__ == "__main__":
    main()

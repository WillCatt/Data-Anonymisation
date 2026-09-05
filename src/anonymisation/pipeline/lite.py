"""
Pipeline Lite — DIRECT-only redaction.

Use case: a law firm wants to send matter notes to a third-party LLM and
needs the names, organisation references, and case-file numbers stripped.
They are *not* trying to defend against a determined re-identification
attack on the document's quasi-identifiers — they just want the obvious
PII out before the document leaves their network.

This is the cheaper, faster, less risky variant. It does not score the
mosaic effect; it just suppresses every span the role classifier marks
as DIRECT.

The output is human-readable, retains every QUASI mention untouched, and
runs in O(text length) regardless of corpus size.

Pseudonymisation
----------------
Construct with `pseudonymise=True` to get referential tokens
(`[PERSON_A]`, `[PERSON_B]`, …) instead of plain `[PERSON]` tags.
The same surface form gets the same token within the document; the
mapping is exposed as `result.pseudonym_vault`. Pair with
`pseudonymise.restore()` to round-trip an LLM answer back to original
names locally, without ever exposing the names to the LLM.
"""
from __future__ import annotations

from typing import List, Optional

from .base import Pipeline
from .pseudonymise import Pseudonymiser
from .types import AuditEntry, RedactionResult


class LitePipeline(Pipeline):
    """DIRECT-only redaction. No mosaic-risk scoring."""

    def __init__(self, ner_provider, *, pseudonymise: bool = False, **kwargs):
        super().__init__(ner_provider, **kwargs)
        self.pseudonymise = pseudonymise

    def _redact(self, text: str) -> RedactionResult:
        spans = self.detect_spans(text)
        audit: List[AuditEntry] = []
        pseudo: Optional[Pseudonymiser] = (
            Pseudonymiser() if self.pseudonymise else None
        )

        for span in spans:
            if span.entity_type in self.exempt_types:
                audit.append(AuditEntry(
                    span=span, action="leave",
                    rationale=(
                        f"{span.entity_type} kept by request; it stays in the "
                        f"document exactly as written."
                    ),
                ))
            elif span.identifier_role == "DIRECT":
                if pseudo is not None:
                    span.replacement = pseudo.token_for(span.entity_type, span.text)
                    rationale = (
                        f"DIRECT identifier ({span.entity_type}); pseudonymised to "
                        f"{span.replacement} (vault keeps the original)."
                    )
                else:
                    span.replacement = f"[{span.entity_type}]"
                    rationale = (
                        f"DIRECT identifier ({span.entity_type}); always suppressed in Lite."
                    )
                span.generalization_level = 3  # immediate suppression
                audit.append(AuditEntry(
                    span=span, action="redact", rationale=rationale,
                ))
            else:
                # QUASI — leave intact, but record the decision
                audit.append(AuditEntry(
                    span=span,
                    action="leave",
                    rationale=(
                        f"QUASI identifier ({span.entity_type}); Lite does not "
                        "touch quasi-identifiers. Run Pro for mosaic-aware redaction."
                    ),
                ))

        redacted_text = self.apply_replacements(text, spans)
        return RedactionResult(
            redacted_text=redacted_text,
            spans=spans,
            audit=audit,
            mosaic_risk_initial=None,
            mosaic_risk_final=None,
            iterations_used=0,
            converged=True,
            pseudonym_vault=(pseudo.vault if pseudo is not None else {}),
            pseudonym_links=(pseudo.decisions if pseudo is not None else []),
        )

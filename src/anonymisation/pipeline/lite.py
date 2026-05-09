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
"""
from __future__ import annotations

from typing import List

from .base import Pipeline
from .types import AuditEntry, RedactionResult, Span


class LitePipeline(Pipeline):
    """DIRECT-only redaction. No mosaic-risk scoring."""

    def _redact(self, text: str) -> RedactionResult:
        spans = self.detect_spans(text)
        audit: List[AuditEntry] = []

        for span in spans:
            if span.identifier_role == "DIRECT":
                span.replacement = f"[{span.entity_type}]"
                span.generalization_level = 3  # immediate suppression
                audit.append(AuditEntry(
                    span=span,
                    action="redact",
                    rationale=f"DIRECT identifier ({span.entity_type}); always suppressed in Lite.",
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
        )

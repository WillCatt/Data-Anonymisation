"""
Shared types for the redaction pipelines.

A `Span` is a single character-offset entity in a document, decorated with
the metadata the pipeline needs to make redaction decisions:

  * `entity_type` — one of TAB's 8 types, plus "REGEX" for things the
    regex pass identified that don't have a TAB equivalent (e.g. emails).
  * `identifier_role` — "DIRECT" (must be removed) or "QUASI" (might be
    generalized; only Pipeline Pro touches these).
  * `source` — which detector found this span: "ner", "regex", or "manual".
  * `confidence` — model confidence if the source produced one.
  * `generalization_level` — for Pipeline Pro, how aggressively the span
    has been generalized (0 = original, increasing means broader, ≥3 = suppressed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Literal, Optional


IdentifierRole = Literal["DIRECT", "QUASI"]
DetectorSource = Literal["ner", "regex", "manual", "coref"]


@dataclass
class Span:
    start: int
    end: int
    entity_type: str
    text: str
    identifier_role: IdentifierRole = "DIRECT"
    source: DetectorSource = "ner"
    confidence: Optional[float] = None
    generalization_level: int = 0  # 0 = original; >0 = generalized; >=3 = suppressed
    replacement: Optional[str] = None  # the actual text we substituted in

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


@dataclass
class AuditEntry:
    """One decision the pipeline made — exposed for compliance review."""
    span: Span
    action: Literal["redact", "generalize", "leave"]
    rationale: str
    iteration: int = 0  # always 0 for Lite; >0 for Pro generalization steps


@dataclass
class RedactionResult:
    """What every pipeline returns when called on a document."""
    redacted_text: str
    spans: List[Span]
    audit: List[AuditEntry]

    # Pipeline Pro only — None for Lite
    mosaic_risk_initial: Optional[int] = None
    mosaic_risk_final: Optional[int] = None
    iterations_used: int = 0
    converged: bool = True  # False if max iterations hit before reaching k_target

    # Pseudonymisation (when pipeline was run with pseudonymise=True). Empty
    # otherwise. Maps tokens like "[PERSON_A]" → original surface form.
    # Suitable for JSON serialisation; pair with `restore()` to round-trip.
    pseudonym_vault: dict = field(default_factory=dict)

    # Convenience for serialisation
    def to_dict(self) -> dict:
        return {
            "redacted_text": self.redacted_text,
            "spans": [
                {
                    "start": s.start, "end": s.end,
                    "entity_type": s.entity_type, "text": s.text,
                    "identifier_role": s.identifier_role, "source": s.source,
                    "generalization_level": s.generalization_level,
                    "replacement": s.replacement,
                }
                for s in self.spans
            ],
            "audit": [
                {
                    "start": e.span.start, "end": e.span.end,
                    "entity_type": e.span.entity_type, "original_text": e.span.text,
                    "action": e.action, "rationale": e.rationale,
                    "iteration": e.iteration,
                    "replacement": e.span.replacement,
                }
                for e in self.audit
            ],
            "mosaic_risk_initial": self.mosaic_risk_initial,
            "mosaic_risk_final": self.mosaic_risk_final,
            "iterations_used": self.iterations_used,
            "converged": self.converged,
            "pseudonym_vault": self.pseudonym_vault,
        }

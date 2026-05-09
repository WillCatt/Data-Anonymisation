"""
Pipeline Pro — DIRECT redaction + mosaic-aware QUASI generalization.

Use case: a law firm with a heavier compliance burden (regulated industries,
GDPR-sensitive jurisdictions, in-house counsel handling employee personal
data). They care not just about names being out, but about the document
being *non-uniquely-identifiable* under a re-identification attack.

Algorithm:
  1. Detect spans (NER + regex) and classify roles.
  2. Suppress every DIRECT span with `[TAB_TYPE]`.
  3. Build the residual QUASI fingerprint and ask the MosaicScorer for k.
  4. If k ≥ k_target, we're done.
  5. Otherwise, advance every QUASI span one generalization level deeper
     and recompute. Repeat up to `max_iterations` times.
  6. If we hit max_iterations without reaching k_target, fall back to full
     QUASI suppression (level = MAX_LEVEL).

Why "advance everything by one level" rather than "generalize the most-
specific span first"?  It's simpler to reason about, and on the corpus we
have it converges in 2–3 iterations for the typical document. A smarter
strategy is one of the obvious Phase-3.1 improvements.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

from .base import Pipeline
from .generalization import MAX_LEVEL, generalize
from .scorer import MosaicScorer
from .types import AuditEntry, RedactionResult, Span


class ProPipeline(Pipeline):
    """DIRECT redaction + iterate-until-safe QUASI generalization."""

    def __init__(
        self,
        ner_provider,
        scorer: Optional[MosaicScorer] = None,
        *,
        k_target: int = 5,
        max_iterations: int = 5,
        **kwargs,
    ):
        super().__init__(ner_provider, **kwargs)
        self.scorer = scorer or MosaicScorer.empty()
        self.k_target = k_target
        self.max_iterations = max_iterations

    # ------------------------------------------------------------------ #
    def _redact(self, text: str) -> RedactionResult:
        spans = self.detect_spans(text)
        audit: List[AuditEntry] = []

        # Phase A — suppress every DIRECT span unconditionally
        direct_spans = [s for s in spans if s.identifier_role == "DIRECT"]
        quasi_spans = [s for s in spans if s.identifier_role == "QUASI"]

        for s in direct_spans:
            s.replacement = f"[{s.entity_type}]"
            s.generalization_level = MAX_LEVEL
            audit.append(AuditEntry(
                span=s,
                action="redact",
                rationale=f"DIRECT identifier ({s.entity_type}); always suppressed.",
                iteration=0,
            ))

        # Phase B — score the initial fingerprint (after DIRECTs gone)
        initial_signature = self._signature_from_quasi(quasi_spans)
        initial_k = self.scorer.k_for(initial_signature)

        # If already safe, no QUASI work needed
        if initial_k >= self.k_target:
            for s in quasi_spans:
                audit.append(AuditEntry(
                    span=s,
                    action="leave",
                    rationale=(
                        f"QUASI identifier ({s.entity_type}); residual k="
                        f"{initial_k} ≥ k_target={self.k_target}, no generalization needed."
                    ),
                    iteration=0,
                ))
            redacted_text = self.apply_replacements(text, spans)
            return RedactionResult(
                redacted_text=redacted_text, spans=spans, audit=audit,
                mosaic_risk_initial=initial_k, mosaic_risk_final=initial_k,
                iterations_used=0, converged=True,
            )

        # Phase C — iterate-until-safe
        current_k = initial_k
        iterations_used = 0
        converged = False

        for level in range(1, MAX_LEVEL + 1):
            iterations_used = level
            if level > self.max_iterations:
                break

            for s in quasi_spans:
                s.generalization_level = level
                s.replacement = generalize(s.entity_type, s.text, level)

            sig = self._signature_from_quasi(quasi_spans)
            current_k = self.scorer.k_for(sig)

            for s in quasi_spans:
                audit.append(AuditEntry(
                    span=s,
                    action="generalize",
                    rationale=(
                        f"Generalized to level {level}: '{s.text}' → "
                        f"'{s.replacement}'. Post-step k={current_k}."
                    ),
                    iteration=level,
                ))

            if current_k >= self.k_target:
                converged = True
                break

        # Phase D — final suppression fallback if we ran out of iterations
        if not converged and iterations_used >= self.max_iterations:
            for s in quasi_spans:
                if s.generalization_level < MAX_LEVEL:
                    s.generalization_level = MAX_LEVEL
                    s.replacement = generalize(s.entity_type, s.text, MAX_LEVEL)
                    audit.append(AuditEntry(
                        span=s, action="generalize",
                        rationale=(
                            f"Iterate loop exhausted (k={current_k} < "
                            f"k_target={self.k_target}); falling back to "
                            f"full suppression."
                        ),
                        iteration=iterations_used + 1,
                    ))
            sig = self._signature_from_quasi(quasi_spans)
            current_k = self.scorer.k_for(sig)
            converged = current_k >= self.k_target

        redacted_text = self.apply_replacements(text, spans)
        return RedactionResult(
            redacted_text=redacted_text, spans=spans, audit=audit,
            mosaic_risk_initial=initial_k, mosaic_risk_final=current_k,
            iterations_used=iterations_used, converged=converged,
        )

    # ------------------------------------------------------------------ #
    @staticmethod
    def _signature_from_quasi(quasi_spans: List[Span]) -> Tuple:
        """
        Build a hashable signature from the *current* representations of
        QUASI spans (after whatever generalization level has been applied).
        Suppressed-to-[TYPE] spans are excluded — once a QUASI is fully
        suppressed it carries no information and shouldn't fingerprint.
        """
        parts = []
        for s in quasi_spans:
            value = s.replacement if s.replacement is not None else s.text
            if value is None or value.startswith("[") and value.endswith("]"):
                continue
            parts.append((s.entity_type, value.strip().lower()))
        return tuple(sorted(set(parts)))

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
from .pseudonymise import Pseudonymiser
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
        pseudonymise: bool = False,
        **kwargs,
    ):
        super().__init__(ner_provider, **kwargs)
        self.scorer = scorer or MosaicScorer.empty()
        self.k_target = k_target
        self.max_iterations = max_iterations
        self.pseudonymise = pseudonymise

    # ------------------------------------------------------------------ #
    def _redact(self, text: str) -> RedactionResult:
        spans = self.detect_spans(text)
        audit: List[AuditEntry] = []
        pseudo: Optional[Pseudonymiser] = (
            Pseudonymiser() if self.pseudonymise else None
        )

        # Phase A — suppress every DIRECT span unconditionally
        #
        # Except those the caller has chosen to keep. An exempt QUASI still
        # goes into `quasi_spans`, because it still fingerprints: keeping a
        # date in the clear makes the document *more* identifying, and the k
        # this loop reports has to say so. What it does not do is get
        # broadened, so it sits at level 0 for the whole run.
        kept_spans = [s for s in spans if s.entity_type in self.exempt_types]
        direct_spans = [s for s in spans if s.identifier_role == "DIRECT"
                        and s.entity_type not in self.exempt_types]
        quasi_spans = [s for s in spans if s.identifier_role == "QUASI"]
        generalisable = [s for s in quasi_spans if s.entity_type not in self.exempt_types]

        for s in kept_spans:
            audit.append(AuditEntry(
                span=s, action="leave",
                rationale=(
                    f"{s.entity_type} kept by request; it stays as written and "
                    f"still counts towards the re-identification risk below."
                ),
            ))

        for s in direct_spans:
            if pseudo is not None:
                s.replacement = pseudo.token_for(s.entity_type, s.text)
                rationale = (
                    f"DIRECT identifier ({s.entity_type}); pseudonymised to "
                    f"{s.replacement} (vault holds the original)."
                )
            else:
                s.replacement = f"[{s.entity_type}]"
                rationale = f"DIRECT identifier ({s.entity_type}); always suppressed."
            s.generalization_level = MAX_LEVEL
            audit.append(AuditEntry(
                span=s, action="redact", rationale=rationale, iteration=0,
            ))

        # Phase B — score the initial fingerprint (after DIRECTs gone)
        initial_signature = self._signature_from_quasi(quasi_spans)
        initial_k = self.scorer.k_for(initial_signature)

        # If already safe, no QUASI work needed
        if initial_k >= self.k_target:
            for s in generalisable:
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
                pseudonym_vault=(pseudo.vault if pseudo is not None else {}),
                pseudonym_links=(pseudo.decisions if pseudo is not None else []),
            )

        # Phase C — iterate-until-safe
        current_k = initial_k
        iterations_used = 0
        converged = False

        for level in range(1, MAX_LEVEL + 1):
            iterations_used = level
            if level > self.max_iterations:
                break

            for s in generalisable:
                s.generalization_level = level
                s.replacement = generalize(s.entity_type, s.text, level)

            sig = self._signature_from_quasi(quasi_spans)
            current_k = self.scorer.k_for(sig)

            for s in generalisable:
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
            for s in generalisable:
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
            # `converged` stays False — once we've fallen back to full
            # suppression we have not "converged" in the natural sense, even
            # though the empty signature trivially satisfies k_target. The
            # caller distinguishes the two cases on the converged flag.

        # Final semantic check: if every QUASI ended up at MAX_LEVEL, this
        # is full suppression regardless of whether it happened inside the
        # iterate loop (k_target trivially satisfied by an empty signature)
        # or via the explicit Phase D fallback. Either way, not "converged".
        if generalisable and all(s.generalization_level >= MAX_LEVEL for s in generalisable):
            converged = False

        redacted_text = self.apply_replacements(text, spans)
        return RedactionResult(
            redacted_text=redacted_text, spans=spans, audit=audit,
            mosaic_risk_initial=initial_k, mosaic_risk_final=current_k,
            iterations_used=iterations_used, converged=converged,
            pseudonym_vault=(pseudo.vault if pseudo is not None else {}),
            pseudonym_links=(pseudo.decisions if pseudo is not None else []),
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

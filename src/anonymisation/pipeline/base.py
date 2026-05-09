"""
Pipeline base class.

Both LitePipeline and ProPipeline share a fair amount of plumbing:
detecting spans (NER + regex), classifying their identifier role,
applying replacements while preserving character offsets, and recording
audit entries. This module factors that out.

Subclasses override `_redact()` to provide their actual redaction policy.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from .types import AuditEntry, IdentifierRole, RedactionResult, Span
from .roles import classify_role, RoleOverride
from .regex_pass import regex_pass, merge_with_ner


# A predictor is anything that maps text to spans of the form
# (start, end, tab_type, span_text).  This matches the interface used
# everywhere else in the package so you can plug in the spaCy, HF,
# Presidio, or fine-tuned predictors built in earlier phases.
SpanPredictor = Callable[[str], List[Tuple[int, int, str, str]]]


class Pipeline:
    """
    Base for both pipeline variants. Don't instantiate directly.
    """

    def __init__(
        self,
        ner_provider: SpanPredictor,
        *,
        run_regex: bool = True,
        role_override: Optional[RoleOverride] = None,
    ):
        self.ner_provider = ner_provider
        self.run_regex = run_regex
        self.role_override = role_override

    # ------------------------------------------------------------------ #
    # Span detection — shared by both pipelines
    # ------------------------------------------------------------------ #
    def detect_spans(self, text: str) -> List[Span]:
        """Run NER + regex, classify roles, return a sorted list of Spans."""
        ner_tuples = self.ner_provider(text)
        ner_spans = [
            Span(
                start=s, end=e, entity_type=t, text=txt,
                source="ner", identifier_role="DIRECT",  # placeholder
            )
            for (s, e, t, txt) in ner_tuples
        ]

        regex_spans = regex_pass(text) if self.run_regex else []
        merged = merge_with_ner(ner_spans, regex_spans)

        # Classify each span's identifier role
        for span in merged:
            span.identifier_role = classify_role(span, override=self.role_override)
        return merged

    # ------------------------------------------------------------------ #
    # Replacement — apply spans to text in reverse offset order
    # ------------------------------------------------------------------ #
    @staticmethod
    def apply_replacements(text: str, spans: List[Span]) -> str:
        """
        Replace each span's range in `text` with `span.replacement`.
        Spans without a replacement are left alone.
        """
        out = text
        for span in sorted(spans, key=lambda s: s.start, reverse=True):
            if span.replacement is None:
                continue
            out = out[: span.start] + span.replacement + out[span.end :]
        return out

    # ------------------------------------------------------------------ #
    # Subclass entry point
    # ------------------------------------------------------------------ #
    def __call__(self, text: str) -> RedactionResult:
        return self._redact(text)

    def _redact(self, text: str) -> RedactionResult:  # pragma: no cover
        raise NotImplementedError("Subclasses must implement _redact().")

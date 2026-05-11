"""
Redaction pipelines — Phase 3 of the Data Anonymisation portfolio project.

Two variants:
  • LitePipeline — DIRECT-only redaction; cheap, fast, no mosaic-risk handling.
  • ProPipeline  — DIRECT redaction + mosaic-aware QUASI generalization with
                   iterate-until-safe. Targets firms with stricter privacy needs.

Both share the same construction surface:

    from anonymisation.pipeline import LitePipeline, ProPipeline

    lite = LitePipeline(ner_provider=my_predictor)
    pro  = ProPipeline(ner_provider=my_predictor, scorer=my_mosaic_scorer)

    result = lite("Maria Petrova lives in Plovdiv ...")
    print(result.redacted_text)
    print(result.audit)
"""
from .base import Pipeline, SpanPredictor
from .lite import LitePipeline
from .pro import ProPipeline
from .scorer import MosaicScorer
from .types import AuditEntry, RedactionResult, Span
from .roles import classify_role, DEFAULT_ROLE_BY_TYPE
from .regex_pass import regex_pass
from .generalization import generalize, MAX_LEVEL
from .pseudonymise import Pseudonymiser, restore, index_to_letters
from .coref import extend_with_coref

__all__ = [
    "Pipeline", "LitePipeline", "ProPipeline",
    "MosaicScorer",
    "SpanPredictor",
    "Span", "AuditEntry", "RedactionResult",
    "classify_role", "DEFAULT_ROLE_BY_TYPE",
    "regex_pass", "generalize", "MAX_LEVEL",
    "Pseudonymiser", "restore", "index_to_letters",
    "extend_with_coref",
]

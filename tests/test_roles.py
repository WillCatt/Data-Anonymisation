"""Identifier-role classification: conservative defaults + override hook."""
from anonymisation.pipeline.roles import classify_role
from anonymisation.pipeline.types import Span


def _span(entity_type):
    return Span(start=0, end=1, entity_type=entity_type, text="x")


def test_direct_defaults():
    for t in ("PERSON", "ORG", "CODE", "MISC"):
        assert classify_role(_span(t)) == "DIRECT"


def test_quasi_defaults():
    for t in ("DATETIME", "LOC", "DEM", "QUANTITY"):
        assert classify_role(_span(t)) == "QUASI"


def test_unknown_type_defaults_to_direct():
    # When in doubt, over-redact rather than leak.
    assert classify_role(_span("SOMETHING_NEW")) == "DIRECT"


def test_override_wins_when_it_returns_a_role():
    force_quasi = lambda span: "QUASI"
    assert classify_role(_span("PERSON"), override=force_quasi) == "QUASI"


def test_override_falls_through_when_it_returns_none():
    abstain = lambda span: None
    assert classify_role(_span("PERSON"), override=abstain) == "DIRECT"

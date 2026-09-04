"""
Keeping an entity type in the clear.

A firm often has a reason to keep something a blanket policy would remove —
dates that carry the limitation period, an amount the other side already has,
a location that is the whole point of the matter. `exempt_types` says so.

The rule that makes it honest rather than convenient: an exempt span is still
detected and still counted in the mosaic fingerprint. Keeping a date does not
stop the date identifying anyone, so the reported re-identification risk has
to get worse, not disappear.
"""
from anonymisation.pipeline import LitePipeline, MosaicScorer, ProPipeline


TEXT = (
    "Maria Petrova, a Bulgarian national, filed claim 4521/2022 against "
    "Acme Holdings Ltd in Manchester on 12 March 2018."
)
ITEMS = [
    ("Maria Petrova", "PERSON"), ("Bulgarian", "DEM"), ("4521/2022", "CODE"),
    ("Acme Holdings Ltd", "ORG"), ("Manchester", "LOC"), ("12 March 2018", "DATETIME"),
]


def predictor(_text=None):
    spans, used = [], 0
    for needle, entity_type in ITEMS:
        start = TEXT.index(needle, used)
        spans.append((start, start + len(needle), entity_type, needle))
        used = start + len(needle)
    return spans


def scorer():
    # A haystack where the broadened forms are common and the exact pair is not.
    haystack = [(("DATETIME", "2018"), ("LOC", "england"))] * 20
    haystack.append((("DATETIME", "12 march 2018"), ("LOC", "manchester")))
    return MosaicScorer(haystack)


def test_an_exempt_type_is_left_exactly_as_written():
    result = LitePipeline(ner_provider=predictor, exempt_types=("DATETIME",))(TEXT)
    assert "12 March 2018" in result.redacted_text
    assert "[PERSON]" in result.redacted_text          # everything else still goes


def test_the_decision_to_keep_is_recorded_not_silent():
    result = LitePipeline(ner_provider=predictor, exempt_types=("DATETIME",))(TEXT)
    kept = [e for e in result.audit if e.span.entity_type == "DATETIME"]
    assert len(kept) == 1
    assert kept[0].action == "leave"
    assert "kept by request" in kept[0].rationale


def test_keeping_a_quasi_identifier_makes_the_risk_worse_not_invisible():
    # Broadened, the document hides in a crowd. Keep the date and it does not,
    # and the scorer has to say so rather than scoring a document it isn't.
    broadened = ProPipeline(predictor, scorer=scorer(), k_target=5, max_iterations=3)(TEXT)
    kept = ProPipeline(predictor, scorer=scorer(), k_target=5, max_iterations=3,
                       exempt_types=("DATETIME",))(TEXT)

    assert broadened.mosaic_risk_final > kept.mosaic_risk_final
    assert kept.mosaic_risk_final == 1          # still uniquely identifiable
    assert not kept.converged
    assert "12 March 2018" in kept.redacted_text


def test_an_exempt_direct_identifier_is_kept_too():
    # The escape hatch is not limited to quasi-identifiers: a firm may want the
    # respondent company named. It is their call, and the audit records it.
    result = ProPipeline(predictor, scorer=scorer(), exempt_types=("ORG",))(TEXT)
    assert "Acme Holdings Ltd" in result.redacted_text
    assert "[PERSON]" in result.redacted_text
    kept = [e for e in result.audit if e.span.entity_type == "ORG"]
    assert kept and kept[0].action == "leave"


def test_exempting_nothing_is_the_default():
    result = LitePipeline(ner_provider=predictor)(TEXT)
    assert "12 March 2018" in result.redacted_text     # QUASI, untouched by Lite
    assert "Maria Petrova" not in result.redacted_text
    assert "Acme Holdings Ltd" not in result.redacted_text


def test_every_span_is_audited_exactly_once():
    # An exempt quasi must not be recorded both as kept and as left-alone.
    result = ProPipeline(predictor, scorer=scorer(), k_target=1,
                         exempt_types=("DATETIME",))(TEXT)
    ids = [(e.span.start, e.span.end) for e in result.audit if e.iteration == 0]
    assert len(ids) == len(set(ids))

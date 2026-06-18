"""
Mosaic re-identification helpers: fact normalisation and the smart-attacker
minimum-facts estimate.

These are the hardened pieces of the re-identification analysis — they let the
fingerprint match past surface form (so "47-year-old" and "aged 47" collide)
and model an attacker who learns the most-discriminating facts first.
"""
from anonymisation.mosaic import (
    min_facts_to_identify,
    normalise_quasi_value,
    quasi_identifier_signature,
)


# ---------------------------------------------------------------------------
# normalise_quasi_value
# ---------------------------------------------------------------------------
def test_datetime_folds_to_year():
    assert normalise_quasi_value("DATETIME", "12 March 2018") == "2018"
    assert normalise_quasi_value("DATETIME", "in 2018") == "2018"
    assert normalise_quasi_value("DATETIME", "since 2010") == "2010"
    # Same year, different surface form → identical normalised value.
    assert normalise_quasi_value("DATETIME", "4 February 2021") == \
        normalise_quasi_value("DATETIME", "February 2021")


def test_age_variants_collapse():
    forms = ["47-year-old", "47 year old", "aged 47", "47-year old"]
    assert {normalise_quasi_value("QUANTITY", f) for f in forms} == {"age:47"}


def test_spelled_age_collapses_with_digits():
    assert normalise_quasi_value("QUANTITY", "forty-seven") == "47"


def test_count_keeps_its_noun():
    # The number folds, but the head noun stays — so a count of three children
    # never collides with three convictions.
    assert normalise_quasi_value("QUANTITY", "three children") == "3 children"
    assert normalise_quasi_value("QUANTITY", "three children") != \
        normalise_quasi_value("QUANTITY", "three convictions")


def test_loc_strips_punctuation_and_case():
    assert normalise_quasi_value("LOC", "Plovdiv,") == "plovdiv"
    assert normalise_quasi_value("LOC", "Plovdiv") == normalise_quasi_value("LOC", "PLOVDIV.")


def test_dem_normalises_whitespace():
    assert normalise_quasi_value("DEM", "  Bulgarian  national ") == "bulgarian national"


# ---------------------------------------------------------------------------
# quasi_identifier_signature(normalise=...)
# ---------------------------------------------------------------------------
def _doc(*quasi):
    """Minimal TAB-shaped doc: each arg is (entity_type, span_text)."""
    return {
        "doc_id": "d",
        "entity_mentions": [
            {"entity_type": t, "span_text": s, "identifier_type": "QUASI",
             "start_offset": i}
            for i, (t, s) in enumerate(quasi)
        ],
    }


def test_signature_normalisation_merges_variants():
    a = _doc(("QUANTITY", "47-year-old"), ("DATETIME", "12 March 2018"))
    b = _doc(("QUANTITY", "aged 47"), ("DATETIME", "in 2018"))
    # Raw surface form: the two docs look different.
    assert quasi_identifier_signature(a) != quasi_identifier_signature(b)
    # Normalised: they are revealed to carry the same identifying facts.
    assert quasi_identifier_signature(a, normalise=True) == \
        quasi_identifier_signature(b, normalise=True)


def test_signature_default_is_unchanged():
    # Backward-compat: default path is still raw lower-cased surface form.
    d = _doc(("LOC", "Plovdiv"))
    assert quasi_identifier_signature(d) == (("LOC", "plovdiv"),)


# ---------------------------------------------------------------------------
# min_facts_to_identify — the greedy smart attacker
# ---------------------------------------------------------------------------
def test_rarest_fact_alone_identifies():
    # 'nurse' is everywhere; 'plovdiv' is unique to the target. A smart attacker
    # needs exactly one fact.
    target = {"nurse", "plovdiv"}
    others = [{"nurse"}, {"nurse", "sofia"}, {"nurse"}]
    assert min_facts_to_identify(target, others) == 1


def test_needs_combination_of_common_facts():
    # No single fact is unique, but the pair is. Document order would reveal
    # them one at a time; the attacker still needs both.
    target = {"nurse", "bulgarian"}
    others = [{"nurse", "german"}, {"teacher", "bulgarian"}]
    assert min_facts_to_identify(target, others) == 2


def test_never_unique_returns_none():
    # Another document has a superset of the target's facts → never singled out.
    target = {"nurse", "bulgarian"}
    others = [{"nurse", "bulgarian", "plovdiv"}]
    assert min_facts_to_identify(target, others) is None


def test_empty_target_is_none():
    assert min_facts_to_identify(set(), [{"a"}]) is None


def test_smart_attacker_never_worse_than_document_order():
    # The greedy estimate must be ≤ the number of facts a document-order
    # attacker would need (here the full fingerprint of 3).
    target = {"a", "b", "c"}
    others = [{"a", "b"}, {"a", "c"}, {"b", "c"}]
    smart = min_facts_to_identify(target, others)
    assert smart is not None and smart <= len(target)

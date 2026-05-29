"""Coreference extender: recover shorthand mentions, stay conservative."""
from anonymisation.pipeline.coref import extend_with_coref
from anonymisation.pipeline.types import Span


def _person(start, end, text):
    return Span(start=start, end=end, entity_type="PERSON", text=text)


def _org(start, end, text):
    return Span(start=start, end=end, entity_type="ORG", text=text)


def test_recovers_bare_surname_mention():
    text = "Maria Petrova testified. Petrova was then cross-examined."
    out = extend_with_coref(text, [_person(0, 13, "Maria Petrova")])
    coref = [s for s in out if s.source == "coref"]
    assert any(s.text == "Petrova" for s in coref)
    # Added spans inherit the parent type and carry reduced confidence.
    assert all(s.entity_type == "PERSON" and s.confidence == 0.7 for s in coref)


def test_does_not_duplicate_the_original_span():
    text = "Maria Petrova spoke. Maria Petrova left."
    out = extend_with_coref(text, [_person(0, 13, "Maria Petrova")])
    # The second full mention isn't added by coref (it would be NER's job);
    # coref only emits non-overlapping shorter forms, none here.
    assert not any(s.source == "coref" and s.text == "Maria Petrova" for s in out)


def test_org_first_word_recovered_but_generic_suffix_is_not():
    text = "Northwind Energy Ltd grew. Northwind expanded. The Ltd filed."
    out = extend_with_coref(text, [_org(0, 20, "Northwind Energy Ltd")])
    coref_texts = {s.text for s in out if s.source == "coref"}
    assert "Northwind" in coref_texts
    assert "Ltd" not in coref_texts  # generic suffix excluded


def test_single_token_entity_generates_no_candidates():
    text = "Northwind grew. Northwind expanded."
    out = extend_with_coref(text, [_org(0, 9, "Northwind")])
    assert not any(s.source == "coref" for s in out)

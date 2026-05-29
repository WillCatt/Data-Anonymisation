"""Pseudonymiser: stable tokens, substring coreference, vault round-trip."""
from anonymisation.pipeline.pseudonymise import (
    Pseudonymiser,
    index_to_letters,
    restore,
)


def test_index_to_letters_rollover():
    assert index_to_letters(0) == "A"
    assert index_to_letters(25) == "Z"
    assert index_to_letters(26) == "AA"
    assert index_to_letters(27) == "AB"
    assert index_to_letters(51) == "AZ"
    assert index_to_letters(52) == "BA"


def test_same_form_gets_same_token():
    ps = Pseudonymiser()
    assert ps.token_for("PERSON", "Maria Petrova") == "[PERSON_A]"
    assert ps.token_for("PERSON", "Maria Petrova") == "[PERSON_A]"


def test_distinct_entities_get_distinct_tokens():
    ps = Pseudonymiser()
    assert ps.token_for("PERSON", "Maria Petrova") == "[PERSON_A]"
    assert ps.token_for("PERSON", "John Doe") == "[PERSON_B]"


def test_substring_coreference_collapses_shorthand():
    ps = Pseudonymiser()
    full = ps.token_for("PERSON", "Maria Petrova")
    # Bare surname and honorific+surname should resolve to the same entity.
    assert ps.token_for("PERSON", "Petrova") == full
    assert ps.token_for("PERSON", "Mrs Petrova") == full


def test_same_form_different_type_is_not_shared():
    ps = Pseudonymiser()
    person = ps.token_for("PERSON", "Sofia")
    org = ps.token_for("ORG", "Sofia")
    assert person == "[PERSON_A]"
    assert org == "[ORG_A]"
    assert person != org


def test_vault_keeps_longest_canonical_form():
    ps = Pseudonymiser()
    ps.token_for("PERSON", "Maria")          # short form seen first
    ps.token_for("PERSON", "Maria Petrova")  # longer form should win
    assert ps.vault["[PERSON_A]"] == "Maria Petrova"


def test_restore_round_trip():
    ps = Pseudonymiser()
    a = ps.token_for("PERSON", "Maria Petrova")
    b = ps.token_for("PERSON", "John Doe")
    redacted = f"{a} sued {b}"
    assert restore(redacted, ps.vault) == "Maria Petrova sued John Doe"


def test_restore_leaves_plain_tags_and_unknown_tokens_untouched():
    # Plain "[PERSON]" tags (non-pseudonymised) don't match the token regex.
    assert restore("[PERSON] spoke", {}) == "[PERSON] spoke"
    # A pseudonym token with no vault entry passes through verbatim.
    assert restore("[PERSON_Z] spoke", {}) == "[PERSON_Z] spoke"

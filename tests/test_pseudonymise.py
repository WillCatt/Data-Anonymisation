"""Pseudonymiser: stable tokens, substring coreference, vault round-trip."""
from anonymisation.pipeline.pseudonymise import (
    LINK_CONFIDENCE,
    LINK_CONFIDENCE_PRIOR,
    LINK_EVIDENCE_TIERS,
    NEW_ENTITY,
    Pseudonymiser,
    classify_link_evidence,
    index_to_letters,
    link_confidence,
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
    ps.token_for("PERSON", "Maria Petrova")
    ps.token_for("PERSON", "Petrova")  # shorthand must not shorten the vault entry
    assert ps.vault["[PERSON_A]"] == "Maria Petrova"


def test_reverse_containment_is_not_merged_under_the_calibrated_policy():
    # "Maria" then "Maria Petrova" is the reverse of the usual legal-document
    # order, and on TAB that direction is right about 20% of the time for
    # PERSON ("Anita" then "Anita De La Cruz" are different people). The
    # calibrated policy keeps them apart; legacy merged them.
    ps = Pseudonymiser()
    assert ps.token_for("PERSON", "Maria") == "[PERSON_A]"
    assert ps.token_for("PERSON", "Maria Petrova") == "[PERSON_B]"

    legacy = Pseudonymiser(link_policy="legacy")
    assert legacy.token_for("PERSON", "Maria") == "[PERSON_A]"
    assert legacy.token_for("PERSON", "Maria Petrova") == "[PERSON_A]"


def test_dates_sharing_a_token_are_not_the_same_date():
    # long_over_short on DATETIME scored 0.001 on 2,121 train+validation
    # links: two dates sharing a year are not one date.
    ps = Pseudonymiser()
    assert ps.token_for("DATETIME", "1989") == "[DATETIME_A]"
    assert ps.token_for("DATETIME", "June 1989") == "[DATETIME_B]"


def test_case_variants_are_not_merged():
    # exact_casefold measured 0.011 — TAB never links case variants.
    ps = Pseudonymiser()
    assert ps.token_for("ORG", "The Board") == "[ORG_A]"
    assert ps.token_for("ORG", "the board") == "[ORG_B]"


def test_classify_link_evidence_names_the_rule():
    assert classify_link_evidence("Petrova", "Petrova") == "exact"
    assert classify_link_evidence("the Board", "The Board") == "exact_casefold"
    assert classify_link_evidence("Nihat Osal", "Mr Nihat Osal") == "honorific_only"
    assert classify_link_evidence("Petrova", "Maria Petrova") == "short_in_long_1tok"
    assert classify_link_evidence("Sofia District", "Sofia District Court") == "short_in_long_multi"
    assert classify_link_evidence("June 1989", "1989") == "long_over_short_1tok"
    assert classify_link_evidence("Northwind Energy Ltd", "Northwind Energy") == "long_over_short_multi"
    assert classify_link_evidence("Petrova", "Kowalski") is None


def test_the_classifier_only_returns_known_tiers():
    # Every tier the classifier can emit must have a calibration entry to look
    # up, or a link would silently fall back to the prior forever.
    pairs = [("Petrova", "Petrova"), ("the Board", "The Board"),
             ("Nihat Osal", "Mr Nihat Osal"), ("Petrova", "Maria Petrova"),
             ("Sofia District", "Sofia District Court"), ("June 1989", "1989"),
             ("Northwind Energy Ltd", "Northwind Energy"), ("", "x"), ("x", "")]
    seen = {classify_link_evidence(new, old) for new, old in pairs} - {None}
    assert seen <= set(LINK_EVIDENCE_TIERS)
    assert seen == set(LINK_EVIDENCE_TIERS)      # all seven are reachable
    assert set(LINK_CONFIDENCE_PRIOR) == set(LINK_EVIDENCE_TIERS)


def test_every_calibrated_confidence_is_a_probability():
    assert all(0.0 <= v <= 1.0 for v in LINK_CONFIDENCE.values())
    # An (evidence, type) pair TAB never produced falls back to the tier rate,
    # not to a free pass.
    assert link_confidence("long_over_short_1tok", "NOT_A_TAB_TYPE") < 0.5


def test_decisions_record_why_each_token_was_assigned():
    ps = Pseudonymiser()
    ps.token_for("PERSON", "Maria Petrova")
    ps.token_for("PERSON", "Mrs Petrova")
    ps.token_for("PERSON", "John Doe")

    minted, merged, other = ps.decisions
    assert minted.evidence == NEW_ENTITY and not minted.merged
    assert minted.confidence is None   # minting is not a measured claim
    assert merged.merged and merged.matched_form == "Maria Petrova"
    assert merged.evidence == "short_in_long_1tok"
    assert merged.confidence == LINK_CONFIDENCE[("short_in_long_1tok", "PERSON")]
    assert "same entity as" in merged.describe()
    assert other.evidence == NEW_ENTITY


def test_a_rejected_link_is_still_recorded():
    ps = Pseudonymiser()
    ps.token_for("DATETIME", "1989")
    ps.token_for("DATETIME", "June 1989")
    rejected = ps.decisions[-1]
    assert not rejected.merged
    assert rejected.refusal == "below_threshold"
    assert rejected.rejected            # the near miss is kept for the audit log
    assert "kept separate" in rejected.describe()


def test_an_ambiguous_link_is_refused():
    # Two Smiths, then a bare surname that matches both equally well. The
    # evidence does not pick between them, so neither is picked.
    ps = Pseudonymiser()
    ps.token_for("PERSON", "John Smith")
    ps.token_for("PERSON", "Jane Smith")
    assert ps.token_for("PERSON", "Smith") == "[PERSON_C]"
    decision = ps.decisions[-1]
    assert decision.ambiguous and decision.tied_candidates == 2
    # The audit line must say it was a tie, not that the evidence was weak.
    assert decision.refusal == "ambiguous"
    assert "matched equally well" in decision.describe()
    assert "below" not in decision.describe()

    # With the guard off, the first-seen Smith wins the coin flip.
    loose = Pseudonymiser(reject_ambiguous=False)
    loose.token_for("PERSON", "John Smith")
    loose.token_for("PERSON", "Jane Smith")
    assert loose.token_for("PERSON", "Smith") == "[PERSON_A]"
    assert loose.decisions[-1].ambiguous


def test_link_policy_must_be_known():
    import pytest
    with pytest.raises(ValueError):
        Pseudonymiser(link_policy="vibes")


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

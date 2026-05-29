"""Generalization rules: per-type level progression and suppression."""
from anonymisation.pipeline.generalization import MAX_LEVEL, generalize


def test_level_zero_is_identity():
    assert generalize("LOC", "Plovdiv", 0) == "Plovdiv"


def test_datetime_year_then_decade_then_suppress():
    assert generalize("DATETIME", "12 March 1998", 1) == "1998"
    assert generalize("DATETIME", "12 March 1998", 2) == "1990s"
    assert generalize("DATETIME", "12 March 1998", 3) == "[DATETIME]"


def test_datetime_without_year_suppresses():
    assert generalize("DATETIME", "last Tuesday", 1) == "[DATETIME]"


def test_quantity_age_heuristic_bands():
    assert generalize("QUANTITY", "47-year-old", 1) == "about 40"
    assert generalize("QUANTITY", "47 years old", 2) == "in their 40s"


def test_quantity_currency_is_not_treated_as_age():
    # The comma-aware regex must read the whole number, not just "2".
    assert generalize("QUANTITY", "£2,300,000", 1) == "about 2,000,000"


def test_loc_city_to_country_to_continent():
    assert generalize("LOC", "Plovdiv", 1) == "Bulgaria"
    assert generalize("LOC", "Plovdiv", 2) == "Europe"


def test_loc_unknown_city_suppresses():
    assert generalize("LOC", "Atlantis", 1) == "[LOC]"


def test_dem_broadening_then_suppress():
    assert generalize("DEM", "Bulgarian", 1) == "European"
    assert generalize("DEM", "Bulgarian", 2) == "[DEM]"


def test_direct_types_always_suppress_regardless_of_level():
    for level in (1, 2, MAX_LEVEL):
        assert generalize("PERSON", "Maria Petrova", level) == "[PERSON]"
        assert generalize("ORG", "Acme Ltd", level) == "[ORG]"
        assert generalize("CODE", "App. No. 1/99", level) == "[CODE]"

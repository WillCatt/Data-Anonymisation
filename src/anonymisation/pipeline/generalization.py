"""
Per-entity-type generalization rules for Pipeline Pro.

Each rule advances a span through a sequence of progressively-broader
representations:

    level 0  → original text
    level 1  → mild generalization (year for dates, decade band for ages)
    level 2  → broader generalization (decade for dates, region for places)
    level 3+ → full suppression as [TAB_TYPE]

The rules here are deliberately simple — domain-specific defaults that
illustrate the iterate-until-safe loop. A real production deployment
would replace many of these with richer logic (taxonomy lookups for
LOC/DEM, an LLM call for free-text MISC fields).
"""
from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# DATETIME — extract year, then decade
# ---------------------------------------------------------------------------
_YEAR_RE = re.compile(r"\b(1\d{3}|20\d{2}|21\d{2})\b")


def _generalize_datetime(text: str, level: int) -> str:
    if level >= 3:
        return "[DATETIME]"
    m = _YEAR_RE.search(text)
    if not m:
        # No identifiable year — fall back to suppression even at level 1
        return "[DATETIME]"
    year = int(m.group(1))
    if level == 1:
        return str(year)
    if level == 2:
        decade = (year // 10) * 10
        return f"{decade}s"
    return "[DATETIME]"


# ---------------------------------------------------------------------------
# QUANTITY — primarily ages; round, then decade-band
# ---------------------------------------------------------------------------
# Match comma-separated numbers ("2,300,000") preferentially, falling back to
# bare digit runs. Without this the previous version saw "£2,300,000" as the
# number 2 and routed it through the age heuristic.
_NUM_RE = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")


def _generalize_quantity(text: str, level: int) -> str:
    if level >= 3:
        return "[QUANTITY]"
    m = _NUM_RE.search(text)
    if not m:
        return "[QUANTITY]"
    n_str = m.group().replace(",", "")
    try:
        n = int(float(n_str))
    except ValueError:
        return "[QUANTITY]"

    # Age heuristic: number must be plausible AND the surrounding text must
    # reference age explicitly. This avoids treating "£18,500" as someone's age.
    is_age = (0 <= n <= 120) and (
        "year" in text.lower() or "old" in text.lower()
    )

    if is_age:
        if level == 1:
            rounded = (n // 10) * 10
            return f"about {rounded}"
        if level == 2:
            band = (n // 10) * 10
            return f"in their {band}s" if band >= 20 else f"under {band + 10}"

    # Generic numeric quantity (currency, weights, durations)
    if level == 1:
        if n >= 1_000_000:
            base = (n // 1_000_000) * 1_000_000
            return f"about {base:,}"
        if n >= 1_000:
            base = (n // 1_000) * 1_000
            return f"about {base:,}"
        if n >= 100:
            base = (n // 100) * 100
            return f"about {base}"
        return f"about {(n // 10) * 10}"
    return "[QUANTITY]"


# ---------------------------------------------------------------------------
# LOC — small lookup of common cities → countries → continents
# ---------------------------------------------------------------------------
_CITY_TO_COUNTRY = {
    # ECHR-corpus heavy hitters
    "plovdiv": "Bulgaria", "sofia": "Bulgaria", "burgas": "Bulgaria",
    "istanbul": "Türkiye", "ankara": "Türkiye",
    "moscow": "Russia", "st. petersburg": "Russia", "rostov": "Russia",
    "warsaw": "Poland", "krakow": "Poland",
    "budapest": "Hungary",
    "bucharest": "Romania",
    "kyiv": "Ukraine", "kiev": "Ukraine", "lviv": "Ukraine",
    "athens": "Greece", "thessaloniki": "Greece",
    "rome": "Italy", "milan": "Italy",
    "paris": "France", "lyon": "France",
    "london": "United Kingdom", "manchester": "United Kingdom",
    "edinburgh": "United Kingdom", "birmingham": "United Kingdom",
    "leeds": "United Kingdom", "glasgow": "United Kingdom",
    "liverpool": "United Kingdom", "bristol": "United Kingdom",
    "berlin": "Germany", "munich": "Germany",
    "madrid": "Spain", "barcelona": "Spain",
    "lisbon": "Portugal",
    "vienna": "Austria",
    "prague": "Czechia",
    "stockholm": "Sweden",
    "oslo": "Norway",
    "helsinki": "Finland",
    "copenhagen": "Denmark",
    "amsterdam": "Netherlands",
    "brussels": "Belgium",
    "dublin": "Ireland",
}

_COUNTRY_TO_CONTINENT = {
    "Bulgaria": "Europe", "Türkiye": "Europe/Asia", "Russia": "Europe/Asia",
    "Poland": "Europe", "Hungary": "Europe", "Romania": "Europe",
    "Ukraine": "Europe", "Greece": "Europe", "Italy": "Europe",
    "France": "Europe", "United Kingdom": "Europe", "Germany": "Europe",
    "Spain": "Europe", "Portugal": "Europe", "Austria": "Europe",
    "Czechia": "Europe", "Sweden": "Europe", "Norway": "Europe",
    "Finland": "Europe", "Denmark": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "Ireland": "Europe",
}


def _generalize_loc(text: str, level: int) -> str:
    if level >= 3:
        return "[LOC]"
    key = text.strip().lower().rstrip(".,;:")
    if level == 1:
        return _CITY_TO_COUNTRY.get(key, "[LOC]")
    if level == 2:
        country = _CITY_TO_COUNTRY.get(key)
        if country:
            return _COUNTRY_TO_CONTINENT.get(country, "[LOC]")
        return "[LOC]"
    return "[LOC]"


# ---------------------------------------------------------------------------
# DEM — demographics; broaden by stripping specifics, then suppress
# ---------------------------------------------------------------------------
_DEM_BROADENING = {
    # Specific nationalities → continent/region
    "bulgarian": "European", "romanian": "European", "hungarian": "European",
    "russian": "Eastern European", "ukrainian": "Eastern European",
    "polish": "European", "german": "European", "french": "European",
    "italian": "European", "spanish": "European", "british": "European",
    "turkish": "Western Asian", "greek": "European",
    # Specific religions → broader categories
    "muslim": "religious minority", "jewish": "religious minority",
    "christian": "religious affiliation", "orthodox": "religious affiliation",
    "catholic": "religious affiliation", "protestant": "religious affiliation",
    "buddhist": "religious affiliation", "hindu": "religious affiliation",
    # Roles
    "asylum-seeker": "person seeking residence", "refugee": "person seeking residence",
    "pensioner": "older person", "retiree": "older person",
    # Ethnic groups
    "roma": "ethnic minority", "kurd": "ethnic minority", "kurdish": "ethnic minority",
}


def _generalize_dem(text: str, level: int) -> str:
    if level >= 3:
        return "[DEM]"
    key = text.strip().lower().rstrip(".,;:")
    if level == 1:
        return _DEM_BROADENING.get(key, "[DEM]")
    return "[DEM]"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
_GENERALIZERS = {
    "DATETIME": _generalize_datetime,
    "QUANTITY": _generalize_quantity,
    "LOC":      _generalize_loc,
    "DEM":      _generalize_dem,
}

# DIRECT-only entity types: never generalized, always suppressed
_ALWAYS_SUPPRESS = {"PERSON", "ORG", "CODE", "MISC"}


def generalize(entity_type: str, text: str, level: int) -> str:
    """
    Return the replacement text for a span at the given generalization level.
    `level` 0 means "leave as-is" (caller should not call us in that case).
    """
    if level <= 0:
        return text
    if entity_type in _ALWAYS_SUPPRESS:
        return f"[{entity_type}]"
    fn = _GENERALIZERS.get(entity_type)
    if fn is None:
        return f"[{entity_type}]"
    return fn(text, level)


# Maximum level any rule supports — useful for the iterate loop
MAX_LEVEL: int = 3

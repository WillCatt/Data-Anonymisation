"""
TAB ↔ spaCy entity-type mapping.

TAB uses legal-domain entity types; spaCy's English models use the OntoNotes
5 label set. Some TAB types (e.g. CODE — case file numbers like
"App. No. 12345/67") have no spaCy equivalent at all. Those gaps are the
core motivation for fine-tuning.
"""
from __future__ import annotations

from typing import Dict, List

# TAB entity type → list of spaCy labels we accept as a match
TAB_TO_SPACY: Dict[str, List[str]] = {
    "PERSON":   ["PERSON"],
    "ORG":      ["ORG"],
    "LOC":      ["GPE", "LOC", "FAC"],
    "DATETIME": ["DATE", "TIME"],
    "QUANTITY": ["QUANTITY", "CARDINAL", "MONEY", "PERCENT", "ORDINAL"],
    "CODE":     [],                                      # ❌ no spaCy equivalent
    "DEM":      ["NORP"],                                # ⚠️  partial only
    "MISC":     ["LAW", "EVENT", "PRODUCT", "WORK_OF_ART", "LANGUAGE"],
}

# Reverse mapping: spaCy label → TAB type
SPACY_TO_TAB: Dict[str, str] = {
    label: tab_type
    for tab_type, spacy_labels in TAB_TO_SPACY.items()
    for label in spacy_labels
}

# Human-readable description of each mapping decision; used by the writeup.
MAPPING_NOTES: Dict[str, str] = {
    "PERSON":   "Reasonable overlap; spaCy misses titles ('Dr', 'Lord Justice'), informal references, and partial names.",
    "ORG":      "Often noisy. spaCy fires ORG on phrases like 'the Court' or 'the Government' which TAB does not always treat as identifiers.",
    "LOC":      "TAB conflates GPE/LOC/FAC into one bucket; spaCy splits them. We accept any of the three.",
    "DATETIME": "Strong overlap. spaCy's DATE + TIME together cover most of TAB's DATETIME mentions.",
    "QUANTITY": "TAB's QUANTITY captures medically/legally-relevant numbers (ages, amounts, sentence lengths). spaCy fires on every number, producing a flood of false positives.",
    "CODE":     "Hard zero. TAB tags case-file numbers like 'Application No. 12345/67' as CODE, but spaCy has no equivalent label, so recall is 0%.",
    "DEM":      "Demographic identifiers (nationality, religion, ethnicity). spaCy's NORP overlaps partially but misses 'pensioner', 'asylum-seeker', 'migrant' style phrases.",
    "MISC":     "TAB's MISC catches anything that uniquely identifies — diagnoses, court orders, named operations. spaCy's LAW/EVENT/etc. cover only a sliver.",
}

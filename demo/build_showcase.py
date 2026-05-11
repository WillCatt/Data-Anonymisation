"""
Build the static side-by-side showcase HTML.

Runs LitePipeline and ProPipeline on the three example inputs in
demo/examples/, renders a single self-contained HTML page at
demo/index.html. No runtime dependencies for visitors — they just
open the file in a browser.

Why hand-crafted entity spans instead of running spaCy here?
    A static showcase wants deterministic, predictable output: the
    same pipelines, the same inputs, the same answers, no off-by-one
    NER drift between runs. We hand-write the spans for these three
    inputs once. The pipeline code under test is exactly the same as
    Phase 3 production — only the NER step is fixed.

Why a synthetic haystack?
    The mosaic scorer needs *something* to compare against. TAB itself
    is the right choice in real use (1,268 ECHR cases) but the download
    is ~50 MB and locks the demo to a network dependency. For the
    static showcase we hand-build a tiny haystack engineered so the
    three samples each demonstrate a different convergence outcome:
        Sample 1 — converges at level 1 (mild generalization)
        Sample 2 — converges at level 0 (mostly DIRECTs; little QUASI work)
        Sample 3 — converges at level 2 (deeper generalization)
    The HTML calls this out explicitly so a reader doesn't conflate
    the showcase haystack with what a deployment would use.

Run with:
    python demo/build_showcase.py

Output:
    demo/index.html
"""
from __future__ import annotations

import html
import json
import sys
from collections import Counter
from pathlib import Path
from typing import List, Tuple

# Allow `from anonymisation.pipeline import ...` whether run from repo
# root or from inside demo/.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from anonymisation.pipeline import (  # noqa: E402
    LitePipeline, ProPipeline, MosaicScorer, RedactionResult,
)


# ---------------------------------------------------------------------------
# Hand-crafted entity spans for each example. Computed at runtime via
# `text.find()` so they stay correct if the example file changes.
# ---------------------------------------------------------------------------
def _spans_via_find(text: str, items: List[Tuple[str, str]]) -> List[Tuple[int, int, str, str]]:
    """
    Locate each (substring, entity_type) and emit (start, end, etype, text).

    Handles substrings that occur multiple times in the document by skipping
    positions already claimed by a previously-assigned span. Avoids the
    overlapping-spans pitfall that my naive first-implementation hit
    (e.g. 'Acme' inside 'Acme Holdings Ltd' vs a later standalone 'Acme').
    """
    out: List[Tuple[int, int, str, str]] = []
    for needle, etype in items:
        # Find the next occurrence whose position doesn't overlap any span we've
        # already assigned. We require the candidate to be a *standalone* match
        # for short surface forms (so 'Acme' doesn't latch onto 'Acme Holdings Ltd').
        search_from = 0
        while True:
            idx = text.find(needle, search_from)
            if idx == -1:
                raise SystemExit(
                    f"Could not find an unclaimed occurrence of {needle!r} in example text"
                )
            end = idx + len(needle)
            # Reject if this position overlaps an already-assigned span
            if any(idx < ge and end > gs for (gs, ge, _, _) in out):
                search_from = idx + 1
                continue
            # Reject if it's a substring of a longer word (e.g. 'Acme' inside
            # 'Acme Holdings Ltd' — the surrounding chars must be non-alphanumeric)
            before = text[idx - 1] if idx > 0 else " "
            after = text[end] if end < len(text) else " "
            if (before.isalnum() and not needle[0].isspace()) or \
               (after.isalnum() and not needle[-1].isspace()):
                # We hit the middle of a word; advance and try again
                search_from = idx + 1
                continue
            out.append((idx, end, etype, needle))
            break
    return out


def make_predictor(text: str, items: List[Tuple[str, str]]):
    """Return a NER-style predictor that always emits the precomputed spans."""
    spans = _spans_via_find(text, items)

    def predict(_text: str):
        return spans
    return predict


CATEGORIES = [
    # ────────────────────────────────────────────────────────────────────
    # Category 1 — Civil rights / human rights (ECHR-style applications)
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "Law — civil rights",
        "summary": (
            "ECHR-style applications: individual complainants, demographic "
            "context, application numbers. The QUASI fingerprint (date + "
            "location + nationality + role) is what the mosaic loop has to "
            "wrestle with."
        ),
        "examples": [
            {
                "title": "Demographic-heavy application",
                "blurb": (
                    "Bulgarian nurse, single applicant, lots of QUASI surface "
                    "area. Pro converges at level 1 (mild generalisation)."
                ),
                "file": "examples/sample_1_demographic_heavy.txt",
                "entities": [
                    ("Maria Petrova",        "PERSON"),
                    ("47-year-old",          "QUANTITY"),
                    ("Bulgarian",            "DEM"),
                    ("Plovdiv",              "LOC"),
                    ("12 March 2018",        "DATETIME"),
                    ("Sofia District Court", "ORG"),
                    ("Roma",                 "DEM"),
                    ("2010",                 "DATETIME"),
                ],
                "k_target": 5,
                "max_iterations": 3,
            },
            {
                "title": "Asylum-seeker — single applicant",
                "blurb": (
                    "Tighter document, fewer QUASI mentions. With this haystack "
                    "the residual fingerprint is moderately distinctive — Pro "
                    "converges at level 1 after one iteration."
                ),
                "file": "examples/sample_civil_rights_b_asylum.txt",
                "entities": [
                    ("Dimitris Karagiannis",          "PERSON"),
                    ("28-year-old",                   "QUANTITY"),
                    ("Greek",                         "DEM"),
                    ("Athens",                        "LOC"),
                    ("5 June 2019",                   "DATETIME"),
                    ("Athens Court of First Instance","ORG"),
                    ("Mr Karagiannis",                "PERSON"),
                ],
                "k_target": 5,
                "max_iterations": 3,
            },
            {
                "title": "Press freedom — Pro fallback to suppression",
                "blurb": (
                    "Highly distinctive document (Russian journalist, named "
                    "publication, specific date, specific location). The "
                    "fingerprint never reaches k=5 in the haystack, so the "
                    "iterate-until-safe loop runs out of room and falls back "
                    "to full suppression. This is the converged=False case — "
                    "graceful degradation rather than silent partial leakage."
                ),
                "file": "examples/sample_civil_rights_c_press.txt",
                "entities": [
                    ("Anastasia Volkova",   "PERSON"),
                    ("41-year-old",         "QUANTITY"),
                    ("Russian",             "DEM"),
                    ("Moscow",              "LOC"),
                    ("14 November 2017",    "DATETIME"),
                    ("Open Forum",          "ORG"),
                    ("Russian Federation",  "ORG"),
                    ("Ms Volkova",          "PERSON"),
                    ("St. Petersburg",      "LOC"),
                ],
                "k_target": 5,
                "max_iterations": 2,   # cap at 2 to force the fallback
            },
        ],
    },

    # ────────────────────────────────────────────────────────────────────
    # Category 2 — Corporate / commercial
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "Law — corporate",
        "summary": (
            "Corporate litigation, M&A confidentiality breaches, IP disputes. "
            "Most of the identifying load is DIRECT (company names, case file "
            "numbers, IBANs) so Lite and Pro often produce similar output. The "
            "regex pass earns its keep here."
        ),
        "examples": [
            {
                "title": "Settlement summary",
                "blurb": (
                    "Settlement agreement: company, named person, case "
                    "number, IBAN, financial amount, multiple dates. "
                    "Initial fingerprint already matches the haystack — Pro "
                    "needs no QUASI work (k_initial ≥ k_target)."
                ),
                "file": "examples/sample_2_corporate.txt",
                "entities": [
                    ("Acme Holdings Ltd", "ORG"),
                    ("4 July 2022",       "DATETIME"),
                    ("John Doe",          "PERSON"),
                    ("January 2019",      "DATETIME"),
                    ("June 2022",         "DATETIME"),
                    ("Manchester",        "LOC"),
                    ("£450,000",          "QUANTITY"),
                ],
                "k_target": 5,
                "max_iterations": 3,
            },
            {
                "title": "M&A confidentiality breach",
                "blurb": (
                    "Former director, named M&A counterparty, multiple dates. "
                    "Pro converges at level 1: dates collapse to the year, "
                    "Edinburgh broadens to the United Kingdom."
                ),
                "file": "examples/sample_corporate_b_ma.txt",
                "entities": [
                    ("Globex Industries Plc", "ORG"),
                    ("Helena Park",           "PERSON"),
                    ("11 January 2024",       "DATETIME"),
                    ("Initech Holdings",      "ORG"),
                    ("Ms Park",               "PERSON"),
                    ("30 September 2023",     "DATETIME"),
                    ("Edinburgh",             "LOC"),
                ],
                "k_target": 5,
                "max_iterations": 3,
            },
            {
                "title": "IP licensing settlement",
                "blurb": (
                    "Patent dispute with a financial figure and an internal "
                    "engineer named. Pro converges at level 1; Manchester "
                    "becomes United Kingdom and the £2.3M becomes a rounded "
                    "amount."
                ),
                "file": "examples/sample_corporate_c_ip.txt",
                "entities": [
                    ("Northwind Energy Ltd", "ORG"),
                    ("Stark Renewables",     "ORG"),
                    ("22 August 2024",       "DATETIME"),
                    ("£2,300,000",           "QUANTITY"),
                    ("Manchester",           "LOC"),
                    ("Marcus Rivers",        "PERSON"),
                    ("2019",                 "DATETIME"),
                ],
                "k_target": 5,
                "max_iterations": 3,
            },
        ],
    },

    # ────────────────────────────────────────────────────────────────────
    # Category 3 — Medical / healthcare
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "Medical",
        "summary": (
            "Patient records and insurance claims. NHS numbers come out "
            "via the regex pass. The QUASI fingerprint (DOB + location + "
            "occupation + admission date) is typically distinctive — Pro "
            "tends to converge at deeper generalisation."
        ),
        "examples": [
            {
                "title": "Patient record (Sarah Khan)",
                "blurb": (
                    "DOB, location, occupation, sensitive medical context, "
                    "and an NHS number. Pro converges at level 2: DOB → "
                    "decade, London → Europe."
                ),
                "file": "examples/sample_3_minimal.txt",
                "entities": [
                    ("Sarah Khan",        "PERSON"),
                    ("14 February 1985",  "DATETIME"),
                    ("London",            "LOC"),
                    ("2020",              "DATETIME"),
                    ("FinTech Co",        "ORG"),
                ],
                "k_target": 5,
                "max_iterations": 3,
            },
            {
                "title": "Hospital incident report",
                "blurb": (
                    "Admission incident with multiple QUASI mentions and "
                    "an NHS number. Pro converges at level 1; the NHS "
                    "number is caught by the regex pass even though "
                    "off-the-shelf NER would never tag it."
                ),
                "file": "examples/sample_medical_b_incident.txt",
                "entities": [
                    ("Thomas Edwards",    "PERSON"),
                    ("62-year-old",       "QUANTITY"),
                    ("St Mary's Hospital","ORG"),
                    ("17 March 2025",     "DATETIME"),
                    ("Mr Edwards",        "PERSON"),
                    ("2014",              "DATETIME"),
                    ("Birmingham",        "LOC"),
                    ("Dr Lin",            "PERSON"),
                ],
                "k_target": 5,
                "max_iterations": 3,
            },
            {
                "title": "Insurance claim review",
                "blurb": (
                    "Policyholder profile with DOB, role, hospital, and "
                    "annual claim amount. Pro converges at level 2 because "
                    "the joint profile is distinctive."
                ),
                "file": "examples/sample_medical_c_insurance.txt",
                "entities": [
                    ("Aisha Rahman",            "PERSON"),
                    ("9 August 1979",           "DATETIME"),
                    ("Leeds",                   "LOC"),
                    ("Leeds General Infirmary", "ORG"),
                    ("2022",                    "DATETIME"),
                    ("£18,500",                 "QUANTITY"),
                    ("Ms Rahman",               "PERSON"),
                    ("2012",                    "DATETIME"),
                ],
                "k_target": 5,
                "max_iterations": 3,
            },
        ],
    },

    # ────────────────────────────────────────────────────────────────────
    # Category 4 — Pseudonymisation + LLM round-trip
    # ────────────────────────────────────────────────────────────────────
    {
        "title": "Pseudonymisation",
        "summary": (
            "Phase 4: instead of every PERSON becoming [PERSON], each distinct "
            "entity gets a stable referential token. The vault stays with the "
            "firm; the LLM only ever sees the tokens. After the LLM responds, "
            "the firm runs restore() locally to swap tokens for real names."
        ),
        "examples": [
            {
                "mode": "round_trip",
                "title": "Multi-party contract dispute",
                "blurb": (
                    "Original Phase 4 demo — two distinct parties referenced "
                    "by multiple surface forms each ('Maria Petrova', 'Mrs "
                    "Petrova', 'Petrova'). Substring + honorific coref "
                    "collapses each set to one token. Round-trip preserves "
                    "the canonical form."
                ),
                "file": "examples/sample_4_round_trip.txt",
                "entities": [
                    ("Maria Petrova",        "PERSON"),
                    ("Acme Holdings Ltd",    "ORG"),
                    ("12 March 2018",        "DATETIME"),
                    ("Mrs Petrova",          "PERSON"),
                    ("Bulgarian",            "DEM"),
                    ("John Smith",           "PERSON"),
                    ("Acme",                 "ORG"),
                    ("Sofia District Court", "ORG"),
                    ("Smith",                "PERSON"),
                    ("Petrova",              "PERSON"),
                    ("€450,000",             "QUANTITY"),
                ],
                "k_target": 5,
                "max_iterations": 3,
                "llm_question": "Who is suing whom, and what court is involved?",
                "llm_answer_template": (
                    "Based on the redacted document, {PERSON_A} is suing {ORG_A} "
                    "(represented by {ORG_B}) over a contract dispute. "
                    "The defendants include {PERSON_B}, the company's CFO."
                ),
            },
            {
                "mode": "round_trip",
                "title": "HR investigation — three parties + tribunal",
                "blurb": (
                    "Internal HR document. Complainant, subject, and witness "
                    "all distinct PERSONs. Mitchell is referenced two ways "
                    "('Sarah Mitchell' / surname only) and coreferred. The "
                    "round-trip shows how an HR partner could ask an LLM "
                    "structural questions about who reports to whom without "
                    "exposing the names."
                ),
                "file": "examples/sample_pseudo_b_hr.txt",
                "entities": [
                    ("Sarah Mitchell",   "PERSON"),
                    ("Acme Holdings Ltd","ORG"),
                    ("2019",             "DATETIME"),
                    ("Daniel Park",      "PERSON"),
                    ("Mr Park",          "PERSON"),
                    ("8 February 2025",  "DATETIME"),
                    ("Lisa Chen",        "PERSON"),
                    ("Park",             "PERSON"),
                    ("Acme",             "ORG"),
                    ("2017",             "DATETIME"),
                    ("Mitchell",         "PERSON"),
                    ("London Employment Tribunal", "ORG"),
                ],
                "k_target": 5,
                "max_iterations": 3,
                "llm_question": "Summarise the structure of the HR complaint — who reported what, against whom?",
                "llm_answer_template": (
                    "{PERSON_A} (the complainant) has alleged that {PERSON_B} "
                    "made discriminatory comments. {PERSON_C} is named as a "
                    "witness. The matter is being handled by {ORG_A} before "
                    "any escalation to {ORG_B}."
                ),
            },
            {
                "mode": "round_trip",
                "title": "Class action — five plaintiffs + letter rollover",
                "blurb": (
                    "Five distinct plaintiffs, each named twice. Demonstrates "
                    "the [PERSON_A]..[PERSON_E] sequence and the letter "
                    "rollover behaviour. Useful for portfolio audiences who "
                    "want to see the pseudonymiser handle non-trivial "
                    "fan-out."
                ),
                "file": "examples/sample_pseudo_c_class_action.txt",
                "entities": [
                    ("4 March 2025",         "DATETIME"),
                    ("MegaCorp International","ORG"),
                    ("Robert Anderson",      "PERSON"),
                    ("Yuki Tanaka",          "PERSON"),
                    ("Priya Sharma",         "PERSON"),
                    ("Carlos Mendez",        "PERSON"),
                    ("Nina Petrov",          "PERSON"),
                    ("Birmingham",           "LOC"),
                    ("2021",                 "DATETIME"),
                    ("2024",                 "DATETIME"),
                    ("Anderson",             "PERSON"),
                    ("Tanaka",               "PERSON"),
                    ("Sharma",               "PERSON"),
                    ("Mendez",               "PERSON"),
                    ("Petrov",               "PERSON"),
                    ("£35,000",              "QUANTITY"),
                    ("MegaCorp",             "ORG"),
                    ("Bishop & Wright LLP",  "ORG"),
                ],
                "k_target": 5,
                "max_iterations": 3,
                "llm_question": "How many plaintiffs are there, and which firm represents the defendant?",
                "llm_answer_template": (
                    "There are five lead plaintiffs: {PERSON_A}, {PERSON_B}, "
                    "{PERSON_C}, {PERSON_D}, and {PERSON_E}. The defendant "
                    "{ORG_A} is represented by {ORG_B}."
                ),
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Synthetic haystack — engineered to drive specific convergence outcomes
# ---------------------------------------------------------------------------
def build_haystack() -> MosaicScorer:
    """
    Build a small synthetic haystack so each example converges at a useful
    level (mostly level 1 or 2). Two examples are deliberately *not* given
    a haystack match: the press-freedom case and the IP licensing case
    use rare-fingerprint shapes that demonstrate Pro fallback.

    The haystack is illustrative, not real. In a deployment it would come
    from the firm's own document corpus (or, for evaluation, from TAB).
    """
    sigs = []

    # Helper — add a sorted+lowered fingerprint to the haystack n times.
    def add(parts, n=6):
        sigs.extend([tuple(sorted({(t, v.lower()) for t, v in parts}))] * n)

    # ── Civil rights, sample (a): Bulgarian nurse — converges at level 1
    add([
        ("DATETIME", "2010"), ("DATETIME", "2018"),
        ("DEM", "european"), ("DEM", "ethnic minority"),
        ("LOC", "bulgaria"), ("QUANTITY", "about 40"),
    ])

    # ── Civil rights, sample (b): Greek asylum-seeker — converges at level 1
    add([
        ("DATETIME", "2019"), ("DEM", "european"),
        ("LOC", "greece"), ("QUANTITY", "about 20"),
    ])

    # ── Civil rights, sample (c): Russian press case — NO match in haystack.
    # This is intentional: with max_iterations=2 and an unmatched fingerprint
    # the iterate-until-safe loop will exit unconverged and fall back to
    # full suppression. Demonstrates graceful degradation.
    # (Nothing added.)

    # ── Corporate, sample (a): Acme settlement — already safe (k_initial)
    add([
        ("DATETIME", "4 july 2022"),
        ("DATETIME", "january 2019"),
        ("DATETIME", "june 2022"),
        ("LOC",      "manchester"),
        ("QUANTITY", "£450,000"),
    ])

    # ── Corporate, sample (b): M&A — converges at level 1
    add([
        ("DATETIME", "2024"), ("DATETIME", "2023"),
        ("LOC", "united kingdom"),
    ])

    # ── Corporate, sample (c): IP licensing — converges at level 1
    # Generalised QUANTITY for £2,300,000 is "about 2,000,000".
    add([
        ("DATETIME", "2024"), ("DATETIME", "2019"),
        ("LOC", "united kingdom"), ("QUANTITY", "about 2,000,000"),
    ])

    # ── Medical, sample (a): Sarah Khan — converges at level 2
    add([
        ("DATETIME", "1980s"), ("DATETIME", "2020s"),
        ("LOC", "europe"),
    ])

    # ── Medical, sample (b): hospital incident — converges at level 1
    add([
        ("DATETIME", "2025"), ("DATETIME", "2014"),
        ("LOC", "united kingdom"), ("QUANTITY", "about 60"),
    ])

    # ── Medical, sample (c): insurance — converges at level 2
    # At level 2 generic QUANTITY suppresses to [QUANTITY] (excluded from
    # signature), so the haystack entry should NOT include a quantity.
    add([
        ("DATETIME", "1970s"), ("DATETIME", "2020s"), ("DATETIME", "2010s"),
        ("LOC", "europe"),
    ])

    # ── Pseudonymisation samples don't go through the mosaic loop directly
    # (they use Lite + pseudonymise=True), so no haystack entries needed.

    # Sprinkle of unrelated noise so the haystack feels less hand-built
    sigs.extend([
        (("DATETIME", "1999"), ("LOC", "athens")),
        (("DEM", "european"), ("LOC", "italy")),
        (("DATETIME", "2005"), ("DEM", "european")),
        (("LOC", "spain"), ("QUANTITY", "about 60")),
        (("LOC", "germany"), ("QUANTITY", "about 30")),
        (("DATETIME", "2012"), ("LOC", "europe")),
    ])

    return MosaicScorer(sigs)


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
CSS = """
:root {
  --bg: #fafbfc;
  --panel: #ffffff;
  --border: #e1e4e8;
  --text: #24292e;
  --muted: #586069;
  --accent: #0366d6;
  --green: #2ea44f;
  --amber: #e36209;
  --red:   #d73a49;
  --code-bg: #f6f8fa;
  --tag-bg: #fff5b1;
  --tag-text: #735c0f;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  margin: 0; padding: 2rem 1rem;
  line-height: 1.5;
}
.container { max-width: 1100px; margin: 0 auto; }
header { margin-bottom: 2rem; }
header h1 { margin: 0 0 0.5rem; font-size: 1.6rem; }
header p { color: var(--muted); margin: 0.25rem 0; }
.tag-row { margin: 0.75rem 0 0; }
.tag { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 1rem;
       background: #eaf5ff; color: var(--accent); font-size: 0.8rem; margin-right: 0.5rem; }

.tabs { display: flex; flex-wrap: wrap; gap: 0.25rem; border-bottom: 1px solid var(--border); margin-bottom: 1rem; }
.tabs label {
  cursor: pointer;
  padding: 0.5rem 1rem;
  border: 1px solid transparent;
  border-bottom: none;
  border-radius: 6px 6px 0 0;
  font-size: 0.95rem;
  color: var(--muted);
}
.tabs input[type=radio] { display: none; }
.tabs label:hover { color: var(--text); }
.panel { display: none; }

/* CSS-only tab switching */
#tab1:checked ~ .tabs label[for=tab1],
#tab2:checked ~ .tabs label[for=tab2],
#tab3:checked ~ .tabs label[for=tab3],
#tab4:checked ~ .tabs label[for=tab4] {
  background: var(--panel); color: var(--text);
  border-color: var(--border); border-bottom-color: var(--panel);
  margin-bottom: -1px;
}
#tab1:checked ~ .panels #panel1,
#tab2:checked ~ .panels #panel2,
#tab3:checked ~ .panels #panel3,
#tab4:checked ~ .panels #panel4 { display: block; }

.sample-blurb {
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 0.75rem 1rem; color: var(--muted); margin-bottom: 1rem; font-size: 0.95rem;
}

.category-summary {
  margin: 0 0 1.25rem; padding: 0.75rem 1rem;
  background: #f1f8ff; border-left: 3px solid var(--accent);
  border-radius: 0 4px 4px 0; font-size: 0.92rem; color: var(--text);
}

.example-card {
  background: transparent;
  margin-bottom: 2rem;
  padding-top: 1rem;
  border-top: 1px solid var(--border);
}
.example-card:first-child { border-top: none; padding-top: 0; }
.example-header {
  display: flex; align-items: center; gap: 0.5rem;
  margin-bottom: 0.5rem;
}
.example-header h2 {
  margin: 0; font-size: 1.05rem; font-weight: 600;
}
.example-header .pill {
  display: inline-block; padding: 0.1rem 0.5rem; border-radius: 1rem;
  background: var(--code-bg); color: var(--muted);
  font-size: 0.75rem; font-family: ui-monospace, SFMono-Regular, monospace;
}
.cols { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1rem; }
@media (max-width: 800px) { .cols { grid-template-columns: 1fr; } }
.col {
  background: var(--panel); border: 1px solid var(--border); border-radius: 6px;
  padding: 1rem; min-height: 200px;
}
.col h3 { margin: 0 0 0.5rem; font-size: 1rem; }
.col h3 .badge {
  display: inline-block; font-size: 0.7rem; padding: 0.1rem 0.45rem;
  border-radius: 3px; background: #f1f8ff; color: var(--accent);
  margin-left: 0.5rem; vertical-align: middle;
}
.col h3 .badge.pro { background: #e6ffed; color: var(--green); }
.col h3 .badge.lite { background: #fff5b1; color: var(--tag-text); }
.col h3 .badge.pseudo { background: #ddf4ff; color: var(--accent); }
.col h3 .badge.vault { background: #f4f1ff; color: #6f42c1; }

.vault-table {
  width: 100%; border-collapse: collapse; font-size: 0.85rem;
  font-family: ui-monospace, SFMono-Regular, monospace;
}
.vault-table th, .vault-table td {
  text-align: left; padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--border);
}
.vault-table th { background: var(--code-bg); color: var(--muted); font-weight: 600; }
.vault-table td.token { color: var(--accent); font-weight: 600; }

.round-trip {
  margin-top: 1.5rem; background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; padding: 1.25rem 1.5rem;
}
.round-trip h3 { margin: 0 0 1rem; font-size: 1rem; }
.round-trip-step {
  margin-bottom: 1rem; padding-left: 1.5rem; position: relative;
  border-left: 2px solid var(--accent);
}
.round-trip-step:last-of-type { margin-bottom: 0; }
.step-label {
  font-size: 0.85rem; color: var(--muted); margin-bottom: 0.4rem; font-weight: 600;
}
.step-body {
  background: var(--code-bg); padding: 0.75rem 1rem; border-radius: 4px;
  font-size: 0.9rem; line-height: 1.55;
}
.round-trip-note {
  margin: 1rem 0 0; font-size: 0.85rem; color: var(--muted); font-style: italic;
}
.col .body { white-space: pre-wrap; font-size: 0.9rem; line-height: 1.55; }
.col .body .tag {
  background: var(--tag-bg); color: var(--tag-text);
  padding: 0 0.25rem; border-radius: 3px; font-weight: 600;
  font-family: ui-monospace, SFMono-Regular, monospace;
  font-size: 0.85em;
}
.mosaic {
  margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px dashed var(--border);
  font-size: 0.85rem; color: var(--muted);
}
.mosaic .k {
  display: inline-block; padding: 0.1rem 0.45rem; border-radius: 3px;
  font-family: ui-monospace, SFMono-Regular, monospace; font-weight: 600;
  background: var(--code-bg); color: var(--text);
}
.mosaic .converged { color: var(--green); }
.mosaic .unconverged { color: var(--red); }

details {
  margin-top: 1.5rem; background: var(--panel); border: 1px solid var(--border);
  border-radius: 6px; padding: 0.5rem 1rem;
}
details summary {
  cursor: pointer; font-size: 0.9rem; color: var(--muted); padding: 0.5rem 0;
}
details[open] summary { color: var(--text); border-bottom: 1px solid var(--border);
                        margin-bottom: 0.75rem; padding-bottom: 0.75rem; }
.audit-table {
  width: 100%; border-collapse: collapse; font-size: 0.82rem;
  font-family: ui-monospace, SFMono-Regular, monospace;
}
.audit-table th, .audit-table td {
  text-align: left; padding: 0.3rem 0.5rem; border-bottom: 1px solid var(--border);
  vertical-align: top;
}
.audit-table th { background: var(--code-bg); color: var(--muted); font-weight: 600; }
.audit-table td.iter { color: var(--muted); }
.audit-table td.action.redact     { color: var(--red); }
.audit-table td.action.generalize { color: var(--amber); }
.audit-table td.action.leave      { color: var(--green); }

footer {
  margin-top: 3rem; padding-top: 1.5rem; border-top: 1px solid var(--border);
  font-size: 0.85rem; color: var(--muted);
}
footer p { margin: 0.5rem 0; }
footer a { color: var(--accent); text-decoration: none; }
footer a:hover { text-decoration: underline; }

.callout {
  background: #fffbea; border-left: 3px solid var(--amber);
  padding: 0.75rem 1rem; margin: 1rem 0; font-size: 0.9rem; color: #735c0f;
  border-radius: 0 4px 4px 0;
}
"""


def _highlight_tags(text: str) -> str:
    """Wrap [TYPE] tokens in a span so CSS can colour them."""
    import re
    escaped = html.escape(text)
    return re.sub(
        r"(\[(?:PERSON|ORG|CODE|MISC|LOC|DATETIME|QUANTITY|DEM)\])",
        r'<span class="tag">\1</span>',
        escaped,
    )


def render_example_card(example: dict, lite_result: RedactionResult,
                        pro_result: RedactionResult, original: str) -> str:
    """Render one example as a card within a category panel (Lite vs Pro)."""
    pro_converged_class = "converged" if pro_result.converged else "unconverged"
    pro_status = "✓ converged" if pro_result.converged else "✗ fallback to suppression"
    audit_rows = []
    for entry in pro_result.audit:
        s = entry.span
        original_t = html.escape(s.text)
        replacement_t = html.escape(s.replacement) if s.replacement else "—"
        audit_rows.append(
            f'<tr><td class="iter">{entry.iteration}</td>'
            f'<td>{html.escape(s.entity_type)}</td>'
            f'<td>{html.escape(s.identifier_role)}</td>'
            f'<td class="action {entry.action}">{entry.action}</td>'
            f'<td>{original_t}</td>'
            f'<td>→ {replacement_t}</td></tr>'
        )
    audit_html = (
        '<table class="audit-table">'
        '<tr><th>iter</th><th>type</th><th>role</th><th>action</th><th>original</th><th>replacement</th></tr>'
        + "".join(audit_rows) +
        '</table>'
    )

    return f"""
    <div class="example-card">
      <div class="example-header">
        <h2>{html.escape(example['title'])}</h2>
        <span class="pill">k_target={example['k_target']} · iters={example['max_iterations']}</span>
      </div>
      <div class="sample-blurb">{html.escape(example['blurb'])}</div>
      <div class="cols">
        <div class="col">
          <h3>Original</h3>
          <div class="body">{html.escape(original)}</div>
        </div>
        <div class="col">
          <h3>Lite <span class="badge lite">DIRECT-only</span></h3>
          <div class="body">{_highlight_tags(lite_result.redacted_text)}</div>
        </div>
        <div class="col">
          <h3>Pro <span class="badge pro">DIRECT + mosaic-aware QUASI</span></h3>
          <div class="body">{_highlight_tags(pro_result.redacted_text)}</div>
          <div class="mosaic">
            <span class="k">k_initial = {pro_result.mosaic_risk_initial}</span>
            <span> → </span>
            <span class="k">k_final = {pro_result.mosaic_risk_final}</span>
            <span> · iterations: {pro_result.iterations_used} · </span>
            <span class="{pro_converged_class}">{pro_status}</span>
          </div>
        </div>
      </div>
      <details>
        <summary>Pro audit log</summary>
        {audit_html}
      </details>
    </div>
    """


def render_round_trip_card(example: dict, pseudo_result: RedactionResult, original: str) -> str:
    """Render a pseudonymisation round-trip example card."""
    from anonymisation.pipeline import restore

    vault_rows = "".join(
        f'<tr><td class="token">{html.escape(token)}</td>'
        f'<td>→ {html.escape(original_text)}</td></tr>'
        for token, original_text in pseudo_result.pseudonym_vault.items()
    )
    vault_html = (
        '<table class="vault-table">'
        '<tr><th>token</th><th>original</th></tr>'
        + vault_rows +
        '</table>'
    )

    # Look up actual tokens by case-insensitive original surface form.
    vault = pseudo_result.pseudonym_vault
    by_original = {v.lower(): k for k, v in vault.items()}

    # The template uses placeholders like {PERSON_A}, {ORG_A}. We need to find
    # which actual surface form corresponds to PERSON_A in this example. The
    # convention: the example's first PERSON entity is PERSON_A, the second is
    # PERSON_B (after coref dedup), etc. Walk the entities list and assign.
    expected_tokens: dict = {}
    seen_per_type: dict = {}
    for surface, etype in example["entities"]:
        if etype not in ("PERSON", "ORG"):
            continue
        # Use the actual token assigned (handles coref correctly)
        token = by_original.get(surface.lower())
        if token is None:
            continue
        # Letter index is determined by the order this distinct entity was first seen
        if token not in seen_per_type.values():
            # First occurrence of this token — assign it the next slot for its type
            type_count = sum(1 for tk in seen_per_type.values() if tk.startswith(f"[{etype}_"))
            slot_letter = chr(ord("A") + type_count)
            expected_tokens[f"{etype}_{slot_letter}"] = token
            seen_per_type[surface] = token

    # Some templates (HR, class action) use placeholders by entity-type slot:
    # {PERSON_A}, {PERSON_B}, ..., {ORG_A}, {ORG_B}
    template_args = {}
    person_slots = sorted([k for k in expected_tokens if k.startswith("PERSON_")])
    org_slots    = sorted([k for k in expected_tokens if k.startswith("ORG_")])
    for slot in person_slots: template_args[slot] = expected_tokens[slot]
    for slot in org_slots:    template_args[slot] = expected_tokens[slot]
    # Backfill any missing slots with literal "[TYPE_X]" so format() doesn't crash
    for placeholder in ["PERSON_A","PERSON_B","PERSON_C","PERSON_D","PERSON_E","ORG_A","ORG_B","ORG_C"]:
        template_args.setdefault(placeholder, f"[{placeholder}]")

    llm_answer = example["llm_answer_template"].format(**template_args)
    restored_answer = restore(llm_answer, vault)

    return f"""
    <div class="example-card">
      <div class="example-header">
        <h2>{html.escape(example['title'])}</h2>
        <span class="pill">pseudonymise=True</span>
      </div>
      <div class="sample-blurb">{html.escape(example['blurb'])}</div>
      <div class="cols">
        <div class="col">
          <h3>Original</h3>
          <div class="body">{html.escape(original)}</div>
        </div>
        <div class="col">
          <h3>Pseudonymised <span class="badge pseudo">DIRECT → tokens</span></h3>
          <div class="body">{_highlight_tags(pseudo_result.redacted_text)}</div>
        </div>
        <div class="col">
          <h3>Vault <span class="badge vault">stays with the firm</span></h3>
          {vault_html}
        </div>
      </div>
      <div class="round-trip">
        <h3>Round-trip — what the LLM workflow looks like</h3>
        <div class="round-trip-step">
          <div class="step-label">1. The firm sends the pseudonymised document + a question to a third-party LLM</div>
          <div class="step-body">
            <strong>Question:</strong> {html.escape(example['llm_question'])}
          </div>
        </div>
        <div class="round-trip-step">
          <div class="step-label">2. The LLM responds, still using only the opaque tokens</div>
          <div class="step-body">{_highlight_tags(llm_answer)}</div>
        </div>
        <div class="round-trip-step">
          <div class="step-label">3. The firm runs <code>restore(answer, vault)</code> locally</div>
          <div class="step-body">{html.escape(restored_answer)}</div>
        </div>
        <p class="round-trip-note">
          The LLM never saw the real names. The restoration step is pure local
          string replacement — no API call, no external dependency on the LLM
          provider's logs or retention policy.
        </p>
      </div>
    </div>
    """


def render_category_panel(idx: int, category: dict, scorer: MosaicScorer, demo_dir: Path) -> str:
    """Render one category as a panel containing multiple stacked example cards."""
    cards = []
    for example in category["examples"]:
        text = (demo_dir / example["file"]).read_text().strip()
        predictor = make_predictor(text, example["entities"])

        if example.get("mode") == "round_trip":
            pseudo_lite = LitePipeline(ner_provider=predictor, pseudonymise=True)
            pseudo_result = pseudo_lite(text)
            cards.append(render_round_trip_card(example, pseudo_result, text))
        else:
            lite = LitePipeline(ner_provider=predictor)
            pro = ProPipeline(
                ner_provider=predictor, scorer=scorer,
                k_target=example["k_target"], max_iterations=example["max_iterations"],
            )
            cards.append(render_example_card(
                example, lite(text), pro(text), text,
            ))

    return f"""
    <div class="panel" id="panel{idx}">
      <div class="category-summary">{html.escape(category['summary'])}</div>
      {''.join(cards)}
    </div>
    """


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    demo_dir = repo_root / "demo"

    scorer = build_haystack()

    panels = []
    radios = []
    tabs = []
    for i, category in enumerate(CATEGORIES, start=1):
        panels.append(render_category_panel(i, category, scorer, demo_dir))
        radios.append(
            f'<input type="radio" name="tab" id="tab{i}"' + (' checked' if i == 1 else '') + '>'
        )
        tabs.append(f'<label for="tab{i}">{html.escape(category["title"])}</label>')

    body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Legal Text Anonymisation — Lite vs Pro</title>
  <style>{CSS}</style>
</head>
<body>
<div class="container">
  <header>
    <h1>Legal Text Anonymisation — Lite vs Pro</h1>
    <p>Two redaction pipelines, side by side, on three illustrative legal-style inputs.</p>
    <div class="tag-row">
      <span class="tag">Lite — DIRECT-only</span>
      <span class="tag">Pro — DIRECT + mosaic-aware QUASI</span>
    </div>
  </header>

  <div class="callout">
    <strong>About this showcase.</strong> The pipelines are the same code as
    in <code>src/anonymisation/pipeline/</code>. Entity spans here are
    hand-curated for deterministic output; in production they come from the
    spaCy / RoBERTa NER from earlier phases. The mosaic scorer's haystack is
    a small synthetic corpus engineered to drive useful convergence behavior
    across the three samples — in deployment it would be the firm's own
    document corpus.
  </div>

  {''.join(radios)}
  <div class="tabs">{''.join(tabs)}</div>
  <div class="panels">
    {''.join(panels)}
  </div>

  <footer>
    <p>Source: <a href="https://github.com/wcatt98/data-anonymisation">github.com/wcatt98/data-anonymisation</a> ·
       Built from the Phase 3 pipelines.
       The interactive Gradio version is on the roadmap once the dependency
       situation calms down — see <code>demo/app.py</code>.</p>
  </footer>
</div>
</body>
</html>
"""
    out = demo_dir / "index.html"
    out.write_text(body)
    print(f"Wrote {out} ({len(body):,} bytes)")


if __name__ == "__main__":
    main()

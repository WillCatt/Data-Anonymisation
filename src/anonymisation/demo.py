"""
Interactive `demo_anonymise` helper.

Given an input string and a spaCy NLP pipeline, prints the entities the
model detected, the entities (if any) that fell into the UNMAPPED bucket,
and an `anonymised` version where every detected entity is replaced with
its TAB type in square brackets, e.g. "[PERSON]".

Used by the live-demo cells in the Phase-1 notebook and (later) the
Gradio / HF Space demo.
"""
from __future__ import annotations


from .mapping import SPACY_TO_TAB


def demo_anonymise(text: str, nlp_model, verbose: bool = True) -> dict:
    """
    Run NER on text and (optionally) print colour-coded results.

    Returns
    -------
    {
        "detected"    : list of (start, end, spacy_label, tab_type, span_text),
        "unmapped"    : list of (start, end, spacy_label, span_text),
        "anonymised"  : the input text with detected entities replaced by [TAB_TYPE]
    }
    """
    doc = nlp_model(text)

    detected = []
    unmapped = []
    for ent in doc.ents:
        tab_type = SPACY_TO_TAB.get(ent.label_)
        if tab_type:
            detected.append((ent.start_char, ent.end_char, ent.label_, tab_type, ent.text))
        else:
            unmapped.append((ent.start_char, ent.end_char, ent.label_, ent.text))

    # Build anonymised output (replace in reverse to preserve offsets)
    anonymised = text
    for ent in sorted(doc.ents, key=lambda e: e.start_char, reverse=True):
        tab_type = SPACY_TO_TAB.get(ent.label_, ent.label_)
        anonymised = anonymised[: ent.start_char] + f"[{tab_type}]" + anonymised[ent.end_char :]

    if verbose:
        bar = "=" * 70
        print(bar)
        print("ENTITIES DETECTED")
        print(bar)
        if not doc.ents:
            print("  (no entities detected)")
        else:
            for s, e, lbl, tt, sp in detected:
                print(f"  [{lbl:10s} → {tt:10s}] \"{sp}\"")
            for s, e, lbl, sp in unmapped:
                print(f"  [{lbl:10s} → UNMAPPED ] \"{sp}\"")

        print(f"\n{bar}\nANONYMISED OUTPUT\n{bar}")
        print(anonymised)

    return {"detected": detected, "unmapped": unmapped, "anonymised": anonymised}

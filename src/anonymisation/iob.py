"""
IOB / BIO tagging utilities for fine-tuning a token-classification model on TAB.

The TAB annotations are character-level spans. To fine-tune a transformer,
we need token-level BIO labels:

    O           outside any entity
    B-PERSON    first sub-token of a PERSON span
    I-PERSON    subsequent sub-token of a PERSON span
    ...

This module:
  * builds the canonical label list (always in the same order so id2label is stable)
  * converts a TAB document + a tokenizer offset_mapping into BIO label IDs
  * converts model predictions back into character-offset spans for evaluation

Only DIRECT and QUASI mentions count — NO_MASK is excluded from gold.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from .mapping import TAB_TO_SPACY

# Canonical TAB entity type list (sorted for stable id2label)
TAB_ENTITY_TYPES: List[str] = sorted(TAB_TO_SPACY.keys())

# BIO label list — index 0 is "O", followed by B-/I- pairs per entity type
BIO_LABELS: List[str] = ["O"] + [
    f"{prefix}-{et}" for et in TAB_ENTITY_TYPES for prefix in ("B", "I")
]

LABEL_TO_ID: Dict[str, int] = {label: i for i, label in enumerate(BIO_LABELS)}
ID_TO_LABEL: Dict[int, str] = {i: label for i, label in enumerate(BIO_LABELS)}


def gold_spans_for_training(doc: dict) -> List[Tuple[int, int, str]]:
    """Pull (start, end, entity_type) for DIRECT/QUASI mentions from a TAB doc."""
    return [
        (em["start_offset"], em["end_offset"], em["entity_type"])
        for em in doc["entity_mentions"]
        if em["identifier_type"] in ("DIRECT", "QUASI")
    ]


def offsets_to_bio(
    offset_mapping: Sequence[Tuple[int, int]],
    gold_spans: Sequence[Tuple[int, int, str]],
    *,
    ignore_index: int = -100,
    label_first_subword_only: bool = True,
    word_ids: Sequence | None = None,
) -> List[int]:
    """
    Project character-offset gold spans onto a tokenizer's offset_mapping.

    Parameters
    ----------
    offset_mapping :
        Per-token (char_start, char_end) tuples from a fast tokenizer.
        Special tokens have offsets (0, 0) — we treat them as O / ignored.
    gold_spans :
        List of (char_start, char_end, entity_type).
    label_first_subword_only :
        If True, subword continuations are labeled `ignore_index` (the
        standard recipe for token classification — only the first sub-token
        of each word contributes to the loss).
    word_ids :
        Optional aligned list from `tokenizer(...).word_ids()`. Required when
        `label_first_subword_only=True` to detect subword continuations.

    Returns
    -------
    List of label IDs the same length as `offset_mapping`.
    """
    labels: List[int] = []
    last_word: int | None = None

    for i, (start, end) in enumerate(offset_mapping):
        # Special tokens (CLS, SEP, PAD) have (0, 0) offsets in HF fast tokenizers
        if start == end == 0:
            labels.append(ignore_index)
            last_word = None
            continue

        # Subword continuation handling
        is_continuation = False
        if label_first_subword_only and word_ids is not None:
            wid = word_ids[i]
            is_continuation = wid is not None and wid == last_word
            last_word = wid

        if is_continuation:
            labels.append(ignore_index)
            continue

        # Find the gold span this token sits inside (if any)
        # We require the *whole* token to be inside the span — this is
        # stricter than character overlap, but avoids tokenizer drift on
        # boundary punctuation.
        match_type: str | None = None
        for gs, ge, gt in gold_spans:
            if start >= gs and end <= ge:
                match_type = gt
                break

        if match_type is None:
            labels.append(LABEL_TO_ID["O"])
            continue

        # Determine B- vs I- by looking at the most recent non-ignored label.
        # If that label belongs to the same entity type, we're a continuation;
        # otherwise we're starting a new span.
        prev_label_id: int | None = None
        for j in range(len(labels) - 1, -1, -1):
            if labels[j] != ignore_index:
                prev_label_id = labels[j]
                break

        b_id = LABEL_TO_ID[f"B-{match_type}"]
        i_id = LABEL_TO_ID[f"I-{match_type}"]
        if prev_label_id in (b_id, i_id):
            labels.append(i_id)
        else:
            labels.append(b_id)

    return labels


def bio_to_spans(
    label_ids: Sequence[int],
    offset_mapping: Sequence[Tuple[int, int]],
    text: str,
) -> List[Tuple[int, int, str, str]]:
    """
    Convert a sequence of predicted BIO label IDs back into character-offset
    spans of the form (start, end, entity_type, span_text) — the standard
    shape consumed by `evaluation.evaluate_document`.

    Tokens with offset (0, 0) are skipped (special tokens).
    """
    spans: List[Tuple[int, int, str, str]] = []
    cur_start: int | None = None
    cur_end: int | None = None
    cur_type: str | None = None

    def _flush() -> None:
        if cur_start is not None and cur_type is not None and cur_end is not None:
            spans.append((cur_start, cur_end, cur_type, text[cur_start:cur_end]))

    for lid, (start, end) in zip(label_ids, offset_mapping):
        if start == end == 0:
            continue
        label = ID_TO_LABEL.get(int(lid), "O")

        if label == "O":
            _flush()
            cur_start = cur_end = cur_type = None
            continue

        prefix, _, etype = label.partition("-")

        if prefix == "B" or etype != cur_type or cur_start is None:
            _flush()
            cur_start, cur_end, cur_type = start, end, etype
        else:  # prefix == "I" and same etype
            cur_end = end

    _flush()
    return spans

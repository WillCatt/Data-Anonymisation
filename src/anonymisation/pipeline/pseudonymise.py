"""
Pseudonymisation — referential redaction.

Instead of replacing every PERSON span with a generic `[PERSON]` (which
collapses every named individual into the same opaque tag), we assign
each *distinct* surface form a stable token: `[PERSON_A]`, `[PERSON_B]`,
`[PERSON_C]` and so on. The same surface form gets the same token every
time it appears, so a downstream LLM can reason coherently about
"what did Person A do" vs "what did Person B do" without ever seeing
the actual names.

Why this matters for the law-firm use case:
    Plain redaction throws away referential identity. After running
    Lite over a contract dispute between Alice and Bob, the redacted
    text reads "[PERSON] sued [PERSON] over [PERSON]'s breach". An
    LLM cannot answer "who sued whom?" from that. Pseudonymisation
    preserves the structure: "[PERSON_A] sued [PERSON_B] over
    [PERSON_B]'s breach" — and crucially, the firm holds the
    `[PERSON_A] → "Alice Smith"` mapping in a vault, so when the
    LLM's answer comes back the firm can swap the tokens for real
    names locally.

This module provides:
  * `Pseudonymiser` — assigns stable tokens to surface forms with a
    simple coreference heuristic (exact match plus word-boundary
    substring rule). Accumulates a `vault` mapping `token → original`.
  * `restore` — given redacted text and a vault, swap tokens back to
    originals. The "round-trip" that makes the pattern useful.

Coreference is deliberately lightweight — string-based, not model-based.
'Maria Petrova' / 'Mrs Petrova' / 'Petrova' all collapse to the same
token in a typical legal document because each shorter form is
contained as a token-aligned substring of the longer one. Edge cases
where this misfires (two different "Smith"s in the same document, for
example) are documented in the walkthrough notebook.

Link evidence and confidence
----------------------------
Every merge is a claim that two surface forms denote the same entity, and
that claim can be wrong in a way plain redaction cannot be: a false merge
gives two people one pseudonym, so `restore()` writes the *wrong* real name
back into the LLM's answer. It is a silent integrity failure, not a leak.

So a merge is no longer a bare yes/no. `classify_link_evidence` names the
rule that fired — exact repeat, honorific-stripped equality, or one form
contained in the other, in either direction — and `LINK_CONFIDENCE` gives
that rule's measured precision for that entity type: the probability, on
TAB, that two forms linked by this evidence really are the same entity.
The table was fitted by `scripts/evaluate_coref_links.py` over the TAB
*train + validation* splits (44,388 link decisions scored against gold
`entity_id` clusters); the test split is held out and reported in
notebook 17. Two results drove the default policy:

  * an exact repeat of a surface form within one document is the same
    entity 35,183 times out of 35,183, and
  * the reverse-containment rule — an already-seen short form turning up
    inside a longer new one, `1989` then `June 1989` — is right about 5%
    of the time.

`Pseudonymiser` therefore takes a `link_policy`. Under `"calibrated"` (the
default) it scores every candidate and merges only on the best evidence
above `min_link_confidence`; under `"legacy"` it reproduces the original
behaviour — first match wins, any evidence, no threshold — which is what
notebook 17 measures against. Either way every call appends a
`LinkDecision` to `.decisions`, so the audit log can say *why* two
mentions share a token and how much that link is worth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Token index → letter sequence (A..Z, AA..ZZ, ...)
# ---------------------------------------------------------------------------
def index_to_letters(idx: int) -> str:
    """0→A, 1→B, …, 25→Z, 26→AA, 27→AB, …, 51→AZ, 52→BA, …"""
    if idx < 0:
        raise ValueError("index must be non-negative")
    out = ""
    n = idx
    while True:
        out = chr(ord("A") + (n % 26)) + out
        n = n // 26 - 1
        if n < 0:
            return out


# ---------------------------------------------------------------------------
# Link evidence — which rule fired, and what that rule is worth
# ---------------------------------------------------------------------------
# The tiers are ordered strongest-first. They are structural: naming them
# needs no data. What the data supplies is LINK_CONFIDENCE below.
#
#   exact                 "Petrova"        then "Petrova"
#   exact_casefold        "The Board"      then "the Board"
#   honorific_only        "Mr Nihat Osal"  then "Nihat Osal"
#   short_in_long_1tok    "Maria Petrova"  then "Petrova"
#   short_in_long_multi   "Sofia District Court" then "Sofia District"
#   long_over_short_1tok  "1989"           then "June 1989"
#   long_over_short_multi "United Kingdom" then "United Kingdom Government"
#
# The two directions are kept apart because they are not symmetric. A new
# form contained in one already seen is the ordinary legal-document
# shorthand — introduce the party in full, then use the surname. The
# reverse, an old short form swallowed by a longer new one, is usually two
# different entities that happen to share a token.
NEW_ENTITY = "new_entity"

LINK_EVIDENCE_TIERS = (
    "exact",
    "exact_casefold",
    "honorific_only",
    "short_in_long_1tok",
    "short_in_long_multi",
    "long_over_short_1tok",
    "long_over_short_multi",
)


def classify_link_evidence(new_form: str, existing_form: str) -> Optional[str]:
    """
    Name the rule under which `new_form` would corefer with `existing_form`.

    Returns None when no rule fires. Direction matters: the arguments are
    (candidate mention, mention already in the vault), in that order.
    """
    new, existing = new_form.strip(), existing_form.strip()
    if not new or not existing:
        return None
    if new == existing:
        return "exact"
    if new.lower() == existing.lower():
        return "exact_casefold"
    new_clean = _strip_honorifics(new).lower()
    existing_clean = _strip_honorifics(existing).lower()
    if new_clean == existing_clean:
        return "honorific_only"
    if _is_word_substring(new_clean, existing_clean):
        return "short_in_long_1tok" if len(new_clean.split()) == 1 else "short_in_long_multi"
    if _is_word_substring(existing_clean, new_clean):
        return "long_over_short_1tok" if len(existing_clean.split()) == 1 else "long_over_short_multi"
    return None


# Measured precision of each (evidence, entity_type) pair: of the links this
# rule made on TAB train+validation, the fraction where the two mentions
# shared a gold `entity_id`. Cells with few observations are shrunk toward
# the tier's own rate (k=20), so a rule seen five times cannot claim 1.0.
# Regenerate with: python scripts/evaluate_coref_links.py --fit
LINK_CONFIDENCE: Dict[Tuple[str, str], float] = {
    ("exact", "ORG"): 1.000, ("exact", "PERSON"): 1.000,
    ("exact", "DATETIME"): 1.000, ("exact", "LOC"): 1.000,
    ("exact", "DEM"): 1.000, ("exact", "MISC"): 1.000,
    ("exact", "CODE"): 1.000, ("exact", "QUANTITY"): 1.000,

    ("exact_casefold", "ORG"): 0.015, ("exact_casefold", "MISC"): 0.007,
    ("exact_casefold", "DEM"): 0.009, ("exact_casefold", "PERSON"): 0.045,
    ("exact_casefold", "LOC"): 0.013, ("exact_casefold", "DATETIME"): 0.014,
    ("exact_casefold", "QUANTITY"): 0.015,

    ("honorific_only", "PERSON"): 0.665,

    ("long_over_short_1tok", "DATETIME"): 0.000, ("long_over_short_1tok", "ORG"): 0.095,
    ("long_over_short_1tok", "PERSON"): 0.222, ("long_over_short_1tok", "LOC"): 0.017,
    ("long_over_short_1tok", "MISC"): 0.008, ("long_over_short_1tok", "DEM"): 0.036,
    ("long_over_short_1tok", "CODE"): 0.063, ("long_over_short_1tok", "QUANTITY"): 0.028,

    ("long_over_short_multi", "ORG"): 0.124, ("long_over_short_multi", "DATETIME"): 0.003,
    ("long_over_short_multi", "MISC"): 0.139, ("long_over_short_multi", "PERSON"): 0.203,
    ("long_over_short_multi", "LOC"): 0.045, ("long_over_short_multi", "QUANTITY"): 0.046,
    ("long_over_short_multi", "DEM"): 0.051,

    ("short_in_long_1tok", "PERSON"): 0.635, ("short_in_long_1tok", "ORG"): 0.625,
    ("short_in_long_1tok", "DATETIME"): 0.018, ("short_in_long_1tok", "LOC"): 0.175,
    ("short_in_long_1tok", "MISC"): 0.400, ("short_in_long_1tok", "DEM"): 0.210,
    ("short_in_long_1tok", "QUANTITY"): 0.293,

    ("short_in_long_multi", "ORG"): 0.339, ("short_in_long_multi", "DATETIME"): 0.014,
    ("short_in_long_multi", "PERSON"): 0.440, ("short_in_long_multi", "MISC"): 0.244,
    ("short_in_long_multi", "DEM"): 0.188, ("short_in_long_multi", "LOC"): 0.264,
    ("short_in_long_multi", "QUANTITY"): 0.202, ("short_in_long_multi", "CODE"): 0.360,
}

# Fallback for an (evidence, entity_type) pair TAB never produced — the
# tier's rate across all types. An unseen pair is not evidence of safety.
LINK_CONFIDENCE_PRIOR: Dict[str, float] = {
    "exact": 1.000,
    "exact_casefold": 0.016,
    "honorific_only": 0.665,
    "long_over_short_1tok": 0.032,
    "long_over_short_multi": 0.076,
    "short_in_long_1tok": 0.468,
    "short_in_long_multi": 0.264,
}

# Default bar for a merge. Set where it is: the two rules that carry real
# legal-document shorthand (a surname or an honorific-stripped name landing
# inside a fuller PERSON/ORG mention) clear it, and nothing else does.
DEFAULT_MIN_LINK_CONFIDENCE = 0.5


def link_confidence(evidence: str, entity_type: str) -> float:
    """Measured probability that a link made on this evidence is correct."""
    if evidence == NEW_ENTITY:
        raise ValueError("minting a new token is not a link; it has no confidence")
    if (evidence, entity_type) in LINK_CONFIDENCE:
        return LINK_CONFIDENCE[(evidence, entity_type)]
    return LINK_CONFIDENCE_PRIOR.get(evidence, 0.0)


@dataclass(frozen=True)
class LinkDecision:
    """One resolution of a surface form to a token — merged or newly minted."""
    entity_type: str
    surface_form: str
    token: str
    evidence: str                      # a LINK_EVIDENCE_TIERS member, or NEW_ENTITY
    # Confidence in the link that was made. None when none was — minting a new
    # token is not a measured claim, and saying 1.0 there would assert one.
    confidence: Optional[float]
    matched_form: Optional[str] = None  # the vault form it was linked to
    rejected: Tuple[Tuple[str, str, float], ...] = ()  # (form, evidence, conf) below the bar
    # How many *distinct* entities matched this form at the winning confidence.
    # More than one means the evidence does not pick between them — "Smith"
    # after both "John Smith" and "Jane Smith" — and the tie was broken by
    # nothing more principled than which was seen first.
    tied_candidates: int = 1
    # Why a candidate that matched was not merged: "below_threshold" when the
    # best evidence was too weak, "ambiguous" when several entities matched
    # equally well. None when nothing was refused.
    refusal: Optional[str] = None

    @property
    def merged(self) -> bool:
        return self.evidence != NEW_ENTITY

    @property
    def ambiguous(self) -> bool:
        """True when two or more entities matched equally well."""
        return self.tied_candidates > 1

    def describe(self) -> str:
        """One line for the audit log."""
        if not self.merged:
            if self.refusal == "ambiguous":
                forms = ", ".join(sorted({r[0] for r in self.rejected}))
                return (
                    f"{self.surface_form!r} → {self.token} (new entity); {self.tied_candidates} "
                    f"entities matched equally well ({forms}), so the evidence picks none of them"
                )
            if self.rejected:
                best = max(self.rejected, key=lambda r: r[2])
                return (
                    f"{self.surface_form!r} → {self.token} (new entity); closest candidate "
                    f"{best[0]!r} linked only by {best[1]} (confidence {best[2]:.2f}, below "
                    f"{DEFAULT_MIN_LINK_CONFIDENCE:.2f}) so it was kept separate"
                )
            return f"{self.surface_form!r} → {self.token} (new entity; no candidate matched)"
        line = (
            f"{self.surface_form!r} → {self.token}, same entity as {self.matched_form!r} "
            f"by {self.evidence} (confidence {self.confidence:.2f})"
        )
        if self.ambiguous:
            line += (
                f"; ambiguous — {self.tied_candidates} entities matched equally well, "
                f"resolved to the first seen"
            )
        return line


# ---------------------------------------------------------------------------
# Pseudonymiser
# ---------------------------------------------------------------------------
@dataclass
class Pseudonymiser:
    """
    Assigns stable [TYPE_A], [TYPE_B], … tokens to entity surface forms.

    Per-document: each instance is independent. Cross-document tokenisation
    is a separate concern (would need a persistent registry); see the
    Phase 4 README for the design notes.

    Use:
        ps = Pseudonymiser()
        ps.token_for("PERSON", "Maria Petrova")   # -> "[PERSON_A]"
        ps.token_for("PERSON", "Maria Petrova")   # -> "[PERSON_A]"   (same)
        ps.token_for("PERSON", "Maria")           # -> "[PERSON_A]"   (substring match)
        ps.token_for("PERSON", "John Doe")        # -> "[PERSON_B]"
        ps.vault                                   # -> {"[PERSON_A]": "Maria Petrova", "[PERSON_B]": "John Doe"}
    """
    # token -> canonical (longest-seen) surface form for that entity
    _vault: Dict[str, str] = field(default_factory=dict)
    # (entity_type, normalised surface form) -> token. Keyed by type so the
    # same surface form (e.g. "Maria") used for two different entity types
    # gets two different tokens.
    _by_form: Dict[Tuple[str, str], str] = field(default_factory=dict)
    # entity_type -> next index
    _next: Dict[str, int] = field(default_factory=dict)

    # "calibrated" scores every candidate and merges only on the best
    # evidence clearing `min_link_confidence`; "legacy" takes the first
    # match in insertion order on any evidence at all, which is what the
    # pipeline did before the links were measured.
    link_policy: str = "calibrated"
    min_link_confidence: float = DEFAULT_MIN_LINK_CONFIDENCE
    # Refuse a merge when two or more entities match equally well — "Mr
    # Ravnsborg" after both "Mr Göran Ravnsborg" and a second Ravnsborg. Such
    # links are right 23% / 25% / 46% of the time on train / validation /
    # test against 97% for unambiguous ones, so the tie is not evidence; it is
    # a coin flip that the first-seen candidate happens to win. It fires on
    # 0.4% of merges. Notebook 17 measures what it costs.
    reject_ambiguous: bool = True
    # Every resolution, in order — merged or minted. The audit trail for
    # "why do these two mentions share a pseudonym?".
    decisions: List[LinkDecision] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.link_policy not in ("calibrated", "legacy"):
            raise ValueError(
                f"link_policy must be 'calibrated' or 'legacy', got {self.link_policy!r}"
            )

    # ------------------------------------------------------------------ #
    @property
    def vault(self) -> Dict[str, str]:
        """The token → original-text mapping. Suitable for JSON serialisation."""
        return dict(self._vault)

    # ------------------------------------------------------------------ #
    def token_for(self, entity_type: str, surface_form: str) -> str:
        """
        Return a stable token for `surface_form`. If we've already assigned
        one (either to this exact form or to a coreferential one), reuse it;
        otherwise mint a new one.
        """
        norm = surface_form.strip()
        if not norm:
            raise ValueError("surface_form must be non-empty")

        key = (entity_type, norm)

        # Exact match (same type + same form) — fast path. On TAB this is
        # right 35,183 times out of 35,183, so it needs no scoring.
        if key in self._by_form:
            existing_token = self._by_form[key]
            self._maybe_promote_canonical(existing_token, norm)
            self.decisions.append(LinkDecision(
                entity_type=entity_type, surface_form=norm, token=existing_token,
                evidence="exact", confidence=link_confidence("exact", entity_type),
                matched_form=norm,
            ))
            return existing_token

        # Coreference match against previously-seen forms of the same type
        (match, evidence, confidence, matched_form, rejected, tied,
         refusal) = self._find_coreferential(entity_type, norm)
        if match is not None:
            self._by_form[key] = match
            self._maybe_promote_canonical(match, norm)
            self.decisions.append(LinkDecision(
                entity_type=entity_type, surface_form=norm, token=match,
                evidence=evidence, confidence=confidence,
                matched_form=matched_form, rejected=rejected, tied_candidates=tied,
            ))
            return match

        # New entity — mint the next token. Under the calibrated policy this
        # branch is also where a *rejected* link lands: the safe default when
        # the evidence is weak is two tokens, not one.
        idx = self._next.get(entity_type, 0)
        self._next[entity_type] = idx + 1
        token = f"[{entity_type}_{index_to_letters(idx)}]"
        self._vault[token] = norm
        self._by_form[key] = token
        self.decisions.append(LinkDecision(
            entity_type=entity_type, surface_form=norm, token=token,
            evidence=NEW_ENTITY, confidence=None, matched_form=None, rejected=rejected,
            tied_candidates=tied, refusal=refusal,
        ))
        return token

    # ------------------------------------------------------------------ #
    def _find_coreferential(self, entity_type: str, new_form: str):
        """
        Look for a previously-seen form of the same type that refers to the
        same entity.

        Returns `(token, evidence, confidence, matched_form, rejected, tied, refusal)`.
        `token` is None in three cases: nothing matched; the best candidate
        fell below `min_link_confidence`; or several entities matched equally
        well and `reject_ambiguous` is set. The last two set `refusal` and fill
        `rejected`, so the audit log can show the near miss rather than
        silently minting a token.

        Honorifics are stripped on both sides before comparison, so "Mrs
        Petrova" collapses to "Petrova" and can then word-substring-match
        "Maria Petrova" — the most common coreference pattern in legal text,
        and one the substring rule alone would not catch.
        """
        candidates = []
        for (existing_type, existing_form), existing_token in self._by_form.items():
            if existing_type != entity_type:
                continue
            evidence = classify_link_evidence(new_form, existing_form)
            if evidence is None:
                continue
            candidates.append(
                (existing_token, evidence, link_confidence(evidence, entity_type), existing_form)
            )

        if not candidates:
            return None, NEW_ENTITY, None, None, (), 1, None

        if self.link_policy == "legacy":
            # First match in insertion order wins, whatever the evidence. The
            # confidence is still reported; it just isn't enforced.
            token, evidence, confidence, matched_form = candidates[0]
            tied = len({c[0] for c in candidates if c[1] == evidence})
            return token, evidence, confidence, matched_form, (), tied, None

        # Calibrated: strongest evidence wins, ties broken by insertion order
        # (`max` is stable), and weak evidence is not evidence.
        token, evidence, confidence, matched_form = max(candidates, key=lambda c: c[2])
        tied = len({c[0] for c in candidates if c[2] == confidence})
        if confidence < self.min_link_confidence:
            rejected = tuple((form, ev, conf) for _, ev, conf, form in candidates)
            return None, NEW_ENTITY, None, None, rejected, tied, "below_threshold"
        if self.reject_ambiguous and tied > 1:
            rejected = tuple(
                (form, ev, conf) for _, ev, conf, form in candidates if conf == confidence
            )
            return None, NEW_ENTITY, None, None, rejected, tied, "ambiguous"
        return token, evidence, confidence, matched_form, (), tied, None

    # ------------------------------------------------------------------ #
    def _maybe_promote_canonical(self, token: str, surface_form: str) -> None:
        """
        Keep the *longest* surface form we've seen as the canonical entry in the
        vault. So if we see "Maria" first then "Maria Petrova", the vault should
        end up with the longer form (the better label for restore output).
        """
        current = self._vault.get(token, "")
        if len(surface_form) > len(current):
            self._vault[token] = surface_form


# ---------------------------------------------------------------------------
# Substring rule used for the coreference heuristic
# ---------------------------------------------------------------------------
# Honorifics stripped before substring comparison — this is what lets
# 'Mrs Petrova' coref with 'Maria Petrova' via the surname, which is the
# most common pattern in legal documents (formal introduction, then short
# references throughout).
_HONORIFICS = {
    "mr", "mr.", "mrs", "mrs.", "ms", "ms.", "miss",
    "dr", "dr.", "prof", "prof.",
    "lord", "lady", "sir", "dame",
    "rev", "rev.", "fr", "fr.",
}


def _strip_honorifics(surface_form: str) -> str:
    """Drop leading honorific tokens. 'Mrs Petrova' -> 'Petrova'."""
    parts = surface_form.split()
    while parts and parts[0].lower().rstrip(".,") in _HONORIFICS:
        parts.pop(0)
    return " ".join(parts) if parts else surface_form


def _is_word_substring(needle: str, haystack: str) -> bool:
    """
    True iff `needle` appears in `haystack` aligned to word boundaries.
    Both inputs are expected to be already lowercased / trimmed.
    """
    if needle == haystack:
        return True
    if not needle or not haystack:
        return False
    # Word-boundary match — protects against e.g. "ko" matching "kosovo"
    pattern = r"\b" + re.escape(needle) + r"\b"
    return re.search(pattern, haystack) is not None


# ---------------------------------------------------------------------------
# Restore — vault round-trip
# ---------------------------------------------------------------------------
TOKEN_RE = re.compile(r"\[[A-Z]+_[A-Z]+\]")


def restore(redacted_text: str, vault: Dict[str, str]) -> str:
    """
    Replace every [TYPE_X] token in `redacted_text` with the corresponding
    original surface form from `vault`.

    Tokens not present in the vault are left untouched (the "[PERSON]" plain
    redactions from non-pseudonymised pipelines won't match the regex
    above, so they pass through; pseudonymised tokens that somehow have no
    vault entry print as-is so the user notices the gap).
    """
    def _swap(match: re.Match) -> str:
        token = match.group(0)
        return vault.get(token, token)
    return TOKEN_RE.sub(_swap, redacted_text)

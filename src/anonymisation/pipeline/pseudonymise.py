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
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


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

        # Exact match (same type + same form) — fast path
        if key in self._by_form:
            existing_token = self._by_form[key]
            self._maybe_promote_canonical(existing_token, norm)
            return existing_token

        # Coreference match against previously-seen forms of the same type
        match = self._find_coreferential(entity_type, norm)
        if match is not None:
            self._by_form[key] = match
            self._maybe_promote_canonical(match, norm)
            return match

        # New entity — mint the next token
        idx = self._next.get(entity_type, 0)
        self._next[entity_type] = idx + 1
        token = f"[{entity_type}_{index_to_letters(idx)}]"
        self._vault[token] = norm
        self._by_form[key] = token
        return token

    # ------------------------------------------------------------------ #
    def _find_coreferential(self, entity_type: str, new_form: str) -> Optional[str]:
        """Return the token of any previously-seen form that refers to the same entity."""
        # Normalise both sides by stripping honorifics — so "Mrs Petrova"
        # collapses to "Petrova", which can then word-substring-match
        # "Maria Petrova". This is the most common coreference pattern
        # in legal text and the substring rule alone wouldn't catch it.
        new_clean = _strip_honorifics(new_form).lower()
        for (existing_type, existing_form), existing_token in self._by_form.items():
            if existing_type != entity_type:
                continue
            existing_clean = _strip_honorifics(existing_form).lower()
            if (
                _is_word_substring(new_clean, existing_clean)
                or _is_word_substring(existing_clean, new_clean)
            ):
                return existing_token
        return None

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

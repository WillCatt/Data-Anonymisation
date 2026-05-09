"""
DIRECT vs QUASI classification.

NER models return entity-type spans without telling us whether each one
is a DIRECT identifier (full name, case file number — must be removed)
or a QUASI identifier (date, location, demographic descriptor — only
identifying in combination).

TAB itself was annotated by humans for this. In production we don't have
that luxury — the pipeline has to make a defensible default per entity
type, with the option for the caller to override.

Defaults below are conservative: when in doubt, treat as DIRECT (over-
redacts but never under-redacts). Pipeline Pro can downgrade selected
DIRECTs to QUASI via a callback if the caller wants finer-grained behaviour.
"""
from __future__ import annotations

from typing import Callable, Optional

from .types import IdentifierRole, Span


# Entity type → conservative default identifier role.
# Rationale: for a *redaction* pipeline, false-positive DIRECT classifications
# (over-redaction) are safer than false-negative ones (under-redaction).
DEFAULT_ROLE_BY_TYPE: dict = {
    "PERSON":   "DIRECT",   # full names — always direct
    "ORG":      "DIRECT",   # organisation names — usually identifying
    "CODE":     "DIRECT",   # case numbers, IDs — pure direct identifiers
    "DATETIME": "QUASI",    # dates — almost always quasi
    "LOC":      "QUASI",    # locations — quasi unless very specific
    "DEM":      "QUASI",    # demographics — classic quasi-identifiers
    "QUANTITY": "QUASI",    # ages and the like
    "MISC":     "DIRECT",   # uncategorized — over-redact to be safe
}


RoleOverride = Callable[[Span], Optional[IdentifierRole]]


def classify_role(
    span: Span,
    override: Optional[RoleOverride] = None,
) -> IdentifierRole:
    """
    Decide DIRECT vs QUASI for a span.

    Resolution order:
      1. Caller-supplied override (if it returns non-None)
      2. The span's existing `identifier_role` if it isn't the default DIRECT
         (so role information that came from upstream — e.g. a regex match —
         is respected)
      3. The default for this entity type
    """
    if override is not None:
        result = override(span)
        if result is not None:
            return result
    # If the span was constructed with a non-default role, trust it.
    # We can't tell "explicit DIRECT" from "default DIRECT" via the dataclass
    # alone, so this is best-effort — overrides are the right place for
    # caller-controlled behaviour.
    return DEFAULT_ROLE_BY_TYPE.get(span.entity_type, "DIRECT")

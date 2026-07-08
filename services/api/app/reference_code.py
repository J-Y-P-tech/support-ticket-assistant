"""Reference-code helper: format and normalize ``TKT-####`` codes (plan Task 5).

A ticket's reference code is what links customer -> ticket -> reply (SPEC §14);
its format is a zero-padded sequence, ``TKT-####`` (SPEC §7 confirmation 3). This
module keeps that one rule in a single pure, unit-tested place (SPEC §10) so route
and store logic stay thin:

- ``format_reference_code`` turns a sequence number into the canonical code.
- ``normalize_reference_code`` cleans a customer-typed code for lookup, case- and
  whitespace-insensitively, and returns ``None`` when the text is not even shaped
  like a code — a not-found sentinel that lets callers skip the database entirely
  (matching the codebase's None-means-not-found convention, e.g. ``get_ticket``).

Task 5 delivers the helper alone; the customer route and email store keep their
inline normalization until a later task adopts this.
"""

from __future__ import annotations

import re

# Prefix + minimum digit width of the canonical code, kept together so the format
# and the validation pattern can never drift apart.
_PREFIX = "TKT-"
_MIN_DIGITS = 4

# A well-shaped code after cleanup: the prefix followed by one or more digits.
# Leading zeros are allowed because customers type the padded form back to us.
_CODE_RE = re.compile(rf"^{re.escape(_PREFIX)}\d+$")


def format_reference_code(sequence: int) -> str:
    """Format a sequence number into the canonical ``TKT-####`` code.

    Deterministic and zero-padded to at least four digits; larger numbers keep
    every digit rather than truncating (e.g. ``10000`` -> ``TKT-10000``). The
    sequence starts at 1, so a zero or negative value is a programmer error and
    raises ``ValueError`` rather than emitting a nonsensical code.
    """
    if sequence < 1:
        raise ValueError(f"reference-code sequence must be >= 1, got {sequence}")
    return f"{_PREFIX}{sequence:0{_MIN_DIGITS}d}"


def normalize_reference_code(raw: str) -> str | None:
    """Normalize a customer-typed reference code for lookup, or return ``None``.

    Trims surrounding whitespace and uppercases so a lightly mistyped code
    (``  tkt-0007 ``) still resolves. Input that is not shaped like a reference
    code at all (wrong/missing prefix, non-digit body, empty) returns ``None`` —
    a not-found sentinel that spares the database a doomed query. A well-shaped
    but unknown code (``TKT-9999``) is returned unchanged: deciding whether it
    exists is the store's job, not this helper's.
    """
    candidate = raw.strip().upper()
    if _CODE_RE.match(candidate):
        return candidate
    return None

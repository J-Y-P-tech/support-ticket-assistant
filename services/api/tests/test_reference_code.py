"""Unit tests for the reference-code helper (plan Task 5, SPEC §10).

The helper is a pure, side-effect-free util shared across services: it formats a
sequence number into the canonical ``TKT-####`` code (SPEC §14 / §7 confirmation
3) and normalizes a customer-typed code for lookup. Normalization is
case/whitespace-insensitive; input that is not even shaped like a reference code
returns a not-found sentinel (``None``) so junk never reaches the database.

No wiring is exercised here — Task 5 delivers the helper alone; the customer
route and email store keep their inline copies until a later task adopts it.
"""

from __future__ import annotations

import pytest

from app.reference_code import format_reference_code, normalize_reference_code


def test_format_pads_to_four_digits() -> None:
    """A small sequence number is zero-padded to the canonical four-digit code."""
    assert format_reference_code(1) == "TKT-0001"
    assert format_reference_code(42) == "TKT-0042"


def test_format_grows_past_four_digits_without_truncation() -> None:
    """At the 9999 boundary the code keeps every digit rather than truncating."""
    assert format_reference_code(9999) == "TKT-9999"
    assert format_reference_code(10000) == "TKT-10000"


def test_format_rejects_non_positive_sequence() -> None:
    """Sequence values start at 1, so zero or negative input is a programmer error."""
    with pytest.raises(ValueError):
        format_reference_code(0)
    with pytest.raises(ValueError):
        format_reference_code(-1)


def test_normalize_is_case_and_whitespace_insensitive() -> None:
    """A lightly mistyped code (lowercase, surrounded by spaces) canonicalizes."""
    assert normalize_reference_code("  tkt-0007 ") == "TKT-0007"
    assert normalize_reference_code("TKT-0007") == "TKT-0007"


def test_normalize_keeps_well_shaped_but_unknown_code() -> None:
    """A properly shaped code is returned as-is; deciding it exists is the DB's job."""
    assert normalize_reference_code("TKT-9999") == "TKT-9999"


def test_normalize_returns_sentinel_for_non_code_shaped_input() -> None:
    """Input that cannot be a reference code yields None, short-circuiting lookup."""
    assert normalize_reference_code("hello") is None
    assert normalize_reference_code("TKT-abcd") is None
    assert normalize_reference_code("0007") is None  # missing the TKT- prefix
    assert normalize_reference_code("TKT-") is None  # no digits
    assert normalize_reference_code("") is None


def test_formatted_code_round_trips_through_normalize() -> None:
    """Every code the formatter emits survives normalization unchanged."""
    for seq in (1, 42, 9999, 10000):
        code = format_reference_code(seq)
        assert normalize_reference_code(code) == code

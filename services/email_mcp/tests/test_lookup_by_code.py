"""Contract tests for reference-code lookup (plan Task 4, supports SPEC §4.8).

The api's customer-lookup route resolves a `TKT-####` code to a ticket, but a
reference code is *not* the ticket id (it comes from its own sequence), so
email_mcp — the sole DB owner — must offer a code lookup. These pin its
behaviour: an exact match loads the ticket (in the same shape as `get_ticket`),
a case/whitespace-messy code still matches, and an unknown code returns a neutral
None rather than leaking whether the code exists.
"""

from __future__ import annotations

from psycopg import Connection

import db
from app.schemas.ticket import TicketRead


def test_lookup_by_code_returns_ticket(conn: Connection) -> None:
    """A known reference code loads the ticket in the customer-facing shape."""
    created = db.create_ticket(conn, message="please help")

    loaded = db.get_ticket_by_code(conn, created["reference_code"])

    assert loaded is not None
    assert TicketRead.model_validate(loaded).reference_code == created["reference_code"]


def test_lookup_by_code_is_case_and_whitespace_insensitive(conn: Connection) -> None:
    """A messy code (lowercase, surrounding whitespace) still resolves the ticket."""
    created = db.create_ticket(conn, message="please help")
    code = created["reference_code"]  # e.g. "TKT-0001"

    loaded = db.get_ticket_by_code(conn, f"  {code.lower()} ")

    assert loaded is not None
    assert loaded["reference_code"] == code


def test_lookup_by_code_unknown_returns_neutral_not_found(conn: Connection) -> None:
    """An unknown reference code yields a neutral None, never an error."""
    assert db.get_ticket_by_code(conn, "TKT-9999") is None

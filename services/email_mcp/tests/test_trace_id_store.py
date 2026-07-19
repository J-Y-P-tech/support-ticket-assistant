"""Contract tests for the ticket trace-id store (plan Task 29 / todo Task 31).

SPEC §7.2 nests a ticket's Langfuse trace under a trace id **stored on the ticket
row**. email_mcp is the only writer of the `tickets` table (SPEC §6), so it owns that
column: `set_trace_id` records the id the api gets back from Langfuse, and `get_ticket`
reads it back. These run against a real throwaway Postgres so the new column behaves as
in production.
"""

from __future__ import annotations

from psycopg import Connection

import db


def test_new_ticket_has_no_trace_id(conn: Connection) -> None:
    """A freshly created ticket carries a NULL trace id until the pipeline traces it."""
    ticket = db.create_ticket(conn, message="please help")

    stored = db.get_ticket(conn, ticket["id"])
    assert stored is not None
    assert stored["trace_id"] is None


def test_set_trace_id_persists_and_reads_back(conn: Connection) -> None:
    """`set_trace_id` records the Langfuse trace id and `get_ticket` reads it back."""
    ticket = db.create_ticket(conn, message="please help")

    row = db.set_trace_id(conn, ticket_id=ticket["id"], trace_id="trace-abc123")

    assert row is not None
    assert row["trace_id"] == "trace-abc123"
    stored = db.get_ticket(conn, ticket["id"])
    assert stored is not None
    assert stored["trace_id"] == "trace-abc123"


def test_set_trace_id_unknown_ticket_returns_none(conn: Connection) -> None:
    """Setting a trace id on an unknown ticket returns a neutral None (no row touched)."""
    assert db.set_trace_id(conn, ticket_id=999_999, trace_id="trace-x") is None

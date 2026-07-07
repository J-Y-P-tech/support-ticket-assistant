"""Contract tests for the create / fetch / get / draft ticket tools (plan Task 3).

Acceptance criterion: "Each tool returns schema-valid output for representative
inputs." Each returned row is validated against the *shared* Pydantic contract
types (`app.schemas`) — the same types the api and frontend consume — so the DB
owner and the rest of the system are proven to agree on shape and enum values.
An unknown ticket id must return a neutral not-found (None), not leak an error.
"""

from __future__ import annotations

from psycopg import Connection

from app.schemas.draft import Draft
from app.schemas.enums import TicketStatus
from app.schemas.ticket import QueueRow, TicketRead

import db


def test_create_ticket_starts_new_with_zero_padded_code(conn: Connection) -> None:
    """A new ticket gets status New and a zero-padded `TKT-####` reference code."""
    ticket = db.create_ticket(conn, message="I can't log in to my account.")

    assert ticket["status"] == TicketStatus.NEW
    assert ticket["reference_code"] == "TKT-0001"
    assert ticket["message"] == "I can't log in to my account."


def test_reference_codes_are_sequential_and_padded(conn: Connection) -> None:
    """Consecutive tickets receive sequential, still zero-padded reference codes."""
    first = db.create_ticket(conn, message="first")
    second = db.create_ticket(conn, message="second")

    assert first["reference_code"] == "TKT-0001"
    assert second["reference_code"] == "TKT-0002"


def test_create_ticket_stores_attachments(conn: Connection) -> None:
    """Attachment references round-trip through creation and reload unchanged."""
    created = db.create_ticket(
        conn,
        message="See the attached statement.",
        attachments=["statement.pdf", "id-front.jpg"],
    )

    reloaded = db.get_ticket(conn, created["id"])

    assert reloaded is not None
    assert reloaded["attachments"] == ["statement.pdf", "id-front.jpg"]


def test_fetch_new_tickets_returns_queue_row_valid_rows(conn: Connection) -> None:
    """Fetched new tickets validate against the shared `QueueRow` contract type."""
    db.create_ticket(conn, message="please help")

    rows = db.fetch_new_tickets(conn)

    assert len(rows) == 1
    row = QueueRow.model_validate(rows[0])
    assert row.status is TicketStatus.NEW
    assert row.reference_code == "TKT-0001"
    # A New ticket has not been triaged yet, so it carries no urgency/category.
    assert row.urgency is None
    assert row.category is None


def test_fetch_new_tickets_excludes_non_new(conn: Connection) -> None:
    """Once a ticket leaves the New status it drops out of the new-ticket feed."""
    created = db.create_ticket(conn, message="please help")
    db.update_status(conn, ticket_id=created["id"], status=TicketStatus.TRIAGED)

    assert db.fetch_new_tickets(conn) == []


def test_get_ticket_returns_customer_and_queue_valid_view(conn: Connection) -> None:
    """A loaded ticket validates against both the customer and queue contracts."""
    created = db.create_ticket(conn, message="please help")

    loaded = db.get_ticket(conn, created["id"])

    assert loaded is not None
    # The same row satisfies the customer-facing and rep-queue shared schemas.
    assert TicketRead.model_validate(loaded).reference_code == "TKT-0001"
    assert QueueRow.model_validate(loaded).status is TicketStatus.NEW
    # No draft has been written yet.
    assert loaded["draft"] is None


def test_get_ticket_unknown_id_returns_neutral_not_found(conn: Connection) -> None:
    """An unknown ticket id yields a neutral None, never an error that leaks state."""
    assert db.get_ticket(conn, 999_999) is None


def test_save_draft_returns_draft_valid_dict(conn: Connection) -> None:
    """A saved draft validates against the shared `Draft` contract type."""
    ticket = db.create_ticket(conn, message="how do I reset my password?")

    saved = db.save_draft(
        conn,
        ticket_id=ticket["id"],
        body="You can reset it from the login screen.",
        citations=[{"source_id": "kb-1", "title": "Reset your password"}],
        unverified=False,
    )

    draft = Draft.model_validate(saved)
    assert draft.body == "You can reset it from the login screen."
    assert draft.citations[0].source_id == "kb-1"
    assert draft.unverified is False


def test_get_ticket_reflects_latest_saved_draft(conn: Connection) -> None:
    """`get_ticket` surfaces the most recently saved draft for the ticket."""
    ticket = db.create_ticket(conn, message="how do I reset my password?")
    db.save_draft(conn, ticket_id=ticket["id"], body="first attempt")
    db.save_draft(conn, ticket_id=ticket["id"], body="second, better attempt")

    loaded = db.get_ticket(conn, ticket["id"])

    assert loaded is not None
    assert loaded["draft"]["body"] == "second, better attempt"

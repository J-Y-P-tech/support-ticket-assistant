"""Contract tests for status transitions and the send/resolve path (plan Task 3).

These pin the core safety invariant of the whole system at its state root
(SPEC §4.7, §13): a case reaches `Resolved` *only* through an explicit rep send
action. `update_status` must refuse to set `Resolved`, and `record_sent_reply`
must refuse to resolve without a non-empty rep-action marker. Every mutation also
leaves an ordered audit trail (SPEC §7.1).
"""

from __future__ import annotations

import pytest
from psycopg import Connection

from app.schemas.enums import TicketStatus

import db


def test_update_status_transitions_and_records_audit(conn: Connection) -> None:
    """A permitted status change is persisted and leaves an audit entry."""
    ticket = db.create_ticket(conn, message="please help")

    updated = db.update_status(
        conn, ticket_id=ticket["id"], status=TicketStatus.TRIAGED, actor="system"
    )

    assert updated["status"] == TicketStatus.TRIAGED
    events = [entry["event"] for entry in db.get_audit_trail(conn, ticket["id"])]
    assert "status_changed" in events


def test_update_status_refuses_to_resolve(conn: Connection) -> None:
    """`update_status` cannot be a back door to Resolved — it must raise."""
    ticket = db.create_ticket(conn, message="please help")

    with pytest.raises(ValueError):
        db.update_status(conn, ticket_id=ticket["id"], status=TicketStatus.RESOLVED)

    # The refusal leaves the ticket untouched.
    reloaded = db.get_ticket(conn, ticket["id"])
    assert reloaded is not None
    assert reloaded["status"] == TicketStatus.NEW


def test_update_status_rejects_unknown_status(conn: Connection) -> None:
    """A status outside the lifecycle enum is rejected rather than stored."""
    ticket = db.create_ticket(conn, message="please help")

    with pytest.raises(ValueError):
        db.update_status(conn, ticket_id=ticket["id"], status="done")


def test_record_sent_reply_with_rep_marker_resolves(conn: Connection) -> None:
    """A rep send saves the reply, resolves the case, and audits the rep actor."""
    ticket = db.create_ticket(conn, message="please help")
    db.save_draft(conn, ticket_id=ticket["id"], body="Here is how to fix it.")

    resolved = db.record_sent_reply(
        conn,
        ticket_id=ticket["id"],
        reply="Here is how to fix it.",
        rep_id="rep-42",
    )

    assert resolved["status"] == TicketStatus.RESOLVED
    assert resolved["reply"] == "Here is how to fix it."
    trail = db.get_audit_trail(conn, ticket["id"])
    assert any(e["event"] == "reply_sent" and e["actor"] == "rep-42" for e in trail)


def test_record_sent_reply_reply_is_visible_on_lookup(conn: Connection) -> None:
    """After a send, the final reply is visible via a normal ticket load (§4.8)."""
    ticket = db.create_ticket(conn, message="please help")
    db.record_sent_reply(
        conn, ticket_id=ticket["id"], reply="All sorted for you.", rep_id="rep-7"
    )

    reloaded = db.get_ticket(conn, ticket["id"])
    assert reloaded is not None
    assert reloaded["reply"] == "All sorted for you."
    assert reloaded["status"] == TicketStatus.RESOLVED


def test_record_sent_reply_without_rep_marker_refuses(conn: Connection) -> None:
    """Without a rep-action marker there is no auto-resolve path — it must raise."""
    ticket = db.create_ticket(conn, message="please help")

    with pytest.raises(ValueError):
        db.record_sent_reply(
            conn, ticket_id=ticket["id"], reply="auto reply", rep_id=""
        )

    # The case must remain unresolved and unmodified.
    reloaded = db.get_ticket(conn, ticket["id"])
    assert reloaded is not None
    assert reloaded["status"] == TicketStatus.NEW
    assert reloaded["reply"] is None

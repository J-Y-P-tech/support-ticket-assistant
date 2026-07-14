"""Contract tests for the compliance audit trail (plan Task 24 / todo Task 25).

These pin the two guarantees SPEC §7.1 makes about the audit trail at its state
root: it is *append-only* — the api records each workflow-node outcome and rep
action through a committing `record_audit` entry point — and it is *immutable* —
a recorded row can never be altered or removed, enforced at the storage layer so
no bug or stray query can rewrite compliance history. Reads come back in insertion
order via `get_audit_trail`.

Like the other contract tests these run against a real throwaway Postgres (the
immutability guarantee is a Postgres trigger, not application code), so the whole
enforcement path is exercised, not mocked.
"""

from __future__ import annotations

import psycopg
import pytest
from psycopg import Connection

import db


def test_record_audit_appends_ordered_entries(conn: Connection) -> None:
    """`record_audit` appends entries a ticket's trail returns in insertion order.

    A freshly created ticket already carries its `ticket_created` submission row;
    each subsequent `record_audit` call appends after it, so the trail reads back
    in the order events happened — the ordered history a compliance review reads.
    """
    ticket = db.create_ticket(conn, message="please help")

    db.record_audit(conn, ticket_id=ticket["id"], event="triaged", actor="system")
    db.record_audit(conn, ticket_id=ticket["id"], event="drafted", actor="system")

    events = [entry["event"] for entry in db.get_audit_trail(conn, ticket["id"])]
    assert events == ["ticket_created", "triaged", "drafted"]


def test_record_audit_preserves_actor_and_detail(conn: Connection) -> None:
    """An audit entry round-trips its actor and its structured JSON detail intact.

    The detail carries the per-event evidence SPEC §7.1 requires (cited sources,
    model tag + prompt version, guardrail decision), so it must survive storage
    unchanged for the trail to document *on what basis* each step acted.
    """
    ticket = db.create_ticket(conn, message="please help")
    detail = {"category": "billing", "model": "gemma4:12b", "prompt_version": "triage-v1"}

    db.record_audit(conn, ticket_id=ticket["id"], event="triaged", actor="system", detail=detail)

    entry = db.get_audit_trail(conn, ticket["id"])[-1]
    assert entry["event"] == "triaged"
    assert entry["actor"] == "system"
    assert entry["detail"] == detail


def test_record_audit_commits_across_connections(_dsn: str, conn: Connection) -> None:
    """A recorded entry is durable — a separate connection sees it immediately.

    The api records one audit entry per node as an independent operation, so
    `record_audit` must commit on its own (unlike the internal helper the other
    tools batch into their own transaction). A second connection reading the same
    ticket proves the write is committed, not merely visible on the writer's
    uncommitted transaction.
    """
    ticket = db.create_ticket(conn, message="please help")
    db.record_audit(conn, ticket_id=ticket["id"], event="reply_sent", actor="rep-9")

    with psycopg.connect(_dsn) as other:
        events = [entry["event"] for entry in db.get_audit_trail(other, ticket["id"])]
    assert "reply_sent" in events


def test_audit_row_cannot_be_updated(conn: Connection) -> None:
    """An `UPDATE` against an audit row is rejected and leaves the row unchanged.

    Immutability is enforced at the storage layer (SPEC §7.1 compliance-grade), so
    even a direct SQL edit is refused rather than silently rewriting history.
    """
    ticket = db.create_ticket(conn, message="please help")

    with pytest.raises(psycopg.Error):
        conn.execute("UPDATE audit SET event = 'tampered' WHERE ticket_id = %s", (ticket["id"],))
    conn.rollback()  # clear the aborted transaction before reading back

    events = [entry["event"] for entry in db.get_audit_trail(conn, ticket["id"])]
    assert events == ["ticket_created"]


def test_audit_row_cannot_be_deleted(conn: Connection) -> None:
    """A `DELETE` against an audit row is rejected and leaves the row in place.

    An append-only trail must not lose an entry, so a delete is refused at the
    storage layer just as an update is; the submission row survives the attempt.
    """
    ticket = db.create_ticket(conn, message="please help")

    with pytest.raises(psycopg.Error):
        conn.execute("DELETE FROM audit WHERE ticket_id = %s", (ticket["id"],))
    conn.rollback()  # clear the aborted transaction before reading back

    events = [entry["event"] for entry in db.get_audit_trail(conn, ticket["id"])]
    assert events == ["ticket_created"]

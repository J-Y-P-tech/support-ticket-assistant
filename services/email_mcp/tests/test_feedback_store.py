"""Contract tests for feedback capture at the state root (plan Task 25 / todo Task 27).

email_mcp is the only writer of the `feedback` table (SPEC §6). `record_feedback`
persists one rep-decision row — the AI draft, the final reply, the edit distance, and
the optional rating/reason (SPEC §4.9) — and `get_feedback` reads a ticket's rows back
in insertion order for the training corpus (§4.9a) and quality loop (§7.4) to consume.
These run against a real throwaway Postgres so the JSONB/timestamp columns behave as
in production.
"""

from __future__ import annotations

from psycopg import Connection

import db


def test_record_feedback_persists_and_reads_back(conn: Connection) -> None:
    """An edited-decision feedback row stores every field and reads back intact."""
    ticket = db.create_ticket(conn, message="please help")

    row = db.record_feedback(
        conn,
        ticket_id=ticket["id"],
        decision="edited",
        ai_draft="reset your password",
        final_reply="please reset your password now",
        edit_distance=11,
        rating=4,
        reason="tightened the tone",
    )

    assert row["decision"] == "edited"
    stored = db.get_feedback(conn, ticket["id"])
    assert len(stored) == 1
    entry = stored[0]
    assert entry["ai_draft"] == "reset your password"
    assert entry["final_reply"] == "please reset your password now"
    assert entry["edit_distance"] == 11
    assert entry["rating"] == 4
    assert entry["reason"] == "tightened the tone"


def test_record_feedback_rejected_allows_null_final_and_distance(conn: Connection) -> None:
    """A rejected draft records the AI draft with no final reply, distance, or rating."""
    ticket = db.create_ticket(conn, message="please help")

    db.record_feedback(
        conn,
        ticket_id=ticket["id"],
        decision="rejected",
        ai_draft="the draft the rep threw away",
    )

    stored = db.get_feedback(conn, ticket["id"])
    assert len(stored) == 1
    entry = stored[0]
    assert entry["ai_draft"] == "the draft the rep threw away"
    assert entry["final_reply"] is None
    assert entry["edit_distance"] is None
    assert entry["rating"] is None
    assert entry["reason"] is None


def test_get_feedback_unknown_ticket_is_empty(conn: Connection) -> None:
    """A ticket with no feedback rows reads back as a neutral empty list."""
    assert db.get_feedback(conn, 999999) == []

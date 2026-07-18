"""Contract tests for the live few-shot lookup query (plan Task 28 / todo Task 30).

The drafting node injects the best recent **approved** replies for a ticket's category
as few-shot examples (SPEC §4.10). email_mcp owns the query that supplies the
candidates: `approved_replies_by_category` returns, for one category, the recent
approved replies — the customer `message`, the human-approved `reply`, the rep
`rating`, and an `example_id` recency key — newest first, so the deterministic
selector (todo Task 29) can rank them. These run against a real throwaway Postgres so
the JOIN to `tickets`, the `category` filter, and the ordering behave as in production.

"Approved" means a rep sent a reply: a feedback row whose decision is not `rejected`
and that carries a `final_reply`. A rejected draft (no reply) and a reply approved in
another category are both excluded.
"""

from __future__ import annotations

from typing import Any

from psycopg import Connection

import db


def _approve(
    conn: Connection,
    *,
    message: str,
    category: str,
    reply: str,
    rating: int | None = None,
    decision: str = "approved_as_is",
) -> int:
    """Create a ticket and record one approved (sent) feedback row for it.

    Mirrors the send path: a ticket is created, then its rep-approved reply is captured
    as a feedback row tagged with the ticket's triage `category`. Returns the ticket id.
    """
    ticket = db.create_ticket(conn, message=message)
    db.record_feedback(
        conn,
        ticket_id=ticket["id"],
        decision=decision,
        ai_draft=reply,
        final_reply=reply,
        rating=rating,
        category=category,
    )
    return int(ticket["id"])


def test_returns_approved_replies_for_category_newest_first(conn: Connection) -> None:
    """Approved replies for the category read back newest-first with all four fields."""
    _approve(conn, message="locked out", category="account_access", reply="reset it here", rating=4)
    _approve(
        conn,
        message="cannot log in",
        category="account_access",
        reply="use the reset link",
        rating=5,
    )

    rows = db.approved_replies_by_category(conn, category="account_access", limit=10)

    assert [row["reply"] for row in rows] == ["use the reset link", "reset it here"]
    newest = rows[0]
    assert newest["message"] == "cannot log in"
    assert newest["reply"] == "use the reset link"
    assert newest["rating"] == 5
    # example_id is the recency/stability key the selector ranks on (higher = newer).
    assert rows[0]["example_id"] > rows[1]["example_id"]


def test_excludes_rejected_and_replies_without_a_final(conn: Connection) -> None:
    """A rejected draft (no sent reply) never appears — only approved, sent replies do."""
    _approve(conn, message="locked out", category="account_access", reply="reset it here")
    rejected = db.create_ticket(conn, message="spam")
    db.record_feedback(
        conn,
        ticket_id=rejected["id"],
        decision="rejected",
        ai_draft="the draft the rep threw away",
        category="account_access",
    )

    rows = db.approved_replies_by_category(conn, category="account_access", limit=10)

    assert [row["reply"] for row in rows] == ["reset it here"]


def test_scopes_to_the_requested_category(conn: Connection) -> None:
    """A reply approved in another category is not returned for this one."""
    _approve(conn, message="locked out", category="account_access", reply="reset it here")
    _approve(conn, message="charged twice", category="payments_billing", reply="we refunded it")

    rows = db.approved_replies_by_category(conn, category="account_access", limit=10)

    assert [row["reply"] for row in rows] == ["reset it here"]


def test_respects_the_limit_keeping_the_newest(conn: Connection) -> None:
    """The candidate pool is capped at `limit`, keeping the most recent approved replies."""
    for i in range(5):
        _approve(conn, message=f"msg {i}", category="account_access", reply=f"reply {i}")

    rows = db.approved_replies_by_category(conn, category="account_access", limit=2)

    assert [row["reply"] for row in rows] == ["reply 4", "reply 3"]


def test_unknown_category_is_empty(conn: Connection) -> None:
    """A category with no approved replies reads back as a neutral empty list."""
    result: list[dict[str, Any]] = db.approved_replies_by_category(
        conn, category="loans_credit", limit=10
    )
    assert result == []

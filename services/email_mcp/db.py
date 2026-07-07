"""Data-access layer for email_mcp — the sole owner of the ticket tables.

Every function here takes an open psycopg3 `Connection` so the same code runs
against the compose Postgres in production and a throwaway `testcontainers`
Postgres in the contract tests. The MCP tools in `server.py` are thin wrappers
over these functions; keeping the logic here means it can be tested directly,
without standing up the MCP protocol (SPEC §10 TDD-first).

Return values are plain JSON-serialisable dicts — never Pydantic models — so the
service stays decoupled from the api package at runtime (the MCP boundary carries
JSON). The shapes deliberately match the shared `app.schemas` contract types,
which the contract tests validate against.

Safety invariant (SPEC §4.7, §13): a case reaches `Resolved` only through
`record_sent_reply` with a non-empty rep-action marker. `update_status` refuses
to set `Resolved`, so there is no automated back door to resolution.
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg
from psycopg import Connection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

# The lifecycle status a freshly-created ticket carries (SPEC §5).
_STATUS_NEW = "New"
# The only status that record_sent_reply may set; forbidden to update_status.
_STATUS_RESOLVED = "Resolved"
# The full closed set of lifecycle statuses (SPEC §5) — spelled exactly.
_VALID_STATUSES = frozenset(
    {
        "New",
        "Triaged",
        "Researching",
        "Drafted",
        "Pending",
        "Resolved",
        "Canceled",
        "NeedsResearch",
    }
)

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


# --------------------------------------------------------------------------- #
# Connection & migrations
# --------------------------------------------------------------------------- #
def connect_from_env() -> Connection:
    """Open a connection to the ticket database using environment configuration.

    email_mcp is the only component with these credentials (SPEC §6). Defaults
    target the docker-compose `postgres` service; every value is overridable.
    """
    dsn = (
        f"postgresql://{os.environ['POSTGRES_USER']}:{os.environ['POSTGRES_PASSWORD']}"
        f"@{os.environ.get('POSTGRES_HOST', 'postgres')}"
        f":{os.environ.get('POSTGRES_PORT', '5432')}"
        f"/{os.environ['POSTGRES_DB']}"
    )
    return psycopg.connect(dsn)


def apply_migrations(conn: Connection) -> None:
    """Apply every `migrations/*.sql` file in filename order, then commit.

    Migration SQL is idempotent (IF NOT EXISTS), so this is safe to re-run — it
    is the code path behind `make migrate`.
    """
    for path in sorted(_MIGRATIONS_DIR.glob("*.sql")):
        conn.execute(path.read_text())
    conn.commit()


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _iso(value: datetime | None) -> str | None:
    """Render a timestamp as an ISO-8601 string (or None) for JSON output."""
    return value.isoformat() if value is not None else None


def _record_audit(
    conn: Connection,
    ticket_id: int,
    event: str,
    *,
    actor: str | None = None,
    detail: dict[str, Any] | None = None,
) -> None:
    """Append one immutable audit row for a ticket mutation (SPEC §7.1)."""
    conn.execute(
        "INSERT INTO audit (ticket_id, event, actor, detail) VALUES (%s, %s, %s, %s)",
        (ticket_id, event, actor, Jsonb(detail) if detail is not None else None),
    )


# --------------------------------------------------------------------------- #
# Ticket tools
# --------------------------------------------------------------------------- #
def create_ticket(
    conn: Connection, *, message: str, attachments: list[str] | None = None
) -> dict[str, Any]:
    """Create a New ticket, assign its `TKT-####` code, and audit the submission.

    Not named in SPEC §2's illustrative tool list, but required: §4.1 creates a
    ticket on intake and §6 makes email_mcp the only writer of these tables.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO tickets (message, attachments) VALUES (%s, %s) "
            "RETURNING id, reference_code, status, message, attachments, created_at",
            (message, Jsonb(attachments or [])),
        )
        row = cur.fetchone()
    assert row is not None  # INSERT ... RETURNING always yields the new row.
    row["created_at"] = _iso(row["created_at"])
    _record_audit(conn, row["id"], "ticket_created", actor="customer")
    conn.commit()
    return row


def fetch_new_tickets(conn: Connection) -> list[dict[str, Any]]:
    """Return New (untriaged) tickets as `QueueRow`-shaped dicts, oldest first."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, reference_code, status, urgency, category "
            "FROM tickets WHERE status = %s ORDER BY created_at, id",
            (_STATUS_NEW,),
        )
        return cur.fetchall()


def get_ticket(conn: Connection, ticket_id: int) -> dict[str, Any] | None:
    """Load one ticket with its latest draft, or None if the id is unknown.

    Returning a neutral None (rather than raising) means an unknown id cannot be
    told apart from a known one via an error — no enumeration leak.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, reference_code, status, message, attachments, category, "
            "urgency, sentiment, reply, created_at, updated_at "
            "FROM tickets WHERE id = %s",
            (ticket_id,),
        )
        ticket = cur.fetchone()
        if ticket is None:
            return None

        cur.execute(
            "SELECT id, ticket_id, body, citations, unverified "
            "FROM drafts WHERE ticket_id = %s ORDER BY id DESC LIMIT 1",
            (ticket_id,),
        )
        draft = cur.fetchone()

    ticket["created_at"] = _iso(ticket["created_at"])
    ticket["updated_at"] = _iso(ticket["updated_at"])
    ticket["draft"] = draft
    return ticket


def save_draft(
    conn: Connection,
    *,
    ticket_id: int,
    body: str,
    citations: list[dict[str, Any]] | None = None,
    unverified: bool = False,
) -> dict[str, Any]:
    """Persist a reply draft for a ticket and return it (`Draft`-shaped)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO drafts (ticket_id, body, citations, unverified) "
            "VALUES (%s, %s, %s, %s) "
            "RETURNING id, ticket_id, body, citations, unverified",
            (ticket_id, body, Jsonb(citations or []), unverified),
        )
        row = cur.fetchone()
    assert row is not None
    _record_audit(conn, ticket_id, "draft_saved", actor="system")
    conn.commit()
    return row


def update_status(
    conn: Connection, *, ticket_id: int, status: str, actor: str | None = None
) -> dict[str, Any] | None:
    """Transition a ticket to a new lifecycle status and audit the change.

    Raises ValueError for an unknown status or for `Resolved`: resolution is
    reserved for `record_sent_reply` behind a rep-action marker (SPEC §4.7), so
    this tool can never be a back door to it. Nothing is written when it raises.
    """
    if status not in _VALID_STATUSES:
        raise ValueError(f"unknown ticket status: {status!r}")
    if status == _STATUS_RESOLVED:
        raise ValueError(
            "update_status cannot set Resolved; a case is resolved only by an "
            "explicit rep send via record_sent_reply"
        )

    conn.execute(
        "UPDATE tickets SET status = %s, updated_at = now() WHERE id = %s",
        (status, ticket_id),
    )
    _record_audit(
        conn, ticket_id, "status_changed", actor=actor, detail={"to": status}
    )
    conn.commit()
    return get_ticket(conn, ticket_id)


def record_sent_reply(
    conn: Connection, *, ticket_id: int, reply: str, rep_id: str
) -> dict[str, Any] | None:
    """Record a rep-sent reply: save it, resolve the case, and audit the rep.

    `rep_id` is the rep-action marker (SPEC §4.7): without a non-empty value the
    function raises before touching the database, so there is no auto-resolve
    path. Feedback/training rows are populated in Tasks 25/26.
    """
    if not rep_id or not rep_id.strip():
        raise ValueError("record_sent_reply requires a rep-action marker (rep_id)")

    conn.execute(
        "UPDATE tickets SET reply = %s, status = %s, updated_at = now() WHERE id = %s",
        (reply, _STATUS_RESOLVED, ticket_id),
    )
    _record_audit(conn, ticket_id, "reply_sent", actor=rep_id)
    conn.commit()
    return get_ticket(conn, ticket_id)


def get_audit_trail(conn: Connection, ticket_id: int) -> list[dict[str, Any]]:
    """Return a ticket's audit entries in insertion order (SPEC §7.1)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT event, actor, detail, created_at "
            "FROM audit WHERE ticket_id = %s ORDER BY id",
            (ticket_id,),
        )
        rows = cur.fetchall()
    for row in rows:
        row["created_at"] = _iso(row["created_at"])
    return rows

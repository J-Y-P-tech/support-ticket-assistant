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

    Parameters are passed as keywords rather than a `postgresql://` URL so a
    password containing URL-special characters (`@`, `/`, `:`, `#`) can't corrupt
    the connection string.
    """
    return psycopg.connect(
        user=os.environ["POSTGRES_USER"],
        password=os.environ["POSTGRES_PASSWORD"],
        host=os.environ.get("POSTGRES_HOST", "postgres"),
        port=os.environ.get("POSTGRES_PORT", "5432"),
        dbname=os.environ["POSTGRES_DB"],
    )


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


# Default rep-queue page size, read from the shared `QUEUE_PAGE_DEFAULT` env var
# (the same value the api uses) so the two sides can't drift; the literal is only a
# fallback when the var is unset. The api always passes an explicit `limit`, so this
# default only applies to direct/other callers.
DEFAULT_PAGE_LIMIT = int(os.environ.get("QUEUE_PAGE_DEFAULT", "50"))


def fetch_new_tickets(
    conn: Connection,
    *,
    limit: int = DEFAULT_PAGE_LIMIT,
    after: tuple[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Return one keyset page of New (untriaged) tickets, oldest first.

    Paged on the stable `(created_at, id)` key so the rep queue can never pull the
    whole table in one call (the Task-4 unbounded-queue gap) and so paging is
    deterministic: `limit` caps the page, and `after` — the `(created_at, id)` of
    the last row a caller has already seen — resumes strictly past it, giving no
    duplicated or skipped tickets even though every row shares the `New` status.
    `id` breaks ties when two tickets share a `created_at`, keeping the order total.

    `created_at` is included in each row (ISO-8601) so the caller can build the
    cursor for the next page; the shared `QueueRow` schema ignores the extra field.
    """
    sql = (
        "SELECT id, reference_code, status, urgency, category, created_at "
        "FROM tickets WHERE status = %s"
    )
    params: list[Any] = [_STATUS_NEW]
    if after is not None:
        # Row-value comparison: the next row after the (created_at, id) cursor.
        sql += " AND (created_at, id) > (%s::timestamptz, %s)"
        params.extend(after)
    sql += " ORDER BY created_at, id LIMIT %s"
    params.append(limit)

    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()
    for row in rows:
        row["created_at"] = _iso(row["created_at"])
    return rows


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
            "SELECT id, ticket_id, body, citations, verified "
            "FROM drafts WHERE ticket_id = %s ORDER BY id DESC LIMIT 1",
            (ticket_id,),
        )
        draft = cur.fetchone()

    ticket["created_at"] = _iso(ticket["created_at"])
    ticket["updated_at"] = _iso(ticket["updated_at"])
    ticket["draft"] = draft
    return ticket


def get_ticket_by_code(conn: Connection, reference_code: str) -> dict[str, Any] | None:
    """Load one ticket by its reference code, or None if the code is unknown.

    Reference codes come from their own sequence, so a code is not the ticket id
    (SPEC §14); the api's customer-lookup route (SPEC §4.8) needs this to resolve
    a `TKT-####` a customer typed. Matching is case- and whitespace-insensitive so
    a lightly mistyped code still resolves; an unknown code returns a neutral None
    (no enumeration leak), mirroring `get_ticket`.
    """
    normalized = reference_code.strip().upper()
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT id FROM tickets WHERE reference_code = %s", (normalized,))
        row = cur.fetchone()
    if row is None:
        return None
    return get_ticket(conn, row["id"])


def save_draft(
    conn: Connection,
    *,
    ticket_id: int,
    body: str,
    citations: list[dict[str, Any]] | None = None,
    verified: bool = True,
) -> dict[str, Any]:
    """Persist a reply draft for a ticket and return it (`Draft`-shaped)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO drafts (ticket_id, body, citations, verified) "
            "VALUES (%s, %s, %s, %s) "
            "RETURNING id, ticket_id, body, citations, verified",
            (ticket_id, body, Jsonb(citations or []), verified),
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
    _record_audit(conn, ticket_id, "status_changed", actor=actor, detail={"to": status})
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


def record_audit(
    conn: Connection,
    *,
    ticket_id: int,
    event: str,
    actor: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one immutable audit entry, commit it, and return the stored row.

    The public, committing entry point the api uses to record each workflow-node
    outcome and rep action through the email_mcp boundary (SPEC §7.1). Unlike the
    internal `_record_audit` helper — which the ticket tools batch inside their own
    transaction — this commits on its own, because the api records one entry per
    node as an independent operation. The audit table is DB-enforced immutable
    (migration 0002), so a recorded entry can never be altered or removed.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO audit (ticket_id, event, actor, detail) VALUES (%s, %s, %s, %s) "
            "RETURNING event, actor, detail, created_at",
            (ticket_id, event, actor, Jsonb(detail) if detail is not None else None),
        )
        row = cur.fetchone()
    assert row is not None  # INSERT ... RETURNING always yields the new row.
    row["created_at"] = _iso(row["created_at"])
    conn.commit()
    return row


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


# --------------------------------------------------------------------------- #
# Feedback tools
# --------------------------------------------------------------------------- #
def record_feedback(
    conn: Connection,
    *,
    ticket_id: int,
    decision: str,
    ai_draft: str,
    final_reply: str | None = None,
    edit_distance: int | None = None,
    rating: int | None = None,
    reason: str | None = None,
    category: str | None = None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """Persist one rep-decision feedback row and return it (SPEC §4.9).

    Records how a rep disposed of an AI draft — approved-as-is / edited (with the diff
    distance between the draft and the final reply) / rejected — plus the optional
    rating and reason. `final_reply`/`edit_distance` are absent for a rejection (no
    reply was sent); `rating`/`reason` are optional throughout. `category` tags the row
    with the ticket's triage category so `approved_replies_by_category` can select an
    approved reply as a same-category few-shot example (SPEC §4.10); it is optional and
    stored NULL when absent. The row feeds the quality loop (§7.4) and the de-identified
    training corpus (§4.9a).
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO feedback "
            "(ticket_id, draft_id, decision, ai_draft, final_reply, edit_distance, "
            "rating, reason, category) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id, ticket_id, draft_id, decision, ai_draft, final_reply, "
            "edit_distance, rating, reason, category, created_at",
            (
                ticket_id,
                draft_id,
                decision,
                ai_draft,
                final_reply,
                edit_distance,
                rating,
                reason,
                category,
            ),
        )
        row = cur.fetchone()
    assert row is not None  # INSERT ... RETURNING always yields the new row.
    row["created_at"] = _iso(row["created_at"])
    conn.commit()
    return row


def approved_replies_by_category(
    conn: Connection, *, category: str, limit: int
) -> list[dict[str, Any]]:
    """Return recent **approved** replies for a category, newest first (SPEC §4.10).

    The candidate pool for the live dynamic few-shot lookup: for `category`, the most
    recent feedback rows a rep actually sent — decision not `rejected` and carrying a
    `final_reply` — joined to their ticket for the customer `message`. Each row is shaped
    for the deterministic selector (todo Task 29): `message`, the approved `reply`, the
    rep `rating`, and `example_id` (the feedback row id — a monotonic recency/stability
    key, higher = newer). Ordered by that id descending and capped at `limit`, so the
    pool is bounded even as the corpus grows. An empty result (no approved replies for
    the category) is a plain empty list — the draft prompt is then left unchanged.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT f.id AS example_id, t.message AS message, "
            "f.final_reply AS reply, f.rating AS rating "
            "FROM feedback f JOIN tickets t ON t.id = f.ticket_id "
            "WHERE f.category = %s AND f.decision <> 'rejected' "
            "AND f.final_reply IS NOT NULL "
            "ORDER BY f.id DESC LIMIT %s",
            (category, limit),
        )
        return cur.fetchall()


def get_feedback(conn: Connection, ticket_id: int) -> list[dict[str, Any]]:
    """Return a ticket's feedback rows in insertion order (SPEC §4.9)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, ticket_id, draft_id, decision, ai_draft, final_reply, "
            "edit_distance, rating, reason, created_at "
            "FROM feedback WHERE ticket_id = %s ORDER BY id",
            (ticket_id,),
        )
        rows = cur.fetchall()
    for row in rows:
        row["created_at"] = _iso(row["created_at"])
    return rows


# --------------------------------------------------------------------------- #
# Training-corpus tools
# --------------------------------------------------------------------------- #
def record_corpus(
    conn: Connection,
    *,
    ticket_id: int,
    record_type: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    """Append one de-identified training-corpus record and return it (SPEC §4.9a).

    The `training_corpus` table is append-only: each resolved case contributes an SFT
    record (`record_type = "sft"`) and, when the rep edited the draft, a preference pair
    (`"preference"`). `payload` is the already-PII-redacted record body, stored in the
    JSONB `payload` column so either shape lives in one table. The rows feed the
    `make export-training-data` JSONL export for future fine-tuning.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "INSERT INTO training_corpus (ticket_id, record_type, payload) "
            "VALUES (%s, %s, %s) "
            "RETURNING id, ticket_id, record_type, payload, created_at",
            (ticket_id, record_type, Jsonb(payload)),
        )
        row = cur.fetchone()
    assert row is not None  # INSERT ... RETURNING always yields the new row.
    row["created_at"] = _iso(row["created_at"])
    conn.commit()
    return row


def get_corpus(conn: Connection, ticket_id: int) -> list[dict[str, Any]]:
    """Return a ticket's training-corpus records in insertion order (SPEC §4.9a)."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "SELECT id, ticket_id, record_type, payload, created_at "
            "FROM training_corpus WHERE ticket_id = %s ORDER BY id",
            (ticket_id,),
        )
        rows = cur.fetchall()
    for row in rows:
        row["created_at"] = _iso(row["created_at"])
    return rows


def export_corpus(conn: Connection) -> list[dict[str, Any]]:
    """Return the whole append-only corpus in insertion order for the JSONL export.

    Reads every ticket's records, ordered as they were written, so the exported dataset
    is a stable stream a downstream fine-tune can consume (SPEC §4.9a). Only the fields
    the export renders — `record_type` and `payload` — are selected.
    """
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute("SELECT record_type, payload FROM training_corpus ORDER BY id")
        return cur.fetchall()

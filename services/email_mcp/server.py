"""email_mcp — the Email/Ticket MCP server (SPEC §2, §3).

Exposes the ticket operations as MCP tools over the official MCP Python SDK
(FastMCP). Each tool is a thin wrapper that opens a connection and delegates to
the tested `db.py` functions; all business logic and the resolve-safety
invariant live there.

Inter-service auth enforcement (SPEC §6) is layered on in Task 23; this task
establishes the tool surface and its behaviour.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

import db

# Served over streamable-HTTP so the api (a separate container) can reach it over
# the network (SPEC §6, api↔MCP over HTTP). Bind and path are configurable; the
# defaults match `.env.example` (EMAIL_MCP_URL=http://email_mcp:8000/mcp).
mcp = FastMCP(
    "email_mcp",
    host=os.environ.get("EMAIL_MCP_HOST", "0.0.0.0"),
    port=int(os.environ.get("EMAIL_MCP_PORT", "8000")),
    streamable_http_path="/mcp",
)


@mcp.tool()
def create_ticket(message: str, attachments: list[str] | None = None) -> dict[str, Any]:
    """Create a New ticket and return it with its assigned reference code."""
    with db.connect_from_env() as conn:
        return db.create_ticket(conn, message=message, attachments=attachments)


@mcp.tool()
def fetch_new_tickets(
    limit: int = db.DEFAULT_PAGE_LIMIT,
    after_created_at: str | None = None,
    after_id: int | None = None,
) -> list[dict[str, Any]]:
    """Return one keyset page of New tickets, oldest first (paged, never the whole table).

    `after_created_at` + `after_id` are the `(created_at, id)` of the last row the
    caller already has; pass both to fetch the following page. They travel as two
    scalars because the MCP boundary carries JSON, not tuples.
    """
    after: tuple[str, int] | None = None
    if after_created_at is not None and after_id is not None:
        after = (after_created_at, after_id)
    with db.connect_from_env() as conn:
        return db.fetch_new_tickets(conn, limit=limit, after=after)


@mcp.tool()
def get_ticket(ticket_id: int) -> dict[str, Any]:
    """Return a ticket with its latest draft, or a neutral not-found result."""
    with db.connect_from_env() as conn:
        ticket = db.get_ticket(conn, ticket_id)
    return ticket if ticket is not None else {"found": False}


@mcp.tool()
def get_ticket_by_code(reference_code: str) -> dict[str, Any]:
    """Return a ticket by its reference code, or a neutral not-found result."""
    with db.connect_from_env() as conn:
        ticket = db.get_ticket_by_code(conn, reference_code)
    return ticket if ticket is not None else {"found": False}


@mcp.tool()
def save_draft(
    ticket_id: int,
    body: str,
    citations: list[dict[str, Any]] | None = None,
    verified: bool = True,
) -> dict[str, Any]:
    """Persist a reply draft for a ticket and return the saved draft."""
    with db.connect_from_env() as conn:
        return db.save_draft(
            conn,
            ticket_id=ticket_id,
            body=body,
            citations=citations,
            verified=verified,
        )


@mcp.tool()
def update_status(ticket_id: int, status: str, actor: str | None = None) -> dict[str, Any] | None:
    """Transition a ticket's status (cannot set Resolved; see record_sent_reply)."""
    with db.connect_from_env() as conn:
        return db.update_status(conn, ticket_id=ticket_id, status=status, actor=actor)


@mcp.tool()
def record_sent_reply(ticket_id: int, reply: str, rep_id: str) -> dict[str, Any] | None:
    """Record a rep-sent reply, resolving the case (requires a rep-action marker)."""
    with db.connect_from_env() as conn:
        return db.record_sent_reply(conn, ticket_id=ticket_id, reply=reply, rep_id=rep_id)


@mcp.tool()
def record_audit(
    ticket_id: int,
    event: str,
    actor: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Append one immutable audit entry for a ticket and return the stored row."""
    with db.connect_from_env() as conn:
        return db.record_audit(conn, ticket_id=ticket_id, event=event, actor=actor, detail=detail)


@mcp.tool()
def get_audit_trail(ticket_id: int) -> list[dict[str, Any]]:
    """Return a ticket's audit entries in insertion order (SPEC §7.1)."""
    with db.connect_from_env() as conn:
        return db.get_audit_trail(conn, ticket_id)


@mcp.tool()
def record_feedback(
    ticket_id: int,
    decision: str,
    ai_draft: str,
    final_reply: str | None = None,
    edit_distance: int | None = None,
    rating: int | None = None,
    reason: str | None = None,
    draft_id: int | None = None,
) -> dict[str, Any]:
    """Persist one rep-decision feedback row and return it (SPEC §4.9)."""
    with db.connect_from_env() as conn:
        return db.record_feedback(
            conn,
            ticket_id=ticket_id,
            decision=decision,
            ai_draft=ai_draft,
            final_reply=final_reply,
            edit_distance=edit_distance,
            rating=rating,
            reason=reason,
            draft_id=draft_id,
        )


@mcp.tool()
def get_feedback(ticket_id: int) -> list[dict[str, Any]]:
    """Return a ticket's feedback rows in insertion order (SPEC §4.9)."""
    with db.connect_from_env() as conn:
        return db.get_feedback(conn, ticket_id)


@mcp.tool()
def record_corpus(ticket_id: int, record_type: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Append one de-identified training-corpus record and return it (SPEC §4.9a)."""
    with db.connect_from_env() as conn:
        return db.record_corpus(conn, ticket_id=ticket_id, record_type=record_type, payload=payload)


@mcp.tool()
def get_corpus(ticket_id: int) -> list[dict[str, Any]]:
    """Return a ticket's training-corpus records in insertion order (SPEC §4.9a)."""
    with db.connect_from_env() as conn:
        return db.get_corpus(conn, ticket_id)


def main() -> None:
    """Run the MCP server over streamable-HTTP (SPEC §6: api↔MCP over HTTP)."""
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

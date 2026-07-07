"""email_mcp — the Email/Ticket MCP server (SPEC §2, §3).

Exposes the ticket operations as MCP tools over the official MCP Python SDK
(FastMCP). Each tool is a thin wrapper that opens a connection and delegates to
the tested `db.py` functions; all business logic and the resolve-safety
invariant live there.

Inter-service auth enforcement (SPEC §6) is layered on in Task 23; this task
establishes the tool surface and its behaviour.
"""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

import db

mcp = FastMCP("email_mcp")


@mcp.tool()
def create_ticket(message: str, attachments: list[str] | None = None) -> dict[str, Any]:
    """Create a New ticket and return it with its assigned reference code."""
    with db.connect_from_env() as conn:
        return db.create_ticket(conn, message=message, attachments=attachments)


@mcp.tool()
def fetch_new_tickets() -> list[dict[str, Any]]:
    """Return the New (untriaged) tickets awaiting processing."""
    with db.connect_from_env() as conn:
        return db.fetch_new_tickets(conn)


@mcp.tool()
def get_ticket(ticket_id: int) -> dict[str, Any]:
    """Return a ticket with its latest draft, or a neutral not-found result."""
    with db.connect_from_env() as conn:
        ticket = db.get_ticket(conn, ticket_id)
    return ticket if ticket is not None else {"found": False}


@mcp.tool()
def save_draft(
    ticket_id: int,
    body: str,
    citations: list[dict[str, Any]] | None = None,
    unverified: bool = False,
) -> dict[str, Any]:
    """Persist a reply draft for a ticket and return the saved draft."""
    with db.connect_from_env() as conn:
        return db.save_draft(
            conn,
            ticket_id=ticket_id,
            body=body,
            citations=citations,
            unverified=unverified,
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


def main() -> None:
    """Run the MCP server (stdio transport by default)."""
    mcp.run()


if __name__ == "__main__":
    main()

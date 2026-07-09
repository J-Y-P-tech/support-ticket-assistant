"""Async client wrapper for the email_mcp server (SPEC §3, plan Task 4).

The api reaches email_mcp — a separate container — over the MCP streamable-HTTP
transport, presenting its `EMAIL_MCP_TOKEN` as a bearer header (enforcement on the
server lands in Task 23; the header is wired now). It builds on the shared
`MCPClient` base (plan Task 8 follow-up), which keeps one session open and reuses it
across calls instead of re-doing the connect/initialize/DELETE handshake per ticket
op. This wrapper adds only the email-specific tool methods.

email_mcp signals "no such ticket" with a neutral `{"found": false}` marker rather
than an error (no enumeration leak); the lookup methods map that back to `None` so
routes can return a uniform 404.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.mcp_clients.base import MCPClient, MCPToolError, _parse_tool_result

# `_parse_tool_result` is re-exported for the wrapper's unit tests, which exercise
# the shared parser directly against the email error type.
__all__ = ["EmailMCPClient", "EmailMCPError", "get_email_client", "_parse_tool_result"]


class EmailMCPError(MCPToolError):
    """Raised when an email_mcp tool call returns an error result."""


def _to_optional(payload: Any) -> dict[str, Any] | None:
    """Map email_mcp's `{"found": false}` not-found marker to None."""
    if isinstance(payload, dict) and payload.get("found") is False:
        return None
    return cast("dict[str, Any] | None", payload)


class EmailMCPClient(MCPClient):
    """Thin async wrapper over email_mcp's ticket tools (shared session base)."""

    _error_cls = EmailMCPError

    async def create_ticket(
        self, message: str, attachments: list[str] | None = None
    ) -> dict[str, Any]:
        """Create a New ticket and return it with its assigned reference code.

        A write: `retry_on_disconnect` is left off so a dropped connection surfaces
        as an error rather than risking a duplicate ticket (see plan Task 8 follow-up
        for the idempotency-key fix).
        """
        payload = await self.call_tool(
            "create_ticket", {"message": message, "attachments": attachments or []}
        )
        return cast("dict[str, Any]", payload)

    async def get_ticket(self, ticket_id: int) -> dict[str, Any] | None:
        """Return a ticket by id, or None for an unknown id (neutral not-found)."""
        return _to_optional(
            await self.call_tool("get_ticket", {"ticket_id": ticket_id}, retry_on_disconnect=True)
        )

    async def get_ticket_by_code(self, reference_code: str) -> dict[str, Any] | None:
        """Return a ticket by reference code, or None for an unknown code."""
        return _to_optional(
            await self.call_tool(
                "get_ticket_by_code",
                {"reference_code": reference_code},
                retry_on_disconnect=True,
            )
        )

    async def fetch_new_tickets(
        self, *, limit: int, after: tuple[str, int] | None = None
    ) -> list[dict[str, Any]]:
        """Return one keyset page of the New (untriaged) rep queue.

        `limit` is required: the caller (the rep route) owns page sizing via config,
        so the wrapper carries no default of its own to drift from it. `after` is the
        `(created_at, id)` of the last row already seen, passed as two scalars over
        the MCP boundary. Each returned row includes `created_at` so the route can
        build the next-page cursor.
        """
        arguments: dict[str, Any] = {"limit": limit}
        if after is not None:
            arguments["after_created_at"] = after[0]
            arguments["after_id"] = after[1]
        payload = await self.call_tool("fetch_new_tickets", arguments, retry_on_disconnect=True)
        return cast("list[dict[str, Any]]", payload)


def get_email_client(
    request: Request, settings: Settings = Depends(get_settings)
) -> EmailMCPClient:
    """FastAPI dependency: return the process-wide shared `EmailMCPClient`.

    Because the client holds a reused session, it must be a singleton rather than
    built per request. It is created lazily on first use and cached on `app.state`
    (closed by the app lifespan on shutdown). Overridden in tests with an in-memory
    fake, so routes never touch the network under test.
    """
    client = getattr(request.app.state, "email_client", None)
    if client is None:
        client = EmailMCPClient(
            url=settings.email_mcp_url,
            token=settings.email_mcp_token.get_secret_value(),
        )
        request.app.state.email_client = client
    return cast("EmailMCPClient", client)

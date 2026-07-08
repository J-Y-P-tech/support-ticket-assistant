"""Async client wrapper for the email_mcp server (SPEC §3, plan Task 4).

The api reaches email_mcp — a separate container — over the MCP streamable-HTTP
transport, presenting its `EMAIL_MCP_TOKEN` as a bearer header (enforcement on
the server lands in Task 23; the header is wired now). Each call opens a short
session, invokes one tool, and parses the JSON payload back out of the result, so
the wrapper is stateless and safe to construct per request.

email_mcp signals "no such ticket" with a neutral `{"found": false}` marker
rather than an error (no enumeration leak); the lookup methods map that back to
`None` so routes can return a uniform 404.
"""

from __future__ import annotations

from typing import Any, cast

from fastapi import Depends
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult

from app.config import Settings, get_settings


class EmailMCPError(RuntimeError):
    """Raised when an email_mcp tool call returns an error result."""


def _parse_tool_result(result: CallToolResult) -> Any:
    """Return the structured payload of a tool result, or raise on a tool error.

    We read `structuredContent`, not the text content blocks, on purpose: FastMCP
    flattens a *list* return into one text block per item (losing the array
    framing), whereas `structuredContent` preserves the value. For our tools that
    return a `dict`, `structuredContent` is that dict directly; for the one tool
    that returns a `list` (`fetch_new_tickets`), FastMCP wraps it as
    `{"result": [...]}` — we unwrap that so callers get the plain list.

    An error result is surfaced as `EmailMCPError` rather than treated as data.
    """
    if result.isError:
        text = result.content[0].text if result.content else ""  # type: ignore[union-attr]
        raise EmailMCPError(f"email_mcp tool error: {text}")
    structured = result.structuredContent
    if structured is None:
        return None
    # FastMCP wraps a non-object (list) return as {"result": <value>}; unwrap it.
    if set(structured) == {"result"}:
        return structured["result"]
    return structured


def _to_optional(payload: Any) -> dict[str, Any] | None:
    """Map email_mcp's `{"found": false}` not-found marker to None."""
    if isinstance(payload, dict) and payload.get("found") is False:
        return None
    return cast("dict[str, Any] | None", payload)


class EmailMCPClient:
    """Thin async wrapper over email_mcp's ticket tools."""

    def __init__(self, url: str, token: str) -> None:
        """Store the streamable-HTTP endpoint and build the bearer auth header."""
        self._url = url
        self._headers = {"Authorization": f"Bearer {token}"}

    async def _call(self, tool: str, arguments: dict[str, Any]) -> Any:
        """Open a session, invoke one tool, and return its parsed JSON payload."""
        async with streamablehttp_client(self._url, headers=self._headers) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                result = await session.call_tool(tool, arguments)
        return _parse_tool_result(result)

    async def create_ticket(
        self, message: str, attachments: list[str] | None = None
    ) -> dict[str, Any]:
        """Create a New ticket and return it with its assigned reference code."""
        payload = await self._call(
            "create_ticket", {"message": message, "attachments": attachments or []}
        )
        return cast("dict[str, Any]", payload)

    async def get_ticket(self, ticket_id: int) -> dict[str, Any] | None:
        """Return a ticket by id, or None for an unknown id (neutral not-found)."""
        return _to_optional(await self._call("get_ticket", {"ticket_id": ticket_id}))

    async def get_ticket_by_code(self, reference_code: str) -> dict[str, Any] | None:
        """Return a ticket by reference code, or None for an unknown code."""
        return _to_optional(
            await self._call("get_ticket_by_code", {"reference_code": reference_code})
        )

    async def fetch_new_tickets(self) -> list[dict[str, Any]]:
        """Return the New (untriaged) tickets awaiting the rep queue."""
        payload = await self._call("fetch_new_tickets", {})
        return cast("list[dict[str, Any]]", payload)


def get_email_client(settings: Settings = Depends(get_settings)) -> EmailMCPClient:
    """FastAPI dependency: build an `EmailMCPClient` from config.

    Overridden in tests with an in-memory fake, so routes never touch the network
    under test.
    """
    return EmailMCPClient(
        url=settings.email_mcp_url,
        token=settings.email_mcp_token.get_secret_value(),
    )

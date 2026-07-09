"""Async client wrapper for the kb_mcp server (SPEC §4.4, plan Task 8).

The agent uses this to call `search_knowledge_base` on kb_mcp — a separate
container — over the MCP streamable-HTTP transport, presenting its `KB_MCP_TOKEN` as
a bearer header. Built on the shared `MCPClient` base (plan Task 8 follow-up), so it
reuses one session across searches instead of reconnecting per call.

It parses the tool payload into a typed `KBSearchResult`: ranked `KBSource` objects
plus the explicit `no_confident_source` signal that routes a case to
needs-human-research (SPEC §4.4). That signal is carried on its own, not inferred
from an empty source list.
"""

from __future__ import annotations

from typing import cast

from fastapi import Depends, Request

from app.config import Settings, get_settings
from app.mcp_clients.base import MCPClient
from app.schemas.kb import KBSearchResult


class KBMCPClient(MCPClient):
    """Thin async wrapper over kb_mcp's `search_knowledge_base` tool."""

    def __init__(self, url: str, token: str, default_limit: int) -> None:
        """Store the endpoint/token and the configured default source count.

        `default_limit` comes from config (`KB_SEARCH_LIMIT`), so how many ranked
        sources a search requests is a tuning knob, not a code constant.
        """
        super().__init__(url, token)
        self._default_limit = default_limit

    async def search(self, query: str, limit: int | None = None) -> KBSearchResult:
        """Search the KB and return typed sources plus the no-confident-source signal.

        `limit` defaults to the configured `KB_SEARCH_LIMIT` when omitted. A search
        is a read with no side effects, so it opts into reconnect+retry on a dropped
        connection.
        """
        payload = await self.call_tool(
            "search_knowledge_base",
            {"query": query, "limit": self._default_limit if limit is None else limit},
            retry_on_disconnect=True,
        )
        return KBSearchResult.model_validate(payload)


def get_kb_client(request: Request, settings: Settings = Depends(get_settings)) -> KBMCPClient:
    """FastAPI dependency: return the process-wide shared `KBMCPClient`.

    Like the email client, this holds a reused session and so is a lazily-built
    singleton cached on `app.state` (closed by the app lifespan on shutdown).
    """
    client = getattr(request.app.state, "kb_client", None)
    if client is None:
        client = KBMCPClient(
            url=settings.kb_mcp_url,
            token=settings.kb_mcp_token.get_secret_value(),
            default_limit=settings.kb_search_limit,
        )
        request.app.state.kb_client = client
    return cast("KBMCPClient", client)

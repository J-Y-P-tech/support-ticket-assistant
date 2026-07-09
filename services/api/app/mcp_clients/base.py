"""Shared session-reusing MCP client base (plan Task 8 follow-up).

The api talks to two MCP servers in separate containers (`email_mcp`, `kb_mcp`)
over the streamable-HTTP transport. The walking-skeleton clients opened a fresh
session per call — every tool invocation paid a connect + `initialize` + call +
`DELETE` round-trip. This base keeps **one** session open and reuses it across
calls; the concrete `EmailMCPClient`/`KBMCPClient` just add their tool methods.

Lifecycle and behaviour (both confirmed with the user 2026-07-08):
- **Lazy, reused session.** The transport is opened on the first call and held via
  an `AsyncExitStack`; later calls reuse it. `aclose()` tears it down (called by the
  app lifespan on shutdown). Because a client now holds a connection, it must be a
  process-wide singleton, not built per request (see `get_email_client`).
- **Reconnect + retry once, for idempotent reads only.** If the held session raises
  a transport error, the dead session is always dropped (so the next call
  reconnects); the failed call itself is retried exactly once **only when the caller
  marks it `retry_on_disconnect=True`** — safe for reads, withheld for writes so a
  half-completed non-idempotent call (e.g. `create_ticket`) is never silently run
  twice. The default is conservative (no retry). A tool *error result* (`isError`)
  is data, not a dropped link, so it is surfaced immediately and never retried. (A
  proper idempotency-key fix for writes is a deferred follow-up — see plan Task 8.)
- **One at a time.** All calls on a client serialise behind a per-client
  `asyncio.Lock`; separate clients (email vs kb) have separate locks, so they never
  block each other.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from typing import Any

import anyio
import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import CallToolResult

# Transport-level failures that mean the held session is gone and a reconnect is
# worth attempting. A tool *error result* is not here — it is parsed as data below.
_RECONNECT_ERRORS: tuple[type[BaseException], ...] = (
    ConnectionError,
    anyio.ClosedResourceError,
    anyio.BrokenResourceError,
    anyio.EndOfStream,
    httpx.HTTPError,
)


class MCPToolError(RuntimeError):
    """Raised when an MCP tool call returns an error result."""


def _parse_tool_result(result: CallToolResult, error_cls: type[MCPToolError] = MCPToolError) -> Any:
    """Return the structured payload of a tool result, or raise `error_cls` on error.

    We read `structuredContent`, not the text content blocks, on purpose: FastMCP
    flattens a *list* return into one text block per item (losing the array framing),
    whereas `structuredContent` preserves the value. For a tool that returns a
    `dict`, `structuredContent` is that dict directly; for a tool that returns a
    `list`, FastMCP wraps it as `{"result": [...]}` — we unwrap that so callers get
    the plain list. An error result is surfaced as `error_cls`, not treated as data.
    """
    if result.isError:
        text = result.content[0].text if result.content else ""  # type: ignore[union-attr]
        raise error_cls(f"MCP tool error: {text}")
    structured = result.structuredContent
    if structured is None:
        return None
    # FastMCP wraps a non-object (list) return as {"result": <value>}; unwrap it.
    if set(structured) == {"result"}:
        return structured["result"]
    return structured


class MCPClient:
    """Base wrapper that keeps one streamable-HTTP MCP session open and reused."""

    # Concrete clients override this so their tool errors surface as a specific type.
    _error_cls: type[MCPToolError] = MCPToolError

    def __init__(self, url: str, token: str) -> None:
        """Store the endpoint and bearer header; the session is opened lazily."""
        self._url = url
        self._headers = {"Authorization": f"Bearer {token}"}
        self._session: ClientSession | None = None
        self._stack: AsyncExitStack | None = None
        self._tools: Any = None
        self._lock = asyncio.Lock()

    async def _open(self) -> ClientSession:
        """Open the transport + session, initialize it, and hold it via an exit stack."""
        stack = AsyncExitStack()
        try:
            read_stream, write_stream, _ = await stack.enter_async_context(
                streamablehttp_client(self._url, headers=self._headers)
            )
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack
        return session

    async def _ensure_session(self) -> ClientSession:
        """Return the held session, opening one on first use. Caller holds the lock."""
        if self._session is None:
            self._session = await self._open()
        return self._session

    async def _reset(self) -> None:
        """Close the held transport and drop the cached session/tool list."""
        stack, self._stack = self._stack, None
        self._session = None
        self._tools = None
        if stack is not None:
            await stack.aclose()

    async def _call_and_maybe_retry(
        self, tool: str, arguments: dict[str, Any], *, retry_on_disconnect: bool
    ) -> CallToolResult:
        """Invoke `tool`; on a transport error drop the dead session, retry if allowed.

        The dead session is always dropped so the *next* call reconnects. Whether we
        retry *this* call depends on `retry_on_disconnect`: safe for idempotent reads,
        withheld for writes so a half-completed, non-idempotent call (e.g.
        `create_ticket`) is never silently run twice — the error is surfaced instead.
        """
        session = await self._ensure_session()
        try:
            return await session.call_tool(tool, arguments)
        except _RECONNECT_ERRORS:
            await self._reset()
            if not retry_on_disconnect:
                raise
        session = await self._ensure_session()
        return await session.call_tool(tool, arguments)

    async def call_tool(
        self, tool: str, arguments: dict[str, Any], *, retry_on_disconnect: bool = False
    ) -> Any:
        """Invoke an MCP tool over the reused session and return its parsed payload.

        `retry_on_disconnect` defaults to False (conservative): callers opt in only
        for idempotent reads. Writes leave it off, so a dropped connection surfaces
        as an error rather than risking a duplicate side effect.
        """
        async with self._lock:
            result = await self._call_and_maybe_retry(
                tool, arguments, retry_on_disconnect=retry_on_disconnect
            )
        return _parse_tool_result(result, self._error_cls)

    async def list_tools(self) -> Any:
        """Return the server's tool list, fetched once over the wire then cached."""
        async with self._lock:
            if self._tools is None:
                session = await self._ensure_session()
                self._tools = await session.list_tools()
            return self._tools

    async def aclose(self) -> None:
        """Close the held session/transport; a later call reconnects lazily."""
        async with self._lock:
            await self._reset()

    def __repr__(self) -> str:
        """Identify the client by type and endpoint (no token — it stays in headers)."""
        return f"{type(self).__name__}(url={self._url!r})"

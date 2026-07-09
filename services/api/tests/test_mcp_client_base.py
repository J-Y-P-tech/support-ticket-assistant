"""Unit tests for the shared session-reusing MCP client base (plan Task 8 follow-up).

These prove the base wrapper's transport lifecycle **without any network**: it opens
the streamable-HTTP session lazily and *reuses* it across calls (instead of the old
connect/initialize/DELETE handshake per call), serialises calls per client behind a
lock (one at a time), reconnects-and-retries once when the kept-open session drops,
does **not** retry a normal tool-error result (that is data, not a dropped
connection), caches the tool list, and closes the transport on `aclose`. The live
transport is exercised separately in `test_integration_email_mcp.py`.

The real `_open` (streamable-HTTP + `ClientSession` + initialize) is replaced with a
scripted `FakeSession`/`FakeStack` so every path is deterministic and offline.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent

from app.mcp_clients.base import MCPClient, MCPToolError


class FakeStack:
    """Stand-in for the AsyncExitStack that holds the transport open; records close."""

    def __init__(self) -> None:
        """Start unclosed."""
        self.closed = False

    async def aclose(self) -> None:
        """Mark the (fake) transport stack closed."""
        self.closed = True


class FakeSession:
    """In-memory `ClientSession` double scripting `call_tool` / `list_tools` behaviour."""

    def __init__(
        self,
        *,
        tool_result: CallToolResult | None = None,
        raises: Exception | None = None,
        tools: Any = None,
    ) -> None:
        """Script one outcome: a tool result, a raised transport error, or a tool list."""
        self._tool_result = tool_result
        self._raises = raises
        self._tools = tools
        self.call_count = 0
        self.list_count = 0

    async def call_tool(self, tool: str, arguments: dict[str, Any]) -> CallToolResult:
        """Return the scripted result, or raise the scripted transport error."""
        self.call_count += 1
        if self._raises is not None:
            raise self._raises
        assert self._tool_result is not None
        return self._tool_result

    async def list_tools(self) -> Any:
        """Return the scripted tool list, counting how often the wire was hit."""
        self.list_count += 1
        return self._tools


def _ok_result(payload: dict[str, Any]) -> CallToolResult:
    """Build a successful dict-returning tool result (structured content = the dict)."""
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=False,
    )


def _install(
    client: MCPClient,
    sessions: list[FakeSession],
    opened: list[bool],
    stacks: list[FakeStack],
) -> None:
    """Replace the client's `_open` with a scripted, offline opener.

    Each `_open` pops the next `FakeSession`, records the open in `opened`, and
    attaches a fresh `FakeStack` (tracked in `stacks`) so reconnect/close can be
    asserted. Shadowing the bound method on the instance is enough: `_ensure_session`
    calls `self._open()`.
    """

    async def fake_open() -> FakeSession:
        opened.append(True)
        stack = FakeStack()
        stacks.append(stack)
        client._stack = stack  # type: ignore[assignment]
        return sessions.pop(0)

    client._open = fake_open  # type: ignore[assignment,method-assign]


@pytest.fixture
def client() -> MCPClient:
    """A base `MCPClient` pointed at a dummy endpoint (no connection is opened)."""
    return MCPClient(url="http://svc:8000/mcp", token="t")


async def test_reuses_session_across_calls(client: MCPClient) -> None:
    """The session is opened once and reused for subsequent calls (no per-call handshake)."""
    opened: list[bool] = []
    stacks: list[FakeStack] = []
    session = FakeSession(tool_result=_ok_result({"ok": True}))
    _install(client, [session], opened, stacks)

    first = await client.call_tool("do", {})
    second = await client.call_tool("do", {})

    assert first == {"ok": True} == second
    assert len(opened) == 1  # opened once...
    assert session.call_count == 2  # ...but used twice


async def test_retryable_call_reconnects_and_retries_once_on_dropped_connection(
    client: MCPClient,
) -> None:
    """A retry-marked call whose session drops is reopened and retried exactly once."""
    opened: list[bool] = []
    stacks: list[FakeStack] = []
    dead = FakeSession(raises=ConnectionError("connection dropped"))
    fresh = FakeSession(tool_result=_ok_result({"ok": True}))
    _install(client, [dead, fresh], opened, stacks)

    result = await client.call_tool("do", {}, retry_on_disconnect=True)

    assert result == {"ok": True}
    assert len(opened) == 2  # reconnected after the drop
    assert stacks[0].closed is True  # the dead transport was closed before retrying


async def test_non_retryable_call_surfaces_error_without_retrying(
    client: MCPClient,
) -> None:
    """A write (default, not retry-marked) surfaces the drop and is NOT re-run.

    The dead session is still dropped so the *next* call reconnects, but this call
    raises rather than risk a duplicate side effect.
    """
    opened: list[bool] = []
    stacks: list[FakeStack] = []
    dead = FakeSession(raises=ConnectionError("connection dropped"))
    fresh = FakeSession(tool_result=_ok_result({"ok": True}))
    _install(client, [dead, fresh], opened, stacks)

    with pytest.raises(ConnectionError):
        await client.call_tool("do", {})  # default: retry_on_disconnect=False

    assert len(opened) == 1  # the call was not retried...
    assert stacks[0].closed is True  # ...but the dead session was dropped

    # The next call reconnects on the fresh session.
    assert await client.call_tool("do", {}) == {"ok": True}
    assert len(opened) == 2


async def test_does_not_retry_on_tool_error(client: MCPClient) -> None:
    """A tool-error *result* raises and is NOT retried — it is data, not a dropped link."""
    opened: list[bool] = []
    stacks: list[FakeStack] = []
    error_result = CallToolResult(content=[TextContent(type="text", text="boom")], isError=True)
    session = FakeSession(tool_result=error_result)
    _install(client, [session], opened, stacks)

    with pytest.raises(MCPToolError):
        await client.call_tool("do", {})

    assert len(opened) == 1  # no reconnect
    assert session.call_count == 1  # no retry


async def test_aclose_closes_transport_and_reopens_next_call(client: MCPClient) -> None:
    """`aclose` closes the held transport; the next call lazily reconnects."""
    opened: list[bool] = []
    stacks: list[FakeStack] = []
    first = FakeSession(tool_result=_ok_result({"n": 1}))
    second = FakeSession(tool_result=_ok_result({"n": 2}))
    _install(client, [first, second], opened, stacks)

    await client.call_tool("do", {})
    await client.aclose()

    assert stacks[0].closed is True

    assert await client.call_tool("do", {}) == {"n": 2}
    assert len(opened) == 2  # reopened after close


async def test_concurrent_calls_open_session_once(client: MCPClient) -> None:
    """Concurrent calls serialise behind the per-client lock and open the session once."""
    opened: list[bool] = []
    stacks: list[FakeStack] = []
    session = FakeSession(tool_result=_ok_result({"ok": True}))

    async def slow_open() -> FakeSession:
        opened.append(True)
        await asyncio.sleep(0.01)  # widen the race window
        stack = FakeStack()
        stacks.append(stack)
        client._stack = stack  # type: ignore[assignment]
        return session

    client._open = slow_open  # type: ignore[assignment,method-assign]

    await asyncio.gather(*(client.call_tool("do", {}) for _ in range(5)))

    assert len(opened) == 1  # one connect despite five concurrent callers
    assert session.call_count == 5


async def test_list_tools_is_cached(client: MCPClient) -> None:
    """The tool list is fetched once over the wire, then served from cache."""
    opened: list[bool] = []
    stacks: list[FakeStack] = []
    session = FakeSession(tools=["a", "b"])
    _install(client, [session], opened, stacks)

    first = await client.list_tools()
    second = await client.list_tools()

    assert first == second == ["a", "b"]
    assert session.list_count == 1  # only the first call hit the wire
    assert len(opened) == 1

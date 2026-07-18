"""Unit tests for the email MCP client wrapper's parsing and mapping (plan Task 4).

These prove how the wrapper turns an MCP `CallToolResult` into typed Python data
*without any network*: the JSON payload is read from the tool result, a tool
error is surfaced (not swallowed), and email_mcp's neutral not-found marker
(`{"found": false}`) is mapped back to `None`. The live transport is exercised
separately in `test_integration_email_mcp.py`.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent

from app.mcp_clients.email import EmailMCPClient, EmailMCPError, _parse_tool_result


def _dict_result(payload: dict[str, Any]) -> CallToolResult:
    """Build a result as FastMCP sends a dict-returning tool: structured = the dict.

    The unstructured text block mirrors the same value; the wrapper reads the
    structured content.
    """
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload))],
        structuredContent=payload,
        isError=False,
    )


def _list_result(items: list[dict[str, Any]]) -> CallToolResult:
    """Build a result as FastMCP sends a list-returning tool.

    FastMCP wraps a non-object return as `{"result": [...]}` in the structured
    content and flattens the unstructured content into one text block per item.
    """
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(item)) for item in items],
        structuredContent={"result": items},
        isError=False,
    )


def test_parse_returns_dict_payload() -> None:
    """A dict-returning tool result yields the dict from its structured content."""
    result = _dict_result({"reference_code": "TKT-0001", "status": "New"})

    assert _parse_tool_result(result) == {"reference_code": "TKT-0001", "status": "New"}


def test_parse_unwraps_wrapped_list_payload() -> None:
    """A list-returning tool's `{"result": [...]}` wrapper is unwrapped to the list."""
    result = _list_result([{"id": 1}, {"id": 2}])

    assert _parse_tool_result(result) == [{"id": 1}, {"id": 2}]


def test_parse_raises_on_tool_error() -> None:
    """A tool result flagged as an error raises the given error class, never returns.

    The shared parser takes the client-specific error class; the email client passes
    `EmailMCPError`, so an email tool error surfaces as `EmailMCPError`.
    """
    result = CallToolResult(content=[TextContent(type="text", text="boom")], isError=True)

    with pytest.raises(EmailMCPError):
        _parse_tool_result(result, EmailMCPError)


@pytest.fixture
def wrapper() -> EmailMCPClient:
    """An `EmailMCPClient` pointed at a dummy endpoint (no connection is opened)."""
    return EmailMCPClient(url="http://email_mcp:8000/mcp", token="test-token")


async def test_get_ticket_maps_found_false_to_none(
    wrapper: EmailMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """email_mcp's `{"found": False}` not-found marker is mapped to None."""

    async def fake_call(tool: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
        return {"found": False}

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    assert await wrapper.get_ticket(999_999) is None


async def test_get_ticket_returns_ticket_dict(
    wrapper: EmailMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A found ticket dict is returned unchanged by `get_ticket`."""
    ticket = {"id": 7, "reference_code": "TKT-0007", "status": "New"}

    async def fake_call(tool: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
        return ticket

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    assert await wrapper.get_ticket(7) == ticket


async def test_get_ticket_by_code_maps_found_false_to_none(
    wrapper: EmailMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unknown reference code resolves to a neutral None."""

    async def fake_call(tool: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
        return {"found": False}

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    assert await wrapper.get_ticket_by_code("TKT-9999") is None


async def test_create_ticket_forwards_arguments(
    wrapper: EmailMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`create_ticket` calls the `create_ticket` tool with message + attachments."""
    seen: dict[str, Any] = {}

    async def fake_call(tool: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
        seen["tool"] = tool
        seen["arguments"] = arguments
        return {"reference_code": "TKT-0001", "status": "New"}

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    await wrapper.create_ticket("help me", ["a.pdf"])

    assert seen["tool"] == "create_ticket"
    assert seen["arguments"] == {"message": "help me", "attachments": ["a.pdf"]}


async def test_fetch_new_tickets_returns_list(
    wrapper: EmailMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fetch_new_tickets` returns the list payload from the tool result."""
    rows = [{"id": 1, "reference_code": "TKT-0001", "status": "New"}]

    async def fake_call(tool: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
        return rows

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    assert await wrapper.fetch_new_tickets(limit=50) == rows


async def test_record_audit_forwards_arguments_as_a_write(
    wrapper: EmailMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`record_audit` calls the `record_audit` tool with the event fields, no retry.

    Recording an audit entry is a non-idempotent write — a reconnect-retry would
    append a duplicate row — so, like `create_ticket`, it must leave
    `retry_on_disconnect` off and let a dropped connection surface as an error.
    """
    seen: dict[str, Any] = {}

    async def fake_call(
        tool: str, arguments: dict[str, Any], *, retry_on_disconnect: bool = False
    ) -> Any:
        seen["tool"] = tool
        seen["arguments"] = arguments
        seen["retry"] = retry_on_disconnect
        return {"event": "triaged", "actor": "system", "detail": None, "created_at": "t"}

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    await wrapper.record_audit(7, "triaged", actor="system", detail={"category": "billing"})

    assert seen["tool"] == "record_audit"
    assert seen["arguments"] == {
        "ticket_id": 7,
        "event": "triaged",
        "actor": "system",
        "detail": {"category": "billing"},
    }
    assert seen["retry"] is False


async def test_get_audit_trail_returns_list_and_opts_into_retry(
    wrapper: EmailMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`get_audit_trail` returns the trail list and opts into the read retry.

    Reading a ticket's trail is an idempotent lookup, so it passes
    `retry_on_disconnect=True` like the other reads — a dropped connection can be
    safely re-run.
    """
    trail = [{"event": "ticket_created", "actor": "customer", "detail": None, "created_at": "t"}]
    seen: dict[str, Any] = {}

    async def fake_call(
        tool: str, arguments: dict[str, Any], *, retry_on_disconnect: bool = False
    ) -> Any:
        seen["retry"] = retry_on_disconnect
        return trail

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    assert await wrapper.get_audit_trail(7) == trail
    assert seen["retry"] is True


async def test_reads_opt_into_retry_but_writes_do_not(
    wrapper: EmailMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reads pass `retry_on_disconnect=True`; the write `create_ticket` does not.

    This is the read-only retry policy: a dropped connection may safely re-run a
    lookup, but never a ticket creation (no silent duplicate).
    """
    seen: dict[str, bool] = {}

    async def fake_call(
        tool: str, arguments: dict[str, Any], *, retry_on_disconnect: bool = False
    ) -> Any:
        seen[tool] = retry_on_disconnect
        if tool == "create_ticket":
            return {"reference_code": "TKT-0001", "status": "New"}
        if tool == "fetch_new_tickets":
            return []
        return {"id": 1, "reference_code": "TKT-0001", "status": "New"}

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    await wrapper.create_ticket("hi")
    await wrapper.get_ticket(1)
    await wrapper.get_ticket_by_code("TKT-0001")
    await wrapper.fetch_new_tickets(limit=50)

    assert seen["create_ticket"] is False  # write: never retried
    assert seen["get_ticket"] is True
    assert seen["get_ticket_by_code"] is True
    assert seen["fetch_new_tickets"] is True


async def test_approved_replies_by_category_returns_list_and_opts_into_retry(
    wrapper: EmailMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`approved_replies_by_category` forwards the category/limit and reads with retry.

    The live few-shot lookup (todo Task 30) fetches candidate approved replies for the
    ticket's category; it is an idempotent read, so — like the other lookups — it passes
    `retry_on_disconnect=True` and returns the tool's row list unchanged.
    """
    rows = [{"example_id": 5, "message": "locked out", "reply": "reset it here", "rating": 4}]
    seen: dict[str, Any] = {}

    async def fake_call(
        tool: str, arguments: dict[str, Any], *, retry_on_disconnect: bool = False
    ) -> Any:
        seen["tool"] = tool
        seen["arguments"] = arguments
        seen["retry"] = retry_on_disconnect
        return rows

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    result = await wrapper.approved_replies_by_category("account_access", limit=3)

    assert result == rows
    assert seen["tool"] == "approved_replies_by_category"
    assert seen["arguments"] == {"category": "account_access", "limit": 3}
    assert seen["retry"] is True


async def test_record_feedback_forwards_the_category(
    wrapper: EmailMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`record_feedback` forwards the record's triage category so approved replies are tagged.

    The category rides along with each stored disposition (todo Task 30), so a later
    `approved_replies_by_category` lookup can find the ticket's approved reply.
    """
    from app.schemas.enums import Category, FeedbackDecision
    from app.schemas.feedback import FeedbackRecord

    seen: dict[str, Any] = {}

    async def fake_call(tool: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
        seen["arguments"] = arguments
        return {"id": 1}

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    record = FeedbackRecord(
        decision=FeedbackDecision.APPROVED_AS_IS,
        ai_draft="reset it",
        final_reply="reset it",
        edit_distance=0,
        category=Category.ACCOUNT_ACCESS,
    )
    await wrapper.record_feedback(7, record)

    assert seen["arguments"]["category"] == "account_access"

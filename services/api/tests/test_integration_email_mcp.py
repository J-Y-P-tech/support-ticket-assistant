"""Live integration test: api's email client against a running email_mcp (Task 4).

The plan calls for "one integration test against live email_mcp". Unlike the rest
of the api suite (which mocks the client), this opens a real streamable-HTTP MCP
connection to a running email_mcp backed by a real database, so it is **opt-in**:
it is skipped unless `RUN_LIVE_MCP=1` and the connection details are supplied via
the environment. The user runs it after `docker compose up email_mcp postgres`
(and `make migrate`); it never runs in the default/CI FakeLLM suite.

Required env when enabled:
- `RUN_LIVE_MCP=1`
- `EMAIL_MCP_URL`   (e.g. http://localhost:8000/mcp)
- `EMAIL_MCP_TOKEN` (bearer; unenforced until Task 23, still sent)
"""

from __future__ import annotations

import os

import pytest

from app.mcp_clients.email import EmailMCPClient

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_LIVE_MCP") != "1",
    reason="live email_mcp integration test; set RUN_LIVE_MCP=1 to enable",
)


async def test_create_then_lookup_roundtrip() -> None:
    """A ticket created via the live tool is retrievable by its reference code.

    Proves the full cross-container path: streamable-HTTP transport, bearer
    header, tool invocation, and JSON result parsing all agree end-to-end.
    """
    client = EmailMCPClient(
        url=os.environ["EMAIL_MCP_URL"],
        token=os.environ["EMAIL_MCP_TOKEN"],
    )

    try:
        created = await client.create_ticket("integration: I can't log in.", [])
        code = created["reference_code"]
        assert code.startswith("TKT-")
        assert created["status"] == "New"

        looked_up = await client.get_ticket_by_code(code)
        assert looked_up is not None
        assert looked_up["reference_code"] == code

        # An unknown code resolves to a neutral not-found, not an error.
        assert await client.get_ticket_by_code("TKT-0000") is None
    finally:
        # The client now holds a reused session; close it so the transport's
        # background tasks shut down cleanly at the end of the test.
        await client.aclose()

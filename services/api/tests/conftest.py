"""Shared fixtures for the api service route/auth tests (plan Task 4).

The route and auth tests exercise the FastAPI app with the email MCP client
*mocked* — they prove the api's own behaviour (auth, validation, not-found
neutrality, response shapes) without standing up email_mcp or a database. A
single dedicated integration test (`test_integration_email_mcp.py`) covers the
live wiring separately.

Every module-under-test import is deferred into the fixture bodies on purpose:
during the RED phase `app.main` and friends do not exist yet, and importing them
at collection time would break the already-green schema/config tests that share
this directory. Deferring the import means only the tests that actually request
these fixtures fail until the implementation lands.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest

# The bearer token the tests present as the frontend->api credential. The app is
# configured (via `test_settings`) to accept exactly this value.
TEST_API_TOKEN = "test-frontend-to-api-token"

# A complete, dummy config set so `Settings` constructs without a real `.env`.
# Mirrors `.env.example`; the only value the tests assert on is the auth token.
_TEST_ENV: dict[str, str] = {
    "LLM_MODEL": "gemma4:12b",
    "OLLAMA_BASE_URL": "http://host.docker.internal:11434",
    "KB_SEARCH_LIMIT": "3",
    "QUEUE_PAGE_DEFAULT": "50",
    "QUEUE_PAGE_MAX": "200",
    "TRIAGE_MAX_ATTEMPTS": "2",
    "GROUNDEDNESS_MIN": "0.6",
    "VALIDATE_MAX_ATTEMPTS": "2",
    "API_AUTH_TOKEN": TEST_API_TOKEN,
    "EMAIL_MCP_URL": "http://email_mcp:8000/mcp",
    "EMAIL_MCP_TOKEN": "api-to-email-token",
    "KB_MCP_URL": "http://kb_mcp:8000/mcp",
    "KB_MCP_TOKEN": "api-to-kb-token",
    "ENCRYPTION_KEY": "dGVzdC1mZXJuZXQta2V5LWRvLW5vdC11c2UtZm9yLXJlYWw=",
    "LANGFUSE_HOST": "http://langfuse:3000",
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
}


class FakeEmailClient:
    """In-memory stand-in for `EmailMCPClient`, configured per test.

    Route tests set the return values (`create_result`, `tickets_by_code`,
    `tickets_by_id`, `queue_rows`) and inspect `calls` to assert what the routes
    forwarded to email_mcp. All methods are async to match the real wrapper.
    """

    def __init__(self) -> None:
        """Start with empty stores and a default 'created ticket' response."""
        self.create_result: dict[str, Any] = {
            "id": 1,
            "reference_code": "TKT-0001",
            "status": "New",
            "message": "",
            "attachments": [],
            "created_at": "2026-07-07T00:00:00+00:00",
        }
        self.tickets_by_code: dict[str, dict[str, Any] | None] = {}
        self.tickets_by_id: dict[int, dict[str, Any] | None] = {}
        self.queue_rows: list[dict[str, Any]] = []
        self.calls: list[tuple[Any, ...]] = []

    async def create_ticket(
        self, message: str, attachments: list[str] | None = None
    ) -> dict[str, Any]:
        """Record the call and echo the message/attachments into `create_result`."""
        self.calls.append(("create_ticket", message, attachments))
        result = dict(self.create_result)
        result["message"] = message
        result["attachments"] = attachments or []
        return result

    async def get_ticket_by_code(self, code: str) -> dict[str, Any] | None:
        """Return the ticket registered for `code`, or None (neutral not-found)."""
        self.calls.append(("get_ticket_by_code", code))
        return self.tickets_by_code.get(code)

    async def get_ticket(self, ticket_id: int) -> dict[str, Any] | None:
        """Return the ticket registered for `ticket_id`, or None."""
        self.calls.append(("get_ticket", ticket_id))
        return self.tickets_by_id.get(ticket_id)

    async def fetch_new_tickets(
        self, *, limit: int, after: tuple[str, int] | None = None
    ) -> list[dict[str, Any]]:
        """Return one keyset page of the configured queue rows.

        Emulates email_mcp's keyset paging so route tests exercise the real
        contract: rows are ordered by `(created_at, id)`, a cursor drops
        everything up to and including that key, and `limit` caps the page. The
        call (with its `limit`/`after`) is recorded so tests can assert the route
        forwarded the cap and cursor it computed.
        """
        self.calls.append(("fetch_new_tickets", limit, after))
        ordered = sorted(self.queue_rows, key=lambda r: (r["created_at"], r["id"]))
        if after is not None:
            ordered = [r for r in ordered if (r["created_at"], r["id"]) > after]
        return ordered[:limit]


@pytest.fixture
def email_client() -> FakeEmailClient:
    """Provide a fresh fake email client for a test to configure."""
    return FakeEmailClient()


@pytest.fixture
def test_settings(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Build a `Settings` from the dummy env, bypassing any real `.env` file.

    The env is set via monkeypatch (auto-undone) and read with `_env_file=None`,
    mirroring `test_config.py`, so no real `.env` and no live secrets are involved.
    """
    from app.config import Settings

    for key, value in _TEST_ENV.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


@pytest.fixture
def client(email_client: FakeEmailClient, test_settings: Any) -> Iterator[Any]:
    """Yield a `TestClient` whose settings and email client are overridden.

    Dependency overrides swap the real config loader and the real email MCP
    client for the test settings and the in-memory fake, so no network or
    database is touched.
    """
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import create_app
    from app.mcp_clients.email import get_email_client

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_email_client] = lambda: email_client
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return an Authorization header carrying the accepted frontend->api token."""
    return {"Authorization": f"Bearer {TEST_API_TOKEN}"}

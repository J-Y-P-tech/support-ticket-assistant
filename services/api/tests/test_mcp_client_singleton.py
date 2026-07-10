"""Unit tests for the app-scoped singleton MCP clients (plan Task 8 follow-up).

Session reuse only pays off if the client is shared across requests — one open
connection per backend — rather than rebuilt per request (the old behaviour). These
prove the FastAPI dependency providers hand back a single shared instance stored on
`app.state`, and that the app lifespan closes any open client on shutdown. No
network: client construction is lazy, so building one never connects.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from app.config import Settings
from app.main import create_app
from app.mcp_clients.email import EmailMCPClient, get_email_client
from app.mcp_clients.kb import KBMCPClient, get_kb_client

# A complete dummy config set so `Settings` constructs without a real `.env`
# (mirrors `conftest._TEST_ENV`); only the MCP URLs/tokens matter here.
_ENV: dict[str, str] = {
    "LLM_MODEL": "gemma4:12b",
    "OLLAMA_BASE_URL": "http://host.docker.internal:11434",
    "KB_SEARCH_LIMIT": "3",
    "QUEUE_PAGE_DEFAULT": "50",
    "QUEUE_PAGE_MAX": "200",
    "TRIAGE_MAX_ATTEMPTS": "2",
    "GROUNDEDNESS_MIN": "0.6",
    "VALIDATE_MAX_ATTEMPTS": "2",
    "API_AUTH_TOKEN": "test-frontend-to-api-token",
    "EMAIL_MCP_URL": "http://email_mcp:8000/mcp",
    "EMAIL_MCP_TOKEN": "api-to-email-token",
    "KB_MCP_URL": "http://kb_mcp:8000/mcp",
    "KB_MCP_TOKEN": "api-to-kb-token",
    "ENCRYPTION_KEY": "dGVzdC1mZXJuZXQta2V5LWRvLW5vdC11c2UtZm9yLXJlYWw=",
    "LANGFUSE_HOST": "http://langfuse:3000",
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
}


@pytest.fixture
def settings(monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Build a `Settings` from the dummy env, bypassing any real `.env` file."""
    for key, value in _ENV.items():
        monkeypatch.setenv(key, value)
    return Settings(_env_file=None)


def test_get_email_client_returns_shared_singleton(settings: Settings) -> None:
    """Two resolutions against the same app return the one shared email client."""
    app = create_app()
    request = SimpleNamespace(app=app)

    first = get_email_client(request, settings)  # type: ignore[arg-type]
    second = get_email_client(request, settings)  # type: ignore[arg-type]

    assert isinstance(first, EmailMCPClient)
    assert first is second


def test_get_kb_client_returns_shared_singleton(settings: Settings) -> None:
    """Two resolutions against the same app return the one shared kb client."""
    app = create_app()
    request = SimpleNamespace(app=app)

    first = get_kb_client(request, settings)  # type: ignore[arg-type]
    second = get_kb_client(request, settings)  # type: ignore[arg-type]

    assert isinstance(first, KBMCPClient)
    assert first is second


async def test_lifespan_closes_open_clients() -> None:
    """On shutdown the lifespan closes any MCP client left open on `app.state`."""
    app = create_app()
    closed: list[str] = []

    class FakeClient:
        """Records that its transport was closed."""

        def __init__(self, name: str) -> None:
            self._name = name

        async def aclose(self) -> None:
            closed.append(self._name)

    async with app.router.lifespan_context(app):
        app.state.email_client = FakeClient("email")
        app.state.kb_client = FakeClient("kb")

    assert sorted(closed) == ["email", "kb"]


async def test_lifespan_noop_when_no_clients() -> None:
    """The lifespan starts and stops cleanly when no client was ever created.

    Pins the "skip absent clients" behaviour so a future edit can't make startup
    eagerly connect (which would break every route test and touch the network).
    """
    app = create_app()

    async with app.router.lifespan_context(app):
        # No request ran, so no client was cached on app.state.
        assert getattr(app.state, "email_client", None) is None
        assert getattr(app.state, "kb_client", None) is None
    # Exiting the lifespan must not raise despite there being nothing to close.

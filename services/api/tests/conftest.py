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
    "DATABASE_URL": "postgresql://support:test@localhost:5432/support_tickets",
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

    `register_ticket` stores one shared ticket dict under both its id and its
    reference code, so a rep action that mutates it by id (send/reject) is visible
    to a later customer lookup by code — the end-to-end approve→lookup path.
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

    def register_ticket(
        self,
        ticket_id: int,
        reference_code: str,
        *,
        status: str = "Drafted",
        message: str = "",
        reply: str | None = None,
    ) -> dict[str, Any]:
        """Register one shared ticket dict under both its id and reference code.

        Returns the dict so a test can inspect it after a rep action mutates it in
        place; the same object is reachable by id (rep routes) and by code
        (customer lookup), so a send/reject write shows up on a later lookup.
        """
        ticket: dict[str, Any] = {
            "id": ticket_id,
            "reference_code": reference_code,
            "status": status,
            "message": message,
            "attachments": [],
            "reply": reply,
        }
        self.tickets_by_id[ticket_id] = ticket
        self.tickets_by_code[reference_code] = ticket
        return ticket

    async def record_sent_reply(
        self, ticket_id: int, reply: str, rep_id: str
    ) -> dict[str, Any] | None:
        """Record a rep-sent reply: resolve the ticket in place and return it.

        Mirrors email_mcp's `record_sent_reply` — the reply is saved and the case
        moves to Resolved. Mutates the registered ticket in place so a later lookup
        by code sees the reply; returns None for an unknown id (neutral not-found).
        """
        self.calls.append(("record_sent_reply", ticket_id, reply, rep_id))
        ticket = self.tickets_by_id.get(ticket_id)
        if ticket is None:
            return None
        ticket["reply"] = reply
        ticket["status"] = "Resolved"
        return ticket

    async def update_status(
        self, ticket_id: int, status: str, actor: str | None = None
    ) -> dict[str, Any] | None:
        """Transition a ticket's status in place and return it (None if unknown).

        Mirrors email_mcp's `update_status` — used by the reject action to route a
        case back to NeedsResearch. It never sets Resolved (that is send-only).
        """
        self.calls.append(("update_status", ticket_id, status, actor))
        ticket = self.tickets_by_id.get(ticket_id)
        if ticket is None:
            return None
        ticket["status"] = status
        return ticket


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
    from app.graph.intake import get_pipeline_starter
    from app.main import create_app
    from app.mcp_clients.email import get_email_client

    async def _noop_pipeline_starter(*args: Any, **kwargs: Any) -> None:
        """Stand in for the submit-time trigger so no Postgres/Ollama is touched."""

    app = create_app()
    app.dependency_overrides[get_settings] = lambda: test_settings
    app.dependency_overrides[get_email_client] = lambda: email_client
    # Submitting schedules the AI pipeline as a background task; the customer-route
    # tests assert the route's own behaviour, so swap the trigger for a no-op.
    app.dependency_overrides[get_pipeline_starter] = lambda: _noop_pipeline_starter
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_headers() -> dict[str, str]:
    """Return an Authorization header carrying the accepted frontend->api token."""
    return {"Authorization": f"Bearer {TEST_API_TOKEN}"}


# --- Rep-action (draft review) test support (plan Task 17) ------------------
#
# The rep-action routes resume a *paused* LangGraph run and then persist the
# outcome via email_mcp. These fixtures drive the real workflow (against the
# deterministic FakeLLM + a confident fake KB) to the human-review pause, so a
# route test exercises the genuine resume→finalize path — no Ollama, no Postgres.

# The happy-path model script, one response per model-using step, in graph order:
# screen_input → triage → draft → validate → screen_output. Mirrors the workflow
# suite so the drafted body a rep sees/sends is a known constant.
HAPPY_DRAFT_BODY = "You can reset your password from the login screen. [KB-1]"
_HAPPY_PATH_SCRIPT = [
    '{"is_injection": false}',
    '{"category": "account_access", "urgency": "normal", "sentiment": "neutral"}',
    HAPPY_DRAFT_BODY,
    '{"score": 1.0, "unsupported_claims": []}',
    '{"has_violation": false}',
]
_BENIGN_MESSAGE = "How do I reset my online banking password?"


class _FakeKBClient:
    """In-memory KB stand-in returning one confident, citable source (id `KB-1`)."""

    async def search(self, query: str, limit: int | None = None) -> Any:
        """Ignore the query and return a single confident source."""
        from app.schemas.kb import KBSearchResult, KBSource

        return KBSearchResult(
            sources=[
                KBSource(
                    id="KB-1",
                    title="Password reset",
                    text="To reset your password, use the login screen.",
                )
            ],
            no_confident_source=False,
        )


@pytest.fixture
def build_paused_workflow(test_settings: Any) -> Any:
    """Return an async factory that drives a fresh workflow to the review pause.

    Each call compiles the workflow with an in-memory checkpointer (the way
    production passes the Postgres saver), runs the happy path for `ticket_id` to
    the `human_review` interrupt, and returns the paused compiled graph — ready for
    a rep-action route to resume via the same `thread_config(ticket_id)` thread.
    """

    async def _factory(ticket_id: int = 7, message: str = _BENIGN_MESSAGE) -> Any:
        """Compile a workflow and run it to the pause for one ticket; return the graph."""
        from langgraph.checkpoint.memory import MemorySaver

        from app.graph.workflow import build_workflow, thread_config
        from app.llm.fake import FakeLLM
        from app.schemas.enums import TicketStatus

        graph = build_workflow(
            llm=FakeLLM(list(_HAPPY_PATH_SCRIPT)),
            kb_client=_FakeKBClient(),
            settings=test_settings,
            checkpointer=MemorySaver(),
        )
        initial = {
            "ticket_id": ticket_id,
            "message": message,
            "attachments": [],
            "extracted_facts": None,
            "flags": [],
            "status": TicketStatus.NEW,
        }
        await graph.ainvoke(initial, thread_config(ticket_id))
        return graph

    return _factory


@pytest.fixture
def rep_client(test_settings: Any, email_client: FakeEmailClient) -> Any:
    """Return an async-context factory: an httpx client wired to a paused workflow.

    Given the compiled graph from `build_paused_workflow`, it builds the app with
    the settings, email client, and workflow dependencies overridden, and yields an
    `httpx.AsyncClient` bound to it. Async (not `TestClient`) because the rep-action
    routes await the graph resume, and the same app carries the customer-lookup
    route, so an approve→send→lookup test runs end to end on one client.
    """
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _make(graph: Any) -> Any:
        """Yield an AsyncClient for an app whose workflow is the given paused graph."""
        import httpx

        from app.config import get_settings
        from app.graph.runtime import get_workflow
        from app.main import create_app
        from app.mcp_clients.email import get_email_client

        app = create_app()
        app.dependency_overrides[get_settings] = lambda: test_settings
        app.dependency_overrides[get_email_client] = lambda: email_client
        app.dependency_overrides[get_workflow] = lambda: graph
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac

    return _make

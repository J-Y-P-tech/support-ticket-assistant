"""Tests for the frontend's typed HTTP client to the api (plan Task 6).

The client is the only place the frontend talks to the api, so its behaviour is
proven here against a mocked transport (`httpx.MockTransport`) — no network, no
running api. We assert the request shape (path, method, bearer header, paging
params) and the response mapping (JSON out, a 404 lookup mapped to `None`).
"""

from __future__ import annotations

import json
from typing import Any

import httpx

from api_client import ApiClient

TOKEN = "frontend-to-api-token"


def _client(handler: Any) -> ApiClient:
    """Build an `ApiClient` whose transport is a `MockTransport` over `handler`."""
    return ApiClient(
        base_url="http://api.test",
        token=TOKEN,
        transport=httpx.MockTransport(handler),
    )


def test_submit_ticket_posts_message_and_returns_body() -> None:
    """Submitting posts the message/attachments and returns the created ticket."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and return a created-ticket response."""
        captured["path"] = request.url.path
        captured["method"] = request.method
        captured["auth"] = request.headers.get("Authorization")
        captured["json"] = json.loads(request.content)
        return httpx.Response(201, json={"reference_code": "TKT-0001", "status": "New"})

    result = _client(handler).submit_ticket("I can't log in", ["id.jpg"])

    assert captured["method"] == "POST"
    assert captured["path"] == "/tickets"
    assert captured["auth"] == f"Bearer {TOKEN}"
    assert captured["json"] == {"message": "I can't log in", "attachments": ["id.jpg"]}
    assert result["reference_code"] == "TKT-0001"


def test_lookup_ticket_returns_body_when_found() -> None:
    """Looking up a known code returns the ticket's customer-facing view."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a ticket for the looked-up reference code."""
        assert request.url.path == "/tickets/TKT-0001"
        return httpx.Response(200, json={"reference_code": "TKT-0001", "status": "New"})

    result = _client(handler).lookup_ticket("TKT-0001")

    assert result is not None
    assert result["status"] == "New"


def test_lookup_ticket_maps_404_to_none() -> None:
    """An unknown code (neutral 404 from the api) becomes a plain `None`."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the api's neutral not-found for any lookup."""
        return httpx.Response(404, json={"detail": "Ticket not found"})

    assert _client(handler).lookup_ticket("TKT-9999") is None


def test_fetch_queue_forwards_paging_and_returns_envelope() -> None:
    """Fetching the queue forwards `limit`/`after` and returns items + next cursor."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the paging query params and return a queue page."""
        captured["path"] = request.url.path
        captured["params"] = dict(request.url.params)
        return httpx.Response(
            200,
            json={
                "items": [{"reference_code": "TKT-0003", "status": "New"}],
                "next_cursor": None,
            },
        )

    page = _client(handler).fetch_queue(limit=2, after="2026-07-07T00:00:02+00:00,2")

    assert captured["path"] == "/rep/queue"
    assert captured["params"]["limit"] == "2"
    assert captured["params"]["after"] == "2026-07-07T00:00:02+00:00,2"
    assert page["items"][0]["reference_code"] == "TKT-0003"
    assert page["next_cursor"] is None


def test_fetch_queue_omits_after_on_the_first_page() -> None:
    """The first page sends no `after` param (there is no cursor yet)."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the query params for a cursor-less first-page request."""
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    _client(handler).fetch_queue()

    assert "after" not in captured["params"]

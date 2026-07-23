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


def test_fetch_queue_omits_limit_when_not_specified() -> None:
    """With no explicit `limit`, the frontend sends none, deferring to the api's default."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the query params for a default-page-size request."""
        captured["params"] = dict(request.url.params)
        return httpx.Response(200, json={"items": [], "next_cursor": None})

    _client(handler).fetch_queue()

    assert "limit" not in captured["params"]


# --- Draft-review: fetch payload + the four rep actions (plan Task 19) --------


def test_fetch_review_gets_the_payload_for_a_ticket_id() -> None:
    """Fetching a review GETs the id's review path and returns the payload body."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and return a draft-review payload."""
        captured["path"] = request.url.path
        captured["method"] = request.method
        captured["auth"] = request.headers.get("Authorization")
        return httpx.Response(
            200,
            json={
                "ticket_id": 7,
                "status": "Drafted",
                "message": "How do I reset my password?",
                "draft": {"body": "Use the login screen.", "citations": [], "verified": True},
                "sources": [{"id": "KB-1", "title": "Password reset", "text": "..."}],
                "flags": [],
                "trace_leak": False,
            },
        )

    result = _client(handler).fetch_review(7)

    assert captured["method"] == "GET"
    assert captured["path"] == "/rep/tickets/7/review"
    assert captured["auth"] == f"Bearer {TOKEN}"
    assert result["draft"]["body"] == "Use the login screen."
    assert result["sources"][0]["id"] == "KB-1"


def test_fetch_review_maps_409_to_none() -> None:
    """A ticket not yet at the review gate (api 409) becomes a plain `None`."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the api's not-awaiting-review conflict for the review path."""
        return httpx.Response(409, json={"detail": "Ticket is not awaiting review"})

    assert _client(handler).fetch_review(3) is None


def test_approve_draft_posts_to_the_approve_action() -> None:
    """Approving POSTs the id's approve path and returns the action result."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the request and return a staged (not-yet-sent) action result."""
        captured["path"] = request.url.path
        captured["method"] = request.method
        return httpx.Response(200, json={"ticket_id": 7, "status": "Drafted", "reply": None})

    result = _client(handler).approve_draft(7)

    assert captured["method"] == "POST"
    assert captured["path"] == "/rep/tickets/7/approve"
    assert result["status"] == "Drafted"


def test_edit_draft_posts_the_edited_reply() -> None:
    """Editing POSTs the id's edit path carrying the rep's replacement reply text."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the edit request and return a staged action result."""
        captured["path"] = request.url.path
        captured["method"] = request.method
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"ticket_id": 7, "status": "Drafted", "reply": None})

    _client(handler).edit_draft(7, "Please reset from the login screen.")

    assert captured["method"] == "POST"
    assert captured["path"] == "/rep/tickets/7/edit"
    assert captured["json"] == {"reply": "Please reset from the login screen."}


def test_send_draft_posts_the_rep_marker_and_returns_the_reply() -> None:
    """Sending POSTs the id's send path with the rep marker and returns the sent reply."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the send request and return a Resolved result with the reply."""
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(
            200, json={"ticket_id": 7, "status": "Resolved", "reply": "Use the login screen."}
        )

    result = _client(handler).send_draft(7, "rep-1")

    assert captured["path"] == "/rep/tickets/7/send"
    assert captured["json"] == {"rep_id": "rep-1"}
    assert result["status"] == "Resolved"
    assert result["reply"] == "Use the login screen."


def test_send_draft_maps_409_to_none_when_nothing_is_staged() -> None:
    """Sending with no approved/edited draft (api 409) becomes a plain `None`."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return the api's nothing-staged conflict for the send path."""
        return httpx.Response(
            409, json={"detail": "No draft has been approved or edited for this ticket"}
        )

    assert _client(handler).send_draft(7, "rep-1") is None


def test_reject_draft_posts_the_marker_and_optional_reason() -> None:
    """Rejecting POSTs the id's reject path with the rep marker and a reason."""
    captured: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the reject request and return a NeedsResearch result."""
        captured["path"] = request.url.path
        captured["json"] = json.loads(request.content)
        return httpx.Response(200, json={"ticket_id": 7, "status": "NeedsResearch", "reply": None})

    result = _client(handler).reject_draft(7, "rep-3", reason="needs a second source")

    assert captured["path"] == "/rep/tickets/7/reject"
    assert captured["json"] == {"rep_id": "rep-3", "reason": "needs a second source"}
    assert result["status"] == "NeedsResearch"

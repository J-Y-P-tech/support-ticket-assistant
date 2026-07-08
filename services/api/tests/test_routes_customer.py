"""Route tests for the customer-facing endpoints (plan Task 4).

Covers SPEC §4.1 (submit; empty message rejected; always returns a reference
code) and §4.8 (reference-code lookup; unknown code yields a neutral not-found
with no enumeration leak). The email MCP client is mocked via the `email_client`
fixture, so these assert the api's own behaviour only.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import FakeEmailClient


def test_submit_ticket_returns_reference_code(
    client: TestClient, email_client: FakeEmailClient, auth_headers: dict[str, str]
) -> None:
    """A valid submission returns the assigned `TKT-####` code and New status."""
    email_client.create_result = {
        "id": 1,
        "reference_code": "TKT-0007",
        "status": "New",
        "message": "",
        "attachments": [],
        "created_at": "2026-07-07T00:00:00+00:00",
    }

    response = client.post("/tickets", json={"message": "I can't log in."}, headers=auth_headers)

    assert response.status_code == 201
    body = response.json()
    assert body["reference_code"] == "TKT-0007"
    assert body["status"] == "New"
    # The message was forwarded to email_mcp unchanged.
    assert ("create_ticket", "I can't log in.", []) in email_client.calls


def test_submit_empty_message_returns_4xx(client: TestClient, auth_headers: dict[str, str]) -> None:
    """An empty message is rejected with a 4xx before any ticket is created (§4.1)."""
    response = client.post("/tickets", json={"message": ""}, headers=auth_headers)

    assert 400 <= response.status_code < 500


def test_submit_whitespace_only_message_returns_4xx(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """A whitespace-only message is treated as empty and rejected (§4.1)."""
    response = client.post("/tickets", json={"message": "   \n\t "}, headers=auth_headers)

    assert 400 <= response.status_code < 500


def test_submit_forwards_attachments(
    client: TestClient, email_client: FakeEmailClient, auth_headers: dict[str, str]
) -> None:
    """Attachment references are passed through to email_mcp on submission."""
    client.post(
        "/tickets",
        json={"message": "See attached.", "attachments": ["statement.pdf"]},
        headers=auth_headers,
    )

    assert ("create_ticket", "See attached.", ["statement.pdf"]) in email_client.calls


def test_submit_requires_auth(client: TestClient) -> None:
    """Submitting without a bearer token is rejected with 401."""
    response = client.post("/tickets", json={"message": "hello"})

    assert response.status_code == 401


def test_lookup_known_code_returns_status(
    client: TestClient, email_client: FakeEmailClient, auth_headers: dict[str, str]
) -> None:
    """Looking up a known code returns its reference code, status, and reply."""
    email_client.tickets_by_code["TKT-0007"] = {
        "id": 7,
        "reference_code": "TKT-0007",
        "status": "Resolved",
        "reply": "All sorted for you.",
    }

    response = client.get("/tickets/TKT-0007", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["reference_code"] == "TKT-0007"
    assert body["status"] == "Resolved"
    assert body["reply"] == "All sorted for you."


def test_lookup_is_case_and_whitespace_insensitive(
    client: TestClient, email_client: FakeEmailClient, auth_headers: dict[str, str]
) -> None:
    """A messy code (lowercase, padded with spaces) still resolves the ticket.

    The route normalizes the code before querying email_mcp, so the fake — which
    only knows the canonical `TKT-0007` — is reached with the normalized value.
    """
    email_client.tickets_by_code["TKT-0007"] = {
        "id": 7,
        "reference_code": "TKT-0007",
        "status": "New",
        "reply": None,
    }

    response = client.get("/tickets/  tkt-0007 ", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["reference_code"] == "TKT-0007"
    assert ("get_ticket_by_code", "TKT-0007") in email_client.calls


def test_lookup_unknown_code_returns_neutral_not_found(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """An unknown code returns a 404 with a generic message (no enumeration leak)."""
    response = client.get("/tickets/TKT-9999", headers=auth_headers)

    assert response.status_code == 404
    # The message must not reveal whether the code ever existed.
    assert "TKT-9999" not in response.text


def test_lookup_requires_auth(client: TestClient) -> None:
    """Looking up a code without a bearer token is rejected with 401."""
    response = client.get("/tickets/TKT-0007")

    assert response.status_code == 401

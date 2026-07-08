"""Route tests for the rep-facing endpoints (plan Task 4).

Covers the rep queue (SPEC §4.3/§4.7 — the New tickets awaiting work) and the
single-ticket detail view a rep opens. Draft-review actions land in Task 17; this
task is queue + detail read-only. The email MCP client is mocked.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.conftest import FakeEmailClient


def test_queue_returns_new_tickets(
    client: TestClient, email_client: FakeEmailClient, auth_headers: dict[str, str]
) -> None:
    """The queue endpoint returns the New/untriaged tickets as queue rows."""
    email_client.queue_rows = [
        {
            "id": 1,
            "reference_code": "TKT-0001",
            "status": "New",
            "urgency": None,
            "category": None,
        }
    ]

    response = client.get("/rep/queue", headers=auth_headers)

    assert response.status_code == 200
    rows = response.json()
    assert len(rows) == 1
    assert rows[0]["reference_code"] == "TKT-0001"
    assert rows[0]["status"] == "New"


def test_queue_requires_auth(client: TestClient) -> None:
    """The rep queue is not readable without a bearer token."""
    response = client.get("/rep/queue")

    assert response.status_code == 401


def test_rep_ticket_detail_returns_ticket(
    client: TestClient, email_client: FakeEmailClient, auth_headers: dict[str, str]
) -> None:
    """Opening a known ticket by id returns its full detail for the rep."""
    email_client.tickets_by_id[7] = {
        "id": 7,
        "reference_code": "TKT-0007",
        "status": "New",
        "message": "I can't log in.",
        "attachments": [],
        "draft": None,
    }

    response = client.get("/rep/tickets/7", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 7
    assert body["message"] == "I can't log in."


def test_rep_ticket_detail_unknown_returns_not_found(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """An unknown ticket id returns a neutral 404, never an error that leaks state."""
    response = client.get("/rep/tickets/999999", headers=auth_headers)

    assert response.status_code == 404


def test_rep_ticket_detail_requires_auth(client: TestClient) -> None:
    """The rep ticket detail is not readable without a bearer token."""
    response = client.get("/rep/tickets/7")

    assert response.status_code == 401

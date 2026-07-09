"""Route tests for the rep-facing endpoints (plan Task 4).

Covers the rep queue (SPEC §4.3/§4.7 — the New tickets awaiting work) and the
single-ticket detail view a rep opens. Draft-review actions land in Task 17; this
task is queue + detail read-only. The email MCP client is mocked.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import FakeEmailClient


def _queue_row(seq: int) -> dict[str, str | int | None]:
    """Build a New queue row with an ordered `created_at`/`id` for keyset tests."""
    return {
        "id": seq,
        "reference_code": f"TKT-{seq:04d}",
        "status": "New",
        "urgency": None,
        "category": None,
        "created_at": f"2026-07-07T00:00:{seq:02d}+00:00",
    }


def test_queue_returns_new_tickets_in_a_paged_envelope(
    client: TestClient, email_client: FakeEmailClient, auth_headers: dict[str, str]
) -> None:
    """The queue endpoint returns New tickets under `items`, with a `next_cursor` key."""
    email_client.queue_rows = [_queue_row(1)]

    response = client.get("/rep/queue", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert [row["reference_code"] for row in body["items"]] == ["TKT-0001"]
    assert body["items"][0]["status"] == "New"
    # A single row is a short (final) page, so there is no next cursor.
    assert body["next_cursor"] is None


def test_queue_caps_results_at_the_hard_max(
    client: TestClient,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
    test_settings: Any,
) -> None:
    """Even asking for far more than the max returns at most the configured cap."""
    max_limit = test_settings.queue_page_max
    email_client.queue_rows = [_queue_row(n) for n in range(1, max_limit + 50)]

    response = client.get("/rep/queue", params={"limit": 100_000}, headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == max_limit
    # A full page means more may remain, so a cursor is handed back.
    assert body["next_cursor"] is not None
    # The route forwarded the capped limit (never the client's oversized ask).
    assert email_client.calls[-1] == ("fetch_new_tickets", max_limit, None)


def test_queue_uses_configured_default_page_size_when_none_requested(
    client: TestClient,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
    test_settings: Any,
) -> None:
    """With no `limit`, the route forwards the configured `QUEUE_PAGE_DEFAULT`."""
    email_client.queue_rows = [_queue_row(1)]

    response = client.get("/rep/queue", headers=auth_headers)

    assert response.status_code == 200
    # The route asked email_mcp for exactly the configured default page size.
    assert email_client.calls[-1] == (
        "fetch_new_tickets",
        test_settings.queue_page_default,
        None,
    )


def test_queue_next_page_has_no_overlap_with_first(
    client: TestClient, email_client: FakeEmailClient, auth_headers: dict[str, str]
) -> None:
    """Following `next_cursor` yields the following tickets with no dupes or gaps."""
    email_client.queue_rows = [_queue_row(n) for n in range(1, 6)]

    first = client.get("/rep/queue", params={"limit": 2}, headers=auth_headers).json()
    second = client.get(
        "/rep/queue",
        params={"limit": 2, "after": first["next_cursor"]},
        headers=auth_headers,
    ).json()

    first_codes = [row["reference_code"] for row in first["items"]]
    second_codes = [row["reference_code"] for row in second["items"]]
    assert first_codes == ["TKT-0001", "TKT-0002"]
    assert second_codes == ["TKT-0003", "TKT-0004"]
    assert set(first_codes).isdisjoint(second_codes)


def test_queue_malformed_cursor_is_rejected(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """A cursor that isn't `<created_at>,<id>` is a client error, not a 500."""
    response = client.get(
        "/rep/queue", params={"after": "not-a-valid-cursor"}, headers=auth_headers
    )

    assert response.status_code == 400


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

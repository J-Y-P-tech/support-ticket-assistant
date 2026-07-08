"""Tests for the bearer-token auth dependency (plan Task 4, SPEC §6).

Every api route sits behind a single frontend->api bearer token. These tests pin
the three cases: a missing token, a wrong token, and the accepted token, using
the rep queue as a representative protected endpoint. Comparison must be constant
-time (see the implementation) so a wrong token cannot be discovered by timing.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_missing_token_is_rejected(client: TestClient) -> None:
    """A request with no Authorization header is rejected with 401."""
    response = client.get("/rep/queue")

    assert response.status_code == 401


def test_wrong_token_is_rejected(client: TestClient) -> None:
    """A request presenting the wrong bearer token is rejected with 401."""
    response = client.get("/rep/queue", headers={"Authorization": "Bearer not-the-real-token"})

    assert response.status_code == 401


def test_non_bearer_scheme_is_rejected(client: TestClient) -> None:
    """A non-bearer Authorization scheme (e.g. Basic) is rejected with 401."""
    response = client.get("/rep/queue", headers={"Authorization": "Basic abc123"})

    assert response.status_code == 401


def test_accepted_token_allows_the_request(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """The configured frontend->api token is accepted and the request proceeds."""
    response = client.get("/rep/queue", headers=auth_headers)

    assert response.status_code == 200

"""Route tests for the api liveness endpoint (plan Task 7 — stack wiring).

The walking skeleton boots as a set of containers; the `api` service needs a
cheap, dependency-free `/health` route so Docker Compose (and any orchestrator)
can tell the container is up. Two things matter and are proven here:

- it reports a simple OK payload, and
- it is exempt from the frontend->api bearer gate, so a health probe with no
  Authorization header still succeeds (the customer/rep routers, by contrast,
  reject an unauthenticated request).

The route touches neither email_mcp nor the database, so the mocked `client`
fixture is reused as-is.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_reports_ok(client: TestClient) -> None:
    """GET /health returns 200 with a minimal `{"status": "ok"}` body."""
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_is_unauthenticated(client: TestClient) -> None:
    """The health probe needs no bearer token, unlike the customer/rep routes.

    A container health check runs without credentials, so /health must answer
    200 with no Authorization header. The same client hitting an authed route
    (`/rep/queue`) without a token is rejected — this contrast proves the health
    route is deliberately exempt from the auth gate, not accidentally open
    because auth is off in the test app.
    """
    assert client.get("/health").status_code == 200

    gated = client.get("/rep/queue")
    assert gated.status_code in (401, 403)

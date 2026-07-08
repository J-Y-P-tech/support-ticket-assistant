"""Liveness endpoint for the api service (SPEC §routes: health, plan Task 7).

A cheap, dependency-free `/health` route so Docker Compose (and any orchestrator)
can probe the running container. It is deliberately **not** behind `require_auth`:
a health check runs without credentials, and the route reveals nothing sensitive.
It touches neither email_mcp nor the database, so it stays green even while those
dependencies are still starting up.
"""

from __future__ import annotations

from fastapi import APIRouter

# No `dependencies=[Depends(require_auth)]` here: the probe must succeed without a
# bearer token, unlike the customer/rep routers.
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Report that the api process is up, as a minimal `{"status": "ok"}` body."""
    return {"status": "ok"}

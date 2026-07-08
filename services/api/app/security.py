"""Authentication for the api service (SPEC §6, plan Task 4).

Every api route sits behind a single frontend->api bearer token (`API_AUTH_TOKEN`
in config). `require_auth` is a FastAPI dependency the routers attach so an
unauthenticated request is rejected with 401 before any handler runs.

Token comparison uses `secrets.compare_digest` (constant-time) so a wrong token
cannot be discovered by timing. Full inter-service auth enforcement on the api↔MCP
hops is hardened in Task 23; this establishes the frontend->api gate.
"""

from __future__ import annotations

import secrets

# Depends(...) tells FastAPI: "before running this function,
# go get this value for me by calling that other function first."
# The key idea: you never call _bearer_scheme(...) or get_settings() yourself — you
# just declare them as parameters, and FastAPI supplies them.
# That's "dependency injection."
# Tests can run without requiring those Depends().
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import Settings, get_settings

# auto_error=False: return None for a missing/non-Bearer credential so we raise a
# single, uniform 401 ourselves (rather than FastAPI's default 403 for no header).
_bearer_scheme = HTTPBearer(auto_error=False)


def require_auth(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    settings: Settings = Depends(get_settings),
) -> None:
    """Reject the request with 401 unless it carries the accepted bearer token.

    A missing header, a non-Bearer scheme, or a token that does not match
    `API_AUTH_TOKEN` all fail identically, so a caller learns only "authenticated
    or not", never why.
    """
    expected = settings.api_auth_token.get_secret_value()
    if credentials is None or not secrets.compare_digest(credentials.credentials, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

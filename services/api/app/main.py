"""FastAPI application factory for the api service (plan Task 4).

`create_app` assembles the customer and rep routers into a single app. It is a
factory (not a module-level singleton) so tests can build an isolated instance and
override dependencies (`get_settings`, `get_email_client`) per test. The LangGraph
agent routes and rep-action routes are added in later phases.

The MCP clients (email_mcp, kb_mcp) hold one reused streamable-HTTP session each and
are cached on `app.state` (plan Task 8 follow-up); the lifespan below closes them on
shutdown so those sessions are torn down cleanly.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes import customer, health, rep


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Close any shared MCP client opened during the app's life, on shutdown.

    The email/kb clients are built lazily on first request and cached on
    `app.state`; here we close their reused sessions when the app stops. Clients
    that were never used (attribute absent) are simply skipped.
    """
    yield
    for attr in ("email_client", "kb_client"):
        client = getattr(app.state, attr, None)
        if client is not None:
            await client.aclose()


def create_app() -> FastAPI:
    """Build and return the FastAPI app with all Task-4 routers mounted."""
    app = FastAPI(title="Support-Ticket Assistant API", lifespan=_lifespan)
    app.include_router(health.router)
    app.include_router(customer.router)
    app.include_router(rep.router)
    return app

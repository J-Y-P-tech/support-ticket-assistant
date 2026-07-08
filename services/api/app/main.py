"""FastAPI application factory for the api service (plan Task 4).

`create_app` assembles the customer and rep routers into a single app. It is a
factory (not a module-level singleton) so tests can build an isolated instance
and override dependencies (`get_settings`, `get_email_client`) per test. The
LangGraph agent routes and rep-action routes are added in later phases.
"""

from __future__ import annotations

from fastapi import FastAPI

from app.routes import customer, health, rep


def create_app() -> FastAPI:
    """Build and return the FastAPI app with all Task-4 routers mounted."""
    app = FastAPI(title="Support-Ticket Assistant API")
    app.include_router(health.router)
    app.include_router(customer.router)
    app.include_router(rep.router)
    return app

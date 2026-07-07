"""Shared pytest fixtures for the email_mcp contract tests.

email_mcp owns real Postgres tables whose migrations use Postgres-specific SQL
(a `SEQUENCE`, `JSONB`, `TIMESTAMPTZ`), so the contract tests run against a real,
throwaway Postgres spun up by `testcontainers` — never SQLite. The container is
started once per test session; migrations are applied once; every test gets a
freshly-truncated database so tests stay independent (SPEC §10 TDD-first).

The user runs the suite (Docker must be available); these fixtures only describe
what a run expects to find.
"""

from __future__ import annotations

from collections.abc import Iterator

import psycopg
import pytest
from psycopg import Connection
from testcontainers.postgres import PostgresContainer

import db

# Tables the migration creates; truncated between tests to isolate them.
_ALL_TABLES = "tickets, drafts, feedback, audit, training_corpus"


@pytest.fixture(scope="session")
def _postgres() -> Iterator[PostgresContainer]:
    """Start a disposable Postgres container for the whole test session."""
    with PostgresContainer("postgres:16-alpine") as container:
        yield container


@pytest.fixture(scope="session")
def _dsn(_postgres: PostgresContainer) -> str:
    """Return a psycopg3 connection string for the throwaway container."""
    host = _postgres.get_container_host_ip()
    port = _postgres.get_exposed_port(5432)
    return (
        f"postgresql://{_postgres.username}:{_postgres.password}"
        f"@{host}:{port}/{_postgres.dbname}"
    )


@pytest.fixture(scope="session", autouse=True)
def _migrated(_dsn: str) -> None:
    """Apply every migration once so the schema exists before any test runs."""
    with psycopg.connect(_dsn) as conn:
        db.apply_migrations(conn)
        conn.commit()


@pytest.fixture()
def conn(_dsn: str, _migrated: None) -> Iterator[Connection]:
    """Yield a clean connection: all tables truncated and the code sequence reset.

    Truncating with `RESTART IDENTITY` and resetting the reference sequence makes
    each test start from `TKT-0001`, so tests can assert exact reference codes.
    """
    with psycopg.connect(_dsn) as connection:
        with connection.cursor() as cur:
            cur.execute(f"TRUNCATE {_ALL_TABLES} RESTART IDENTITY CASCADE")
            cur.execute("ALTER SEQUENCE ticket_reference_seq RESTART WITH 1")
        connection.commit()
        yield connection

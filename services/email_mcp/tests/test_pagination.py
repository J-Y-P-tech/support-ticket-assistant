"""Keyset-pagination contract tests for `fetch_new_tickets` (plan Task 6).

Closes the Task-4 unbounded-queue gap at the state root: `fetch_new_tickets`
must page the New queue by a stable keyset on `(created_at, id)` so a caller can
never pull the whole table in one call, and so paging is deterministic — no
duplicated or skipped tickets across pages even though rows share a status.

These run against the throwaway Postgres from `conftest.py` (real `TIMESTAMPTZ`
ordering, not an in-memory emulation).
"""

from __future__ import annotations

from psycopg import Connection

import db


def _make_tickets(conn: Connection, count: int) -> list[int]:
    """Create `count` New tickets and return their ids in creation order."""
    return [db.create_ticket(conn, message=f"help #{n}")["id"] for n in range(count)]


def test_fetch_new_tickets_caps_results_at_limit(conn: Connection) -> None:
    """A limit caps how many rows one call returns, even when more New exist."""
    _make_tickets(conn, 5)

    page = db.fetch_new_tickets(conn, limit=2)

    assert len(page) == 2


def test_fetch_new_tickets_defaults_are_ordered_oldest_first(conn: Connection) -> None:
    """Without a cursor the first page is the oldest New tickets, in order."""
    ids = _make_tickets(conn, 3)

    page = db.fetch_new_tickets(conn, limit=10)

    assert [row["id"] for row in page] == ids


def test_fetch_new_tickets_paging_covers_all_without_dupes_or_gaps(conn: Connection) -> None:
    """Walking pages by the `(created_at, id)` cursor yields every id exactly once."""
    created = _make_tickets(conn, 5)

    seen: list[int] = []
    after: tuple[str, int] | None = None
    while True:
        page = db.fetch_new_tickets(conn, limit=2, after=after)
        if not page:
            break
        seen.extend(row["id"] for row in page)
        last = page[-1]
        after = (last["created_at"], last["id"])

    assert seen == created  # every ticket once, in order, no dupes/gaps


def test_fetch_new_tickets_exposes_cursor_fields(conn: Connection) -> None:
    """Each row carries the `created_at` and `id` a caller needs to build a cursor."""
    _make_tickets(conn, 1)

    row = db.fetch_new_tickets(conn, limit=1)[0]

    assert row["id"] is not None
    assert row["created_at"] is not None


def test_fetch_new_tickets_after_last_row_is_empty(conn: Connection) -> None:
    """A cursor past the final ticket returns an empty page (end of the queue)."""
    ids = _make_tickets(conn, 2)
    last = db.get_ticket(conn, ids[-1])
    assert last is not None

    page = db.fetch_new_tickets(conn, limit=10, after=(last["created_at"], ids[-1]))

    assert page == []

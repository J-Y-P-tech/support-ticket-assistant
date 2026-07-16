"""Contract tests for the training-corpus store at the state root (todo Task 28).

email_mcp is the only writer of the `training_corpus` table (SPEC §6). `record_corpus`
appends one de-identified record — an SFT record or a preference pair (SPEC §4.9a) —
and `export_corpus` reads every ticket's records back in insertion order for the
`make export-training-data` JSONL export to consume. These run against a real throwaway
Postgres so the JSONB `payload` column behaves as in production.
"""

from __future__ import annotations

from psycopg import Connection

import db


def test_record_corpus_persists_and_reads_back(conn: Connection) -> None:
    """One SFT record stores its type + JSONB payload and reads back intact."""
    ticket = db.create_ticket(conn, message="please help")
    payload = {
        "input": {"message": "reset my password", "facts": None, "sources": []},
        "output": "Use the login screen.",
        "metadata": {"category": "account_access", "groundedness": 1.0},
    }

    row = db.record_corpus(conn, ticket_id=ticket["id"], record_type="sft", payload=payload)

    assert row["record_type"] == "sft"
    stored = db.get_corpus(conn, ticket["id"])
    assert len(stored) == 1
    assert stored[0]["record_type"] == "sft"
    assert stored[0]["payload"] == payload


def test_export_corpus_returns_every_record_in_insertion_order(conn: Connection) -> None:
    """`export_corpus` returns all tickets' records in append order for the JSONL export.

    Two tickets each contribute records; the export is the whole append-only corpus,
    ordered as it was written, so a downstream fine-tune sees a stable stream.
    """
    first = db.create_ticket(conn, message="ticket one")
    second = db.create_ticket(conn, message="ticket two")

    db.record_corpus(conn, ticket_id=first["id"], record_type="sft", payload={"n": 1})
    db.record_corpus(conn, ticket_id=first["id"], record_type="preference", payload={"n": 2})
    db.record_corpus(conn, ticket_id=second["id"], record_type="sft", payload={"n": 3})

    exported = db.export_corpus(conn)

    assert [r["record_type"] for r in exported] == ["sft", "preference", "sft"]
    assert [r["payload"]["n"] for r in exported] == [1, 2, 3]


def test_get_corpus_unknown_ticket_is_empty(conn: Connection) -> None:
    """A ticket with no corpus records reads back as a neutral empty list."""
    assert db.get_corpus(conn, 999999) == []


def test_export_corpus_empty_is_empty(conn: Connection) -> None:
    """An empty corpus exports as a neutral empty list."""
    assert db.export_corpus(conn) == []

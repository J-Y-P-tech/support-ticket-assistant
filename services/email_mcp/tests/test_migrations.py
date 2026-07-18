"""Contract tests for the email_mcp schema migrations (plan Task 3).

Acceptance criterion: "Migrations create all tables." These tests assert the
five owned tables and the reference-code sequence exist after `apply_migrations`
runs (which the session-scoped `_migrated` fixture already did).
"""

from __future__ import annotations

from psycopg import Connection

# The tables email_mcp is the sole owner of (SPEC §6 least-privilege).
_EXPECTED_TABLES = {"tickets", "drafts", "feedback", "audit", "training_corpus"}


def test_migrations_create_all_owned_tables(conn: Connection) -> None:
    """Every table email_mcp owns exists in the public schema after migration."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT table_name FROM information_schema.tables " "WHERE table_schema = 'public'"
        )
        present = {row[0] for row in cur.fetchall()}

    assert _EXPECTED_TABLES <= present


def test_migrations_create_reference_code_sequence(conn: Connection) -> None:
    """The zero-padded reference-code sequence exists so codes can be assigned."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM information_schema.sequences "
            "WHERE sequence_name = 'ticket_reference_seq'"
        )
        assert cur.fetchone() is not None


def test_feedback_category_index_exists(conn: Connection) -> None:
    """The live few-shot lookup filters feedback by category, so that column is indexed.

    `approved_replies_by_category` reads approved replies for one category at a time; the
    index (added in migration 0003) keeps that per-category filter from scanning the whole
    feedback table as the corpus grows (todo Task 30).
    """
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_indexes WHERE indexname = 'feedback_category_idx'")
        assert cur.fetchone() is not None

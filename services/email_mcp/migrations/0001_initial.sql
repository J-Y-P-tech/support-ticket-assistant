-- Migration 0001 — initial ticket schema (plan Task 3).
--
-- email_mcp is the SOLE owner of these tables (SPEC §6 least-privilege): no other
-- service holds credentials to touch them. Statements are idempotent
-- (IF NOT EXISTS) so `make migrate` can be re-run safely.
--
-- Reference codes come from a dedicated SEQUENCE, zero-padded to `TKT-####`
-- (SPEC §14). Triage fields on `tickets` are nullable: a New ticket has not been
-- triaged yet. Only an explicit rep send (record_sent_reply) sets `Resolved`
-- (SPEC §4.7) — enforced in db.py, not by the schema.

-- Reference-code sequence: TKT-0001, TKT-0002, ...
CREATE SEQUENCE IF NOT EXISTS ticket_reference_seq START WITH 1;

CREATE TABLE IF NOT EXISTS tickets (
    id              BIGSERIAL PRIMARY KEY,
    reference_code  TEXT UNIQUE NOT NULL
                        DEFAULT ('TKT-' || lpad(nextval('ticket_reference_seq')::text, 4, '0')),
    status          TEXT NOT NULL DEFAULT 'New',
    message         TEXT NOT NULL,
    attachments     JSONB NOT NULL DEFAULT '[]'::jsonb,
    category        TEXT,
    urgency         TEXT,
    sentiment       TEXT,
    reply           TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- fetch_new_tickets filters on status; index the lookup.
CREATE INDEX IF NOT EXISTS tickets_status_idx ON tickets (status);

-- Reply drafts. A ticket may accumulate several drafts; the latest one wins.
CREATE TABLE IF NOT EXISTS drafts (
    id          BIGSERIAL PRIMARY KEY,
    ticket_id   BIGINT NOT NULL REFERENCES tickets (id),
    body        TEXT NOT NULL,
    citations   JSONB NOT NULL DEFAULT '[]'::jsonb,
    unverified  BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS drafts_ticket_idx ON drafts (ticket_id);

-- Rep disposition of a draft (populated in Task 25; table created now).
CREATE TABLE IF NOT EXISTS feedback (
    id             BIGSERIAL PRIMARY KEY,
    ticket_id      BIGINT NOT NULL REFERENCES tickets (id),
    draft_id       BIGINT REFERENCES drafts (id),
    decision       TEXT NOT NULL,
    ai_draft       TEXT NOT NULL,
    final_reply    TEXT,
    edit_distance  INTEGER,
    rating         INTEGER,
    reason         TEXT,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Immutable, ordered audit trail: every mutation appends one row (SPEC §7.1).
CREATE TABLE IF NOT EXISTS audit (
    id          BIGSERIAL PRIMARY KEY,
    ticket_id   BIGINT NOT NULL REFERENCES tickets (id),
    event       TEXT NOT NULL,
    actor       TEXT,
    detail      JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS audit_ticket_idx ON audit (ticket_id);

-- De-identified fine-tuning corpus (populated in Task 26; table created now).
CREATE TABLE IF NOT EXISTS training_corpus (
    id           BIGSERIAL PRIMARY KEY,
    ticket_id    BIGINT NOT NULL REFERENCES tickets (id),
    record_type  TEXT NOT NULL,
    payload      JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

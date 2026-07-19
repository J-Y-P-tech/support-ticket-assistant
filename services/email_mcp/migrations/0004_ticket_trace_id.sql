-- Migration 0004 — Langfuse trace id on the ticket (plan Task 29 / todo Task 31).
--
-- SPEC §7.2 nests a ticket's Langfuse trace under a trace id stored on the ticket
-- row, so a rep or auditor can jump from a case to its trace. email_mcp is the sole
-- owner of the tickets table (SPEC §6), so the column and its writer live here.
-- Nullable: a New ticket has not been traced yet, and an offline run (no Langfuse)
-- never gets one. Idempotent (IF NOT EXISTS) so `make migrate` can be re-run.

ALTER TABLE tickets ADD COLUMN IF NOT EXISTS trace_id TEXT;

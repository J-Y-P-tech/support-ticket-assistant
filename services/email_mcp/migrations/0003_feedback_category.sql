-- Migration 0003 — tag feedback with its ticket's triage category (todo Task 30).
--
-- The live dynamic few-shot lookup injects the best recent APPROVED replies for a
-- drafting ticket's category (SPEC §4.10). To find them, each approved reply must
-- carry the category it was approved under: a `category` column on `feedback`, set at
-- send time from the ticket's triage result. It is nullable — older rows and the
-- no-draft hand-off carry none — and `approved_replies_by_category` filters on it.
--
-- The index keeps that per-category filter from scanning the whole feedback table as
-- the corpus grows. Idempotent like 0001/0002 (IF NOT EXISTS), so `make migrate` is
-- safe to re-run.

ALTER TABLE feedback ADD COLUMN IF NOT EXISTS category TEXT;

CREATE INDEX IF NOT EXISTS feedback_category_idx ON feedback (category);

-- Migration 0002 — make the audit trail physically immutable (SPEC §7.1).
--
-- The audit table (0001) is append-only by convention; compliance-grade §7.1
-- asks for more: a recorded event must be impossible to alter or remove, so the
-- guarantee cannot depend on every caller behaving. A row-level trigger rejects
-- any UPDATE or DELETE of an audit row at the storage layer — even a direct SQL
-- edit is refused. INSERT is untouched (the trail is append-only, not read-only).
--
-- Scope is deliberately per-row: the trigger fires FOR EACH ROW, so it does not
-- touch table-level TRUNCATE, which the contract-test fixture uses to reset the
-- schema between tests. Purging is an operator/DDL concern, not a row mutation.
--
-- Idempotent like 0001: CREATE OR REPLACE for both the function and the triggers
-- (Postgres 14+), so `make migrate` can be re-run safely.

CREATE OR REPLACE FUNCTION audit_reject_mutation() RETURNS trigger AS $$
BEGIN
    RAISE EXCEPTION 'audit rows are immutable: the compliance trail is append-only';
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER audit_no_update
    BEFORE UPDATE ON audit
    FOR EACH ROW EXECUTE FUNCTION audit_reject_mutation();

CREATE OR REPLACE TRIGGER audit_no_delete
    BEFORE DELETE ON audit
    FOR EACH ROW EXECUTE FUNCTION audit_reject_mutation();

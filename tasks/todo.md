# TODO — Support-Ticket Assistant

Task detail, acceptance criteria, and verification live in [plan.md](plan.md).
Order is dependency-correct; do not start a task before its dependencies are checked off.

## Phase 0 — Foundation & scaffold
- [x] **1.** Repo scaffold, config, Compose skeleton (postgres only) — *deps: none*
- [x] **2.** Shared Pydantic v2 schemas — *deps: 1*
- [x] **3.** email_mcp: schema + migrations + ticket CRUD tools (STATE ROOT) — *deps: 1,2*
- [x] **Checkpoint A** — foundation: lint/test green, migrations + email_mcp contract tests pass

## Phase 1 — Walking skeleton (no AI)
- [x] **4.** api: email MCP client + core routes (submit/lookup/queue) + auth — *deps: 3*
- [ ] **5.** reference-code generate/lookup util — *deps: 2*
- [ ] **6.** frontend: api_client + 3 views (queue list-only) + AppTest + queue pagination (closes Task-4 unbounded-queue gap) — *deps: 4*
- [ ] **Checkpoint B** — walking skeleton: submit → store → rep queue → lookup, no AI

## Phase 2 — KB connector
- [ ] **7.** kb_mcp: provider interface + MockKB + search tool + mock_kb data — *deps: 1,2*
- [ ] **8.** api: kb MCP client wrapper — *deps: 7*

## Phase 3 — LLM plumbing + triage
- [ ] **9.** LLM client + FakeLLM + thinking-trace stripper — *deps: 1*
- [ ] **10.** triage node (validated, retry-once) — *deps: 2,9*

## Phase 4 — Retrieval + grounded drafting
- [ ] **11.** retrieve node + groundedness gate — *deps: 8,10*
- [ ] **12.** draft node (grounded, cited; unverified flag) — *deps: 11*
- [ ] **13.** validate node (schema + groundedness scorer) — *deps: 12*

## Phase 5 — Guardrails
- [ ] **14.** input guards: prompt-injection screening — *deps: 9*
- [ ] **15.** output guards: forbidden promises / PII / tone — *deps: 12*

## Phase 6 — Workflow assembly + human-in-the-loop
- [ ] **16.** LangGraph workflow + human interrupt + Postgres checkpointer + safety-invariant test — *deps: 3,10–15*
- [ ] **17.** api: rep-action routes (edit/approve/reject/send) + finalize — *deps: 16*
- [ ] **18.** frontend: rep workspace draft review — *deps: 17*
- [ ] **Checkpoint C** — text pipeline end-to-end on FakeLLM, human gate enforced

## Phase 7 — Document digitization
- [ ] **19.** OCR vision-transcription pass (strip thinking) — *deps: 9*
- [ ] **20.** structured extraction pass (retry-once, flag on fail) — *deps: 2,19*
- [ ] **21.** fused search-query pass + wire ocr_extract node — *deps: 16,20*

## Phase 8 — Security & encryption
- [ ] **22.** PII encryption at rest (AES-GCM/Fernet) — *deps: 3,20*
- [ ] **23.** structlog + PII redaction + input hardening + auth enforcement — *deps: 4,22*
- [ ] **24.** compliance audit trail — *deps: 3,17*

## Phase 9 — Feedback, training corpus, dynamic prompting
- [ ] **25.** feedback capture (approved/edited-diff/rejected + rating) — *deps: 17,24*
- [ ] **26.** training corpus (SFT + preference pairs) + export JSONL + PII test — *deps: 25*
- [ ] **27.** dynamic prompting: Langfuse prompts + deterministic few-shot — *deps: 12,25*

## Phase 10 — Observability
- [ ] **28.** Langfuse service + one PII-redacted trace per ticket + scores — *deps: 16,23,25*

## Phase 11 — Evals + CI
- [ ] **29.** eval + red-team suites + runner + eval-gated promotion — *deps: 14,15,27*
- [ ] **30.** CI pipeline (.github/workflows/ci.yml) — *deps: 29*
- [ ] **Checkpoint D** — complete: all §13 invariants covered, eval + security gates green

---

## Tooling decisions (resolved)
- [x] **Q1.** Python dependency manager → **`uv`** (committed `uv.lock`)
- [x] **Q2.** DB migrations → **plain SQL files + apply script** (no Alembic)

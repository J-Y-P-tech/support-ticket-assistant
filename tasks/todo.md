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
- [x] **5.** reference-code generate/lookup util — *deps: 2*
- [x] **6.** frontend: api_client + 3 views (queue list-only) + AppTest + queue pagination (closes Task-4 unbounded-queue gap) — *deps: 4*
- [x] **7.** stack wiring: compose services (email_mcp/api/frontend) + api Dockerfile so the walking skeleton boots — *deps: 3,4,6*
- [ ] **Checkpoint B** — walking skeleton: submit → store → rep queue → lookup, no AI

## Phase 2 — KB connector
- [ ] **8.** kb_mcp: provider interface + MockKB + search tool + mock_kb data — *deps: 1,2*
- [ ] **9.** api: kb MCP client wrapper — *deps: 8*
  - follow-up (Checkpoint B): factor out a shared, session-reusing MCP client and retrofit the email client — today each call re-does the connect/list-tools/DELETE handshake (~5 round-trips). See plan.md.

## Phase 3 — LLM plumbing + triage
- [ ] **10.** LLM client + FakeLLM + thinking-trace stripper — *deps: 1*
- [ ] **11.** triage node (validated, retry-once) — *deps: 2,10*

## Phase 4 — Retrieval + grounded drafting
- [ ] **12.** retrieve node + groundedness gate — *deps: 9,11*
- [ ] **13.** draft node (grounded, cited; unverified flag) — *deps: 12*
- [ ] **14.** validate node (schema + groundedness scorer) — *deps: 13*

## Phase 5 — Guardrails
- [ ] **15.** input guards: prompt-injection screening — *deps: 10*
- [ ] **16.** output guards: forbidden promises / PII / tone — *deps: 13*

## Phase 6 — Workflow assembly + human-in-the-loop
- [ ] **17.** LangGraph workflow + human interrupt + Postgres checkpointer + safety-invariant test — *deps: 3,11–16*
- [ ] **18.** api: rep-action routes (edit/approve/reject/send) + finalize — *deps: 17*
- [ ] **19.** frontend: rep workspace draft review — *deps: 18*
- [ ] **Checkpoint C** — text pipeline end-to-end on FakeLLM, human gate enforced

## Phase 7 — Document digitization
- [ ] **20.** OCR vision-transcription pass (strip thinking) — *deps: 10*
- [ ] **21.** structured extraction pass (retry-once, flag on fail) — *deps: 2,20*
- [ ] **22.** fused search-query pass + wire ocr_extract node — *deps: 17,21*

## Phase 8 — Security & encryption
- [ ] **23.** PII encryption at rest (AES-GCM/Fernet) — *deps: 3,21*
- [ ] **24.** structlog + PII redaction + input hardening + auth enforcement — *deps: 4,23*
- [ ] **25.** compliance audit trail — *deps: 3,18*

## Phase 9 — Feedback, training corpus, dynamic prompting
- [ ] **26.** feedback capture (approved/edited-diff/rejected + rating) — *deps: 18,25*
- [ ] **27.** training corpus (SFT + preference pairs) + export JSONL + PII test — *deps: 26*
- [ ] **28.** dynamic prompting: Langfuse prompts + deterministic few-shot — *deps: 13,26*

## Phase 10 — Observability
- [ ] **29.** Langfuse service + one PII-redacted trace per ticket + scores — *deps: 17,24,26*

## Phase 11 — Evals + CI
- [ ] **30.** eval + red-team suites + runner + eval-gated promotion — *deps: 15,16,28*
- [ ] **31.** CI pipeline (.github/workflows/ci.yml) — *deps: 30*
- [ ] **Checkpoint D** — complete: all §13 invariants covered, eval + security gates green

---

## Tooling decisions (resolved)
- [x] **Q1.** Python dependency manager → **`uv`** (committed `uv.lock`)
- [x] **Q2.** DB migrations → **plain SQL files + apply script** (no Alembic)

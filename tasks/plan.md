# Implementation Plan — Support-Ticket Assistant

> Derived from `SPEC.md` (v-current, 2026-07-06). No code exists yet.
> Working agreement: assistant **writes/edits code only**; the user runs all tests, the app,
> Docker, and Ollama. Pause and ask before any decision not already in SPEC.md.

## Overview

A human-in-the-loop support desk for a financial institution. A customer submits a message
(+ optional attachments) through Streamlit; a LangGraph agent on a FastAPI backend triages it,
digitizes attachments (vision OCR + structured extraction with `gemma4:12b`), searches a
pluggable KB connector (mock provider, no RAG), drafts a grounded reply with citations, and
**pauses for a human rep** who edits/approves/sends. Two MCP servers (`kb_mcp`, `email_mcp`),
Postgres for state + checkpointer, self-hosted Langfuse for tracing. Everything is Dockerized
except host Ollama (`host.docker.internal`).

## Sequencing strategy

The build follows a **walking-skeleton-first, then vertical thickening** shape:

1. Stand up the plumbing end-to-end with **no AI** (submit → store → rep queue → lookup) so the
   state root and all four services talk to each other early.
2. Add the KB connector and the LLM client as isolated, independently-tested units.
3. Thicken one AI capability at a time (triage → retrieve+ground → draft → guardrails), then wire
   the LangGraph workflow with the human interrupt.
4. Layer the harder-to-test / cross-cutting concerns last (OCR, encryption, audit, feedback,
   training corpus, dynamic prompting, Langfuse tracing, evals + CI).

`email_mcp` (ticket state) is the dependency root and is built first. The single riskiest
invariant — **no code path sends without a rep action** — is proven the moment the workflow is
assembled (Task 16), not at the end.

## Dependency graph

```
repo scaffold + config + compose skeleton  (Task 1)
   │
   ├── shared Pydantic schemas (Task 2)
   │
   ├── postgres ── email_mcp: schema + migrations + ticket CRUD tools (Task 3)   ← STATE ROOT
   │        │
   │        ├── api: email MCP client + core routes (Task 4) ── frontend: client + views (Task 5,6)
   │        │                                                        │
   │        │                          [Checkpoint B: walking skeleton, no AI]
   │        │
   │        ├── audit trail (Task 24) · feedback (Task 25) · training corpus (Task 26)
   │        └── LangGraph Postgres checkpointer (Task 16)
   │
   ├── kb_mcp: provider iface + MockKB + search tool (Task 7) ── api kb client (Task 8)
   │
   └── LLM client + FakeLLM + thinking-stripper (Task 9)
          ├── triage node (Task 10)
          ├── retrieve + groundedness gate (Task 11) ── draft (Task 12) ── validate (Task 13)
          ├── input guards (Task 14) · output guards (Task 15)
          ├── ocr: transcribe (19) → extract (20) → fuse (21)
          └── LangGraph workflow + human interrupt (Task 16) ── rep-action routes (17) ── rep UI (18)
                 └── observability (28) · dynamic prompting (27) · evals (29) · CI (30)
```

## Architecture decisions carried from SPEC (not re-opened)

- Postgres (not Mongo); LangChain + LangGraph; **two** MCP servers; single multimodal
  `gemma4:12b` for vision **and** text (`LLM_MODEL` in `.env`).
- KB = pluggable connector, **mock provider only, no RAG/vectors**; every KB source is an
  eligible, citable answer — drafts phrase from them and cite them.
- Simulated send; reference-code (`TKT-####`) customer lookup; human approval is a hard
  state-machine invariant.
- TDD-first for **all** logic including frontend; every service dir has its own `tests/`.
- `email_mcp` is the sole holder of ticket-DB credentials.

## Tooling decisions (confirmed 2026-07-06, on top of SPEC)

Two choices SPEC.md left open, now decided by the user:
- **Dependency manager: `uv`** — installs + lockfile; used by the Makefile and CI.
- **Migrations: plain SQL files + a small apply script** (no Alembic); consistent with
  `email_mcp` owning the DB.

---

## Task list

### Phase 0 — Foundation & scaffold

#### Task 1: Repo scaffold, config, and Compose skeleton
**Description:** Create the repo layout from SPEC §8, `pyproject.toml` managed by **`uv`** (tooling:
black/ruff/mypy/pytest) with a committed `uv.lock`, `.env.example`, `Makefile` targets (lint/format/typecheck/test/eval/migrate/seed/
export-training-data/security), and a `docker-compose.yml` skeleton with **postgres only** wired
so far. Add `services/api/app/config.py` (`pydantic-settings`: `LLM_MODEL`, tokens, service URLs,
encryption key, Langfuse keys) with no hard-coded values.
**Acceptance criteria:**
- [ ] `docker compose config` validates; postgres service present with named volume.
- [ ] `config.py` loads from env and fails clearly when a required var is missing.
- [ ] `.env.example` documents every var referenced by config (no real secrets).
**Verification:** `docker compose config`; `make lint` clean on the empty tree; a unit test loads
config from a sample env and asserts a missing-required-var raises.
**Dependencies:** None. **Scope:** M. **Files:** `pyproject.toml`, `docker-compose.yml`,
`.env.example`, `Makefile`, `services/api/app/config.py`, `services/api/tests/test_config.py`.

#### Task 2: Shared Pydantic v2 schemas
**Description:** Define the contract types every layer shares: `ExtractionResult`, `TriageResult`
(category/urgency/sentiment enums), `KBSource` (`id/title/text/source_kind`), `Draft` (+ citations),
`FeedbackRecord`, and ticket DTOs (create/read/queue-row). Enums for status and urgency.
**Acceptance criteria:**
- [ ] Each schema round-trips valid JSON and rejects invalid enum values.
- [ ] Urgency and status enums match SPEC §4.3 / §5 exactly.
**Verification:** `make test` — unit tests per schema (valid + invalid cases).
**Dependencies:** Task 1. **Scope:** M. **Files:** `services/api/app/schemas/*.py`, `tests/`.

#### Task 3: email_mcp — schema, migrations, ticket CRUD tools (STATE ROOT)
**Description:** The only DB owner. Define tables (tickets, drafts, feedback, audit, training_corpus),
**plain SQL migration files applied by a small script** (`make migrate`), `db.py`, and MCP tools `fetch_new_tickets`, `get_ticket`,
`save_draft`, `record_sent_reply`, `update_status`. Reference-code column with a zero-padded
sequence.
**Acceptance criteria:**
- [ ] Migrations create all tables; `make migrate` documented path works.
- [ ] Each tool returns schema-valid output for representative inputs (contract tests).
- [ ] `record_sent_reply` only transitions a case to Resolved when given a rep action marker
      (no auto-resolve path).
**Verification:** contract tests against a throwaway test DB; assert an unknown ticket id returns a
neutral not-found, not an error leak.
**Dependencies:** Tasks 1, 2. **Scope:** L (candidate to split tools-vs-migrations if needed).
**Files:** `services/email_mcp/{server.py,db.py,migrations/,tests/,Dockerfile}`.

> **Checkpoint A — Foundation:** `make lint`/`make test` green; `docker compose up postgres` +
> migrations succeed; email_mcp contract tests pass. Review before Phase 1.

### Phase 1 — Walking skeleton (text-only, no AI)

#### Task 4: api — email MCP client + core routes
**Description:** MCP client wrapper for `email_mcp`; FastAPI routes: `POST /tickets` (creates New,
returns `TKT-####`), `GET /tickets/{code}` (customer lookup, neutral not-found), `GET /rep/queue`,
`GET /rep/tickets/{id}`. Bearer-token auth dependency. Reject empty message (SPEC §4.1).
**Acceptance criteria:**
- [ ] Submitting with no message → 4xx with a clear error; success always returns a code.
- [ ] Unknown reference code → neutral not-found (no enumeration leak).
- [ ] Unauthenticated request → 401.
**Verification:** route tests with the MCP client mocked; one integration test against live email_mcp.
**Dependencies:** Task 3. **Scope:** M. **Files:** `services/api/app/{routes/,mcp_clients/email.py,
security.py}`, `tests/`.

#### Task 5: reference-code generate/lookup util
**Description:** Pure `TKT-####` zero-padded sequence generator + lookup normalizer, unit-tested,
used by Task 4. (Small, extracted so route logic stays thin.)
**Acceptance criteria:** deterministic formatting; case/whitespace-insensitive lookup; unknown →
not-found sentinel. **Verification:** unit tests. **Dependencies:** Task 2. **Scope:** S.

#### Task 6: frontend — api_client + 3 views (list-only) + queue pagination
**Description:** `api_client.py` (typed HTTP to api, token from config); views: *Customer* (submit +
attach), *Check my case* (code lookup), *Rep workspace* (queue table only — draft review comes in
Task 18). AppTest headless harness.
**Also close the Task-4 unbounded-queue gap here, where it naturally surfaces:** `GET /rep/queue`
and email_mcp's `fetch_new_tickets` currently return *every* New ticket with no cap. Add
pagination end-to-end — a `LIMIT`/`OFFSET` (keyset on `created_at, id` preferred) in
`fetch_new_tickets`, `?limit=&offset=` (or a cursor) on the api route with a **server-side default
and hard max** (e.g. default 50, max 200) so a client can never request the whole table, and a
"next page" affordance in the rep queue view. (Not specified in SPEC; scoping decision confirmed
2026-07-07.)
**Acceptance criteria:**
- [ ] Submitting a ticket in AppTest shows a returned reference code.
- [ ] Looking up that code shows status `New`.
- [ ] api_client tests pass against mocked HTTP.
- [ ] `GET /rep/queue` caps results at the configured max even when more New tickets exist, and
      paginates deterministically (stable order, no dupes/gaps across pages).
**Verification:** `make test` (frontend/tests: api_client mocked + AppTest submit/lookup; api route
test for the limit cap + a pagination boundary; email_mcp contract test for the paged query).
**Dependencies:** Task 4. **Scope:** M. **Files:** `services/frontend/{app.py,views/,api_client.py,
tests/,Dockerfile}`, plus `services/api/app/routes/rep.py`, `services/api/app/mcp_clients/email.py`,
`services/email_mcp/{db.py,server.py}` (pagination).

> **Checkpoint B — Walking skeleton:** customer submit → stored → rep sees it in the queue →
> customer looks it up by code, **with no AI in the loop**. All four containers talk. Review.

### Phase 2 — KB connector

#### Task 7: kb_mcp — provider interface + MockKB + search tool
**Description:** `search_knowledge_base` tool over a pluggable `KBProvider` interface; ship
`MockKBProvider` reading curated answers from `mock_kb/`, returning ranked `KBSource` chunks with
`source_kind`. I provide the curated answer set (SPEC §14.2).
**Acceptance criteria:**
- [ ] Tool returns schema-valid ranked chunks for a matching query.
- [ ] No match → explicit "no confident source" signal (drives needs-research later).
- [ ] Provider interface is swappable (a second dummy provider can register with no tool change).
**Verification:** contract tests over representative queries incl. the no-match case.
**Dependencies:** Task 1, 2. **Scope:** M. **Files:** `services/kb_mcp/{server.py,providers/,mock_kb/,
tests/,Dockerfile}`.

#### Task 8: api — kb MCP client wrapper
**Description:** Client wrapper the agent uses to call `search_knowledge_base`.
**Acceptance criteria:** returns typed `KBSource[]`; surfaces the no-match signal distinctly.
**Verification:** unit tests with kb_mcp mocked. **Dependencies:** Task 7. **Scope:** S.
**Follow-up (from Checkpoint B, plan Task 7):** the email MCP client opens a fresh
streamable-HTTP session per call — every ticket op costs a connect + `ListToolsRequest`
+ call + `DELETE` (~5 round-trips), observed live in the walking-skeleton logs. As
this is the second MCP client wrapper, factor out a shared client that reuses the
session/transport (and caches the tool list) across calls, and retrofit the email
client onto it, instead of duplicating the per-call handshake. Correctness is fine
today; this is a latency/chattiness optimization.
**Follow-up (deferred, from Task 8 impl 2026-07-09):** the shared client's
reconnect+retry is scoped to idempotent reads; writes (`create_ticket`, and later
`save_draft`/`record_sent_reply`/`update_status`) deliberately do **not** retry so a
half-completed call can't silently duplicate. The production-grade fix is an
idempotency key — the write carries a client-generated UUID and `email_mcp` dedupes
via a unique constraint (needs an email_mcp tool-signature change + a migration).
Out of scope for Task 8; revisit before this goes anywhere near real traffic.

### Phase 3 — LLM plumbing + triage

#### Task 9: LLM client + FakeLLM + thinking-trace stripper
**Description:** Async Ollama client (host via `host.docker.internal`, model from `LLM_MODEL`);
a deterministic `FakeLLM` for tests/CI; a `strip_thinking()` util that removes
`Thinking… …done thinking.` traces / narration and keeps only the final answer (SPEC App. A).
**Acceptance criteria:**
- [ ] `strip_thinking` on a stubbed response containing a thinking block yields only the final text.
- [ ] FakeLLM returns scripted responses deterministically; no network in tests.
**Verification:** unit tests. **Dependencies:** Task 1. **Scope:** M. **Files:**
`services/api/app/llm/*.py`, `tests/`.

#### Task 10: triage node
**Description:** Node producing validated `TriageResult` (category/urgency/sentiment); retry once on
schema-invalid output.
**Acceptance criteria:** valid FakeLLM output → typed result; invalid-then-valid → retried and
passes; invalid-twice → surfaced, not silently dropped. **Verification:** unit tests with FakeLLM.
**Dependencies:** Tasks 2, 9. **Scope:** S/M.

### Phase 4 — Retrieval + grounded drafting

#### Task 11: retrieve node + groundedness gate
**Description:** `retrieve` calls the kb client with the query; `groundedness_gate` routes to
`flag_needs_research` when no `authoritative` source is returned.
**Acceptance criteria:** authoritative match → proceeds to draft; no-match → needs-research flag,
never a drafted answer (SPEC §4.4). **Verification:** unit tests both branches.
**Dependencies:** Tasks 8, 10. **Scope:** M.

#### Task 12: draft node (grounded, cited)
**Description:** Draft written **only** from authoritative sources, with citations to source
`id/title`; in-repo prompt for now (Langfuse fetch deferred to Task 27). A draft built on a
`model_generated` source is marked **"AI-suggested, unverified."**
**Acceptance criteria:** draft includes at least one citation when authoritative sources exist;
model_generated path sets the unverified flag. **Verification:** unit tests with FakeLLM.
**Dependencies:** Task 11. **Scope:** M.

#### Task 13: validate node (schema + groundedness scorer)
**Description:** Validate the draft schema and compute a groundedness score; below-threshold →
flagged for the rep. **Acceptance criteria:** low-groundedness draft is flagged, not passed as
sourced fact. **Verification:** unit tests around the threshold. **Dependencies:** Task 12.
**Scope:** S/M.

### Phase 5 — Guardrails

#### Task 14: input guards — prompt-injection screening
**Description:** Screen customer text and OCR output for prompt-injection before it reaches the LLM
nodes. **Acceptance criteria:** a known injection string is flagged/neutralized (eval-style test).
**Verification:** unit tests. **Dependencies:** Task 9. **Scope:** S/M.

#### Task 15: output guards — forbidden promises / PII leak / tone
**Description:** Block financial/legal commitments (refund promises, guarantees), PII leakage, and
tone violations in the draft. **Acceptance criteria:** a draft containing a forbidden promise or PII
is blocked/flagged (eval-style test). **Verification:** unit tests + feeds Task 30 red-team.
**Dependencies:** Task 12. **Scope:** M.

### Phase 6 — Workflow assembly + human-in-the-loop

#### Task 16: LangGraph workflow + human interrupt + Postgres checkpointer
**Description:** Wire `ingest → (ocr_extract) → triage → retrieve → groundedness_gate → draft →
validate → human_review(interrupt) → finalize` with the Postgres checkpointer. (OCR node stubbed
until Phase 7.)
**Acceptance criteria:**
- [ ] Integration test with FakeLLM runs to the `human_review` pause and stops there.
- [ ] **Safety invariant:** no code path reaches sent/Resolved without an explicit rep action.
- [ ] State resumes across the pause via the checkpointer.
**Verification:** workflow integration test; explicit safety-invariant test (SPEC §10).
**Dependencies:** Tasks 3, 10–15. **Scope:** L. **Files:** `services/api/app/graph/{state.py,nodes/,
workflow.py}`, `tests/`.
> **Thinking-trace leak → human flag (from Task 10 review).** The LLM client reads
> Ollama's trace-free `response` field, but a model that ignores `think` can still
> leak a `<think>`/`…done thinking` trace into it. `contains_thinking_trace` (Task 10,
> in `app/llm/thinking.py`) detects this. Free-text node output (draft; OCR in Phase 7)
> that trips the detector must set a `trace_leak` flag on the state so `human_review`
> surfaces it to the rep — never silently strip. Structured nodes (triage/extraction)
> already fail schema-validation on a leak and retry; detection there is for logging.

#### Task 17: api — rep-action routes + finalize
**Description:** Routes for edit/approve/reject/send that resume the graph; `finalize` saves the
reply, sets status Resolved (send), writes the audit record, records feedback + training record
(feedback/training bodies land in Phase 9; wire the hooks now).
**Acceptance criteria:** approve+send → Resolved and reply visible via lookup; reject → needs-research/
back; unauthenticated → 401. **Verification:** route tests; end-to-end approve→lookup test.
**Dependencies:** Task 16. **Scope:** M.

#### Task 18: frontend — rep workspace draft review
**Description:** Rep view: original message, extracted facts, retrieved sources, draft, warning
banners; edit/approve/reject/send controls. **Acceptance criteria:** AppTest approves a draft and the
case shows Resolved; unverified banner renders when flagged. **Verification:** AppTest.
**Dependencies:** Task 17. **Scope:** M.

> **Checkpoint C — Text pipeline end-to-end:** submit → triage → retrieve → draft → **rep approves**
> → resolved → customer lookup, all on FakeLLM, human gate enforced. Review before Phase 7.
>
> **Wiring the checkpoint needs.** Nothing started the workflow — `submit_ticket` only stored the
> ticket, and the rep-action routes only *resume* an already-paused run, so no ticket ever reached
> `human_review`; kb_mcp was also missing from `docker-compose.yml`, so the retrieve step had no
> connector. Kick off the graph as a **FastAPI background task** on submit (`app/graph/intake.py`):
> the customer's submit returns the reference code immediately while `screen_input → triage →
> retrieve → draft → validate → screen_output` run behind it and the graph pauses at the human gate,
> persisted for the rep to review (SPEC §4.7). Failures fail safe: the exception is logged and the
> ticket stays New. Add the `kb_mcp` service to compose and make `api` depend on it.
> **Checks:** submitting produces a run paused at `human_review` under `thread-<id>` (nothing sent);
> the rep review route returns its draft; a pipeline failure leaves the ticket New, never a 500 on
> submit; `docker compose config` shows five services including kb_mcp.
> **Files:** `services/api/app/graph/{intake.py,runtime.py}`, `services/api/app/routes/customer.py`,
> `docker-compose.yml`, `tests/`.

### Phase 7 — Document digitization (attachments)

#### Task 19: OCR vision-transcription pass
**Description:** Verbatim, description-free transcription prompt + strip thinking/narration (uses
Task 9 util). **Acceptance criteria:** stubbed model response with a thinking block → only the
verbatim transcription survives (SPEC §4.2 / App. A). **Verification:** unit test. **Dependencies:**
Task 9. **Scope:** S/M.

#### Task 20: structured extraction pass
**Description:** Transcription → validated `ExtractionResult` (doc_type/amounts/dates/names/
references/raw_text/low_confidence); retry once, then flag "could not extract" + surface raw text —
never silently drop. **Acceptance criteria:** schema-fail-twice → flagged with raw text preserved.
**Verification:** unit tests. **Dependencies:** Tasks 2, 19. **Scope:** M.

#### Task 21: fused search-query pass + wire ocr_extract node
**Description:** Produce the concise fused query (question + attachment summary; message-only when no
attachment) and replace the stubbed `ocr_extract` node in the graph so retrieval uses the fused
query. **Acceptance criteria:** fused query non-empty whenever a message or transcription exists;
attachment-less ticket fuses from message alone. **Verification:** unit + workflow test.
**Dependencies:** Tasks 16, 20. **Scope:** M.

### Phase 8 — Security & encryption

#### Task 22: PII encryption at rest
**Description:** Authenticated encryption (AES-GCM/Fernet) for extracted PII fields + stored
attachments; app-level key from config; email_mcp stores ciphertext.
**Acceptance criteria:** PII fields are ciphertext at rest and decrypt round-trip in-app; wrong key
fails closed. **Verification:** unit tests. **Dependencies:** Tasks 3, 20. **Scope:** M.

#### Task 23: structured logging + redaction + input hardening
**Description:** `structlog` + PII-redaction processor; upload size/type limits + api rate limiting;
enforce bearer auth on every api↔MCP call. **Acceptance criteria:** logs never contain full account/
card numbers, IDs, or raw attachment text; oversized/disallowed upload rejected. **Verification:**
redaction unit test on a log record with PII. **Dependencies:** Task 4, 22. **Scope:** M.
> **Thinking-trace leak monitoring (from Task 10 review).** At the `OllamaLLM.generate`
> choke point, call `contains_thinking_trace` on the model's `response`; on a hit emit a
> structured `llm_trace_leak` warning (model tag + node, PII-redacted) and increment a
> counter, so a model that stops honouring `think` is visible/alertable in production
> rather than silently degrading. Detection only — handling (human flag) is Task 16.

#### Task 24: compliance audit trail
**Description:** Immutable per-ticket audit rows (submission, each node outcome, cited sources, model
tag + prompt version, guardrail decisions, rep edits, approve/send) with actor + timestamp, written
via email_mcp. **Acceptance criteria:** a resolved case has an ordered, immutable audit trail linking
customer ↔ ticket ↔ final reply. **Verification:** integration test asserts audit completeness.
**Dependencies:** Tasks 3, 17. **Scope:** M.

### Phase 9 — Feedback, training corpus, dynamic prompting

#### Task 25: feedback capture
**Description:** On finalize record approved-as-is / edited (with AI-vs-final diff) / rejected +
rating + reason into the feedback table. **Acceptance criteria:** every resolved case yields a
feedback row with draft, final, edit distance, rating. **Verification:** unit + integration test.
**Dependencies:** Tasks 17, 24. **Scope:** M.

#### Task 26: training corpus + export
**Description:** Append-only de-identified corpus: SFT record (message+facts+sources → approved
reply) and a preference pair when the draft was edited; `make export-training-data` → JSONL. PII
redacted before storage. **Acceptance criteria:** each resolved case → one SFT record (+ pref pair
when edited); a test asserts exported records contain **no** configured PII patterns (SPEC §4.9a).
**Verification:** unit tests on builder + redaction; export smoke test. **Dependencies:** Task 25.
**Scope:** M.

#### Task 27: dynamic prompting — Langfuse prompts + few-shot
**Description:** Resolve drafting/triage/extraction prompts from Langfuse with a pinned in-repo
fallback; deterministic dynamic few-shot selector picks best recent approved replies by category.
**Acceptance criteria:** drafting node resolves prompt from Langfuse with in-repo fallback (Langfuse
stubbed); few-shot selection is deterministic and unit-tested (SPEC §4.10). **Verification:** unit
tests with Langfuse stubbed. **Dependencies:** Tasks 12, 25. **Scope:** M.

#### Task 28: live dynamic few-shot lookup — wire into drafting
**Description:** The follow-up to Task 27, which built the deterministic selector + Langfuse-fallback
resolver but not the live retrieval. Add an email_mcp query returning recent **approved** replies for
a category (DB query + a category index migration + MCP tool + api client method); the workflow
`draft_node` selects the best of them with the Task 27 selector and injects them into the drafting
prompt, so a running ticket gets real few-shot examples end-to-end. **Acceptance criteria:** a
resolved ticket's draft is built with category-matched approved examples end-to-end; selection stays
deterministic; no examples → the prompt is unchanged (SPEC §4.10). **Verification:** unit + workflow
test (FakeLLM). **Dependencies:** Tasks 26, 27 (+ workflow assembly). **Scope:** M.

### Phase 10 — Observability (Langfuse)

#### Task 29: Langfuse service + one PII-redacted trace per ticket
**Description:** Add self-hosted `langfuse` Docker service (own datastore); LangChain/LangGraph
callback emits one trace/ticket (nodes, model calls, tokens, latency, retrieval, guardrail
outcomes), PII-redacted; store trace id on the ticket; attach rep feedback + eval results as scores.
**Acceptance criteria:** a run produces a trace with the trace id persisted; redaction test asserts
no PII in trace payloads (SPEC §7.2). **Verification:** unit test on the redacting callback; compose
config valid. **Dependencies:** Tasks 16, 23, 25. **Scope:** M/L.
> **Thinking-trace leak score (from Task 10 review).** Attach a `trace_leak` boolean
> score to the ticket trace (from the Task 23 detection signal) so leaks are trended on
> the Langfuse dashboard per model tag — closes the loop from pre-deploy grounding
> (the `test_llm_capture.py` fixture) to production observability.

### Phase 11 — Evals + CI

#### Task 30: eval + red-team suites + runner
**Description:** `evals/` golden set (expected category/urgency) + groundedness assertions +
red-team cases (prompt injection, PII-leak, forbidden-promise) that MUST be blocked; `make eval`
runner; eval-gated prompt-promotion hook (Task 27). **Acceptance criteria:** golden cases pass;
every red-team case is blocked/flagged; groundedness threshold enforced (SPEC §10, §12.3).
**Verification:** `make eval`. **Dependencies:** Tasks 14, 15, 27. **Scope:** M/L.

#### Task 31: CI pipeline
**Description:** `.github/workflows/ci.yml`: lint/format/type → unit+contract+workflow (FakeLLM,
no model) → AI eval gate → security scan (bandit/pip-audit/gitleaks/trivy) → build images →
optional gated deploy. **Acceptance criteria:** pipeline mirrors SPEC §12 stage order; eval/security
gates fail the build on regression. (User owns git/GitHub; I only write the workflow file.)
**Verification:** the user runs it in CI. **Dependencies:** Task 30. **Scope:** M.

> **Checkpoint D — Complete:** all §13 "Always" invariants covered; eval + security gates green;
> every SPEC §4 acceptance criterion has a passing test. Final review.

## Risks & mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Human-gate invariant accidentally bypassable | High | Prove it the moment the graph is assembled (Task 16) with a dedicated safety-invariant test; re-assert in CI. |
| PII leaking into logs / Langfuse / training corpus | High | Single shared redaction processor reused by logs (23), traces (28), corpus (26); PII-pattern tests on each sink. |
| `gemma4:12b` non-determinism breaks CI | High | All CI logic runs on FakeLLM; no model download in CI (SPEC §10/§12). |
| OCR hallucinates text not in the image | Med | Extracted facts are unverified input shown to the rep; never used as authoritative grounding (Task 20/21). |
| Task 3 and Task 16 are L-sized | Med | Split if they exceed one focused session (Task 3: migrations vs tools; Task 16: wiring vs safety tests). |
| MCP transport/auth wiring between separate containers | Med | Nail it in Tasks 4/7/8 with a live integration test before the agent depends on it. |

## Resolved (was open)

1. **Python dependency manager → `uv`** (with committed `uv.lock`). Drives Makefile + CI installs.
2. **DB migrations → plain SQL files applied by a small script** (`make migrate`); no Alembic.

No open questions remain. Plan is ready for a go-ahead to start Task 1.

# SPEC — Support-Ticket Assistant

A human-in-the-loop support assistant for a financial institution. A customer submits a
message (and optional document attachments) through a web front end. An AI agent reads it,
figures out what it is about and how urgent it is, digitizes any attachments (OCR +
structured extraction), searches the company knowledge base for the relevant answer, and
drafts a reply grounded in what it found. A human support rep reviews, edits or approves the
draft, and only a person clicking "send" resolves the case. **Nothing reaches a customer
automatically.**

> Status: specification approved-pending. No application code written yet.
> Date: 2026-07-06.

---

## 1. Objective & Users

**Objective.** Cut support handling time while keeping a human in control of every outgoing
reply, and use OCR + retrieval-grounded drafting so answers are accurate and traceable to
source documents. Demonstrate a production-grade, safety-first agentic pipeline.

**Users.**
- **Customer** — writes a support message, attaches documents (e.g. a photo of a statement,
  a screenshot, an ID), submits it, and later looks up the outcome with a reference code.
- **Support rep (human-in-the-loop)** — works a queue of cases, reviews the AI draft and the
  extracted facts, edits or approves, sends the reply, and sets case status.

**Success looks like.** Every case that reaches a customer was approved by a rep; every draft
cites the knowledge-base article(s) it used; cases with no confident answer are flagged for a
human instead of guessed; sensitive customer data is encrypted at rest and never logged in
the clear.

---

## 2. Locked Decisions (from the spec conversation)

| Area | Decision |
|---|---|
| Intake | Customer-facing Streamlit tab acts as the "mailbox": customer writes a message + uploads attachments; the submission becomes a ticket row. No mail server. |
| Rep workspace | Streamlit tab showing all cases as a table with statuses (New / Pending / Resolved / Canceled), the AI draft, extracted facts, and edit/approve/send controls. |
| Customer notification | **Reference-code lookup.** On submit the customer receives a code (e.g. `TKT-1042`) and types it into a "check my case" box to see status and the final reply. |
| Sending | **Simulated.** "Send" marks the reply saved and the case Resolved; the reply becomes visible via reference-code lookup. Nothing is emailed out. |
| MCP servers | **Two.** (1) **Knowledge-Base MCP** — a **pluggable connector/gateway** exposing `search_knowledge_base`, with a swappable provider interface (open door to plug real KB systems in via API later). (2) **Email/Ticket MCP** exposes ticket operations (`fetch_new_tickets`, `get_ticket`, `save_draft`, `record_sent_reply`, `update_status`) and is the sole owner of the ticket tables. |
| KB provider (demo) | **Mock-KB provider**: a curated list of canned help answers matched by simple lookup and returned as **citable sources**. gemma4 only *phrases* the reply from the matched source — it is never the source. **No RAG / no vectors** in this project (deliberately deferred to a future project). Replies with no real source are labeled "AI-suggested, unverified". |
| Attachments | OCR the file, then LLM structured extraction of key facts (doc type, amounts, dates, names, references) feeds classification and drafting. |
| Storage | **PostgreSQL** running as a Docker service. |
| LLM (single, multimodal) | Ollama model tag **`gemma4:12b`** used for **everything** — vision/OCR of attachments *and* all text reasoning (triage, extraction, drafting, phrasing). Runs on the host (outside Docker), reached via `host.docker.internal`. The tag is a single config value in `.env` (`LLM_MODEL`) so it can be swapped later. No separate OCR model. |
| Workflow engine | **LangChain + LangGraph** (multi-step flow, persisted state, human-in-the-loop interrupt). |
| Frontend | **Streamlit.** |
| Packaging | **Docker / Docker Compose.** Ollama is NOT containerized. |

---

## 3. Architecture

Docker Compose services (Ollama runs on the host):

```
           host Ollama (gemma4:12b — vision + text, one model)  ◄── host.docker.internal
                                   ▲
                                   │
  ┌───────────┐   HTTP(+auth)  ┌───┴────────────────────────┐
  │ frontend  │ ─────────────► │ api  (FastAPI backend)      │
  │ Streamlit │                │  • LangGraph orchestrator   │
  │ 3 views   │ ◄───────────── │  • OCR + extraction         │
  └───────────┘                │  • guardrails / validation  │
                               │  • MCP CLIENT               │
                               └──┬──────────────────┬───────┘
                        MCP (auth)│                  │MCP (auth)
                         ┌────────▼──────┐   ┌────────▼─────────┐
                         │ kb_mcp        │   │ email_mcp        │
                         │ pluggable     │   │ ticket CRUD =    │
                         │ KB connector  │   │ sole DB owner    │
                         │ (mock-KB,     │   └────────┬─────────┘
                         │  no RAG)      │            │
                         └───────┬───────┘     ┌──────▼──────┐
                    mock_kb/ answers list      │ postgres    │
                    (+ future API providers)   └─────────────┘
```

- **frontend (Streamlit)** — UI only. Three views: *Customer* (submit + attach), *Check my
  case* (reference-code lookup), *Rep workspace* (queue table, draft review, edit/approve/send,
  status change). Talks only to `api` over HTTP with an API token.
- **api (FastAPI)** — backend for the front end and host of the LangGraph agent. Acts as the
  **MCP client** to both MCP servers. Owns OCR post-processing, guardrails, encryption of PII
  fields, and workflow orchestration. No direct SQL to ticket tables — it goes through
  `email_mcp`.
- **kb_mcp** — MCP server exposing a single `search_knowledge_base` tool in front of a
  **pluggable provider interface**. Ships one **mock-KB provider**: a curated list of canned help
  answers (`mock_kb/`) matched by simple lookup and returned as citable sources with an ID/title.
  The interface is the "open door" — future providers (Confluence/Zendesk/ServiceNow/internal doc
  APIs) can be added behind the same tool with no change to the agent. **No RAG / no vectors here.**
- **email_mcp** — MCP server that is the **only** component touching the ticket/draft/feedback/
  audit tables in Postgres, exposed as MCP tools.
- **postgres** — persistence + LangGraph checkpointer store.
- **langfuse** — self-hosted LLM/agent observability (own datastore). The `api` sends one trace per
  ticket (nodes, model calls, tokens, latency, guardrail outcomes, rep-feedback scores), PII-redacted.
  See §7.2. Not on the customer/reply path; purely for traceability and quality.

---

## 4. Core Features & Acceptance Criteria

### 4.1 Customer intake
- Given a customer fills the message form and optionally attaches files, when they submit,
  then a ticket is created with status `New`, a reference code is returned, and attachments
  are stored encrypted.
- Acceptance: submitting with no message is rejected with a clear error; a reference code is
  always shown on success.

### 4.2 Document digitization (vision OCR + structured extraction)
`gemma4:12b` is **multimodal**, so it reads the attachment image directly — there is no separate
OCR engine and no bounding-box output to clean. Verified behaviour (user test, Appendix A): given
an image and a question, the model returns a natural-language answer and, in "thinking" mode, also
emits a reasoning trace. Two consequences drive the design:
- We must **constrain the OCR prompt** so the model returns the transcribed text only (no narration),
  and **strip any reasoning/`thinking` trace**, keeping only the final answer.
- The model can add commentary or mis-read; the rep always reviews the extracted facts.

Digitization is a **three-pass flow** (all passes use the one model, `gemma4:12b`):
1. **Vision transcription (OCR pass)** — send the image with a strict "transcribe the visible text
   verbatim, output text only" prompt → raw transcription; discard any thinking trace / preamble.
2. **Structured extraction (text pass)** — feed the transcription to the model to produce a
   **validated Pydantic structure** (`doc_type`, `amounts`, `dates`, `names`, `references`,
   `raw_text`, `low_confidence`).
3. **Search-intent fusion (text pass)** — the attachment is *part of the message*, not a separate
   thing. So the model produces a concise **fused search query** that combines the customer's
   question with a short summary of the attachment content. This fused query — not the raw
   transcription — is what the agent hands to the KB connector for retrieval (§4.4). Tickets with no
   attachment skip passes 1–2 and fuse from the message text alone.

- Acceptance: the OCR pass returns transcription text with no reasoning trace or narration
  (verified by a unit test on a stubbed model response containing a `thinking` block); extraction
  that fails schema validation is retried once, then the attachment is flagged "could not extract"
  and surfaced to the rep with the raw transcription — never silently dropped; the fused query is
  non-empty whenever there is a message or a transcription.
- Safety note: a multimodal model can hallucinate text that isn't in the image, so the extracted
  facts are always shown to the rep next to the draft for correction, `low_confidence` is surfaced,
  and extracted document facts are treated as *unverified input*, never as authoritative KB grounding.

### 4.3 Triage (classification + urgency)
- Given ticket text (+ extracted facts), when triaged, then the agent produces a validated
  `category`, `urgency` (low/normal/high/critical), and `sentiment`.
- Acceptance: output conforms to the enum schema or the node retries; urgency is shown in the
  rep queue and used for sort order.

### 4.4 Knowledge retrieval (MCP connector)
- Given a triaged ticket, when the agent calls `search_knowledge_base` with the **fused search
  query** (§4.2 pass 3 — customer question + attachment summary), then it receives matching
  **sources ("chunks")** from the active provider. Each source carries an `id`, `title`, `text`,
  and a `source_kind` (`authoritative` for the mock-KB canned answers, or `model_generated` if a
  future/fallback provider has no real source).
- The tool contract already returns ranked chunks, so a future provider can do real embedding
  search behind the same interface with no change to the agent (the demo provider does keyword
  lookup; **no RAG/vectors in this project**).
- Acceptance: when the mock-KB provider finds no match, the tool returns "no confident source,"
  which routes the case to a **needs-human-research** flag rather than a drafted answer. A
  `model_generated` source never counts as authoritative grounding (see 4.5).

### 4.5 Grounded drafting
- Given `authoritative` sources, when the agent drafts, then the reply is written **only** from
  those sources and includes citations (which source `id`/`title` it used). gemma4 phrases the
  answer from the source; it is never the source itself.
- Acceptance: a groundedness check runs on the draft; low groundedness, or a draft built on a
  `model_generated` source, is flagged for the rep as **"AI-suggested, unverified"** with a
  warning banner and cannot be presented as sourced fact.

### 4.6 Guardrails & validation
- Every draft passes output guardrails before reaching the rep: no invented policies/facts, no
  financial/legal commitments (refund promises, guarantees) without human, no PII leakage, tone
  check.
- Prompt-injection checks run on incoming customer text and OCR output.
- Acceptance: a red-team ticket attempting prompt injection or eliciting a forbidden promise is
  blocked/flagged, proven by an eval test.

### 4.7 Human-in-the-loop review
- The workflow **pauses** (LangGraph interrupt) at the review node. The rep sees the original
  message, extracted facts, retrieved sources, the draft, and any warnings.
- The rep can **edit**, **approve**, **reject** (send back / mark needs-research), and set
  status. Only an explicit rep **send** action can move a case to `Resolved`.
- Acceptance: there is no code path that transitions a case to sent/resolved without a rep action.

### 4.8 Customer follow-up
- Given a resolved case, when the customer enters their reference code, then they see the status
  and the final reply. Unknown codes return a neutral "not found" (no enumeration leak).

### 4.9 Feedback loop & feedback-driven improvement
- Every rep decision is recorded: approved-as-is / edited (with diff between AI draft and final)
  / rejected, plus an optional rating and reason.
- Approved high-rated replies become curated few-shot exemplars for the drafting prompt (§4.10); rep
  corrections grow the evaluation golden set. A report summarizes where the agent underperforms
  (by category and groundedness).
- Acceptance: the feedback table captures draft, final, edit distance, and rating for every
  resolved case; the eval runner can consume the golden set.

### 4.9a Training-data capture (for future fine-tuning)
The project does **not** fine-tune the model at runtime, but it **captures a fine-tuning-ready
dataset from day one** so that, after months of use, the collected data can train an improved model.
- Every resolved case is written to an **append-only, de-identified training corpus**:
  - **SFT record:** input = `(customer message + extracted facts + cited sources)` → output =
    the **human-approved final reply**.
  - **Preference pair:** `AI original draft` (rejected) vs `rep-corrected final` (chosen) → usable
    for DPO/ORPO-style preference tuning.
  - Metadata: category, urgency, groundedness score, rep rating, model tag + prompt version.
- **PII is redacted before a record enters the corpus** (same redaction as logs/traces); records
  carry no raw account/card numbers, IDs, or sensitive attachment text.
- Exportable to **JSONL** via a `make export-training-data` script.
- Acceptance: each resolved case yields one SFT record (and a preference pair when the draft was
  edited); a unit test asserts exported records contain no configured PII patterns.

### 4.10 Dynamic prompting & prompt improvement
Prompts improve without code changes, through three levers:
- **Langfuse-managed prompts** — drafting/triage/extraction prompts are stored and **versioned in
  Langfuse** and fetched at runtime; a prompt can be edited in the UI and versions compared by score
  with no redeploy. A pinned fallback prompt ships in-repo for offline/CI runs.
- **Dynamic few-shot** — at draft time the agent injects the best recent **approved** replies for the
  ticket's category (from §4.9) as few-shot examples, so quality rises as reps work.
- **Eval-gated promotion** — a new prompt version must beat the eval suite (§10) before it becomes
  the active version; regressions are rejected.
- Acceptance: the drafting node resolves its prompt from Langfuse with an in-repo fallback (test with
  Langfuse stubbed); few-shot selection is deterministic and unit-tested; no prompt version is marked
  active unless it passes the eval gate.

---

## 5. AI Workflow (LangGraph)

State machine, persisted via a Postgres checkpointer (resumable across the human pause):

1. `ingest` — load ticket + attachment refs.
2. `ocr_extract` — vision transcription (thinking-trace stripped) → structured extraction → **fused
   search query** (customer question + attachment summary) → validate (§4.2).
3. `triage` — category + urgency + sentiment (validated).
4. `retrieve` — KB MCP connector `search_knowledge_base` called with the **fused query** (mock-KB
   provider returns ranked chunks).
5. `groundedness_gate` — authoritative source found? if not → `flag_needs_research`.
6. `draft` — grounded reply with citations; prompt resolved from Langfuse + dynamic few-shot (§4.10).
7. `validate` — schema + guardrails (PII, forbidden promises, tone, groundedness score). PII (Personally Identifiable Information).
8. `human_review` — **interrupt / pause** for the rep.
9. `finalize` — on approve/edit → save reply, status `Resolved`, record feedback + training record.

Case lifecycle statuses (customer-visible subset in **bold**): **New** → Triaged → Researching
→ Drafted(awaiting review) → **Pending** / **Resolved** / **Canceled** / NeedsResearch.

---

## 6. Security & Encryption Standards

- **AuthN/Z between components:** Streamlit→api and api→MCP servers require bearer/API tokens.
  MCP servers reject unauthenticated calls.
- **In transit:** TLS between services (or an internal-only network + documented TLS-termination
  path); no plaintext secrets on the wire. TLS = Transport Layer Security.
- **At rest:** sensitive PII extracted from documents (IDs, full account/card numbers, statement
  details) is encrypted at the application layer (authenticated encryption, e.g. AES-GCM/Fernet)
  before being stored; attachments stored encrypted.
- **Secrets:** never committed. `.env.example` documents required vars; real values via env /
  Docker secrets. Model tags, tokens, and keys are config, not code.
- **Logging:** structured logs with PII redaction (no full account numbers, IDs, or raw
  attachment text in logs). Full audit trail of who approved/sent each reply and when.
- **Input hardening:** request validation, size limits on uploads, allowed file types, rate
  limiting on the api, prompt-injection screening on customer/OCR text.
- **Least privilege:** only `email_mcp` holds ticket-table DB credentials.

---

## 7. Traceability, Observability & Quality Control

Two traceability layers, plus a layered quality-control path.

### 7.1 Business / audit traceability (compliance-grade)
- Every ticket carries an immutable audit trail in Postgres: customer submission, each workflow
  node's outcome, the source(s) cited, the model tag + prompt version used, guardrail decisions,
  rep edits, and the approve/send action — each with actor + timestamp.
- The reference code links customer ↔ ticket ↔ final reply, so a reviewer can answer "who sent
  this, when, and on what documented basis" for any case.

### 7.2 LLM / agent observability (Langfuse, self-hosted)
- **Self-hosted Langfuse** runs as a Docker service with its own datastore; no data leaves the
  machine — consistent with the local-Ollama posture.
- LangChain/LangGraph emit **one trace per ticket** via the Langfuse callback handler: every node,
  every model call (prompt, response, token counts, latency, cost), the vision/OCR pass, retrieval
  results, and guardrail outcomes — nested under a trace id stored on the ticket row.
- **Prompts are versioned** in Langfuse so any reply ties back to the exact prompt that produced it.
- Rep feedback (approved / edited / rejected + rating) and offline eval results attach to the trace
  as **scores**, closing the loop to §4.9 and giving quality trends over time.
- **PII redaction applies before data reaches Langfuse** (same redaction as logs) — traces must not
  carry full account/card numbers, IDs, or raw sensitive attachment text.

### 7.3 Quality control — the layered path to quality
Ordered gates every ticket passes; a failure at any gate flags or stops rather than proceeding
silently:
1. **Input gate** — validation + prompt-injection screening on customer text and OCR transcription.
2. **Structured-output validation** — every LLM output validated against Pydantic; retry on failure.
3. **Grounding gate** — no authoritative source → needs-human-research; model-generated → unverified.
4. **Output guardrails** — PII, forbidden financial/legal promises, tone, groundedness/faithfulness.
5. **Human gate** — rep approval mandatory (hard state-machine invariant; nothing sends without it).
6. **Observability gate** — every run traced in Langfuse; latency/token/cost + guardrail trips visible.
7. **Offline eval gate (CI)** — golden + red-team suites must pass; groundedness threshold enforced;
   regressions fail the build; scores logged to Langfuse for trend tracking.

### 7.4 Quality metrics (surfaced in Langfuse + rep dashboard)
Draft acceptance rate (approved-as-is %), edit-distance trend, groundedness-score distribution,
guardrail-trip rate, needs-research rate, low-confidence-OCR rate, average handle time, and
red-team block rate — the KPIs the feedback loop (§4.9) is measured against.

---

## 8. Project Structure

```
support-ticket-assistant/
├─ SPEC.md
├─ README.md
├─ docker-compose.yml
├─ .env.example
├─ Makefile
├─ pyproject.toml
├─ .github/workflows/ci.yml
├─ services/
│  ├─ api/                      # FastAPI backend + LangGraph orchestrator (MCP client)
│  │  ├─ app/
│  │  │  ├─ main.py
│  │  │  ├─ config.py           # pydantic-settings; model tag, tokens, urls
│  │  │  ├─ security.py         # auth deps, encryption helpers
│  │  │  ├─ schemas/            # Pydantic models (extraction, triage, draft, feedback)
│  │  │  ├─ ocr/                # vision.py (transcribe + strip thinking) + extract.py + fuse.py (search query)
│  │  │  ├─ graph/              # LangGraph: state.py, nodes/, workflow.py
│  │  │  ├─ guardrails/         # input/output guards, groundedness, PII
│  │  │  ├─ mcp_clients/        # kb + email MCP client wrappers
│  │  │  └─ routes/             # customer, rep, lookup, health
│  │  ├─ tests/
│  │  └─ Dockerfile
│  ├─ kb_mcp/                   # Knowledge-Base MCP connector (pluggable providers, no RAG)
│  │  ├─ server.py              # exposes search_knowledge_base
│  │  ├─ providers/            # provider interface + MockKBProvider (+ future API providers)
│  │  ├─ mock_kb/              # curated canned help answers (the "mock RAG" source data)
│  │  ├─ tests/
│  │  └─ Dockerfile
│  ├─ email_mcp/                # Email/Ticket MCP server (sole ticket-table owner)
│  │  ├─ server.py
│  │  ├─ db.py
│  │  ├─ migrations/
│  │  ├─ tests/
│  │  └─ Dockerfile
│  └─ frontend/                 # Streamlit
│     ├─ app.py
│     ├─ views/                 # customer, check_my_case, rep_workspace
│     ├─ api_client.py
│     ├─ tests/                 # api_client (mocked) + view helpers + AppTest headless UI tests
│     └─ Dockerfile
├─ data/                        # sample attachments / seed tickets
├─ evals/                       # golden datasets + eval runner (triage, groundedness, red-team)
└─ scripts/                     # seed, migrate, export-training-data helpers
```

**Container images:** 4 app services have their own Dockerfile (`api`, `kb_mcp`, `email_mcp`,
`frontend`); `postgres` and `langfuse` use official images. Each MCP server stays a separate
authenticated service, and `email_mcp` is the only container with ticket-DB credentials
(least-privilege). Every service directory has its own `tests/` — this project is built **TDD-first**
(§10).

---

## 9. Code Style

- **Python 3.12**, full type hints, **Pydantic v2** for every LLM structured output and API
  boundary. No unvalidated LLM JSON reaches business logic.
- **Async** for I/O-bound paths (FastAPI routes, MCP clients, Ollama calls).
- **Formatting/lint/types:** `black`, `ruff`, `mypy` (strict where practical).
- **Config:** 12-factor via `pydantic-settings`; no hard-coded urls/tokens/model tags.
- **Logging:** `structlog` with a PII-redaction processor; structured, no secrets.
- Small, single-responsibility modules; docstrings on public functions; clear node boundaries
  in the LangGraph workflow.
- **TDD-first:** write a failing test before the implementation for every unit of logic, including
  frontend logic. Keep logic out of hard-to-test surfaces (thin Streamlit views / FastAPI routes;
  put real logic in pure, unit-testable helpers).

---

## 10. Testing Strategy

**The project is built TDD-first: a failing test is written before each unit of implementation**
(§9). Every service directory carries its own `tests/`.

- **Unit** (`pytest`): extraction validators, triage parsers, fused-query builder, guardrail
  functions, groundedness scorer, encryption/redaction utils, reference-code generation/lookup,
  training-record builder + PII-redaction, dynamic few-shot selector.
- **Frontend** (`services/frontend/tests/`): `api_client.py` against mocked HTTP; pure view/format
  helpers; and the app driven headlessly via Streamlit's `AppTest` harness (submit a ticket, look up
  a reference code, approve a draft) asserting on rendered state.
- **MCP contract tests:** each MCP tool returns schema-valid output for representative inputs.
- **Workflow integration:** run the LangGraph flow against a **deterministic fake LLM** (stubbed
  Ollama) so CI does not depend on the real model; verify the human-review pause and that no path
  reaches "sent" without an approval action.
- **AI evals** (`evals/`): golden ticket set with expected category/urgency; groundedness
  assertions; **red-team** cases (prompt injection, PII-leak attempts, forbidden-promise
  elicitation) that MUST be blocked/flagged. Also gate prompt-version promotion (§4.10).
- **Safety invariant test:** assert there is no code path that sends to a customer without a rep
  action, and that "no confident source" never produces a customer-facing answer.
- **Training-corpus test:** exported records contain no configured PII patterns.
- No test ever performs a real outbound send.

> Per project working agreement, **the user runs all tests, the app, Docker, and Ollama.**
> This spec documents the commands; the assistant writes/edits code only and does not execute them.

---

## 11. Commands (for the user to run)

```bash
# Prereqs (host): Ollama running with models pulled
ollama pull gemma4:12b                       # single multimodal model: vision/OCR + all text
# Tag is set in .env as LLM_MODEL=gemma4:12b and can be changed later.

# Build & run the stack
docker compose build
docker compose up            # postgres, langfuse, email_mcp, kb_mcp, api, frontend

# Configuration (.env, copied from .env.example) includes at least:
#   LLM_MODEL=gemma4:12b          # single multimodal model tag; change to swap models
#   LANGFUSE_HOST / LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY   # tracing
#   plus API tokens + the app-level encryption key for PII-at-rest

# One-time / maintenance
make migrate                 # apply DB migrations (via email_mcp)
make seed                    # load mock-KB answers + sample tickets
make export-training-data    # export de-identified SFT + preference JSONL for future fine-tuning

# Quality gates
make lint                    # ruff
make format                  # black
make typecheck               # mypy
make test                    # pytest (fake-LLM, no real model needed)
make eval                    # run AI eval + red-team suite
make security                # bandit + pip-audit + gitleaks + image scan
```

(Exact Make targets/compose service names are finalized during implementation, kept consistent
with this list.)

---

## 12. CI/CD (AI application pipeline)

GitHub Actions on push/PR:
1. **Lint/format/type** — ruff, black --check, mypy.
2. **Unit + contract + workflow tests** — pytest with the fake LLM (no model download in CI).
3. **AI eval gate** — run `evals/` golden + red-team suite; fail the build if guardrail/red-team
   cases regress or groundedness drops below threshold.
4. **Security scan** — bandit (SAST), pip-audit (deps), gitleaks (secrets), image scan (e.g.
   trivy) on built images.
5. **Build** — build all service images.
6. Optional gated **deploy/publish** stage (manual approval) — never auto-deploys a change that
   fails the eval or security gates.

---

## 13. Boundaries

**Always**
- Require an explicit human rep action before any reply is marked sent / a case Resolved.
- Ground every draft in retrieved KB passages and include citations; if no confident source,
  flag for human research instead of answering.
- Validate every LLM structured output against a Pydantic schema.
- Encrypt PII (from OCR/attachments) at rest; redact PII in logs; keep a send/approval audit trail.
- Emit a PII-redacted Langfuse trace + a Postgres audit record for every ticket (§7).
- Write a failing test before the implementation (TDD-first), for backend **and** frontend logic.
- De-identify every training-corpus record before it is stored (§4.9a).
- Point all model calls at the host Ollama via `host.docker.internal`.
- Assistant **writes/edits code only**; the user runs tests, the app, Docker, and Ollama.

**Ask first**
- Any decision not already captured in this SPEC.md (no silent defaults).
- Adding a new dependency, service, or external integration.
- Changing the DB schema, the model tag, or the workflow node structure.
- Enabling real outbound email (SMTP) or any real-send behavior.
- Anything that would let content reach a customer without rep approval.

**Never**
- Send anything to a real customer automatically or auto-approve a draft.
- Invent policies, numbers, or facts not present in the knowledge base.
- Store secrets in the repo or log full PII (IDs, full account/card numbers, raw attachment text)
  — in logs **or in Langfuse traces**.
- Run the app, tests, Docker, or Ollama on the user's behalf.

---

## 14. Resolved Decisions (previously open)

All four items flagged during the spec conversation are now decided:

1. **KB approach — DECIDED:** the KB MCP is a **pluggable connector**; the demo provider is a
   **curated list of canned help answers ("mock RAG")** returned as citable sources; gemma4 only
   phrases the reply from the matched source. **No RAG / no vectors** in this project — real
   retrieval is deliberately deferred to a future project behind the same provider interface.
2. **KB content — DECIDED:** I provide the curated mock-KB answer set in `services/kb_mcp/mock_kb/`.
3. **Reference-code format — CONFIRMED:** `TKT-####` (zero-padded sequence).
4. **api layer — CONFIRMED:** a FastAPI backend sits between Streamlit and the MCP servers, giving
   security/guardrails/orchestration one home.
5. **Model — CONFIRMED (2026-07-06):** single multimodal `gemma4:12b` for vision/OCR **and** all
   text; tag lives in `.env` as `LLM_MODEL`. No separate OCR model.
6. **Traceability — CONFIRMED (2026-07-06):** self-hosted **Langfuse** is the tracing/observability
   backend (§7), running as one Docker service with its own datastore; all data stays local.
7. **Container topology — CONFIRMED (2026-07-06):** 4 app services; MCP servers stay separate
   authenticated services; `email_mcp` is the sole DB-credential holder.
8. **KB retrieval — CONFIRMED (2026-07-06):** keep the mock KB (no RAG); adopt the fused
   question+attachment-summary query (§4.2/§4.4). Real embedding search remains a future provider.
9. **TDD + frontend tests — CONFIRMED (2026-07-06):** TDD-first for all logic, including frontend.
10. **Fine-tuning data — CONFIRMED (2026-07-06):** capture a de-identified SFT + preference corpus
    now (§4.9a) for future fine-tuning; the model itself is still not fine-tuned by this project.
11. **Dynamic prompting — CONFIRMED (2026-07-06):** Langfuse-versioned prompts + dynamic few-shot +
    eval-gated promotion (§4.10).

---

## 15. Out of Scope (this version)

- Real outbound email (SMTP) — simulated send only.
- Live mailbox (IMAP) intake — intake is via the customer Streamlit view.
- **Running** a fine-tune of `gemma4:12b` — this project *collects* a fine-tuning-ready corpus
  (§4.9a) and improves quality via dynamic prompting + eval-driven iteration (§4.10), but does not
  perform weight updates itself. Actual fine-tuning is a later step, fed by the exported corpus.
- **RAG / vector retrieval / embeddings** — the KB MCP uses a mock provider; real embedding search is
  a future provider that plugs into the same connector interface (the fused query already targets it).
- Multi-tenant / SSO / production identity provider.

---

## Appendix A — `gemma4:12b` vision behaviour (verified)

The user tested `gemma4:12b` on a screenshot. Key observations that shape §4.2:
- The model **reads the image directly** and answers in natural language (it correctly identified a
  Langfuse "Organizations" onboarding page from the screenshot).
- In "thinking" mode it first emits a **reasoning trace** (`Thinking... ...done thinking.`) before the
  final answer. The OCR pass must request text-only output and **strip everything except the final
  answer**.
- Left to its own prompt it *describes/interprets* the image rather than transcribing verbatim, so the
  OCR prompt must explicitly ask for a verbatim transcription of visible text, not a description.

Implication: the digitization module needs (a) a prompt that forces verbatim, description-free output,
and (b) a parser that removes any `thinking` block / narration. A unit test feeds a stubbed model
response containing a `thinking` block plus a final answer and asserts only the final transcription
survives.

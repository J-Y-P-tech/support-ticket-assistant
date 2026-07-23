"""Shared state for the LangGraph workflow (plan Task 16 / todo Task 17).

The single typed channel the workflow's nodes read and write as a ticket moves
`screen_input → triage → retrieve → groundedness_gate → draft → validate →
screen_output → human_review → finalize` (SPEC §5). Every field is optional
(`total=False`): a node fills in only the slice it produces, LangGraph merges that
partial into the running state, and the checkpointer persists the whole thing so the
case survives the human pause (and a process restart) and resumes days later.

Two conventions matter:

- **`flags` accumulates.** It carries the plain-language warnings the rep sees at
  review (an injection block, a no-source hand-off, a low-groundedness draft, a
  leaked reasoning trace). Several nodes contribute, so it uses an additive reducer
  (`operator.add`): each node returns only its *new* flags and LangGraph concatenates
  them, rather than a later node clobbering an earlier one's warning.
- **The rep-decision fields are written during the pause, not by the pipeline.**
  `rep_decision` / `rep_edited_reply` are supplied out of band (the rep-action routes,
  plan Task 17) while the graph is interrupted before `human_review`; `finalize` reads
  them. They start absent, which is exactly what lets `finalize` fail closed — no
  decision means no send (SPEC §10 safety invariant).
"""

from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from app.schemas.draft import Draft
from app.schemas.enums import FeedbackDecision, TicketStatus
from app.schemas.guardrails import InjectionScreenResult, OutputScreenResult
from app.schemas.kb import KBSearchResult
from app.schemas.triage import TriageResult
from app.schemas.validation import ValidationResult


class WorkflowState(TypedDict, total=False):
    """The typed state channel threaded through every workflow node (SPEC §5).

    Grouped by lifecycle stage:

    - **Intake** — `ticket_id`, `reference_code` (the customer-facing `TKT-####` code),
      `message`, and `attachments` are set at invocation; `extracted_facts` (the OCR
      digest, `None` for text-only tickets) and `search_query` (the fused KB query —
      message + attachment summary) are produced by the `ocr_extract` node before triage
      and retrieval.
    - **Pipeline products** — `injection_screen`, `triage`, `kb_result`, `draft`,
      `validation`, and `output_screen` are each written by the node that owns them,
      so downstream nodes and the rep see validated artefacts, never raw model output.
    - **Human-facing signals** — `status` is the case lifecycle status (SPEC §5);
      `trace_leak` marks a draft that leaked a reasoning trace; `flags` accumulates the
      review warnings (additive reducer).
    - **Rep decision** — `rep_decision` / `rep_edited_reply` are injected during the
      pause; `finalize` turns them into `final_reply` and the terminal status. Absent
      until the rep acts, which is what makes the human gate unbypassable.
    """

    # --- Intake (set at invocation) ---
    ticket_id: int
    reference_code: str
    message: str
    attachments: list[str]
    extracted_facts: str | None
    search_query: str

    # --- Pipeline products (each written by its owning node) ---
    injection_screen: InjectionScreenResult
    triage: TriageResult
    kb_result: KBSearchResult
    draft: Draft
    validation: ValidationResult
    output_screen: OutputScreenResult

    # --- Human-facing signals ---
    status: TicketStatus
    trace_leak: bool
    flags: Annotated[list[str], operator.add]

    # --- Rep decision (supplied during the pause; read by finalize) ---
    rep_decision: FeedbackDecision
    rep_edited_reply: str
    final_reply: str | None

"""Turn a finished workflow run into its PII-redacted Langfuse trace (plan Task 29 / todo Task 31).

SPEC §7.2 wants **one trace per ticket**: every node, the model calls, retrieval, and
guardrail outcomes, nested under a trace id stored on the ticket row, with rep feedback
and the reasoning-trace-leak signal attached as **scores** — and no account/card/ID
number or raw attachment text ever reaching Langfuse.

This module is the *emission* half, mirroring `graph/audit.py` and `graph/feedback.py`:

- `build_trace` is **pure** — a finished (paused) LangGraph state in, a redacted
  `TicketTrace` out, no I/O — so the "no PII in the payload" guarantee is provable
  deterministically without standing up Langfuse. Its spans are built straight from
  `build_audit_entries`, so the trace and the compliance trail can never drift on which
  nodes ran or what they recorded; the only text a span carries is the *redacted* draft
  response, scrubbed with the same `redact_pii` the logs use (SPEC §6/§7.2 single
  redaction source).
- `build_scores` maps a state (plus the rep's rating) to the numeric `TraceScore`s
  Langfuse trends (SPEC §7.4): groundedness and the trace-leak flag at the human-review
  pause, then — once the rep has acted — whether the draft was accepted, the AI-vs-final
  edit distance (reusing the feedback builder), and the rep's rating.
- `Tracer` is the seam: a two-method protocol (`emit` a trace, `add_scores` to one) the
  submit-time trigger and the rep routes depend on. `NoOpTracer` is the offline/CI
  fallback — it emits nothing and returns an empty trace id, so a run without Langfuse
  configured traces to nothing and persists nothing. The real Langfuse adapter lives in
  `app/observability/langfuse_tracer.py` and is wired in `graph/runtime.py`.
- `emit_ticket_trace` / `attach_scores` are the thin write steps: the first builds and
  emits the trace and persists the returned id on the ticket (via email_mcp); the second
  reads that id back and attaches the disposition scores. Persistence lives at the
  service boundary, never inside a graph node — exactly as audit and feedback do.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.graph.audit import build_audit_entries
from app.graph.feedback import build_feedback_record
from app.logging_config import redact_pii
from app.schemas.enums import FeedbackDecision

# The trace name every ticket's run is recorded under in Langfuse — a stable label so
# every ticket's story sits under the same trace kind in the dashboard.
_TRACE_NAME = "ticket-run"


@dataclass(frozen=True)
class TraceSpan:
    """One observation on a ticket's trace: a node outcome, optionally with a response.

    `name` is the node-outcome name (`"triaged"`, `"drafted"`, ...) shared with the
    audit trail; `detail` is the same PII-safe evidence the audit row carries (ids,
    titles, categories, model + prompt version, guardrail decisions); `output` is the
    model's *redacted* response text, set only on the drafted span. Frozen: a built span
    is a fact about a run that happened.
    """

    name: str
    detail: dict[str, Any] | None = None
    output: str | None = None


@dataclass(frozen=True)
class TraceScore:
    """One numeric quality score attached to a ticket's trace (SPEC §7.4).

    `name` is the metric (`"groundedness"`, `"trace_leak"`, `"draft_accepted"`,
    `"edit_distance"`, `"rating"`); `value` is its float value; `comment` is an optional
    note. Numeric so Langfuse can trend it over time and per model tag.
    """

    name: str
    value: float
    comment: str | None = None


@dataclass(frozen=True)
class TicketTrace:
    """A ticket's whole run as one PII-redacted trace destined for Langfuse (SPEC §7.2).

    `ticket_id` keys it to the case; `input` is the *redacted* customer message; `spans`
    are the ordered node outcomes; `scores` are the metrics known at build time;
    `metadata` names the host model. Nothing here carries a full account/card number or
    raw attachment text — the builder scrubs before it constructs the trace.
    """

    ticket_id: int
    input: str
    spans: list[TraceSpan]
    scores: list[TraceScore] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    name: str = _TRACE_NAME

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable view of the trace for the tracer and PII scans.

        The Langfuse adapter sends this shape, and the redaction tests scan its
        serialization end to end — so a single representation is both what ships and
        what is asserted PII-free.
        """
        return {
            "name": self.name,
            "ticket_id": self.ticket_id,
            "input": self.input,
            "metadata": self.metadata,
            "spans": [
                {"name": span.name, "detail": span.detail, "output": span.output}
                for span in self.spans
            ],
            "scores": [
                {"name": score.name, "value": score.value, "comment": score.comment}
                for score in self.scores
            ],
        }


class Tracer(Protocol):
    """The two-method seam the trace write steps depend on (SPEC §7.2).

    Typed structurally so the submit-time trigger and the rep routes depend only on
    `emit`/`add_scores`, not on the Langfuse SDK — the real adapter and the tests' fakes
    both satisfy it without inheritance.
    """

    async def emit(self, trace: TicketTrace) -> str:
        """Send a ticket's trace to Langfuse and return its trace id (empty if untraced)."""
        ...

    async def add_scores(self, trace_id: str, scores: list[TraceScore]) -> None:
        """Attach quality scores to an already-emitted trace by its id."""
        ...


class NoOpTracer:
    """The offline/CI tracer: emits nothing and scores nothing (SPEC §10/§12).

    A run without Langfuse configured must still work end to end, so this fallback
    accepts a trace and returns an empty id — the write step then persists nothing and
    the desk drafts exactly as before. Matches the `Tracer` protocol.
    """

    async def emit(self, trace: TicketTrace) -> str:
        """Discard the trace and report an empty id (nothing was traced)."""
        return ""

    async def add_scores(self, trace_id: str, scores: list[TraceScore]) -> None:
        """Discard the scores (nothing was traced to attach them to)."""
        return None


class _TraceIdWriter(Protocol):
    """The one method `emit_ticket_trace` needs from the email_mcp client."""

    async def set_trace_id(self, ticket_id: int, trace_id: str) -> dict[str, Any] | None:
        """Persist a ticket's Langfuse trace id and return the updated row (or None)."""
        ...


class _TicketReader(Protocol):
    """The one method `attach_scores` needs from the email_mcp client."""

    async def get_ticket(self, ticket_id: int) -> dict[str, Any] | None:
        """Return a ticket row (carrying its `trace_id`), or None if unknown."""
        ...


def build_trace(state: Mapping[str, Any], *, model: str) -> TicketTrace:
    """Map a finished workflow state to its PII-redacted trace (SPEC §7.2).

    Reuses `build_audit_entries` for the ordered node spans, so the trace records the
    exact same branch of the pipeline the compliance trail does (a blocked injection, a
    triage failure, and a no-source hand-off each stop at the same point, with no later
    spans invented). The customer message becomes the trace `input`, number-scrubbed;
    the drafted span additionally carries the model's response body, also scrubbed — the
    only free text on the trace, so no account/card number reaches Langfuse. Pure: no
    I/O, mutates nothing.
    """
    draft = state.get("draft")
    spans: list[TraceSpan] = []
    for entry in build_audit_entries(state, model=model):
        # The drafted span carries the model's response text — redacted, like the logs.
        output = redact_pii(draft.body) if entry.event == "drafted" and draft is not None else None
        spans.append(TraceSpan(name=entry.event, detail=entry.detail, output=output))
    return TicketTrace(
        ticket_id=state["ticket_id"],
        input=redact_pii(state.get("message", "") or ""),
        spans=spans,
        scores=build_scores(state),
        metadata={"model": model},
    )


def build_scores(state: Mapping[str, Any], *, rating: int | None = None) -> list[TraceScore]:
    """Map a state (plus the rep's rating) to the trace's numeric quality scores (SPEC §7.4).

    Returns every score derivable from what is in `state`, so it works at both call
    sites. At the human-review pause the validation groundedness and the boolean
    reasoning-trace-leak flag (0.0 clean / 1.0 leaked) are known. Once the rep has acted
    (`rep_decision` present) the disposition scores attach too: `draft_accepted` (1.0
    only for an unedited approval), the AI-vs-final `edit_distance` when there is a final
    reply, and the rep's `rating` when given. Pure: no I/O, mutates nothing.
    """
    scores: list[TraceScore] = []

    validation = state.get("validation")
    if validation is not None:
        scores.append(TraceScore("groundedness", float(validation.groundedness)))
    # `trace_leak` is set by the draft node; absent on branches that never drafted.
    if "trace_leak" in state:
        scores.append(TraceScore("trace_leak", 1.0 if state["trace_leak"] else 0.0))

    decision = state.get("rep_decision")
    if decision is not None:
        accepted = 1.0 if decision == FeedbackDecision.APPROVED_AS_IS else 0.0
        scores.append(TraceScore("draft_accepted", accepted))
        # Reuse the feedback builder so the edit distance on the trace and in the
        # feedback table are the identical measure (zero for approved-as-is, absent for
        # a rejection with no final reply).
        record = build_feedback_record(state, rating=rating)
        if record is not None and record.edit_distance is not None:
            scores.append(TraceScore("edit_distance", float(record.edit_distance)))

    if rating is not None:
        scores.append(TraceScore("rating", float(rating)))

    return scores


async def emit_ticket_trace(
    tracer: Tracer,
    email: _TraceIdWriter,
    *,
    ticket_id: int,
    state: Mapping[str, Any],
    model: str,
) -> str:
    """Emit a finished run's trace and persist its id on the ticket (SPEC §7.2).

    The thin write step behind the submit-time pipeline trigger: it builds the trace for
    the paused run, emits it, and — when the tracer returned a real id (not the NoOp
    fallback's empty string) — stores that id on the ticket row via email_mcp, so a rep
    or auditor can jump from the case to its Langfuse trace. Returns the trace id (empty
    when untraced).
    """
    trace_id = await tracer.emit(build_trace(state, model=model))
    if trace_id:
        await email.set_trace_id(ticket_id, trace_id)
    return trace_id


async def attach_scores(
    tracer: Tracer,
    email: _TicketReader,
    *,
    ticket_id: int,
    state: Mapping[str, Any],
    rating: int | None = None,
) -> None:
    """Attach a finished run's quality scores to its Langfuse trace (SPEC §7.2/§7.4).

    The thin write step behind the rep-action routes: it reads the trace id persisted on
    the ticket and, when the case was traced, attaches the disposition scores under it
    (draft accepted, edit distance, rating). A ticket that was never traced (an offline
    run) has no id, so nothing is attached — the send/reject still succeeds.
    """
    ticket = await email.get_ticket(ticket_id)
    trace_id = ticket.get("trace_id") if ticket else None
    if not trace_id:
        return
    scores = build_scores(state, rating=rating)
    if scores:
        await tracer.add_scores(trace_id, scores)

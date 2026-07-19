"""Unit tests for the Langfuse trace-emission layer (plan Task 29 / todo Task 31).

SPEC §7.2 wants **one PII-redacted trace per ticket**: every node, the model calls,
retrieval, and guardrail outcomes, nested under a trace id stored on the ticket, with
rep feedback and the reasoning-trace-leak signal attached as **scores** — and no
account/card/ID number or raw attachment text ever reaching Langfuse.

This suite covers the seam, which mirrors `graph/audit.py` and `graph/feedback.py`:

- `build_trace` is a **pure** function — a finished (paused) LangGraph state in, a
  redacted `TicketTrace` out, no I/O — so the "no PII in the payload" guarantee is
  provable deterministically without standing up Langfuse. It reuses the audit layer's
  branch logic so the trace's spans and the compliance trail never drift.
- `build_scores` maps a state (plus the rep's rating) to the numeric `TraceScore`s
  Langfuse trends: groundedness, the trace-leak flag, and — once the rep has acted —
  whether the draft was accepted, the edit distance, and the rating.
- `NoOpTracer` is the offline/CI fallback: it emits nothing and returns an empty trace
  id, so a run without Langfuse configured traces to nothing and persists nothing.
- `emit_ticket_trace` / `attach_scores` are the thin write steps the submit-time
  pipeline trigger and the rep-action routes call; their tests prove they push the
  trace, persist its id, and attach the scores through injected fakes — no network.
"""

from __future__ import annotations

import json
from typing import Any

from app.schemas.draft import Citation, Draft
from app.schemas.enums import Category, FeedbackDecision, Sentiment, Urgency
from app.schemas.guardrails import InjectionScreenResult, OutputScreenResult
from app.schemas.kb import KBSearchResult, KBSource
from app.schemas.triage import TriageResult
from app.schemas.validation import ValidationResult

# The single host model tag (SPEC §4.2); the trace metadata must carry exactly this.
_MODEL = "gemma4:12b"

# A 16-digit card number and a long account number planted in customer/model text. The
# redaction assertions scan the whole serialized trace for these; neither may survive.
_CARD = "4111 1111 1111 1111"
_ACCOUNT = "1234567890123"


def _clean_screen() -> InjectionScreenResult:
    """Build the input-guard verdict for a benign message: nothing caught, nothing flagged."""
    return InjectionScreenResult(flagged=False, detector="none")


def _triage_ok() -> TriageResult:
    """Build a valid triage classification (account-access, normal, neutral)."""
    return TriageResult(
        category=Category.ACCOUNT_ACCESS,
        urgency=Urgency.NORMAL,
        sentiment=Sentiment.NEUTRAL,
    )


def _confident_kb() -> KBSearchResult:
    """Build a KB result with one confident, citable source (id `KB-1`)."""
    return KBSearchResult(
        sources=[KBSource(id="KB-1", title="Password reset", text="Use the login screen.")],
        no_confident_source=False,
    )


_DRAFT_BODY = "You can reset your password from the login screen. [KB-1]"


def _verified_draft(body: str = _DRAFT_BODY) -> Draft:
    """Build a grounded, verified draft that cites the KB-1 source (body overridable)."""
    return Draft(
        body=body,
        citations=[Citation(source_id="KB-1", title="Password reset")],
        verified=True,
    )


def _validation_ok(draft: Draft | None = None) -> ValidationResult:
    """Build a validate-node result: fully grounded, not flagged."""
    return ValidationResult(draft=draft or _verified_draft(), groundedness=1.0, flagged=False)


def _clean_output_screen() -> OutputScreenResult:
    """Build the output-guard verdict for a clean draft: nothing caught, nothing flagged."""
    return OutputScreenResult(flagged=False, detector="none")


def _happy_state() -> dict[str, Any]:
    """Build a finished workflow state for a clean text-only happy-path run.

    Mirrors the state `build_trace` sees at the human-review pause: every pipeline
    product present, no rep decision yet (the rep acts later).
    """
    return {
        "ticket_id": 1,
        "message": "How do I reset my online banking password?",
        "extracted_facts": None,
        "injection_screen": _clean_screen(),
        "triage": _triage_ok(),
        "kb_result": _confident_kb(),
        "draft": _verified_draft(),
        "validation": _validation_ok(),
        "output_screen": _clean_output_screen(),
        "trace_leak": False,
    }


def _span_names(trace: Any) -> list[str]:
    """Pull just the span names out of a built trace, in order."""
    return [span.name for span in trace.spans]


def _span(trace: Any, name: str) -> Any:
    """Return the single span with the given name (fails if absent)."""
    return next(span for span in trace.spans if span.name == name)


def _score(scores: list[Any], name: str) -> Any:
    """Return the single score with the given name (fails if absent)."""
    return next(score for score in scores if score.name == name)


# --- build_trace: shape and ordering -----------------------------------------------


def test_build_trace_carries_ticket_id_model_and_ordered_spans() -> None:
    """A clean run builds one trace with the ticket id, model tag, and node spans in order.

    The trace is the run's story: its spans follow pipeline order (input screen →
    triage → retrieval → draft → validation → output screen), it is keyed to the
    ticket, and its metadata names the host model (SPEC §7.2).
    """
    from app.graph.trace import build_trace

    trace = build_trace(_happy_state(), model=_MODEL)

    assert trace.ticket_id == 1
    assert trace.metadata["model"] == _MODEL
    assert _span_names(trace) == [
        "input_screened",
        "triaged",
        "retrieved",
        "drafted",
        "validated",
        "output_screened",
    ]


def test_build_trace_drafted_span_carries_the_redacted_response_body() -> None:
    """The drafted span records the model's response text, PII-redacted (SPEC §7.2).

    A trace captures the model call's response; the draft body is the response, so it
    rides on the drafted span — but scrubbed, so an account number the reply echoed
    never reaches Langfuse.
    """
    from app.graph.trace import build_trace

    state = _happy_state()
    state["draft"] = _verified_draft(body=f"Your account {_ACCOUNT} is now unlocked. [KB-1]")

    trace = build_trace(state, model=_MODEL)

    drafted = _span(trace, "drafted")
    assert drafted.output is not None
    assert _ACCOUNT not in drafted.output
    assert "[REDACTED]" in drafted.output


def test_build_trace_follows_the_no_confident_source_branch() -> None:
    """A no-source hand-off traces the screen, triage, and hand-off — and no draft span.

    The trace reuses the audit branch logic, so a run that never drafted has no drafted
    span invented for it — the trace reads exactly as the case ran.
    """
    from app.graph.trace import build_trace

    state = {
        "ticket_id": 4,
        "message": "obscure question with no answer",
        "injection_screen": _clean_screen(),
        "triage": _triage_ok(),
        "kb_result": KBSearchResult(sources=[], no_confident_source=True),
    }

    trace = build_trace(state, model=_MODEL)

    assert _span_names(trace) == ["input_screened", "triaged", "no_confident_source"]


# --- build_trace: PII redaction (the graded guarantee) -----------------------------


def test_build_trace_redacts_pii_from_the_customer_message() -> None:
    """The trace input is the customer message, number-scrubbed before it reaches Langfuse.

    SPEC §7.2: a trace must not carry a full card/account number. The customer message
    is the trace's input, so any such number in it is redacted first.
    """
    from app.graph.trace import build_trace

    state = _happy_state()
    state["message"] = f"My card {_CARD} was declined, please help"

    trace = build_trace(state, model=_MODEL)

    assert _CARD not in trace.input
    assert "[REDACTED]" in trace.input


def test_build_trace_carries_no_pii_anywhere_in_the_payload() -> None:
    """No account/card number appears anywhere in the serialized trace (SPEC §7.2).

    The strongest form of the guarantee: with PII planted in both the customer message
    and the drafted reply, scanning the whole serialized trace payload (input, every
    span, every score) finds neither number — the single redaction guarantee logs and
    traces share holds end to end.
    """
    from app.graph.trace import build_trace

    state = _happy_state()
    state["message"] = f"My card {_CARD} was declined"
    state["draft"] = _verified_draft(body=f"Your account {_ACCOUNT} is unlocked. [KB-1]")

    trace = build_trace(state, model=_MODEL)

    serialized = json.dumps(trace.as_dict())
    assert _CARD not in serialized
    assert _ACCOUNT not in serialized
    # The digit runs are gone; the redaction marker is what remains in their place.
    assert "[REDACTED]" in serialized


def test_build_trace_does_not_carry_raw_attachment_text() -> None:
    """A digitized attachment adds an extraction span but never its raw transcribed text.

    Like the audit trail, the trace records *that* a document was digitized (model +
    prompt) but not its raw text, so sensitive attachment content never reaches Langfuse
    (SPEC §7.2).
    """
    from app.graph.trace import build_trace

    state = _happy_state()
    # A digest that itself contains a sensitive number, to prove nothing text-like leaks.
    state["extracted_facts"] = f"Document: bank statement\nAccount: {_ACCOUNT}"

    trace = build_trace(state, model=_MODEL)

    extracted = _span(trace, "attachment_extracted")
    assert _ACCOUNT not in json.dumps(extracted.detail)
    assert _ACCOUNT not in json.dumps(trace.as_dict())


# --- build_scores ------------------------------------------------------------------


def test_build_scores_at_the_pause_carries_groundedness_and_trace_leak() -> None:
    """At the human-review pause the trace scores are groundedness and the trace-leak flag.

    Before the rep acts, the derivable scores are the validation groundedness and the
    boolean reasoning-trace-leak signal (0.0 clean / 1.0 leaked) — the leak trend the
    Task 10 note asks Langfuse to surface per model tag.
    """
    from app.graph.trace import build_scores

    scores = build_scores(_happy_state())

    assert _score(scores, "groundedness").value == 1.0
    assert _score(scores, "trace_leak").value == 0.0
    # No rep decision yet, so no acceptance/edit-distance/rating score.
    assert {score.name for score in scores} == {"groundedness", "trace_leak"}


def test_build_scores_flags_a_leaked_reasoning_trace() -> None:
    """A drafted reply that leaked a reasoning trace scores trace_leak = 1.0."""
    from app.graph.trace import build_scores

    state = _happy_state()
    state["trace_leak"] = True

    assert _score(build_scores(state), "trace_leak").value == 1.0


def test_build_scores_after_approval_records_acceptance_and_rating() -> None:
    """An approved-as-is send scores draft_accepted = 1.0, distance 0, and the rating.

    Once the rep sends an unchanged draft, the feedback scores attach: the draft was
    accepted (1.0), the edit distance from AI draft to final reply is zero, and the
    rep's rating rides along (SPEC §7.2 rep feedback as scores).
    """
    from app.graph.trace import build_scores

    state = _happy_state()
    state["rep_decision"] = FeedbackDecision.APPROVED_AS_IS
    state["final_reply"] = state["draft"].body

    scores = build_scores(state, rating=5)

    assert _score(scores, "draft_accepted").value == 1.0
    assert _score(scores, "edit_distance").value == 0.0
    assert _score(scores, "rating").value == 5.0


def test_build_scores_after_edit_records_non_acceptance_and_distance() -> None:
    """An edited send scores draft_accepted = 0.0 and the non-zero edit distance.

    An edited reply was not accepted as-is, so draft_accepted is 0.0, and the edit
    distance is the character diff between the AI draft and the rep's final reply.
    """
    from app.graph.feedback import edit_distance
    from app.graph.trace import build_scores

    state = _happy_state()
    ai = state["draft"].body
    final = ai + " Let me know if that helps."
    state["rep_decision"] = FeedbackDecision.EDITED
    state["final_reply"] = final

    scores = build_scores(state)

    assert _score(scores, "draft_accepted").value == 0.0
    assert _score(scores, "edit_distance").value == float(edit_distance(ai, final))


def test_build_scores_after_rejection_records_non_acceptance_without_distance() -> None:
    """A rejected draft scores draft_accepted = 0.0 with no edit distance (no final reply)."""
    from app.graph.trace import build_scores

    state = _happy_state()
    state["rep_decision"] = FeedbackDecision.REJECTED
    state["final_reply"] = None

    scores = build_scores(state)

    assert _score(scores, "draft_accepted").value == 0.0
    assert "edit_distance" not in {score.name for score in scores}


# --- NoOpTracer --------------------------------------------------------------------


async def test_noop_tracer_emits_nothing_and_returns_an_empty_trace_id() -> None:
    """The offline fallback emits no trace and returns an empty id, so nothing persists."""
    from app.graph.trace import NoOpTracer, build_trace

    tracer = NoOpTracer()

    trace_id = await tracer.emit(build_trace(_happy_state(), model=_MODEL))

    assert trace_id == ""
    # add_scores is a no-op that never raises.
    await tracer.add_scores("anything", [])


# --- emit_ticket_trace / attach_scores: the thin write steps -----------------------


class _RecordingTracer:
    """A tracer fake that records the emitted trace and any attached scores.

    Returns a fixed non-empty trace id from `emit`, so the wiring's persistence branch
    is exercised without a real Langfuse — the tests read `emitted` / `scored` back.
    """

    def __init__(self, trace_id: str = "trace-abc") -> None:
        """Start with nothing emitted; hand back `trace_id` from every `emit`."""
        self._trace_id = trace_id
        self.emitted: list[Any] = []
        self.scored: list[tuple[str, list[Any]]] = []

    async def emit(self, trace: Any) -> str:
        """Record the trace and return the fixed trace id."""
        self.emitted.append(trace)
        return self._trace_id

    async def add_scores(self, trace_id: str, scores: list[Any]) -> None:
        """Record the scores attached under a trace id."""
        self.scored.append((trace_id, scores))


class _RecordingEmail:
    """A minimal email_mcp fake for the trace wiring: stores + reads back a trace id."""

    def __init__(self, trace_id: str | None = None) -> None:
        """Seed the ticket's stored trace id (what `get_ticket` reports) and record writes."""
        self._trace_id = trace_id
        self.set_calls: list[tuple[int, str]] = []

    async def set_trace_id(self, ticket_id: int, trace_id: str) -> dict[str, Any]:
        """Capture a set_trace_id write and update what `get_ticket` will report."""
        self.set_calls.append((ticket_id, trace_id))
        self._trace_id = trace_id
        return {"id": ticket_id, "trace_id": trace_id}

    async def get_ticket(self, ticket_id: int) -> dict[str, Any] | None:
        """Return a ticket carrying the stored trace id (or None if never set)."""
        return {"id": ticket_id, "trace_id": self._trace_id}


async def test_emit_ticket_trace_emits_and_persists_the_trace_id() -> None:
    """`emit_ticket_trace` pushes the built trace and stores the returned id on the ticket.

    The submit-time write step: build the trace for the paused run, emit it, and persist
    the id Langfuse returned onto the ticket row (SPEC §7.2 "nested under a trace id
    stored on the ticket").
    """
    from app.graph.trace import emit_ticket_trace

    tracer = _RecordingTracer(trace_id="trace-xyz")
    email = _RecordingEmail()
    # The passed ticket id and the state's ticket id are the same case in real use.
    state = _happy_state()
    state["ticket_id"] = 7

    trace_id = await emit_ticket_trace(tracer, email, ticket_id=7, state=state, model=_MODEL)

    assert trace_id == "trace-xyz"
    assert len(tracer.emitted) == 1
    assert tracer.emitted[0].ticket_id == 7
    assert email.set_calls == [(7, "trace-xyz")]


async def test_emit_ticket_trace_persists_nothing_when_untraced() -> None:
    """With the NoOp fallback (empty trace id) nothing is written to the ticket.

    An offline run traces to nothing, so there is no id to persist and the ticket row
    is never touched — the desk works with or without Langfuse running.
    """
    from app.graph.trace import NoOpTracer, emit_ticket_trace

    email = _RecordingEmail()

    trace_id = await emit_ticket_trace(
        NoOpTracer(), email, ticket_id=7, state=_happy_state(), model=_MODEL
    )

    assert trace_id == ""
    assert email.set_calls == []


async def test_attach_scores_scores_the_ticket_trace() -> None:
    """`attach_scores` looks up the ticket's trace id and attaches the feedback scores.

    The rep-action write step: after a send it reads the trace id persisted on the
    ticket and attaches the disposition scores under it (SPEC §7.2 feedback as scores).
    """
    from app.graph.trace import attach_scores

    tracer = _RecordingTracer()
    email = _RecordingEmail(trace_id="trace-abc")
    state = _happy_state()
    state["rep_decision"] = FeedbackDecision.APPROVED_AS_IS
    state["final_reply"] = state["draft"].body

    await attach_scores(tracer, email, ticket_id=7, state=state, rating=5)

    assert len(tracer.scored) == 1
    trace_id, scores = tracer.scored[0]
    assert trace_id == "trace-abc"
    names = {score.name for score in scores}
    assert {"draft_accepted", "rating"} <= names


async def test_attach_scores_is_a_no_op_when_the_ticket_was_never_traced() -> None:
    """With no trace id on the ticket (offline run), no scores are attached."""
    from app.graph.trace import attach_scores

    tracer = _RecordingTracer()
    email = _RecordingEmail(trace_id=None)
    state = _happy_state()
    state["rep_decision"] = FeedbackDecision.REJECTED

    await attach_scores(tracer, email, ticket_id=7, state=state)

    assert tracer.scored == []

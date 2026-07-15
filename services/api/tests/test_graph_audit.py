"""Unit tests for the audit-emission layer (plan Task 24 / todo Task 26).

Task 25 built the *store* (an immutable audit table + a committing `record_audit`
tool + api client). This suite covers the *emission* half: turning a finished
workflow run into the ordered list of compliance rows SPEC §7.1 requires — each
node's outcome, the sources it cited, the model tag + prompt version it used, and
the guardrail decisions — and writing them through the email_mcp client.

`build_audit_entries` is a pure function: it reads a finished (paused) LangGraph
state and returns the ordered `AuditEntry` list, with no I/O, so every branch of the
pipeline (a clean run, a blocked injection, a triage failure, a no-source hand-off,
an attachment) is exercised in isolation without a database or a model. A thin
`record_node_audits` then writes that list through the client; its one test proves it
forwards each entry, in order, under the ticket id.
"""

from __future__ import annotations

from typing import Any

from app.schemas.draft import Citation, Draft
from app.schemas.enums import Category, Sentiment, Urgency
from app.schemas.guardrails import InjectionScreenResult, OutputScreenResult
from app.schemas.kb import KBSearchResult, KBSource
from app.schemas.triage import TriageResult
from app.schemas.validation import ValidationResult

# The model tag the tests pass in — the single host model's name (SPEC §4.2). The audit
# rows for model-using nodes must carry exactly this value.
_MODEL = "gemma4:12b"


def _clean_screen() -> InjectionScreenResult:
    """Build the input-guard verdict for a benign message: nothing caught, nothing flagged."""
    # detector "none" means neither the rules floor nor the LLM layer flagged the text.
    return InjectionScreenResult(flagged=False, detector="none")


def _triage_ok() -> TriageResult:
    """Build a valid triage classification (account-access, normal, neutral)."""
    # A fully classified ticket: topic, urgency, and customer mood all present.
    return TriageResult(
        category=Category.ACCOUNT_ACCESS,
        urgency=Urgency.NORMAL,
        sentiment=Sentiment.NEUTRAL,
    )


def _confident_kb() -> KBSearchResult:
    """Build a KB result with one confident, citable source (id `KB-1`)."""
    # One authoritative source the draft can cite; no "no confident source" flag.
    return KBSearchResult(
        sources=[KBSource(id="KB-1", title="Password reset", text="Use the login screen.")],
        no_confident_source=False,
    )


def _verified_draft() -> Draft:
    """Build a grounded, verified draft that cites the KB-1 source."""
    # A drafted reply that stayed grounded, so its `verified` flag is True.
    return Draft(
        body="You can reset your password from the login screen. [KB-1]",
        citations=[Citation(source_id="KB-1", title="Password reset")],
        verified=True,
    )


def _validation_ok() -> ValidationResult:
    """Build a validate-node result: fully grounded, not flagged."""
    # Groundedness 1.0 and not flagged — the draft is fully backed by its sources.
    return ValidationResult(draft=_verified_draft(), groundedness=1.0, flagged=False)


def _clean_output_screen() -> OutputScreenResult:
    """Build the output-guard verdict for a clean draft: nothing caught, nothing flagged."""
    # detector "none": neither the promise/PII floor nor the tone LLM layer flagged it.
    return OutputScreenResult(flagged=False, detector="none")


def _happy_state() -> dict[str, Any]:
    """Build a finished workflow state for a clean text-only happy-path run.

    Every pipeline product a full run produces is present: the input screen, triage,
    a confident KB result, the draft, its validation, and the output screen. This is
    the state `build_audit_entries` sees at the human-review pause on the happy path.
    """
    # Assemble the same channel the workflow threads through its nodes, filled as a
    # clean run leaves it just before the human gate.
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


def _events(entries: list[Any]) -> list[str]:
    """Pull just the event names out of a list of audit entries, in order."""
    # Each entry has an `event` name; collect them so a test can assert the sequence.
    return [entry.event for entry in entries]


def _by_event(entries: list[Any], event: str) -> Any:
    """Return the single audit entry with the given event name (fails if absent)."""
    # Find the one entry a test wants to inspect the detail of.
    return next(entry for entry in entries if entry.event == event)


def test_happy_path_records_each_node_outcome_in_order() -> None:
    """A clean run yields one ordered audit row per node outcome, all by the system.

    The trail reads back in pipeline order — input screen, triage, retrieval, draft,
    validation, output screen — so a reviewer follows the case exactly as it ran, and
    every automated step is attributed to the `system` actor (SPEC §7.1).
    """
    from app.graph.audit import build_audit_entries

    # Turn a finished happy-path run into its audit entries.
    entries = build_audit_entries(_happy_state(), model=_MODEL)

    # The events appear in the order the nodes ran — the ordered history §7.1 requires.
    assert _events(entries) == [
        "input_screened",
        "triaged",
        "retrieved",
        "drafted",
        "validated",
        "output_screened",
    ]
    # Every automated node outcome is recorded as the system's action, not a rep's.
    assert all(entry.actor == "system" for entry in entries)


def test_model_using_nodes_record_model_tag_and_prompt_version() -> None:
    """Triage, draft, and validate each record the model tag and the prompt version used.

    This is the "on what basis" half of §7.1: any reply can be tied back to the exact
    model and the exact prompt version that produced each step.
    """
    from app.graph.audit import build_audit_entries

    # Build the audit rows for a clean run.
    entries = build_audit_entries(_happy_state(), model=_MODEL)

    # The triage row names the host model and triage's prompt version.
    triaged = _by_event(entries, "triaged")
    assert triaged.detail["model"] == _MODEL
    assert triaged.detail["prompt_version"] == "triage-v1"
    # The draft row names the host model and the draft prompt version.
    drafted = _by_event(entries, "drafted")
    assert drafted.detail["model"] == _MODEL
    assert drafted.detail["prompt_version"] == "draft-v1"
    # The validation row names the host model and the groundedness-judge prompt version.
    validated = _by_event(entries, "validated")
    assert validated.detail["model"] == _MODEL
    assert validated.detail["prompt_version"] == "validate-v1"


def test_triaged_entry_records_the_classification() -> None:
    """The triage row records the category, urgency, and sentiment the model chose."""
    from app.graph.audit import build_audit_entries

    # Build the rows and pull out the triage outcome.
    triaged = _by_event(build_audit_entries(_happy_state(), model=_MODEL), "triaged")

    # The three classification fields are recorded as their plain string values.
    assert triaged.detail["category"] == "account_access"
    assert triaged.detail["urgency"] == "normal"
    assert triaged.detail["sentiment"] == "neutral"


def test_retrieved_entry_lists_the_cited_sources() -> None:
    """The retrieval row records exactly which KB sources were cited (SPEC §7.1).

    "The source(s) cited" is a named §7.1 requirement: the row lists each source's id
    and title and the count, so a reviewer sees the documented basis for the reply.
    """
    from app.graph.audit import build_audit_entries

    # Pull out the retrieval outcome from a clean run.
    retrieved = _by_event(build_audit_entries(_happy_state(), model=_MODEL), "retrieved")

    # The cited source's id and title are on the record.
    assert retrieved.detail["sources"] == [{"id": "KB-1", "title": "Password reset"}]
    # The count of confident sources found is recorded alongside them.
    assert retrieved.detail["count"] == 1


def test_drafted_entry_records_citations_and_verified_flag() -> None:
    """The draft row records the citations it drew on and whether it stayed verified."""
    from app.graph.audit import build_audit_entries

    # Pull out the draft outcome from a clean run.
    drafted = _by_event(build_audit_entries(_happy_state(), model=_MODEL), "drafted")

    # The draft's citations are recorded (source id + title).
    assert drafted.detail["citations"] == [{"source_id": "KB-1", "title": "Password reset"}]
    # A grounded draft is recorded as verified.
    assert drafted.detail["verified"] is True


def test_clean_input_screen_records_the_guardrail_decision_without_a_model() -> None:
    """A clean input screen records the guardrail decision but no model tag.

    When nothing was flagged the detector is "none" — no LLM verdict *fired the
    decision* — so the row carries the decision (`detector`/`flagged`) but omits a
    model/prompt version there is nothing to attribute.
    """
    from app.graph.audit import build_audit_entries

    # Pull out the input-screen outcome from a clean run.
    screened = _by_event(build_audit_entries(_happy_state(), model=_MODEL), "input_screened")

    # The guardrail decision is recorded: which layer fired, and that nothing was flagged.
    assert screened.detail["detector"] == "none"
    assert screened.detail["flagged"] is False
    # No layer's verdict drove the decision, so no model is attributed on this row.
    assert "model" not in screened.detail


def test_llm_flagged_input_screen_records_the_model_and_blocks() -> None:
    """An LLM-caught injection records the model+prompt on the screen row, then stops.

    When the LLM layer fires the decision (`detector == "llm"`) the guardrail row
    attributes the model and the input-guard prompt version. A flagged injection then
    routes straight to the human gate, so the trail ends at `injection_blocked` — no
    triage, retrieval, or draft ever ran.
    """
    from app.graph.audit import build_audit_entries

    # A finished state where the LLM layer flagged an injection attempt.
    state = {
        "ticket_id": 1,
        "injection_screen": InjectionScreenResult(
            flagged=True,
            categories=["instruction_override"],
            evidence=["ignore all"],
            detector="llm",
        ),
    }
    # Build the audit rows for that blocked run.
    entries = build_audit_entries(state, model=_MODEL)

    # Only the screen decision and the block are recorded — the pipeline went no further.
    assert _events(entries) == ["input_screened", "injection_blocked"]
    # The LLM layer fired the decision, so the model and prompt version are attributed.
    screened = _by_event(entries, "input_screened")
    assert screened.detail["model"] == _MODEL
    assert screened.detail["prompt_version"] == "input_guard-v1"
    # The block row names the attack family the guard caught.
    blocked = _by_event(entries, "injection_blocked")
    assert blocked.detail["categories"] == ["instruction_override"]


def test_rules_blocked_injection_records_no_model_and_stops() -> None:
    """A signatures-caught injection records the block with no model call attributed.

    The deterministic floor short-circuits before any model runs, so `detector` is
    "rules": the screen row carries the decision but no model, and the trail stops at
    the block.
    """
    from app.graph.audit import build_audit_entries

    # A finished state where the rules floor caught the attack (no model consulted).
    state = {
        "ticket_id": 1,
        "injection_screen": InjectionScreenResult(
            flagged=True,
            categories=["role_manipulation"],
            evidence=["you are now"],
            detector="rules",
        ),
    }
    # Build the audit rows for that blocked run.
    entries = build_audit_entries(state, model=_MODEL)

    # The trail is just the screen decision and the block.
    assert _events(entries) == ["input_screened", "injection_blocked"]
    # The floor fired without a model, so no model is attributed on the screen row.
    assert "model" not in _by_event(entries, "input_screened").detail


def test_triage_failure_records_and_stops() -> None:
    """A ticket triage could not classify records `triage_failed` and goes no further.

    When triage exhausts its retries the case routes to a human, so the trail records
    the failed triage attempt (with the model/prompt that ran) and stops — no
    retrieval or draft row follows.
    """
    from app.graph.audit import build_audit_entries

    # A finished state: clean screen, but triage never produced a classification.
    state = {"ticket_id": 1, "injection_screen": _clean_screen()}
    # Build the audit rows for that unclassifiable run.
    entries = build_audit_entries(state, model=_MODEL)

    # The trail records the screen then the triage failure, and stops there.
    assert _events(entries) == ["input_screened", "triage_failed"]
    # The failed triage still records which model and prompt version were tried.
    failed = _by_event(entries, "triage_failed")
    assert failed.detail["model"] == _MODEL
    assert failed.detail["prompt_version"] == "triage-v1"


def test_no_confident_source_records_and_stops() -> None:
    """A no-confident-source retrieval records the hand-off and produces no draft row.

    When retrieval finds no confident source the case is handed to a human for
    research, so the trail records `no_confident_source` after triage and stops — the
    draft/validate/output rows never appear.
    """
    from app.graph.audit import build_audit_entries

    # A finished state: clean screen, good triage, but retrieval found no confident source.
    state = {
        "ticket_id": 1,
        "injection_screen": _clean_screen(),
        "triage": _triage_ok(),
        "kb_result": KBSearchResult(sources=[], no_confident_source=True),
    }
    # Build the audit rows for that no-source hand-off.
    entries = build_audit_entries(state, model=_MODEL)

    # The trail stops at the no-source hand-off; no draft was ever attempted.
    assert _events(entries) == ["input_screened", "triaged", "no_confident_source"]


def test_attachment_extraction_is_recorded_before_triage() -> None:
    """A ticket whose attachment was digitized records `attachment_extracted`.

    When the OCR/extraction passes ran (the state carries an `extracted_facts` digest),
    the trail records that a document was digitized and with which model/prompt version,
    slotted between the input screen and triage — documenting the AI read a customer
    document, without dumping its raw text into the compliance row.
    """
    from app.graph.audit import build_audit_entries

    # A happy-path state, but with an attachment digest present (an attachment ran).
    state = _happy_state()
    # The rendered OCR/extraction digest a ticket with an attachment carries.
    state["extracted_facts"] = "Document: cheque\nReferences: CHK-4471"
    # Build the audit rows for that attachment run.
    entries = build_audit_entries(state, model=_MODEL)

    # The extraction row sits right after the input screen and before triage.
    assert _events(entries)[:3] == ["input_screened", "attachment_extracted", "triaged"]
    # It records the model and the extraction prompt version, not the raw document text.
    extracted = _by_event(entries, "attachment_extracted")
    assert extracted.detail == {"model": _MODEL, "prompt_version": "extract-v1"}


def test_flagged_output_screen_records_the_violation_categories() -> None:
    """A flagged output screen records the guardrail decision and the categories found.

    The output guard's decision is part of §7.1's "guardrail decisions": a flagged
    draft records `flagged` True and the violation families, and — when the LLM tone
    layer fired the decision — the model and output-guard prompt version.
    """
    from app.graph.audit import build_audit_entries

    # A happy-path state, but the output guard's LLM layer flagged a tone problem.
    state = _happy_state()
    # Replace the clean screen with a flagged one caught by the LLM tone layer.
    state["output_screen"] = OutputScreenResult(
        flagged=True, categories=["dismissive"], evidence=["that's your problem"], detector="llm"
    )
    # Build the audit rows for that flagged run.
    screened = _by_event(build_audit_entries(state, model=_MODEL), "output_screened")

    # The guardrail decision records the flag and the tone family it found.
    assert screened.detail["flagged"] is True
    assert screened.detail["categories"] == ["dismissive"]
    # The LLM tone layer fired the decision, so the model + prompt version are attributed.
    assert screened.detail["model"] == _MODEL
    assert screened.detail["prompt_version"] == "output_guard-v1"


class _RecordingEmailClient:
    """Minimal stand-in that records every `record_audit` call in order.

    `record_node_audits` should forward each built entry through the client; this fake
    captures the `(ticket_id, event, actor, detail)` of each call so the test can
    assert what was written and in what order — no email_mcp or database involved.
    """

    def __init__(self) -> None:
        """Start with an empty list of captured audit calls."""
        # Each element is one `record_audit` call, in the order it was made.
        self.calls: list[tuple[int, str, str | None, Any]] = []

    async def record_audit(
        self, ticket_id: int, event: str, *, actor: str | None = None, detail: Any = None
    ) -> dict[str, Any]:
        """Capture one audit call and return a stub row (matches the real signature)."""
        # Remember exactly what the emitter asked us to write.
        self.calls.append((ticket_id, event, actor, detail))
        return {"event": event}


async def test_record_node_audits_writes_each_entry_in_order() -> None:
    """`record_node_audits` forwards every built entry through the client, in order.

    The thin write step: it turns a finished state into entries and writes each one via
    `record_audit` under the ticket id, preserving order — so the persisted trail is the
    ordered node history, one row per outcome.
    """
    from app.graph.audit import record_node_audits

    # A recording fake standing in for the email_mcp client.
    email = _RecordingEmailClient()
    # Emit the audit trail for a finished happy-path run under ticket id 7.
    await record_node_audits(email, ticket_id=7, state=_happy_state(), model=_MODEL)

    # One call per node outcome, in pipeline order, all under ticket 7 by the system.
    assert [(tid, event, actor) for tid, event, actor, _ in email.calls] == [
        (7, "input_screened", "system"),
        (7, "triaged", "system"),
        (7, "retrieved", "system"),
        (7, "drafted", "system"),
        (7, "validated", "system"),
        (7, "output_screened", "system"),
    ]

"""Audit-completeness integration tests (plan Task 24 / todo Task 26).

SPEC §7.1's acceptance criterion is that *a resolved case has an ordered, immutable
audit trail linking customer ↔ ticket ↔ final reply*. Two tests prove the emission
path end to end against the in-memory fakes (no Ollama, no Postgres):

1. the submit-time pipeline trigger records the node-outcome rows when a run reaches
   the human-review pause; and
2. a full case — customer submission, the automated pipeline, the rep's approval, and
   the send — leaves one ordered trail whose last row is the sent reply attributed to
   the rep, so a reviewer can answer "who sent this, and on what documented basis".

These compose the *real* emission function and the *real* rep-action routes against
the shared `FakeEmailClient`, whose in-memory ledger mirrors email_mcp's immutable
`audit` table (the immutability itself is proven at the storage layer in
`email_mcp/tests/test_audit.py`).
"""

from __future__ import annotations

from typing import Any

from tests.conftest import HAPPY_DRAFT_BODY, FakeEmailClient

# The benign customer message the happy-path model script answers.
_BENIGN_MESSAGE = "How do I reset my online banking password?"

# The happy-path model script: one response per model-using step, in graph order —
# input screen, search-intent fusion, triage, draft, groundedness judge, tone screen.
_HAPPY_PATH_SCRIPT = [
    '{"is_injection": false}',
    "reset online banking password",
    '{"category": "account_access", "urgency": "normal", "sentiment": "neutral"}',
    HAPPY_DRAFT_BODY,
    '{"score": 1.0, "unsupported_claims": []}',
    '{"has_violation": false}',
]

# The ordered node-outcome events a clean text-only run records at the hand-off.
_NODE_EVENTS = [
    "input_screened",
    "triaged",
    "retrieved",
    "drafted",
    "validated",
    "output_screened",
]


class _FakeKBClient:
    """In-memory KB stand-in returning one confident, citable source (id `KB-1`)."""

    async def search(self, query: str, limit: int | None = None) -> Any:
        """Ignore the query and return a single confident source."""
        # Import here so the module still collects before the schemas are importable.
        from app.schemas.kb import KBSearchResult, KBSource

        # One authoritative source the draft can cite; not a no-source hand-off.
        return KBSearchResult(
            sources=[KBSource(id="KB-1", title="Password reset", text="Use the login screen.")],
            no_confident_source=False,
        )


def _fresh_workflow(test_settings: Any) -> Any:
    """Compile an un-run happy-path workflow with the fakes and an in-memory saver.

    Mirrors production's wiring (which passes the Postgres saver) but with the
    deterministic `FakeLLM`, the confident fake KB, and a `MemorySaver`, so the
    pipeline trigger can drive a real run to the pause with no Ollama or database.
    """
    # Import here so collection doesn't require the implementation during RED.
    from langgraph.checkpoint.memory import MemorySaver

    from app.graph.workflow import build_workflow
    from app.llm.fake import FakeLLM

    # Build the same graph the app builds, against the test fakes.
    return build_workflow(
        llm=FakeLLM(list(_HAPPY_PATH_SCRIPT)),
        kb_client=_FakeKBClient(),
        email_client=FakeEmailClient(),
        settings=test_settings,
        checkpointer=MemorySaver(),
    )


async def test_start_pipeline_records_node_audits_at_the_pause(
    test_settings: Any,
    email_client: FakeEmailClient,
    monkeypatch: Any,
) -> None:
    """The submit-time trigger writes the node-outcome audit rows when the run pauses.

    Driving a fresh ticket through the pipeline to the human-review pause must leave
    the ordered node history on the ticket's trail (SPEC §7.1) — proving the emission
    is wired into the background run, not only into the pure builder.
    """
    from app.graph import intake
    from app.main import create_app

    # A fresh, un-run happy-path workflow the trigger will drive to the pause.
    graph = _fresh_workflow(test_settings)

    async def _return_graph(app: Any, settings: Any) -> Any:
        """Stand in for the runtime builder, handing back our fake-backed graph."""
        # The trigger asks for the app's workflow; give it our in-memory one.
        return graph

    # Swap the real (Postgres/Ollama) workflow builder for the fake-backed graph.
    monkeypatch.setattr(intake, "get_workflow_for_app", _return_graph)
    # Build an app and pin our fake email client on it, so emission writes to the fake.
    app = create_app()
    app.state.email_client = email_client

    # Run the submit-time pipeline for a freshly submitted ticket, to the pause.
    await intake.start_pipeline(
        app, test_settings, ticket_id=42, message=_BENIGN_MESSAGE, attachments=[]
    )

    # The ticket's trail carries exactly the ordered node-outcome rows, all by the system.
    trail = await email_client.get_audit_trail(42)
    assert [row["event"] for row in trail] == _NODE_EVENTS
    assert all(row["actor"] == "system" for row in trail)


async def test_resolved_case_has_a_complete_ordered_trail(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    test_settings: Any,
    auth_headers: dict[str, str],
) -> None:
    """A sent case yields one ordered trail linking customer, ticket, and final reply.

    The §7.1 acceptance criterion end to end: the customer submission, every node
    outcome, the rep's approval, and the send appear in order on one immutable trail;
    the last row is the sent reply attributed to the rep, and the same reply is on the
    resolved ticket — the customer ↔ ticket ↔ reply link a reviewer follows.
    """
    from app.graph.audit import record_node_audits
    from app.graph.workflow import thread_config

    # A ticket already exists (registered), and its intake wrote the customer submission
    # row — mirror email_mcp's `ticket_created` entry from create_ticket at §4.1.
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    await email_client.record_audit(7, "ticket_created", actor="customer")

    # Drive the automated pipeline to the human-review pause for ticket 7.
    graph = await build_paused_workflow(ticket_id=7)
    # Emit the node-outcome audit rows the pipeline records at the hand-off.
    state = (await graph.aget_state(thread_config(7))).values
    await record_node_audits(email_client, ticket_id=7, state=state, model=test_settings.llm_model)

    # The rep approves the draft, then explicitly sends it.
    async with rep_client(graph) as ac:
        await ac.post("/rep/tickets/7/approve", headers=auth_headers)
        await ac.post("/rep/tickets/7/send", json={"rep_id": "rep-1"}, headers=auth_headers)

    # The whole lifecycle reads back in order on one trail.
    trail = await email_client.get_audit_trail(7)
    assert [row["event"] for row in trail] == [
        "ticket_created",
        *_NODE_EVENTS,
        "draft_approved",
        "reply_sent",
    ]
    # The customer opens the trail; the rep closes it with the send.
    assert trail[0]["actor"] == "customer"
    assert trail[-1]["event"] == "reply_sent"
    assert trail[-1]["actor"] == "rep-1"
    # The reply on the resolved ticket is the very draft the trail says was sent —
    # the customer ↔ ticket ↔ final reply link (SPEC §7.1).
    ticket = email_client.tickets_by_id[7]
    # The send registered a ticket, so it is present (not the neutral not-found None).
    assert ticket is not None
    assert ticket["status"] == "Resolved"
    assert ticket["reply"] == HAPPY_DRAFT_BODY

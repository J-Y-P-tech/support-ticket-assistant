"""Tests for the submit-time pipeline trigger (plan Task 19a / Checkpoint C).

`start_pipeline` is the background task the submit route schedules: it drives a
freshly submitted ticket's run to the `human_review` interrupt and — critically —
must **fail safe**, swallowing any pipeline error so a background crash never affects
the already-returned submit. Both behaviours are proven here against the deterministic
`FakeLLM` and a stub app, so the trigger is guarded without the live walk (which cannot
prove the safe-failure path).

The trigger resolves the workflow via `get_workflow_for_app`, which returns whatever is
cached on `app.state.workflow`. A `SimpleNamespace` app carrying a pre-built graph
stands in for the real FastAPI app, so no Postgres, Ollama, or kb_mcp is touched.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

from app.graph.intake import start_pipeline
from app.graph.workflow import build_workflow, thread_config
from app.llm.fake import FakeLLM
from app.schemas.enums import TicketStatus
from app.schemas.kb import KBSearchResult, KBSource

# Happy-path model script, one response per model-using step in graph order:
# screen_input → ocr_extract (search-intent fusion) → triage → draft → validate →
# screen_output. The ocr_extract node runs on every ticket; a text-only submission
# skips transcription/extraction and only fuses the query from the message.
_DRAFT_BODY = "You can reset your password from the login screen. [KB-1]"
_HAPPY_PATH_SCRIPT = [
    '{"is_injection": false}',
    "reset online banking password",
    '{"category": "account_access", "urgency": "normal", "sentiment": "neutral"}',
    _DRAFT_BODY,
    '{"score": 1.0, "unsupported_claims": []}',
    '{"has_violation": false}',
]
_BENIGN_MESSAGE = "How do I reset my online banking password?"


class _FakeKBClient:
    """In-memory KB stand-in returning one confident, citable source (id `KB-1`)."""

    async def search(self, query: str, limit: int | None = None) -> KBSearchResult:
        """Ignore the query and return a single confident source."""
        return KBSearchResult(
            sources=[
                KBSource(
                    id="KB-1",
                    title="Password reset",
                    text="To reset your password, use the login screen.",
                )
            ],
            no_confident_source=False,
        )


def _app_with_workflow(workflow: Any) -> Any:
    """Return a stub app whose `state.workflow` is the given (pre-built) graph.

    Typed `Any` so it stands in for the `FastAPI` app `start_pipeline` expects without
    a cast — only `app.state.workflow` is ever read here.
    """
    return SimpleNamespace(state=SimpleNamespace(workflow=workflow))


async def test_start_pipeline_runs_a_submission_to_the_human_gate(test_settings: Any) -> None:
    """A submitted ticket is driven to the review pause with a verified draft, unsent.

    Mirrors what the submit route schedules: the trigger runs the automated pipeline on
    the ticket's checkpoint thread and stops before `human_review` — nothing Resolved,
    no reply, but the draft ready for the rep. This is the run the rep-action routes
    later resume.
    """
    graph = build_workflow(
        llm=FakeLLM(list(_HAPPY_PATH_SCRIPT)),
        kb_client=_FakeKBClient(),
        settings=test_settings,
        checkpointer=MemorySaver(),
    )
    app = _app_with_workflow(graph)

    await start_pipeline(app, test_settings, ticket_id=42, message=_BENIGN_MESSAGE, attachments=[])

    snapshot = graph.get_state(thread_config(42))
    assert snapshot.next == ("human_review",)
    values = snapshot.values
    assert values["status"] != TicketStatus.RESOLVED
    assert values.get("final_reply") is None
    assert values["draft"].body == _DRAFT_BODY


async def test_start_pipeline_swallows_a_pipeline_failure(test_settings: Any) -> None:
    """A pipeline that raises never propagates out of the background trigger.

    The submit response is already sent by the time this runs, so a failure (Ollama or
    kb_mcp unreachable) must be logged and swallowed, leaving the ticket New — never
    crash the worker or surface to the customer. A workflow whose `ainvoke` raises
    stands in for any such failure.
    """

    class _BoomWorkflow:
        async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("ollama unreachable")

    app = _app_with_workflow(_BoomWorkflow())

    # Must return normally despite the raise inside the run.
    await start_pipeline(app, test_settings, ticket_id=1, message="x", attachments=[])

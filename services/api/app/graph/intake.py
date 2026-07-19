"""Start the workflow when a ticket is submitted — the pipeline's entry trigger.

Submitting a ticket stores it and returns the reference code immediately (SPEC §4.1);
the AI pipeline then runs *out of band* as a FastAPI background task, so the
customer's submit stays instant while `screen_input → triage → retrieve → draft →
validate → screen_output` run behind it. The run drives the compiled workflow to its
`human_review` interrupt and the Postgres checkpointer persists the paused state, so a
rep opens a case whose draft is already waiting (SPEC §4.7). Nothing here resolves or
sends — the graph stops at the human gate; only a rep action resumes it.

Failures fail *safe and quiet*: if the pipeline raises (Ollama or kb_mcp unreachable,
a malformed run) the exception is logged and the ticket simply stays New in the rep
queue for manual handling — the customer's submit already succeeded, so a background
crash must never surface to them or take down the worker.

The trigger is injected as a dependency (`get_pipeline_starter`) rather than imported
directly by the route, so tests override it the same way they override the settings
and MCP clients — the customer-route suite swaps in a no-op and touches no Postgres or
Ollama.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from fastapi import FastAPI

from app.config import Settings
from app.graph.audit import record_node_audits
from app.graph.runtime import get_tracer_for_app, get_workflow_for_app
from app.graph.trace import emit_ticket_trace
from app.graph.workflow import thread_config
from app.mcp_clients.email import email_client_for_app
from app.schemas.enums import TicketStatus

_logger = logging.getLogger(__name__)


class PipelineStarter(Protocol):
    """The submit-time trigger's call shape, so the route can depend on it abstractly.

    Typing the dependency structurally lets tests supply a no-op with the same
    signature without importing or subclassing the real `start_pipeline`.
    """

    async def __call__(
        self,
        app: FastAPI,
        settings: Settings,
        *,
        ticket_id: int,
        message: str,
        attachments: list[str],
    ) -> None:
        """Run the automated pipeline for one freshly submitted ticket."""
        ...


async def start_pipeline(
    app: FastAPI,
    settings: Settings,
    *,
    ticket_id: int,
    message: str,
    attachments: list[str],
) -> None:
    """Run the automated pipeline for a freshly submitted ticket, to the human gate.

    Invoked as a background task after the submit response is sent. Drives the graph to
    its `human_review` interrupt on the ticket's checkpoint thread; the paused state is
    what the rep-action routes later resume. Once the run reaches the pause, the
    node-outcome audit rows are written to the ticket's immutable trail (SPEC §7.1) —
    each node's outcome, its cited sources, the model tag + prompt version, and the
    guardrail decisions. Any failure is swallowed and logged, so a background crash
    never affects the already-returned submit — the ticket stays New for a rep to pick
    up by hand.
    """
    initial: dict[str, Any] = {
        "ticket_id": ticket_id,
        "message": message,
        "attachments": attachments,
        "extracted_facts": None,
        "flags": [],
        "status": TicketStatus.NEW,
    }
    config = thread_config(ticket_id)
    try:
        workflow = await get_workflow_for_app(app, settings)
        await workflow.ainvoke(initial, config)
        # The run is now paused at the human gate; record what the pipeline did on the
        # ticket's compliance trail. Reading the paused snapshot (rather than ainvoke's
        # return) keeps this robust to how the interrupt surfaces the state.
        snapshot = await workflow.aget_state(config)
        email = email_client_for_app(app, settings)
        await record_node_audits(
            email, ticket_id=ticket_id, state=snapshot.values, model=settings.llm_model
        )
        # Emit the one PII-redacted Langfuse trace for this ticket's run and persist its
        # id on the ticket (SPEC §7.2). Best-effort: with no Langfuse configured the
        # no-op tracer returns no id and nothing is stored — the case is unaffected.
        tracer = get_tracer_for_app(app, settings)
        await emit_ticket_trace(
            tracer, email, ticket_id=ticket_id, state=snapshot.values, model=settings.llm_model
        )
    except Exception:
        _logger.exception("pipeline failed for ticket %s; left New for manual handling", ticket_id)


def get_pipeline_starter() -> PipelineStarter:
    """FastAPI dependency: return the submit-time pipeline trigger.

    Returns the real `start_pipeline`; the customer-route tests override this with a
    no-op so submitting a ticket schedules nothing that would reach Postgres or Ollama.
    """
    return start_pipeline

"""Rep-facing routes: the work queue, ticket detail, and draft-review actions.

- `GET /rep/queue` (SPEC §4.3/§4.7): the New/untriaged tickets awaiting a rep.
- `GET /rep/tickets/{ticket_id}`: the full detail a rep opens; an unknown id
  returns a neutral 404.
- `POST /rep/tickets/{ticket_id}/{edit,approve,reject,send}` (SPEC §4.7, plan
  Task 17): the rep's disposition of an AI draft at the human-review pause.

**The action routes are the second half of the human gate.** The workflow pauses
before `human_review`; these routes write the rep's decision into that paused run
and resume it. `edit` and `approve` only *stage* a decision (no resume) — approve
alone never sends. `send` resumes the run through `finalize` (the only node that can
resolve a case) and then persists the reply via email_mcp, so there is no path to
Resolved without an explicit rep send (SPEC §10). `reject` resumes with a rejection,
routing the case back to NeedsResearch with nothing sent.

All routes require the frontend->api bearer token.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot

from app.config import Settings, get_settings
from app.graph.runtime import get_workflow
from app.graph.workflow import thread_config
from app.mcp_clients.email import EmailMCPClient, get_email_client
from app.schemas.enums import FeedbackDecision, TicketStatus
from app.schemas.rep import (
    RepActionResult,
    RepEditRequest,
    RepRejectRequest,
    RepSendRequest,
)
from app.schemas.ticket import QueuePage, QueueRow
from app.security import require_auth

router = APIRouter(prefix="/rep", dependencies=[Depends(require_auth)])


def _parse_cursor(after: str | None) -> tuple[str, int] | None:
    """Parse an `after` cursor (`<created_at>,<id>`) into `(created_at, id)`.

    `None` (no cursor) means the first page. A present-but-malformed cursor is a
    client mistake, surfaced as 400 rather than a 500 or a silent first page.
    """
    if after is None:
        return None
    created_at, _, id_str = after.rpartition(",")
    if not created_at or not id_str:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid cursor")
    try:
        return created_at, int(id_str)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid cursor"
        ) from None


@router.get("/queue", response_model=QueuePage)
async def rep_queue(
    limit: int | None = Query(default=None),
    after: str | None = Query(default=None),
    settings: Settings = Depends(get_settings),
    email: EmailMCPClient = Depends(get_email_client),
) -> QueuePage:
    """Return one keyset page of the New-ticket rep queue.

    An omitted `limit` falls back to the configured `QUEUE_PAGE_DEFAULT`; any value
    is clamped to `[1, QUEUE_PAGE_MAX]` so an oversized ask is capped rather than
    rejected; `after` resumes past the last row already seen. A full page yields a
    `next_cursor` (more may remain); a short page yields `None`.
    """
    requested = settings.queue_page_default if limit is None else limit
    capped = min(max(requested, 1), settings.queue_page_max)
    cursor = _parse_cursor(after)
    rows = await email.fetch_new_tickets(limit=capped, after=cursor)
    items = [QueueRow.model_validate(row) for row in rows]
    next_cursor = None
    if len(rows) == capped and rows:
        last = rows[-1]
        next_cursor = f"{last['created_at']},{last['id']}"
    return QueuePage(items=items, next_cursor=next_cursor)


@router.get("/tickets/{ticket_id}")
async def rep_ticket_detail(
    ticket_id: int,
    email: EmailMCPClient = Depends(get_email_client),
) -> dict[str, Any]:
    """Return one ticket's full detail, or a neutral 404 if the id is unknown."""
    ticket = await email.get_ticket(ticket_id)
    if ticket is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return ticket


async def _require_review_pause(workflow: CompiledStateGraph, ticket_id: int) -> StateSnapshot:
    """Return the ticket's paused run state, or 409 if it is not awaiting review.

    A rep action is only valid while the run is interrupted before `human_review`.
    An unknown ticket (no checkpoint) or one already finalized has a different `next`,
    so it is refused with 409 rather than resuming a run that isn't at the gate.
    """
    snapshot = await workflow.aget_state(thread_config(ticket_id))
    if snapshot.next != ("human_review",):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Ticket is not awaiting review"
        )
    return snapshot


@router.post("/tickets/{ticket_id}/edit", response_model=RepActionResult)
async def rep_edit(
    ticket_id: int,
    payload: RepEditRequest,
    workflow: CompiledStateGraph = Depends(get_workflow),
) -> RepActionResult:
    """Stage the rep's edited reply into the paused case (held until send).

    Records the edited text and an `edited` decision on the paused run without
    resuming it — the edit becomes the customer reply only when the rep sends. Nothing
    is written to email_mcp here; the case stays awaiting review.
    """
    await _require_review_pause(workflow, ticket_id)
    config = thread_config(ticket_id)
    await workflow.aupdate_state(
        config,
        {"rep_edited_reply": payload.reply, "rep_decision": FeedbackDecision.EDITED},
    )
    values = (await workflow.aget_state(config)).values
    return RepActionResult(ticket_id=ticket_id, status=values["status"], reply=None)


@router.post("/tickets/{ticket_id}/approve", response_model=RepActionResult)
async def rep_approve(
    ticket_id: int,
    workflow: CompiledStateGraph = Depends(get_workflow),
) -> RepActionResult:
    """Stage an approve-as-is decision on the paused case (does not send).

    Approve alone never resolves a case (SPEC §4.7): it records the decision on the
    paused run and leaves it awaiting review. A later `send` commits it.
    """
    await _require_review_pause(workflow, ticket_id)
    config = thread_config(ticket_id)
    await workflow.aupdate_state(config, {"rep_decision": FeedbackDecision.APPROVED_AS_IS})
    values = (await workflow.aget_state(config)).values
    return RepActionResult(ticket_id=ticket_id, status=values["status"], reply=None)


@router.post("/tickets/{ticket_id}/send", response_model=RepActionResult)
async def rep_send(
    ticket_id: int,
    payload: RepSendRequest,
    workflow: CompiledStateGraph = Depends(get_workflow),
    email: EmailMCPClient = Depends(get_email_client),
) -> RepActionResult:
    """Send the approved/edited reply: resume through `finalize`, then persist it.

    Refuses with 409 if no rep has approved or edited the draft — with no staged
    decision `finalize` fails closed, so there is no path to Resolved without an
    explicit approve/edit first (SPEC §10). On success the resumed run resolves the
    case and its reply is recorded via email_mcp under the rep's audit marker.
    """
    snapshot = await _require_review_pause(workflow, ticket_id)
    if snapshot.values.get("rep_decision") is None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No draft has been approved or edited for this ticket",
        )
    final = await workflow.ainvoke(None, thread_config(ticket_id))
    reply = final.get("final_reply")
    if final.get("status") != TicketStatus.RESOLVED or reply is None:
        # A staged rejection (or a missing reply) is not a send; refuse rather than
        # resolve. reject() is the route for routing a case back.
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Ticket cannot be sent")
    await email.record_sent_reply(ticket_id, reply, payload.rep_id)
    return RepActionResult(ticket_id=ticket_id, status=TicketStatus.RESOLVED, reply=reply)


@router.post("/tickets/{ticket_id}/reject", response_model=RepActionResult)
async def rep_reject(
    ticket_id: int,
    payload: RepRejectRequest,
    workflow: CompiledStateGraph = Depends(get_workflow),
    email: EmailMCPClient = Depends(get_email_client),
) -> RepActionResult:
    """Reject the draft: resume with a rejection and route the case back for research.

    Resumes the paused run with a rejection; `finalize` routes it to NeedsResearch and
    produces no reply. The status is persisted via `update_status` (never a send), so
    nothing reaches the customer.
    """
    await _require_review_pause(workflow, ticket_id)
    config = thread_config(ticket_id)
    await workflow.aupdate_state(config, {"rep_decision": FeedbackDecision.REJECTED})
    final = await workflow.ainvoke(None, config)
    result_status = final.get("status", TicketStatus.NEEDS_RESEARCH)
    await email.update_status(ticket_id, TicketStatus.NEEDS_RESEARCH.value, actor=payload.rep_id)
    return RepActionResult(ticket_id=ticket_id, status=result_status, reply=None)

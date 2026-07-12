"""Customer-facing routes: submit a ticket, look one up by code (plan Task 4).

- `POST /tickets` (SPEC §4.1): create a New ticket from a non-empty message
  (validated by `TicketCreate`) and return its `TKT-####` reference code.
- `GET /tickets/{code}` (SPEC §4.8): resolve a reference code to its status/reply;
  an unknown code returns a neutral 404 that never echoes the code back.

Both sit behind the frontend->api bearer token (`require_auth`): a customer only
ever reaches them through the token-holding frontend.
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Request, status

from app.config import Settings, get_settings
from app.graph.intake import PipelineStarter, get_pipeline_starter
from app.mcp_clients.email import EmailMCPClient, get_email_client
from app.schemas.ticket import TicketCreate, TicketRead
from app.security import require_auth

router = APIRouter(dependencies=[Depends(require_auth)])


def _normalize_code(code: str) -> str:
    """Normalize a customer-typed reference code (strip + uppercase).

    Provisional: Task 5 extracts a shared reference-code util that will replace
    this inline helper. Keeping normalization at the api boundary means a lightly
    mistyped code (`  tkt-0007 `) still resolves.
    """
    return code.strip().upper()


@router.post("/tickets", status_code=status.HTTP_201_CREATED, response_model=TicketRead)
async def submit_ticket(
    payload: TicketCreate,
    request: Request,
    background_tasks: BackgroundTasks,
    email: EmailMCPClient = Depends(get_email_client),
    settings: Settings = Depends(get_settings),
    start_pipeline: PipelineStarter = Depends(get_pipeline_starter),
) -> TicketRead:
    """Create a New ticket, kick off the AI pipeline, and return its code (SPEC §4.1).

    The reply is returned as soon as the ticket is stored; the AI pipeline (triage →
    retrieve → draft, pausing at the human gate) runs afterwards as a background task,
    so the customer's submit stays instant. The pipeline persists a paused run the rep
    later reviews — see `app.graph.intake`.
    """
    created = await email.create_ticket(payload.message, payload.attachments)
    background_tasks.add_task(
        start_pipeline,
        request.app,
        settings,
        ticket_id=created["id"],
        message=created["message"],
        attachments=created.get("attachments") or [],
    )
    # model_validate reads the fields from `created` and validates them against TicketRead.
    return TicketRead.model_validate(created)


@router.get("/tickets/{code}", response_model=TicketRead)
async def lookup_ticket(
    code: str,
    email: EmailMCPClient = Depends(get_email_client),
) -> TicketRead:
    """Return a ticket by reference code, or a neutral 404 if unknown (SPEC §4.8)."""
    ticket = await email.get_ticket_by_code(_normalize_code(code))
    if ticket is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ticket not found")
    return TicketRead.model_validate(ticket)

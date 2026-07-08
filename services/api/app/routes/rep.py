"""Rep-facing routes: the work queue and single-ticket detail (plan Task 4).

- `GET /rep/queue` (SPEC §4.3/§4.7): the New/untriaged tickets awaiting a rep.
- `GET /rep/tickets/{ticket_id}`: the full detail a rep opens; an unknown id
  returns a neutral 404.

Draft review/approve/send actions land in Task 17; this task is read-only. Both
routes require the frontend->api bearer token.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status

from app.mcp_clients.email import EmailMCPClient, get_email_client
from app.schemas.ticket import QueueRow
from app.security import require_auth

router = APIRouter(prefix="/rep", dependencies=[Depends(require_auth)])


@router.get("/queue", response_model=list[QueueRow])
async def rep_queue(
    email: EmailMCPClient = Depends(get_email_client),
) -> list[QueueRow]:
    """Return the New tickets as queue rows for the rep workspace."""
    rows = await email.fetch_new_tickets()
    return [QueueRow.model_validate(row) for row in rows]


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

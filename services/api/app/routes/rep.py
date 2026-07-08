"""Rep-facing routes: the work queue and single-ticket detail (plan Task 4).

- `GET /rep/queue` (SPEC §4.3/§4.7): the New/untriaged tickets awaiting a rep.
- `GET /rep/tickets/{ticket_id}`: the full detail a rep opens; an unknown id
  returns a neutral 404.

Draft review/approve/send actions land in Task 17; this task is read-only. Both
routes require the frontend->api bearer token.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.mcp_clients.email import EmailMCPClient, get_email_client
from app.schemas.ticket import QueuePage, QueueRow
from app.security import require_auth

# Server-side page sizing (plan Task 6, closing the Task-4 unbounded-queue gap):
# a default page and a hard ceiling the client can never exceed, so no request can
# pull the whole New queue in one call.
DEFAULT_QUEUE_LIMIT = 50
MAX_QUEUE_LIMIT = 200

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
    limit: int = Query(default=DEFAULT_QUEUE_LIMIT),
    after: str | None = Query(default=None),
    email: EmailMCPClient = Depends(get_email_client),
) -> QueuePage:
    """Return one keyset page of the New-ticket rep queue.

    `limit` is clamped to `[1, MAX_QUEUE_LIMIT]` so an oversized ask is capped
    rather than rejected; `after` resumes past the last row already seen. A full
    page yields a `next_cursor` (more may remain); a short page yields `None`.
    """
    capped = min(max(limit, 1), MAX_QUEUE_LIMIT)
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

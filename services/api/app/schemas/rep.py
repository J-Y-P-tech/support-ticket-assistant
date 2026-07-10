"""Request/response bodies for the rep draft-review actions (SPEC §4.7, plan Task 17).

The rep dispositions an AI draft at the human-review pause: `edit` and `approve` stage
a decision into the paused run; `send` commits it (the only path to Resolved); `reject`
routes the case back for research. `send`/`reject` carry a `rep_id` — the audit marker
email_mcp requires to attribute the action — validated non-blank here so a blank marker
is a 422 at the boundary, never a silent send. Per-rep identity is not modelled yet
(one shared api token), so the frontend supplies the marker with the request.
"""

from __future__ import annotations

from pydantic import BaseModel, field_validator

from app.schemas.enums import TicketStatus


class RepEditRequest(BaseModel):
    """A rep's edited reply text, staged into the paused case until send."""

    reply: str

    @field_validator("reply")
    @classmethod
    def _reply_not_blank(cls, value: str) -> str:
        """Reject an empty or whitespace-only edit — there is nothing to stage."""
        if not value.strip():
            raise ValueError("reply must not be empty")
        return value


class RepSendRequest(BaseModel):
    """The rep marker authorising a send; required to resolve the case (SPEC §4.7)."""

    rep_id: str

    @field_validator("rep_id")
    @classmethod
    def _rep_id_not_blank(cls, value: str) -> str:
        """Reject a blank marker: email_mcp will not resolve without a real one."""
        if not value.strip():
            raise ValueError("rep_id must not be empty")
        return value


class RepRejectRequest(BaseModel):
    """The rep marker for a rejection, plus an optional reason for the audit trail."""

    rep_id: str
    reason: str | None = None

    @field_validator("rep_id")
    @classmethod
    def _rep_id_not_blank(cls, value: str) -> str:
        """Reject a blank marker so every rejection is attributable."""
        if not value.strip():
            raise ValueError("rep_id must not be empty")
        return value


class RepActionResult(BaseModel):
    """The outcome of a rep action: the case's new status and the reply, if sent.

    `reply` is the customer-facing text on a send and `None` otherwise (a staged
    approve/edit, or a rejection that produced no reply).
    """

    ticket_id: int
    status: TicketStatus
    reply: str | None = None

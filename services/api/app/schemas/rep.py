"""Request/response bodies for the rep draft-review actions (SPEC §4.7, plan Task 17).

The rep dispositions an AI draft at the human-review pause: `edit` and `approve` stage
a decision into the paused run; `send` commits it (the only path to Resolved); `reject`
routes the case back for research. `send`/`reject` carry a `rep_id` — the audit marker
email_mcp requires to attribute the action — validated non-blank here so a blank marker
is a 422 at the boundary, never a silent send. Per-rep identity is not modelled yet
(one shared api token), so the frontend supplies the marker with the request.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.draft import Draft
from app.schemas.enums import TicketStatus
from app.schemas.kb import KBSource
from app.schemas.triage import TriageResult


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


class RepReview(BaseModel):
    """The draft-review payload a rep opens: the paused run projected for the workspace.

    Assembled from the LangGraph state while a case is interrupted before `human_review`
    (SPEC §4.7) — it is not the email_mcp ticket record. It gathers everything the rep
    weighs before dispositioning the AI draft: the original `message`, the OCR
    `extracted_facts` (`None` for text-only tickets), the `triage` classification, the
    retrieved KB `sources`, the `draft` itself (with its citations and `verified` flag),
    and the human-facing signals — `flags` (the accumulated plain-language warnings) and
    `trace_leak`. Every pipeline product is optional because a case can reach the gate
    early (a blocked injection, an unclassifiable ticket, a no-source hand-off) with no
    draft ever written.
    """

    ticket_id: int
    status: TicketStatus
    message: str
    extracted_facts: str | None = None
    triage: TriageResult | None = None
    sources: list[KBSource] = Field(default_factory=list)
    draft: Draft | None = None
    flags: list[str] = Field(default_factory=list)
    trace_leak: bool = False


class RepActionResult(BaseModel):
    """The outcome of a rep action: the case's new status and the reply, if sent.

    `reply` is the customer-facing text on a send and `None` otherwise (a staged
    approve/edit, or a rejection that produced no reply).
    """

    ticket_id: int
    status: TicketStatus
    reply: str | None = None

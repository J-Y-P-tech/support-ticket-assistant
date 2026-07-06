"""Ticket data-transfer objects exchanged by the api routes (plan Task 4).

- `TicketCreate` — customer intake (SPEC §4.1): a non-empty message is required;
  attachments are optional.
- `TicketRead` — customer follow-up (SPEC §4.8): reference code + status + the
  final reply once resolved.
- `QueueRow` — the rep queue table (SPEC §4.3 / §4.7): identity plus the triage
  fields used to sort and open a case (absent until the ticket is triaged).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from app.schemas.enums import Category, TicketStatus, Urgency


class TicketCreate(BaseModel):
    """A customer's new ticket submission."""

    message: str
    attachments: list[str] = Field(default_factory=list)

    @field_validator("message")
    @classmethod
    def _message_not_blank(cls, value: str) -> str:
        """Reject an empty or whitespace-only message (SPEC §4.1)."""
        if not value.strip():
            raise ValueError("message must not be empty")
        return value


class TicketRead(BaseModel):
    """The customer-facing view of a ticket, returned on reference-code lookup."""

    reference_code: str
    status: TicketStatus
    reply: str | None = None


class QueueRow(BaseModel):
    """A single row in the rep's work queue.

    `urgency` and `category` are populated once the ticket is triaged; a New
    ticket carries neither yet.
    """

    id: int
    reference_code: str
    status: TicketStatus
    urgency: Urgency | None = None
    category: Category | None = None

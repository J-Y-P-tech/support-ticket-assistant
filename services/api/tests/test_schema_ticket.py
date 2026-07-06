"""Unit tests for the ticket DTOs (`app.schemas`): create / read / queue-row.

Pins the boundary shapes the api routes (plan Task 4) exchange:
- `TicketCreate` — customer intake (SPEC §4.1): a non-empty message is required
  (empty/whitespace-only is rejected); attachments are optional.
- `TicketRead` — customer follow-up (SPEC §4.8): reference code + status + the
  final reply (present once resolved).
- `QueueRow` — the rep queue table (SPEC §4.3 / §4.7): enough to triage-sort and
  open a ticket; `urgency`/`category` are absent until the ticket is triaged.

Field choices beyond what SPEC fixes were surfaced to the user at the RED review.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.enums import Category, TicketStatus, Urgency
from app.schemas.ticket import QueueRow, TicketCreate, TicketRead


def test_ticket_create_requires_non_empty_message() -> None:
    """A whitespace-only message is rejected (SPEC §4.1 empty-message rule)."""
    with pytest.raises(ValidationError):
        TicketCreate(message="   ")


def test_ticket_create_defaults_to_no_attachments() -> None:
    """A minimal create has a message and an empty attachments list."""
    ticket = TicketCreate(message="I can't log in to my account.")

    assert ticket.message == "I can't log in to my account."
    assert ticket.attachments == []


def test_ticket_create_round_trips_json_with_attachments() -> None:
    """A create carrying attachments survives a JSON round-trip unchanged."""
    ticket = TicketCreate(
        message="See the attached statement.",
        attachments=["statement.pdf", "id-front.jpg"],
    )

    restored = TicketCreate.model_validate_json(ticket.model_dump_json())

    assert restored == ticket


def test_ticket_read_round_trips_and_defaults_reply_none() -> None:
    """A fresh read shows code + status with no reply yet; it round-trips through JSON."""
    read = TicketRead(reference_code="TKT-1042", status=TicketStatus.NEW)

    restored = TicketRead.model_validate_json(read.model_dump_json())

    assert restored == read
    assert restored.reply is None
    assert restored.status is TicketStatus.NEW


def test_ticket_read_rejects_invalid_status() -> None:
    """The customer-facing read rejects a status outside the lifecycle enum."""
    with pytest.raises(ValidationError):
        TicketRead.model_validate({"reference_code": "TKT-1", "status": "done"})


def test_queue_row_round_trips_with_triage_fields() -> None:
    """A triaged queue row carries urgency + category and round-trips through JSON."""
    row = QueueRow(
        id=7,
        reference_code="TKT-1042",
        status=TicketStatus.TRIAGED,
        urgency=Urgency.HIGH,
        category=Category.CARD_ISSUES,
    )

    restored = QueueRow.model_validate_json(row.model_dump_json())

    assert restored == row
    assert restored.urgency is Urgency.HIGH


def test_queue_row_allows_untriaged_ticket() -> None:
    """A New (untriaged) ticket has no urgency/category yet, so both may be omitted."""
    row = QueueRow(id=1, reference_code="TKT-1043", status=TicketStatus.NEW)

    assert row.urgency is None
    assert row.category is None

"""Shared enumerations for triage, ticket lifecycle, sources, and feedback.

These are the single source of truth for the closed value sets the whole system
exchanges. `Urgency`, `TicketStatus`, and `FeedbackDecision` are copied verbatim
from SPEC (§4.3, §5, §4.9). `Category` and `Sentiment` are not enumerated in SPEC;
their values were confirmed with the user (2026-07-06).

Every enum is a `StrEnum`, so a value serializes to its plain string in JSON and
compares equal to that string, while still validating a closed set on input.
"""

from __future__ import annotations

from enum import StrEnum


class Urgency(StrEnum):
    """Triage urgency level, used for rep-queue sort order (SPEC §4.3)."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"


class TicketStatus(StrEnum):
    """Case lifecycle status; spelled exactly as in SPEC §5.

    Customer-visible subset (SPEC §5): New, Pending, Resolved, Canceled.
    """

    NEW = "New"
    TRIAGED = "Triaged"
    RESEARCHING = "Researching"
    DRAFTED = "Drafted"
    PENDING = "Pending"
    RESOLVED = "Resolved"
    CANCELED = "Canceled"
    NEEDS_RESEARCH = "NeedsResearch"


class Category(StrEnum):
    """Support topic a ticket is triaged into (finance-desk set, confirmed 2026-07-06)."""

    ACCOUNT_ACCESS = "account_access"
    PAYMENTS_BILLING = "payments_billing"
    CARD_ISSUES = "card_issues"
    TRANSACTION_DISPUTE_FRAUD = "transaction_dispute_fraud"
    LOANS_CREDIT = "loans_credit"
    TECHNICAL_PROBLEM = "technical_problem"
    GENERAL_INQUIRY = "general_inquiry"
    OTHER = "other"


class Sentiment(StrEnum):
    """Customer mood inferred at triage (confirmed 2026-07-06)."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


class FeedbackDecision(StrEnum):
    """The rep's disposition of an AI draft at review (SPEC §4.9)."""

    APPROVED_AS_IS = "approved_as_is"
    EDITED = "edited"
    REJECTED = "rejected"

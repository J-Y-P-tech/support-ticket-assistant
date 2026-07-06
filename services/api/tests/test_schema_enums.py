"""Unit tests for the shared triage/status enums (`app.schemas`).

These tests pin the Task 2 acceptance criteria (plan Task 2):
- `Urgency` and `TicketStatus` match SPEC §4.3 / §5 *exactly* (values and spelling).
- `SourceKind`, `Category`, `Sentiment`, and `FeedbackDecision` expose the agreed
  value sets, and every enum rejects an unknown value.

The value sets are asserted here — rather than in each model's test — so the
contract that the rest of the system shares has a single, explicit source of truth.
Category and Sentiment values were confirmed with the user (2026-07-06); the others
are copied verbatim from SPEC.
"""

from __future__ import annotations

import pytest

from app.schemas.enums import (
    Category,
    FeedbackDecision,
    Sentiment,
    SourceKind,
    TicketStatus,
    Urgency,
)


def test_urgency_values_match_spec() -> None:
    """`Urgency` is exactly low/normal/high/critical (SPEC §4.3)."""
    assert [u.value for u in Urgency] == ["low", "normal", "high", "critical"]


def test_ticket_status_values_match_spec() -> None:
    """`TicketStatus` is exactly the SPEC §5 lifecycle set, spelled as in the spec."""
    assert [s.value for s in TicketStatus] == [
        "New",
        "Triaged",
        "Researching",
        "Drafted",
        "Pending",
        "Resolved",
        "Canceled",
        "NeedsResearch",
    ]


def test_source_kind_values_match_spec() -> None:
    """`SourceKind` is exactly authoritative/model_generated (SPEC §4.4)."""
    assert [k.value for k in SourceKind] == ["authoritative", "model_generated"]


def test_category_values_match_agreement() -> None:
    """`Category` is the finance-desk topic set confirmed with the user (2026-07-06)."""
    assert [c.value for c in Category] == [
        "account_access",
        "payments_billing",
        "card_issues",
        "transaction_dispute_fraud",
        "loans_credit",
        "technical_problem",
        "general_inquiry",
        "other",
    ]


def test_sentiment_values_match_agreement() -> None:
    """`Sentiment` is negative/neutral/positive, confirmed with the user (2026-07-06)."""
    assert [s.value for s in Sentiment] == ["negative", "neutral", "positive"]


def test_feedback_decision_values_match_spec() -> None:
    """`FeedbackDecision` is exactly the SPEC §4.9 rep outcomes."""
    assert [d.value for d in FeedbackDecision] == ["approved_as_is", "edited", "rejected"]


@pytest.mark.parametrize(
    ("enum_cls", "bad_value"),
    [
        (Urgency, "urgent"),
        (TicketStatus, "new"),  # wrong case: must be "New"
        (SourceKind, "trusted"),
        (Category, "billing"),  # not in the agreed set (payments_billing)
        (Sentiment, "angry"),
        (FeedbackDecision, "approved"),
    ],
)
def test_enum_rejects_unknown_value(enum_cls: type, bad_value: str) -> None:
    """Constructing any enum from a value outside its set raises `ValueError`."""
    with pytest.raises(ValueError):
        enum_cls(bad_value)


def test_str_enums_serialize_to_plain_strings() -> None:
    """Enums are `StrEnum`, so each member is its plain value string (for JSON)."""
    assert Urgency.HIGH.value == "high"
    assert TicketStatus.NEW.value == "New"
    assert Category.CARD_ISSUES.value == "card_issues"
    # `StrEnum` also renders as the bare value via `str()`, not "Urgency.HIGH".
    assert str(Urgency.HIGH) == "high"

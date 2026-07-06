"""Unit tests for `TriageResult` (`app.schemas`).

Pins the Task 2 acceptance criterion "each schema round-trips valid JSON and
rejects invalid enum values" for the triage output the `triage` node (SPEC §5.3)
must return: a validated `category` + `urgency` + `sentiment` (SPEC §4.3).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.enums import Category, Sentiment, Urgency
from app.schemas.triage import TriageResult


def test_valid_triage_round_trips_json() -> None:
    """A valid triage result survives a dump→load JSON round-trip unchanged."""
    result = TriageResult(
        category=Category.CARD_ISSUES,
        urgency=Urgency.HIGH,
        sentiment=Sentiment.NEGATIVE,
    )

    restored = TriageResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.category is Category.CARD_ISSUES
    assert restored.urgency is Urgency.HIGH
    assert restored.sentiment is Sentiment.NEGATIVE


def test_triage_accepts_plain_string_enum_values() -> None:
    """Raw LLM JSON with plain string enum values validates into typed members."""
    result = TriageResult.model_validate(
        {"category": "loans_credit", "urgency": "critical", "sentiment": "neutral"}
    )

    assert result.category is Category.LOANS_CREDIT
    assert result.urgency is Urgency.CRITICAL
    assert result.sentiment is Sentiment.NEUTRAL


def test_triage_rejects_invalid_enum_value() -> None:
    """An out-of-set category value is rejected rather than silently accepted."""
    with pytest.raises(ValidationError):
        TriageResult.model_validate(
            {"category": "mortgage", "urgency": "high", "sentiment": "negative"}
        )


def test_triage_requires_all_three_fields() -> None:
    """Omitting any of category/urgency/sentiment raises a validation error."""
    with pytest.raises(ValidationError):
        TriageResult.model_validate({"category": "other", "urgency": "low"})

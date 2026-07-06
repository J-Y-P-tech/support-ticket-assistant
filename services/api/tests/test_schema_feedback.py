"""Unit tests for `FeedbackRecord` (`app.schemas`).

Pins the feedback-capture contract (SPEC §4.9): every rep decision is recorded as
approved-as-is / edited (with the AI-vs-final diff) / rejected, plus an optional
rating and reason. The record keeps the AI draft, the final reply, and the edit
distance so the quality loop (§7.4) and training corpus (§4.9a) can consume it.

`rating` is left as an unbounded optional integer here: SPEC does not define a
rating scale, so no range is invented (flagged to the user at the RED review).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.enums import FeedbackDecision
from app.schemas.feedback import FeedbackRecord


def test_approved_as_is_feedback_round_trips_json() -> None:
    """An approved-as-is record (zero edit distance) round-trips through JSON."""
    record = FeedbackRecord(
        decision=FeedbackDecision.APPROVED_AS_IS,
        ai_draft="Here is how to reset your access ...",
        final_reply="Here is how to reset your access ...",
        edit_distance=0,
        rating=5,
        reason=None,
    )

    restored = FeedbackRecord.model_validate_json(record.model_dump_json())

    assert restored == record


def test_edited_feedback_captures_diff_distance() -> None:
    """An edited record keeps both drafts and a positive edit distance (SPEC §4.9)."""
    record = FeedbackRecord(
        decision=FeedbackDecision.EDITED,
        ai_draft="draft text",
        final_reply="final edited text",
        edit_distance=9,
    )

    assert record.decision is FeedbackDecision.EDITED
    assert record.edit_distance == 9
    assert record.rating is None


def test_rejected_feedback_allows_no_final_reply() -> None:
    """A rejected draft has no final reply and no edit distance."""
    record = FeedbackRecord(
        decision=FeedbackDecision.REJECTED,
        ai_draft="draft the rep threw away",
        reason="off-topic; needs research",
    )

    assert record.final_reply is None
    assert record.edit_distance is None


def test_feedback_rejects_invalid_decision() -> None:
    """A decision value outside the enum is rejected."""
    with pytest.raises(ValidationError):
        FeedbackRecord.model_validate({"decision": "approved", "ai_draft": "x"})


def test_feedback_rejects_negative_edit_distance() -> None:
    """Edit distance cannot be negative."""
    with pytest.raises(ValidationError):
        FeedbackRecord.model_validate(
            {"decision": "edited", "ai_draft": "x", "final_reply": "y", "edit_distance": -1}
        )


def test_feedback_requires_decision_and_draft() -> None:
    """The rep `decision` and the `ai_draft` it applies to are mandatory."""
    with pytest.raises(ValidationError):
        FeedbackRecord.model_validate({"reason": "no decision given"})

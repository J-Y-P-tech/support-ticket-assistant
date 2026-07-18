"""Unit tests for feedback capture: edit distance + record construction (todo Task 27).

SPEC §4.9 records every rep decision as approved-as-is / edited (with the AI-vs-final
diff) / rejected, plus an optional rating and reason. These pin the two *pure* pieces
of the capture path: `edit_distance` (character-level Levenshtein — the diff measure
confirmed with the user) and `build_feedback_record`, which turns a finished workflow
state plus the rep's rating/reason into the `FeedbackRecord` the email_mcp feedback
table stores. No I/O, no model, no database.
"""

from __future__ import annotations

from typing import Any

from app.graph.feedback import build_feedback_record, edit_distance
from app.schemas.draft import Draft
from app.schemas.enums import FeedbackDecision


def _finished_state(
    *,
    decision: FeedbackDecision,
    draft_body: str = "AI draft body",
    final_reply: str | None = None,
) -> dict[str, Any]:
    """Build a minimal finished-run state slice for `build_feedback_record`.

    Mirrors what `finalize` leaves in LangGraph state: the drafted reply, the rep's
    decision, and the `final_reply` (the sent text, or `None` for a rejection).
    """
    state: dict[str, Any] = {"rep_decision": decision, "draft": Draft(body=draft_body)}
    state["final_reply"] = final_reply
    return state


def test_edit_distance_of_identical_strings_is_zero() -> None:
    """An unedited (approved-as-is) reply has an edit distance of zero from the draft."""
    assert edit_distance("reset via the login screen", "reset via the login screen") == 0


def test_edit_distance_counts_a_single_substitution() -> None:
    """One changed character is one edit."""
    assert edit_distance("draft", "graft") == 1


def test_edit_distance_from_empty_is_the_other_length() -> None:
    """Building a string from nothing (or clearing it) costs one edit per character."""
    assert edit_distance("", "hello") == 5
    assert edit_distance("hello", "") == 5


def test_edit_distance_classic_kitten_sitting_is_three() -> None:
    """The textbook Levenshtein example: `kitten` → `sitting` is three edits."""
    assert edit_distance("kitten", "sitting") == 3


def test_approved_as_is_record_has_zero_edit_distance() -> None:
    """An approved-as-is draft yields a record whose final matches the draft (distance 0)."""
    body = "You can reset your password from the login screen."
    record = build_feedback_record(
        _finished_state(
            decision=FeedbackDecision.APPROVED_AS_IS, draft_body=body, final_reply=body
        ),
        rating=5,
        reason=None,
    )

    assert record is not None
    assert record.decision is FeedbackDecision.APPROVED_AS_IS
    assert record.ai_draft == body
    assert record.final_reply == body
    assert record.edit_distance == 0
    assert record.rating == 5


def test_edited_record_captures_positive_distance_and_the_edited_text() -> None:
    """An edited draft keeps both texts and the character distance between them (SPEC §4.9)."""
    draft_body = "Reset your password."
    edited = "Please reset your password from the login screen."
    record = build_feedback_record(
        _finished_state(
            decision=FeedbackDecision.EDITED, draft_body=draft_body, final_reply=edited
        ),
        rating=None,
        reason="tightened the tone",
    )

    assert record is not None
    assert record.decision is FeedbackDecision.EDITED
    assert record.ai_draft == draft_body
    assert record.final_reply == edited
    assert record.edit_distance == edit_distance(draft_body, edited)
    assert record.edit_distance > 0
    assert record.reason == "tightened the tone"


def test_rejected_record_has_no_final_reply_or_distance() -> None:
    """A rejected draft keeps the AI draft but records no final reply and no distance."""
    record = build_feedback_record(
        _finished_state(decision=FeedbackDecision.REJECTED, draft_body="the discarded draft"),
        rating=2,
        reason="off-topic; needs research",
    )

    assert record is not None
    assert record.decision is FeedbackDecision.REJECTED
    assert record.ai_draft == "the discarded draft"
    assert record.final_reply is None
    assert record.edit_distance is None
    assert record.rating == 2
    assert record.reason == "off-topic; needs research"


def test_record_captures_the_triage_category() -> None:
    """A finished run's triage category rides onto the feedback record (todo Task 30).

    The category tags the approved reply so the live few-shot lookup can later select it
    for a same-category drafting ticket (SPEC §4.10). It is read from the `triage` result
    `finalize` left in state.
    """
    from app.schemas.enums import Category, Sentiment, Urgency
    from app.schemas.triage import TriageResult

    state = _finished_state(
        decision=FeedbackDecision.APPROVED_AS_IS, draft_body="reset it", final_reply="reset it"
    )
    state["triage"] = TriageResult(
        category=Category.ACCOUNT_ACCESS, urgency=Urgency.NORMAL, sentiment=Sentiment.NEUTRAL
    )

    record = build_feedback_record(state)

    assert record is not None
    assert record.category is Category.ACCOUNT_ACCESS


def test_record_without_triage_leaves_category_none() -> None:
    """A finished run missing its triage result records no category (stays None)."""
    record = build_feedback_record(
        _finished_state(decision=FeedbackDecision.EDITED, draft_body="a", final_reply="b")
    )

    assert record is not None
    assert record.category is None


def test_no_draft_yields_no_feedback_record() -> None:
    """A case handed to a human with no draft has no AI draft to rate — no record.

    A no-confident-source hand-off or a blocked injection reaches the human gate
    without a draft ever being written. Rejecting such a case must not fabricate a
    feedback row whose `ai_draft` would be empty; there is simply nothing to capture.
    """
    record = build_feedback_record(
        {"rep_decision": FeedbackDecision.REJECTED, "final_reply": None},
        rating=None,
        reason=None,
    )

    assert record is None

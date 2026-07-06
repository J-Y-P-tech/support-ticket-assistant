"""Rep feedback captured at review time (SPEC §4.9 / §4.9a).

Records how the rep disposed of an AI draft, keeping the original draft, the final
reply, and the edit distance so the quality loop (§7.4) and the training corpus
(§4.9a) can consume it. `rating` is an unbounded optional integer: SPEC defines no
rating scale, so none is invented here.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.enums import FeedbackDecision


class FeedbackRecord(BaseModel):
    """One rep decision on a draft: approved-as-is / edited / rejected.

    `final_reply` and `edit_distance` are absent for a rejected draft; for an
    approved-as-is draft the edit distance is zero and the final matches the draft.
    """

    decision: FeedbackDecision
    ai_draft: str
    final_reply: str | None = None
    edit_distance: int | None = Field(default=None, ge=0)
    rating: int | None = None
    reason: str | None = None

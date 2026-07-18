"""Turn a finished workflow run into its rep-feedback row (plan Task 25 / todo Task 27).

SPEC §4.9 records every rep decision — approved-as-is / edited (with the diff between
the AI draft and the final reply) / rejected, plus an optional rating and reason — so
the quality loop (§7.4) and the training corpus (§4.9a) can consume it. This module is
the *emission* half, mirroring `graph/audit.py`: it reads a finished (resumed) LangGraph
state and produces the `FeedbackRecord` the email_mcp feedback table stores.

The split is deliberate, exactly as for audit. `edit_distance` and
`build_feedback_record` are **pure** — no I/O, no model, no database — so every
disposition (approved-as-is / edited / rejected, and the no-draft hand-off) is
unit-testable in isolation. `record_feedback` is the thin write step the send/reject
routes call once `finalize` has run; it just forwards the built record through the
email_mcp client. Persistence lives at the service boundary, never inside a graph node.

The diff measure is the classic character-level Levenshtein edit distance (confirmed
with the user): the fewest single-character insertions, deletions, and substitutions
that turn the AI draft into the rep's final reply. An approved-as-is reply is unchanged,
so its distance is zero; a rejection produces no final reply, so it has no distance.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from app.schemas.enums import FeedbackDecision
from app.schemas.feedback import FeedbackRecord


class _FeedbackRecorder(Protocol):
    """The one method `record_feedback` needs from the email_mcp client.

    Typed structurally so the write step depends only on `record_feedback`, not on the
    whole `EmailMCPClient` — the tests pass a lightweight recording fake.
    """

    async def record_feedback(self, ticket_id: int, record: FeedbackRecord) -> dict[str, Any]:
        """Persist one rep-decision feedback row and return the stored row."""
        ...


def edit_distance(a: str, b: str) -> int:
    """Return the character-level Levenshtein distance between two strings.

    The fewest single-character insertions, deletions, and substitutions that turn `a`
    into `b` — the diff measure SPEC §4.9 records between the AI draft and the rep's
    final reply. Identical strings are distance zero. Computed with the standard
    two-row dynamic-programming recurrence (O(len(a)·len(b)) time, O(len(b)) space), so
    it needs no third-party dependency.
    """
    # Distance from the empty prefix of `a` to each prefix of `b`: all insertions.
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        # Row i starts with the cost of deleting `a`'s first i characters.
        current = [i]
        for j, cb in enumerate(b, start=1):
            substitution = previous[j - 1] + (ca != cb)
            deletion = previous[j] + 1
            insertion = current[j - 1] + 1
            current.append(min(substitution, deletion, insertion))
        previous = current
    return previous[-1]


def build_feedback_record(
    state: Mapping[str, Any], *, rating: int | None = None, reason: str | None = None
) -> FeedbackRecord | None:
    """Map a finished workflow state to its rep-feedback row, or `None` if none applies.

    Reads the disposition `finalize` left in `state` — the rep's `rep_decision`, the AI
    `draft`, and the `final_reply` (the sent text, or `None` for a rejection) — and pairs
    it with the rep's optional `rating`/`reason`. For an approved/edited case the edit
    distance is the character diff between the AI draft and the final reply (zero when
    approved unchanged); a rejection carries the discarded draft with no final reply and
    no distance. Returns `None` when there is no draft to rate: a no-confident-source
    hand-off or a blocked injection reaches the human gate without a draft, so a rejection
    there has no AI reply to give feedback on. Pure: no I/O, mutates nothing.
    """
    draft = state.get("draft")
    if draft is None:
        return None
    decision: FeedbackDecision = state["rep_decision"]
    final_reply = state.get("final_reply")
    # A rejection has no final reply, so there is nothing to diff against the draft.
    distance = None if final_reply is None else edit_distance(draft.body, final_reply)
    # Tag the row with the ticket's triage category so an approved reply can be selected
    # as a same-category few-shot example later (SPEC §4.10). Absent when the run never
    # triaged (e.g. a blocked injection reaching the gate with no triage result).
    triage = state.get("triage")
    return FeedbackRecord(
        decision=decision,
        ai_draft=draft.body,
        final_reply=final_reply,
        edit_distance=distance,
        rating=rating,
        reason=reason,
        category=triage.category if triage is not None else None,
    )


async def record_feedback(
    email: _FeedbackRecorder,
    *,
    ticket_id: int,
    state: Mapping[str, Any],
    rating: int | None = None,
    reason: str | None = None,
) -> None:
    """Write the rep-feedback row for a finished run through the email_mcp client.

    The thin write step behind the send/reject routes: it builds the record for `state`
    and, when there is one (a case with a draft), records it under `ticket_id` so the
    feedback table gains the rep's disposition. A no-draft hand-off produces no record,
    so nothing is written.
    """
    record = build_feedback_record(state, rating=rating, reason=reason)
    if record is not None:
        await email.record_feedback(ticket_id, record)

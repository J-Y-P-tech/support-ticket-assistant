"""Outcome of the validate node: a scored, possibly-flagged draft (SPEC §4.5)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.draft import Draft


class GroundednessVerdict(BaseModel):
    """The groundedness judge's structured verdict on a draft (SPEC §4.5).

    `score` is the fraction of the reply's factual claims the cited sources support,
    in [0, 1]; `unsupported_claims` names the claims the judge could not back. The
    validate node validates the model's JSON against this schema and retries once on
    failure, so no unvalidated judgement flows into the flag decision. An out-of-range
    score fails validation here and drives a retry, not a silently clamped result.
    """

    score: float = Field(ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)


class ValidationResult(BaseModel):
    """The validate node's verdict on a draft before it reaches the rep.

    `draft` is the reply carried forward, with its `verified` flag downgraded to
    False when the draft is flagged. `groundedness` is the LLM-as-judge score in
    [0, 1] measuring how much of the reply is backed by its cited sources — stored
    as case metadata (SPEC §7.2). `flagged` is True when the draft must be surfaced
    to the rep as "AI-suggested, unverified" rather than passed as sourced fact, and
    `reasons` lists why (low groundedness with the unsupported claims, a judge that
    could not run, an already-unverified draft, or an empty body). Flagging never
    blocks the flow — every draft still reaches human review.
    """

    draft: Draft
    groundedness: float
    flagged: bool
    reasons: list[str] = Field(default_factory=list)

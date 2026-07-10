"""Grounded reply draft and its source citations (SPEC §4.5)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A reference to the KB source a draft drew from (its id and title)."""

    source_id: str
    title: str


class Draft(BaseModel):
    """A drafted reply written from authoritative sources, awaiting rep review.

    `verified` is True for a draft that stays grounded in its cited sources. The
    validate node (todo Task 14) sets it False when the draft's groundedness is low
    — the model drifting off its sources; such a draft is shown as "AI-suggested,
    unverified" and cannot be presented as sourced fact (SPEC §4.5).
    """

    body: str
    citations: list[Citation] = Field(default_factory=list)
    verified: bool = True

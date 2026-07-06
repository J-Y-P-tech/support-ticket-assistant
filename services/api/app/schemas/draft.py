"""Grounded reply draft and its source citations (SPEC §4.5)."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Citation(BaseModel):
    """A reference to the KB source a draft drew from (its id and title)."""

    source_id: str
    title: str


class Draft(BaseModel):
    """A drafted reply written from authoritative sources, awaiting rep review.

    `unverified` marks a draft built on a `model_generated` source or with low
    groundedness; such a draft is shown as "AI-suggested, unverified" and cannot
    be presented as sourced fact (SPEC §4.5).
    """

    body: str
    citations: list[Citation] = Field(default_factory=list)
    unverified: bool = False

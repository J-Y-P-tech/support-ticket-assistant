"""Unit tests for `Draft` and `Citation` (`app.schemas`).

Pins the grounded-drafting contract (SPEC §4.5): a draft is written from
authoritative sources and cites which source `id`/`title` it used; a draft built
on a `model_generated` source (or otherwise low-groundedness) is flagged
"AI-suggested, unverified" via `verified=False` and cannot be presented as
sourced fact.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.draft import Citation, Draft


def test_grounded_draft_round_trips_json() -> None:
    """A cited, verified draft survives a dump→load JSON round-trip unchanged."""
    draft = Draft(
        body="You can reset your access from the login page ...",
        citations=[Citation(source_id="kb-042", title="Resetting online-banking access")],
        verified=True,
    )

    restored = Draft.model_validate_json(draft.model_dump_json())

    assert restored == draft
    assert restored.citations[0].source_id == "kb-042"


def test_draft_defaults_to_no_citations_and_verified_true() -> None:
    """A bare draft has no citations and is verified by default."""
    draft = Draft(body="hello")

    assert draft.citations == []
    assert draft.verified is True


def test_unverified_draft_carries_the_flag() -> None:
    """A model-generated/low-groundedness draft records `verified=False` (SPEC §4.5)."""
    draft = Draft(body="best-effort answer", verified=False)

    assert draft.verified is False


def test_draft_requires_body() -> None:
    """The reply `body` is mandatory."""
    with pytest.raises(ValidationError):
        Draft.model_validate({"citations": []})


def test_citation_requires_source_id_and_title() -> None:
    """A citation must name both the source id and its title."""
    with pytest.raises(ValidationError):
        Citation.model_validate({"source_id": "kb-1"})

"""Unit tests for `KBSource` (`app.schemas`).

Pins the knowledge-retrieval chunk contract (SPEC §4.4): each source carries an
`id`, `title`, `text`, and a `source_kind` that is either `authoritative` (mock-KB
canned answers) or `model_generated` (never counts as grounding — SPEC §4.5).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.enums import SourceKind
from app.schemas.kb import KBSource


def test_valid_kb_source_round_trips_json() -> None:
    """A valid KB source survives a dump→load JSON round-trip unchanged."""
    source = KBSource(
        id="kb-042",
        title="Resetting online-banking access",
        text="To reset access, verify identity then ...",
        source_kind=SourceKind.AUTHORITATIVE,
    )

    restored = KBSource.model_validate_json(source.model_dump_json())

    assert restored == source
    assert restored.source_kind is SourceKind.AUTHORITATIVE


def test_kb_source_rejects_invalid_source_kind() -> None:
    """An unknown `source_kind` is rejected (only authoritative/model_generated allowed)."""
    with pytest.raises(ValidationError):
        KBSource.model_validate({"id": "kb-1", "title": "t", "text": "x", "source_kind": "trusted"})


def test_kb_source_requires_all_fields() -> None:
    """Every field (id/title/text/source_kind) is mandatory."""
    with pytest.raises(ValidationError):
        KBSource.model_validate({"id": "kb-1", "title": "t", "text": "x"})

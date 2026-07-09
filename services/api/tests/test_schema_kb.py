"""Unit tests for `KBSource` and `KBSearchResult` (`app.schemas`).

Pins the knowledge-retrieval chunk contract (SPEC §4.4): each source carries an
`id`, `title`, `text`, and a `source_kind` that is either `authoritative` (mock-KB
canned answers) or `model_generated` (never counts as grounding — SPEC §4.5).
`KBSearchResult` bundles the ranked sources with the explicit `no_confident_source`
signal the kb client hands back to the agent.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.enums import SourceKind
from app.schemas.kb import KBSearchResult, KBSource


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


def test_kb_search_result_round_trips_json() -> None:
    """A search result (sources + signal) survives a dump→load JSON round-trip."""
    result = KBSearchResult(
        sources=[
            KBSource(
                id="kb-1",
                title="Reset access",
                text="Verify identity, then reset ...",
                source_kind=SourceKind.AUTHORITATIVE,
            )
        ],
        no_confident_source=False,
    )

    restored = KBSearchResult.model_validate_json(result.model_dump_json())

    assert restored == result
    assert restored.no_confident_source is False


def test_kb_search_result_carries_signal_with_empty_sources() -> None:
    """`no_confident_source` is an explicit field, valid even when no sources matched."""
    result = KBSearchResult(sources=[], no_confident_source=True)

    assert result.sources == []
    assert result.no_confident_source is True

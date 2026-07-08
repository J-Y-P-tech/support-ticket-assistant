"""Contract tests for the MockKBProvider keyword lookup (plan Task 7 / todo Task 8).

Acceptance: a matching query returns ranked, schema-valid `KBSource` chunks; a
query with no real overlap returns nothing (the no-confident-source path); and the
provider is swappable behind the `KBProvider` interface. Returned chunks validate
against the *shared* `app.schemas.kb.KBSource` — the same type the agent consumes —
so kb_mcp and the rest of the system are proven to agree on shape and provenance.
"""

from __future__ import annotations

from typing import Any

from app.schemas.enums import SourceKind
from app.schemas.kb import KBSource
from providers.base import KBProvider
from providers.mock_kb import MockKBProvider


def test_matching_query_returns_ranked_kbsource_chunks() -> None:
    sources = MockKBProvider().search("I forgot my password and can't log in", limit=3)

    assert sources, "a password question should match at least one curated answer"
    validated = [KBSource.model_validate(source) for source in sources]
    # Mock-KB answers are always authoritative grounding (SPEC §4.4).
    assert all(source.source_kind is SourceKind.AUTHORITATIVE for source in validated)
    # The password-reset answer overlaps the query most, so it ranks first.
    assert validated[0].id == "kb-reset-password"


def test_limit_caps_the_number_of_chunks() -> None:
    sources = MockKBProvider().search("card payment declined transaction dispute", limit=2)

    assert len(sources) <= 2


def test_no_overlap_returns_no_sources() -> None:
    # None of these words appear in any curated answer's indexed terms.
    assert MockKBProvider().search("photosynthesis chlorophyll sunlight", limit=3) == []


def test_stopword_only_query_returns_no_sources() -> None:
    # A query made only of stopwords has no significant terms, so nothing matches.
    assert MockKBProvider().search("how do I get this to the", limit=3) == []


def test_provider_is_swappable_behind_the_interface() -> None:
    """A second provider implementing `KBProvider` needs no change elsewhere."""

    class DummyProvider(KBProvider):
        def search(self, query: str, limit: int) -> list[dict[str, Any]]:
            return [{"id": "dummy-1", "title": "T", "text": "B", "source_kind": "authoritative"}]

    sources = DummyProvider().search("anything", limit=1)

    assert KBSource.model_validate(sources[0]).id == "dummy-1"

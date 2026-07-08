"""Contract tests for the `search_knowledge_base` tool logic (plan Task 7 / todo Task 8).

These exercise `kb_search.run_search` — the exact payload the MCP tool returns —
over a stub provider, so the envelope contract (ranked sources + the
`no_confident_source` signal) is verified independently of the curated data and
proven to hold behind *any* `KBProvider`.
"""

from __future__ import annotations

from typing import Any

import kb_search
from app.schemas.kb import KBSource
from providers.base import KBProvider

_AUTHORITATIVE = {
    "id": "kb-1",
    "title": "Reset your password",
    "text": "Open Forgot password and follow the emailed link.",
    "source_kind": "authoritative",
}
_MODEL_GENERATED = {
    "id": "gen-1",
    "title": "Suggested answer",
    "text": "A model-drafted guess with no real source behind it.",
    "source_kind": "model_generated",
}


class _StubProvider(KBProvider):
    def __init__(self, sources: list[dict[str, Any]]) -> None:
        self._sources = sources

    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        return self._sources[:limit]


def test_match_returns_sources_and_clears_no_confident_flag() -> None:
    payload = kb_search.run_search(_StubProvider([_AUTHORITATIVE]), "reset password", limit=3)

    assert payload["no_confident_source"] is False
    assert [KBSource.model_validate(s).id for s in payload["sources"]] == ["kb-1"]


def test_no_sources_signals_no_confident_source() -> None:
    payload = kb_search.run_search(_StubProvider([]), "nothing matches here", limit=3)

    assert payload["sources"] == []
    assert payload["no_confident_source"] is True


def test_model_generated_only_still_signals_no_confident_source() -> None:
    """A model-generated fallback never counts as authoritative grounding (SPEC §4.5)."""
    payload = kb_search.run_search(_StubProvider([_MODEL_GENERATED]), "q", limit=3)

    assert payload["sources"], "the fallback source is still returned to the caller"
    assert payload["no_confident_source"] is True

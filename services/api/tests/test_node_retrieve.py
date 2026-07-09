"""Unit tests for the retrieve node + groundedness gate (plan Task 11 / todo Task 12).

`retrieve` asks the kb client for sources for a query; `groundedness_gate` decides
whether the case may proceed to drafting or must be routed to a human for research.
Both are standalone functions here — the LangGraph wiring that turns the gate into a
conditional edge is added when the workflow is assembled (todo Task 17) — so these
tests exercise them directly, with the kb client faked (no network, SPEC §10/§12).

The behaviours pinned here are the acceptance criteria (SPEC §4.4 / §4.5):

- `retrieve` forwards the query to `search_knowledge_base` and returns the typed
  `KBSearchResult` unchanged (sources + the explicit no-confident-source signal);
- an **authoritative** match routes to `draft`;
- **no confident source** routes to `flag_needs_research` — never a drafted answer;
- a `model_generated`-only result routes to `flag_needs_research` too: such a source
  never counts as grounding (SPEC §4.5), even if the provider claims confidence;
- the gate treats an empty source list defensively — no sources, no drafting.
"""

from __future__ import annotations

import pytest

from app.graph.nodes.retrieve import DRAFT, FLAG_NEEDS_RESEARCH, groundedness_gate, retrieve
from app.mcp_clients.kb import KBMCPClient
from app.schemas.enums import SourceKind
from app.schemas.kb import KBSearchResult, KBSource


def _authoritative(id_: str = "kb-1") -> KBSource:
    """Build an authoritative KB source — the kind that may ground a draft."""
    return KBSource(
        id=id_,
        title="Reset access",
        text="Verify identity, then reset ...",
        source_kind=SourceKind.AUTHORITATIVE,
    )


def _model_generated(id_: str = "kb-9") -> KBSource:
    """Build a model_generated KB source — a fallback that never grounds (SPEC §4.5)."""
    return KBSource(
        id=id_,
        title="Best guess",
        text="Unverified suggestion ...",
        source_kind=SourceKind.MODEL_GENERATED,
    )


@pytest.fixture
def kb_client() -> KBMCPClient:
    """A `KBMCPClient` at a dummy endpoint; tests monkeypatch its `search`.

    Mirrors `test_mcp_client_kb.py`: no connection is opened, so `retrieve` runs
    entirely offline against a canned search result.
    """
    return KBMCPClient(url="http://kb_mcp:8000/mcp", token="test-token", default_limit=3)


async def test_retrieve_forwards_query_and_returns_result(
    kb_client: KBMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`retrieve` calls the client's search with the query and returns its result verbatim."""
    seen: dict[str, str] = {}
    canned = KBSearchResult(sources=[_authoritative()], no_confident_source=False)

    async def fake_search(query: str, limit: int | None = None) -> KBSearchResult:
        seen["query"] = query
        return canned

    monkeypatch.setattr(kb_client, "search", fake_search)

    result = await retrieve("how do I reset my password", kb_client)

    assert seen["query"] == "how do I reset my password"
    assert result is canned


async def test_retrieve_passes_through_no_confident_source(
    kb_client: KBMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A no-match search result is returned intact so the gate can route it."""

    async def fake_search(query: str, limit: int | None = None) -> KBSearchResult:
        return KBSearchResult(sources=[], no_confident_source=True)

    monkeypatch.setattr(kb_client, "search", fake_search)

    result = await retrieve("qwerty zzz nonsense", kb_client)

    assert result.sources == []
    assert result.no_confident_source is True


def test_gate_routes_authoritative_to_draft() -> None:
    """An authoritative source with provider confidence proceeds to drafting."""
    result = KBSearchResult(sources=[_authoritative()], no_confident_source=False)
    assert groundedness_gate(result) == DRAFT


def test_gate_routes_no_confident_source_to_needs_research() -> None:
    """No confident source → needs-human-research, never a drafted answer (SPEC §4.4)."""
    result = KBSearchResult(sources=[], no_confident_source=True)
    assert groundedness_gate(result) == FLAG_NEEDS_RESEARCH


def test_gate_routes_model_generated_only_to_needs_research() -> None:
    """A model_generated fallback never grounds a reply: route to needs-research (SPEC §4.5).

    Even when the provider populates `sources` and claims confidence, a source that
    is not authoritative must not be presented as sourced fact — the committed kb
    client contract routes this case to a human.
    """
    result = KBSearchResult(sources=[_model_generated()], no_confident_source=False)
    assert groundedness_gate(result) == FLAG_NEEDS_RESEARCH


def test_gate_ignores_model_generated_when_flag_set() -> None:
    """The no-confident-source flag routes to needs-research even alongside sources."""
    result = KBSearchResult(sources=[_model_generated()], no_confident_source=True)
    assert groundedness_gate(result) == FLAG_NEEDS_RESEARCH


def test_gate_drafts_on_mixed_sources_with_authoritative() -> None:
    """A confident result carrying at least one authoritative source may draft.

    A model_generated chunk sitting beside a real authoritative one does not block
    drafting — the authoritative source grounds the reply.
    """
    result = KBSearchResult(
        sources=[_model_generated(), _authoritative()], no_confident_source=False
    )
    assert groundedness_gate(result) == DRAFT


def test_gate_treats_empty_sources_as_needs_research() -> None:
    """Empty sources route to needs-research even if the flag is unset (defensive).

    Nothing to ground from means nothing to draft from, regardless of the provider's
    self-reported confidence.
    """
    result = KBSearchResult(sources=[], no_confident_source=False)
    assert groundedness_gate(result) == FLAG_NEEDS_RESEARCH

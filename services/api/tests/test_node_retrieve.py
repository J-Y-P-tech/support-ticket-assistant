"""Unit tests for the retrieve node + groundedness gate (plan Task 11 / todo Task 12).

`retrieve` asks the kb client for sources for a query; `groundedness_gate` decides
whether the case may proceed to drafting or must be routed to a human for research.
Both are standalone functions here — the LangGraph wiring that turns the gate into a
conditional edge is added when the workflow is assembled (todo Task 17) — so these
tests exercise them directly, with the kb client faked (no network, SPEC §10/§12).

The behaviours pinned here are the acceptance criteria (SPEC §4.4):

- `retrieve` forwards the query to `search_knowledge_base` and returns the typed
  `KBSearchResult` unchanged (sources + the explicit no-confident-source signal);
- a confident result carrying a source routes to `draft`;
- **no confident source** routes to `flag_needs_research` — never a drafted answer;
- the gate honours the explicit `no_confident_source` flag even when a source is
  present, and treats an empty source list defensively — no sources, no drafting.
"""

from __future__ import annotations

import pytest

from app.graph.nodes.retrieve import DRAFT, FLAG_NEEDS_RESEARCH, groundedness_gate, retrieve
from app.mcp_clients.kb import KBMCPClient
from app.schemas.kb import KBSearchResult, KBSource


def _source(id_: str = "kb-1") -> KBSource:
    """Build a KB source — every KB source is an eligible, citable answer."""
    return KBSource(id=id_, title="Reset access", text="Verify identity, then reset ...")


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
    canned = KBSearchResult(sources=[_source()], no_confident_source=False)

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


def test_gate_routes_confident_source_to_draft() -> None:
    """A confident result carrying a source proceeds to drafting."""
    result = KBSearchResult(sources=[_source()], no_confident_source=False)
    assert groundedness_gate(result) == DRAFT


def test_gate_routes_no_confident_source_to_needs_research() -> None:
    """No confident source → needs-human-research, never a drafted answer (SPEC §4.4)."""
    result = KBSearchResult(sources=[], no_confident_source=True)
    assert groundedness_gate(result) == FLAG_NEEDS_RESEARCH


def test_gate_honours_flag_even_when_a_source_is_present() -> None:
    """The explicit no-confident-source flag routes to research even with a source.

    The gate trusts the provider's confidence signal rather than inferring from the
    source list, so a flagged result never drafts even if a weak source came back.
    """
    result = KBSearchResult(sources=[_source()], no_confident_source=True)
    assert groundedness_gate(result) == FLAG_NEEDS_RESEARCH


def test_gate_treats_empty_sources_as_needs_research() -> None:
    """Empty sources route to needs-research even if the flag is unset (defensive).

    Nothing to ground from means nothing to draft from, regardless of the provider's
    self-reported confidence.
    """
    result = KBSearchResult(sources=[], no_confident_source=False)
    assert groundedness_gate(result) == FLAG_NEEDS_RESEARCH

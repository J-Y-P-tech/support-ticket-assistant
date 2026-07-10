"""Unit tests for the kb MCP client wrapper (plan Task 8).

Prove the wrapper turns `search_knowledge_base`'s JSON payload into a typed
`KBSearchResult` **without network**: ranked `KBSource` objects plus the distinct
`no_confident_source` signal that routes a case to needs-human-research (SPEC §4.4).
Crucially, the client carries that signal through exactly as the provider set it —
it does *not* re-derive it from the source list — so the provider stays the
authority on confidence. The shared transport (`call_tool`) is faked here; it is
exercised for real in `test_mcp_client_base.py`.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.mcp_clients.kb import KBMCPClient
from app.schemas.kb import KBSearchResult, KBSource


@pytest.fixture
def wrapper() -> KBMCPClient:
    """A `KBMCPClient` pointed at a dummy endpoint (no connection is opened).

    `default_limit=3` stands in for the configured `KB_SEARCH_LIMIT`.
    """
    return KBMCPClient(url="http://kb_mcp:8000/mcp", token="test-token", default_limit=3)


async def test_search_returns_typed_kb_sources(
    wrapper: KBMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A matching search yields a `KBSearchResult` of typed `KBSource` objects."""
    payload = {
        "sources": [
            {
                "id": "kb-1",
                "title": "Reset access",
                "text": "Verify identity, then reset ...",
            }
        ],
        "no_confident_source": False,
    }

    async def fake_call(tool: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
        return payload

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    result = await wrapper.search("how do I reset my password")

    assert isinstance(result, KBSearchResult)
    assert result.no_confident_source is False
    assert result.sources == [
        KBSource(
            id="kb-1",
            title="Reset access",
            text="Verify identity, then reset ...",
        )
    ]


async def test_search_surfaces_no_confident_source_when_empty(
    wrapper: KBMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No matches → empty sources with `no_confident_source` set True."""

    async def fake_call(tool: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
        return {"sources": [], "no_confident_source": True}

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    result = await wrapper.search("qwerty zzz nonsense")

    assert result.sources == []
    assert result.no_confident_source is True


async def test_no_confident_source_is_carried_through_verbatim(
    wrapper: KBMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The client trusts the provider's flag: it does not re-derive it from the list.

    Given a (contrived) payload where the provider reports `no_confident_source`
    while still returning a source, the client passes the flag through unchanged
    rather than inferring confidence from the non-empty source list.
    """

    async def fake_call(tool: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
        return {
            "sources": [
                {
                    "id": "kb-9",
                    "title": "Best guess",
                    "text": "A weak partial match ...",
                }
            ],
            "no_confident_source": True,
        }

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    result = await wrapper.search("obscure edge case")

    assert len(result.sources) == 1
    # The signal is carried through — not inferred from the (non-empty) list.
    assert result.no_confident_source is True


async def test_search_forwards_query_and_limit(
    wrapper: KBMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`search` calls the `search_knowledge_base` tool with the query and limit."""
    seen: dict[str, Any] = {}

    async def fake_call(tool: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
        seen["tool"] = tool
        seen["arguments"] = arguments
        return {"sources": [], "no_confident_source": True}

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    await wrapper.search("query text", limit=5)

    assert seen["tool"] == "search_knowledge_base"
    assert seen["arguments"] == {"query": "query text", "limit": 5}


async def test_search_uses_configured_default_limit_when_omitted(
    wrapper: KBMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no explicit `limit`, the client sends its configured default (KB_SEARCH_LIMIT)."""
    seen: dict[str, Any] = {}

    async def fake_call(tool: str, arguments: dict[str, Any], **_kwargs: Any) -> Any:
        seen["arguments"] = arguments
        return {"sources": [], "no_confident_source": True}

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    await wrapper.search("query text")

    assert seen["arguments"] == {"query": "query text", "limit": 3}  # fixture default_limit


async def test_search_opts_into_retry(
    wrapper: KBMCPClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A KB search is a side-effect-free read, so it opts into reconnect+retry."""
    seen: dict[str, bool] = {}

    async def fake_call(
        tool: str, arguments: dict[str, Any], *, retry_on_disconnect: bool = False
    ) -> Any:
        seen["retry_on_disconnect"] = retry_on_disconnect
        return {"sources": [], "no_confident_source": True}

    monkeypatch.setattr(wrapper, "call_tool", fake_call)

    await wrapper.search("anything")

    assert seen["retry_on_disconnect"] is True

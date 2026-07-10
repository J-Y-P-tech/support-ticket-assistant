"""Retrieve node + groundedness gate (plan Task 11 / todo Task 12).

Two steps sit between triage and drafting. `retrieve` asks the kb client for
sources for the (fused) search query and hands back the typed `KBSearchResult`
untouched. `groundedness_gate` then decides the path: a case may only proceed to
`draft` when the KB returned a source; otherwise it is routed to
`flag_needs_research` for a human, never a drafted answer (SPEC §4.4).

Every source the KB returns is an eligible, citable answer (the provider surfaces
nothing else), so the gate's rule is simply: draft only when the KB found a
confident source. A result that flags `no_confident_source`, or that is empty,
routes to research. The gate does not trust the flag alone — it independently
refuses to draft with no source in hand, mirroring the committed kb client
contract (`app/mcp_clients/kb.py`), where the no-confident-source signal is carried
explicitly rather than inferred.

Both are plain functions, independent of LangGraph; the state adapter that wraps
`retrieve` into a node and `groundedness_gate` into a conditional edge is added when
the workflow is assembled (todo Task 17).
"""

from __future__ import annotations

from typing import Final, Literal

from app.mcp_clients.kb import KBMCPClient
from app.schemas.kb import KBSearchResult

# The two routes the gate can pick, named for the workflow nodes they lead to
# (todo Task 17). Kept as constants so callers and tests reference one source of
# truth rather than bare string literals.
DRAFT: Final = "draft"
FLAG_NEEDS_RESEARCH: Final = "flag_needs_research"

Route = Literal["draft", "flag_needs_research"]


async def retrieve(query: str, kb_client: KBMCPClient) -> KBSearchResult:
    """Search the KB for `query` and return the typed result for the gate to route.

    A thin pass-through over `kb_client.search`: it does not interpret the result,
    so the ranked sources and the explicit `no_confident_source` signal reach
    `groundedness_gate` intact. The query is the fused search query (§4.2 —
    customer question + attachment summary); composing it is the caller's job.
    """
    return await kb_client.search(query)


def groundedness_gate(result: KBSearchResult) -> Route:
    """Route to `draft` only with a confident source in hand; otherwise needs-research.

    Drafting is permitted **only** when the provider is confident
    (`no_confident_source` is False) *and* actually returned a source. A
    no-confident-source signal or an empty source list routes to
    `flag_needs_research` — the case goes to a human rather than producing an
    ungrounded answer (SPEC §4.4).
    """
    if result.no_confident_source:
        return FLAG_NEEDS_RESEARCH
    if result.sources:
        return DRAFT
    return FLAG_NEEDS_RESEARCH

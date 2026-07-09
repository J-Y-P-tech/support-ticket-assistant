"""Retrieve node + groundedness gate (plan Task 11 / todo Task 12).

Two steps sit between triage and drafting. `retrieve` asks the kb client for
sources for the (fused) search query and hands back the typed `KBSearchResult`
untouched. `groundedness_gate` then decides the path: a case may only proceed to
`draft` when the KB returned a genuinely **authoritative** source; otherwise it is
routed to `flag_needs_research` for a human, never a drafted answer (SPEC §4.4).

The gate enforces the project's grounding rule: only `authoritative` sources count.
A `model_generated` fallback never grounds a reply (SPEC §4.5) — so a result that
carries only model-generated sources, or that flags `no_confident_source`, or that
is empty, all route to research. This mirrors the committed kb client contract
(`app/mcp_clients/kb.py`): the no-confident-source signal is carried explicitly and
is not the only trigger — the gate independently refuses to draft without an
authoritative source.

Both are plain functions, independent of LangGraph; the state adapter that wraps
`retrieve` into a node and `groundedness_gate` into a conditional edge is added when
the workflow is assembled (todo Task 17).
"""

from __future__ import annotations

from typing import Final, Literal

from app.mcp_clients.kb import KBMCPClient
from app.schemas.enums import SourceKind
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
    """Route to `draft` only with an authoritative source; otherwise needs-research.

    Drafting is permitted **only** when the provider is confident
    (`no_confident_source` is False) *and* at least one returned source is
    `authoritative`. A no-confident-source signal, a result carrying only
    `model_generated` fallbacks, or an empty source list all route to
    `flag_needs_research` — the case goes to a human rather than producing an
    ungrounded answer (SPEC §4.4 / §4.5).
    """
    if result.no_confident_source:
        return FLAG_NEEDS_RESEARCH
    if any(source.source_kind is SourceKind.AUTHORITATIVE for source in result.sources):
        return DRAFT
    return FLAG_NEEDS_RESEARCH

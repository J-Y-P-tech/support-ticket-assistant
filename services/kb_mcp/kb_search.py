"""Query orchestration for `search_knowledge_base` (SPEC §4.4).

Keeps the tool's contract logic — run the active provider, then derive the
"no confident source" signal — in one tested place, so `server.py` stays a thin
transport wrapper (mirroring email_mcp's db/server split). The signal is what
routes a case to needs-human-research instead of a drafted answer (SPEC §4.4).
"""

from __future__ import annotations

from typing import Any

from providers.base import KBProvider


def run_search(provider: KBProvider, query: str, limit: int) -> dict[str, Any]:
    """Run the active provider and attach the no-confident-source signal.

    Returns the tool payload: `sources` (ranked `KBSource` dicts, possibly empty)
    and `no_confident_source` — True when the provider surfaced no source, which
    routes the case to needs-human-research rather than a drafted answer (SPEC
    §4.4). Every returned source is an eligible, citable answer, so any non-empty
    result clears the flag.
    """
    sources = provider.search(query, limit=limit)
    return {"sources": sources, "no_confident_source": not sources}

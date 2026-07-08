"""Query orchestration for `search_knowledge_base` (SPEC §4.4).

Keeps the tool's contract logic — run the active provider, then derive the
"no confident source" signal — in one tested place, so `server.py` stays a thin
transport wrapper (mirroring email_mcp's db/server split). The signal is what
routes a case to needs-human-research instead of a drafted answer (SPEC §4.4).
"""

from __future__ import annotations

from typing import Any

from providers.base import SOURCE_KIND_AUTHORITATIVE, KBProvider


def run_search(provider: KBProvider, query: str, limit: int) -> dict[str, Any]:
    """Run the active provider and attach the no-confident-source signal.

    Returns the tool payload: `sources` (ranked `KBSource` dicts, possibly empty)
    and `no_confident_source` — True when no *authoritative* source matched, which
    routes the case to needs-human-research rather than a drafted answer (SPEC
    §4.4). A `model_generated` fallback never clears this flag (SPEC §4.5).
    """
    sources = provider.search(query, limit=limit)
    no_confident_source = not any(
        source["source_kind"] == SOURCE_KIND_AUTHORITATIVE for source in sources
    )
    return {"sources": sources, "no_confident_source": no_confident_source}

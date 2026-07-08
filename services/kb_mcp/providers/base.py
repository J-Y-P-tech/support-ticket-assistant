"""Pluggable knowledge-base provider interface (SPEC §4.4, §14.1).

`search_knowledge_base` is a thin gateway over a `KBProvider`, so a real
embedding/API backend can replace the demo provider with no change to the tool
or the agent. The demo `MockKBProvider` does keyword lookup — no RAG / no vectors
in this project (SPEC §14.1); real retrieval is a future drop-in here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

# Provenance strings a provider stamps on each source. These mirror
# `app.schemas.enums.SourceKind`; kb_mcp's runtime image stays free of the shared
# `app` package (the MCP boundary carries plain JSON, like email_mcp), so the
# values are declared here and kept honest by the contract tests, which validate
# every returned source against the shared `KBSource` schema.
SOURCE_KIND_AUTHORITATIVE = "authoritative"
SOURCE_KIND_MODEL_GENERATED = "model_generated"


class KBProvider(ABC):
    """A swappable knowledge-base backend behind `search_knowledge_base`."""

    @abstractmethod
    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Return up to `limit` ranked source chunks, best match first.

        Each chunk is a plain dict with the `KBSource` shape (`id`, `title`,
        `text`, `source_kind`). An empty list means the provider found no
        confident source; a provider with only a model-generated fallback returns
        chunks marked `model_generated`, which never count as grounding (SPEC §4.5).
        """
        raise NotImplementedError

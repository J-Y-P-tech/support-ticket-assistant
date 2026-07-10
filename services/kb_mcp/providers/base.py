"""Pluggable knowledge-base provider interface (SPEC §4.4, §14.1).

`search_knowledge_base` is a thin gateway over a `KBProvider`, so a real
embedding/API backend can replace the demo provider with no change to the tool
or the agent. The demo `MockKBProvider` does keyword lookup — no RAG / no vectors
in this project (SPEC §14.1); real retrieval is a future drop-in here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class KBProvider(ABC):
    """A swappable knowledge-base backend behind `search_knowledge_base`."""

    @abstractmethod
    def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        """Return up to `limit` ranked source chunks, best match first.

        Each chunk is a plain dict with the `KBSource` shape (`id`, `title`,
        `text`) — every returned chunk is an eligible, citable source. An empty
        list means the provider found no confident source, which routes the case to
        needs-human-research (SPEC §4.4).
        """
        raise NotImplementedError

"""Knowledge-base source chunk returned by `search_knowledge_base` (SPEC §4.4)."""

from __future__ import annotations

from pydantic import BaseModel


class KBSource(BaseModel):
    """A ranked KB chunk the agent may cite when drafting.

    Every chunk the KB returns is an eligible, citable source — the provider only
    ever surfaces authoritative answers (SPEC §4.4), so there is no provenance flag
    to weigh here.
    """

    id: str
    title: str
    text: str


class KBSearchResult(BaseModel):
    """The kb client's typed reply: ranked sources plus the grounding signal.

    `no_confident_source` is carried explicitly in the tool payload rather than the
    client re-deriving it: the provider is the authority on whether it found a
    confident match, and a no-confident result must route to needs-human-research
    rather than a drafted answer (SPEC §4.4).
    """

    sources: list[KBSource]
    no_confident_source: bool

"""Knowledge-base source chunk returned by `search_knowledge_base` (SPEC §4.4)."""

from __future__ import annotations

from pydantic import BaseModel

from app.schemas.enums import SourceKind


class KBSource(BaseModel):
    """A ranked KB chunk the agent may cite when drafting.

    `source_kind` decides whether the chunk can ground a reply: only
    `authoritative` sources count; `model_generated` never does (SPEC §4.5).
    """

    id: str
    title: str
    text: str
    source_kind: SourceKind

"""Structured output of the document-extraction pass (SPEC §4.2).

The vision model transcribes an attachment, then a text pass produces this
validated structure. `raw_text` (the verbatim transcription) is always required
so it can be surfaced to the rep and never silently dropped; the list fields
default to empty and `low_confidence` defaults off.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ExtractionResult(BaseModel):
    """Facts extracted from a customer attachment, treated as unverified input.

    These facts are shown to the rep for correction and are never used as
    authoritative KB grounding (SPEC §4.2 safety note).
    """

    raw_text: str
    doc_type: str | None = None
    amounts: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    names: list[str] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    low_confidence: bool = False

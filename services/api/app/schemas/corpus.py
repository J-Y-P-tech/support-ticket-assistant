"""De-identified training-corpus records captured from resolved cases (SPEC §4.9a).

The project does not fine-tune at runtime but captures a fine-tuning-ready dataset
from day one: every resolved case yields one **SFT record** — input = `(customer
message + extracted facts + cited sources)`, output = the human-approved final reply —
and, when the rep edited the draft, a **preference pair** — the AI's original draft
(rejected) vs the rep-corrected final (chosen), usable for DPO/ORPO-style tuning. Each
record carries the case metadata (category, urgency, groundedness, rep rating, model
tag + prompt version). PII is redacted into every text field before a record is built,
so no raw account/card number or ID reaches the corpus (§4.9a, same scrubber as §6).

`CorpusRecord` is the storage envelope the email_mcp `training_corpus` table persists:
a `record_type` discriminator plus the `payload` (an `SFTRecord` or `PreferenceRecord`
rendered to a JSON-safe dict). The typed sub-models exist so the payload shape is
validated at build time rather than assembled as a loose dict.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class CorpusRecordType(StrEnum):
    """Which kind of training record a corpus row holds (SPEC §4.9a)."""

    SFT = "sft"
    PREFERENCE = "preference"


class CorpusSource(BaseModel):
    """One cited KB source carried into a corpus record's input (redacted text)."""

    id: str
    title: str
    text: str


class CorpusInput(BaseModel):
    """The grounding input a reply was written from (SPEC §4.9a).

    The customer `message`, the OCR-extracted `facts` (`None` for a text-only ticket),
    and the cited KB `sources` — the exact context a fine-tune should learn to map to
    the approved reply. Every free-text field is PII-redacted before it is set here.
    """

    message: str
    facts: str | None = None
    sources: list[CorpusSource] = Field(default_factory=list)


class CorpusMetadata(BaseModel):
    """Per-case metadata attached to every corpus record (SPEC §4.9a).

    `category`/`urgency` come from triage and `groundedness` from validation, so they
    are optional — a case that skipped those stages carries `None`. `rating` is the
    rep's optional score. `model` + `prompt_version` attribute which model and drafting
    prompt produced the reply, so a later fine-tune can filter or weight by them.
    """

    category: str | None = None
    urgency: str | None = None
    groundedness: float | None = None
    rating: int | None = None
    model: str
    prompt_version: str


class SFTRecord(BaseModel):
    """A supervised fine-tuning example: grounding input → human-approved reply."""

    input: CorpusInput
    output: str
    metadata: CorpusMetadata


class PreferenceRecord(BaseModel):
    """A preference pair: the rep-corrected `chosen` reply vs the AI `rejected` draft.

    Built only when the rep edited the draft, so a DPO/ORPO-style tune has a concrete
    positive/negative pair over the same grounding input (SPEC §4.9a).
    """

    input: CorpusInput
    chosen: str
    rejected: str
    metadata: CorpusMetadata


class CorpusRecord(BaseModel):
    """The storage envelope the `training_corpus` table persists.

    `record_type` discriminates the two shapes; `payload` is the `SFTRecord` or
    `PreferenceRecord` rendered to a JSON-safe dict, ready for the JSONB column and the
    JSONL export. Keeping the payload opaque here lets email_mcp store either shape in
    one append-only table without a per-type column set.
    """

    record_type: CorpusRecordType
    payload: dict[str, Any]

"""Turn a finished workflow run into its training-corpus records (todo Task 28).

SPEC §4.9a captures a fine-tuning-ready dataset from day one: every resolved case
yields one **SFT record** (grounding input → the human-approved final reply) and, when
the rep edited the draft, a **preference pair** (the AI's original draft = rejected vs
the rep-corrected final = chosen). This module is the *emission* half, mirroring
`graph/feedback.py` and `graph/audit.py`: it reads a finished (resumed) LangGraph state
and produces the `CorpusRecord`s the email_mcp `training_corpus` table stores.

The split is deliberate, exactly as for feedback and audit. `build_corpus_records` is
**pure** — no I/O, no model, no database — so every disposition (approved-as-is / edited
/ rejected / no-draft hand-off) is unit-testable in isolation, and so is the PII
redaction. `record_corpus` is the thin write step the send route calls once `finalize`
has run; it just forwards each built record through the email_mcp client. Persistence
lives at the service boundary, never inside a graph node.

**Redaction happens here, before a record is built.** Every customer-derived text field
— the message, the OCR-extracted facts, the cited source titles/text, and the reply
texts — is passed through the same `redact_pii` scrubber the logs and Langfuse traces
use (SPEC §6), so no raw account/card number or ID can enter the corpus (§4.9a). KB
source *ids* (e.g. `KB-1`) are identifiers, not customer PII, and are kept intact so a
record stays traceable to its sources.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from app.logging_config import redact_pii
from app.prompts.registry import get_prompt_version
from app.schemas.corpus import (
    CorpusInput,
    CorpusMetadata,
    CorpusRecord,
    CorpusRecordType,
    CorpusSource,
    PreferenceRecord,
    SFTRecord,
)
from app.schemas.enums import FeedbackDecision

# The drafting prompt whose version every corpus record attributes: the corpus captures
# what the *drafting* model produced, so its metadata records the draft prompt in force.
_DRAFT_PROMPT = "draft"


class _CorpusRecorder(Protocol):
    """The one method `record_corpus` needs from the email_mcp client.

    Typed structurally so the write step depends only on `record_corpus`, not on the
    whole `EmailMCPClient` — the tests pass a lightweight recording fake.
    """

    async def record_corpus(self, ticket_id: int, record: CorpusRecord) -> dict[str, Any]:
        """Persist one de-identified corpus record and return the stored row."""
        ...


def _build_input(state: Mapping[str, Any]) -> CorpusInput:
    """Build the redacted grounding input (message + facts + cited sources).

    Every free-text field is scrubbed with `redact_pii` before it is set, so the corpus
    input carries no raw account/card number or ID (SPEC §4.9a). Source ids are kept as
    identifiers; a text-only ticket has no facts and carries `None`.
    """
    facts = state.get("extracted_facts")
    kb_result = state.get("kb_result")
    sources = (
        [
            CorpusSource(id=s.id, title=redact_pii(s.title), text=redact_pii(s.text))
            for s in kb_result.sources
        ]
        if kb_result is not None
        else []
    )
    return CorpusInput(
        message=redact_pii(state["message"]),
        facts=redact_pii(facts) if facts is not None else None,
        sources=sources,
    )


def _build_metadata(state: Mapping[str, Any], *, model: str, rating: int | None) -> CorpusMetadata:
    """Build the per-case metadata (category, urgency, groundedness, rating, model, prompt).

    `category`/`urgency`/`groundedness` are read from triage and validation when those
    stages ran, else `None`. `model` is the host model tag and the prompt version is the
    drafting prompt in force, so a later fine-tune can attribute or filter by them.
    """
    triage = state.get("triage")
    validation = state.get("validation")
    return CorpusMetadata(
        category=triage.category.value if triage is not None else None,
        urgency=triage.urgency.value if triage is not None else None,
        groundedness=validation.groundedness if validation is not None else None,
        rating=rating,
        model=model,
        prompt_version=get_prompt_version(_DRAFT_PROMPT),
    )


def build_corpus_records(
    state: Mapping[str, Any], *, model: str, rating: int | None = None
) -> list[CorpusRecord]:
    """Map a finished workflow state to its training-corpus records, or `[]` if none apply.

    Returns one SFT record for every resolved case that produced an approved reply, plus
    a preference pair when the rep edited the draft. Returns `[]` when there is nothing to
    learn from: a case with no draft (a no-confident-source hand-off or a blocked
    injection never reaches one), or a rejection — SPEC §4.9a ties the SFT output to the
    *human-approved* final reply, and a rejection produces none. Every text field is
    PII-redacted before the record is built. Pure: no I/O, mutates nothing.
    """
    draft = state.get("draft")
    final_reply = state.get("final_reply")
    # No draft, or no approved reply (a rejection) — nothing enters the corpus.
    if draft is None or final_reply is None:
        return []

    corpus_input = _build_input(state)
    metadata = _build_metadata(state, model=model, rating=rating)
    approved = redact_pii(final_reply)

    records = [
        CorpusRecord(
            record_type=CorpusRecordType.SFT,
            payload=SFTRecord(input=corpus_input, output=approved, metadata=metadata).model_dump(),
        )
    ]
    # The rep rewrote the draft: capture the positive/negative preference pair too.
    if state.get("rep_decision") == FeedbackDecision.EDITED:
        records.append(
            CorpusRecord(
                record_type=CorpusRecordType.PREFERENCE,
                payload=PreferenceRecord(
                    input=corpus_input,
                    chosen=approved,
                    rejected=redact_pii(draft.body),
                    metadata=metadata,
                ).model_dump(),
            )
        )
    return records


async def record_corpus(
    email: _CorpusRecorder,
    *,
    ticket_id: int,
    state: Mapping[str, Any],
    model: str,
    rating: int | None = None,
) -> None:
    """Write the training-corpus records for a finished run through the email_mcp client.

    The thin write step behind the send route: it builds the records for `state` and,
    for each (an SFT record, and a preference pair when the draft was edited), records it
    under `ticket_id` so the append-only corpus grows. A case with no approved reply
    produces no records, so nothing is written.
    """
    for record in build_corpus_records(state, model=model, rating=rating):
        await email.record_corpus(ticket_id, record)

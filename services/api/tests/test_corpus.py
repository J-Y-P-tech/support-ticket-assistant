"""Unit tests for the training-corpus builder + its PII redaction (todo Task 28).

SPEC §4.9a captures a fine-tuning-ready dataset from day one: every resolved case
yields one **SFT record** (input = customer message + extracted facts + cited sources
→ output = the human-approved final reply) and, when the rep edited the draft, a
**preference pair** (AI original draft = rejected vs rep-corrected final = chosen).
PII is redacted before a record enters the corpus, reusing the same scrubber as logs
and traces (§6). These pin the two pure pieces of the capture path: `build_corpus_records`
turns a finished workflow state into the `CorpusRecord`s the email_mcp `training_corpus`
table stores, and the redaction guarantee that no configured PII pattern survives.
No I/O, no model, no database.
"""

from __future__ import annotations

import json
from typing import Any

from app.graph.corpus import build_corpus_records
from app.schemas.corpus import CorpusRecordType
from app.schemas.draft import Citation, Draft
from app.schemas.enums import Category, FeedbackDecision, Sentiment, Urgency
from app.schemas.kb import KBSearchResult, KBSource
from app.schemas.triage import TriageResult
from app.schemas.validation import ValidationResult

_MODEL = "gemma4:12b"


def _finished_state(
    *,
    decision: FeedbackDecision,
    message: str = "How do I reset my online banking password?",
    draft_body: str = "You can reset your password from the login screen. [KB-1]",
    final_reply: str | None = None,
    extracted_facts: str | None = None,
    source_text: str = "To reset your password, use the login screen.",
) -> dict[str, Any]:
    """Build a minimal finished-run state slice for `build_corpus_records`.

    Mirrors what a happy-path run leaves in LangGraph state at the review pause plus
    what `finalize` adds on send: the customer message, the triage classification, the
    retrieved KB source, the drafted reply, its groundedness verdict, the rep's
    decision, and the `final_reply` (the sent text, or `None` for a rejection).
    """
    draft = Draft(body=draft_body, citations=[Citation(source_id="KB-1", title="Password reset")])
    return {
        "message": message,
        "extracted_facts": extracted_facts,
        "triage": TriageResult(
            category=Category.ACCOUNT_ACCESS,
            urgency=Urgency.NORMAL,
            sentiment=Sentiment.NEUTRAL,
        ),
        "kb_result": KBSearchResult(
            sources=[KBSource(id="KB-1", title="Password reset", text=source_text)],
            no_confident_source=False,
        ),
        "draft": draft,
        "validation": ValidationResult(draft=draft, groundedness=1.0, flagged=False),
        "rep_decision": decision,
        "final_reply": final_reply,
    }


def test_approved_as_is_yields_one_sft_record_only() -> None:
    """An approved-as-is case yields exactly one SFT record and no preference pair.

    The SFT input carries the customer message, the (absent) extracted facts, and the
    cited KB source; the output is the human-approved final reply; the metadata carries
    the triage category/urgency, the groundedness score, the rep rating, and the model
    tag + prompt version (SPEC §4.9a).
    """
    reply = "You can reset your password from the login screen. [KB-1]"
    records = build_corpus_records(
        _finished_state(decision=FeedbackDecision.APPROVED_AS_IS, final_reply=reply),
        model=_MODEL,
        rating=5,
    )

    assert len(records) == 1
    sft = records[0]
    assert sft.record_type is CorpusRecordType.SFT
    assert sft.payload["input"]["message"] == "How do I reset my online banking password?"
    assert sft.payload["input"]["facts"] is None
    assert sft.payload["input"]["sources"] == [
        {
            "id": "KB-1",
            "title": "Password reset",
            "text": "To reset your password, use the login screen.",
        }
    ]
    assert sft.payload["output"] == reply
    meta = sft.payload["metadata"]
    assert meta["category"] == "account_access"
    assert meta["urgency"] == "normal"
    assert meta["groundedness"] == 1.0
    assert meta["rating"] == 5
    assert meta["model"] == _MODEL
    assert meta["prompt_version"] == "draft-v1"


def test_edited_case_yields_sft_and_a_preference_pair() -> None:
    """An edited case yields an SFT record plus a preference pair (SPEC §4.9a).

    The SFT output is the rep's corrected final reply; the preference pair keeps the
    rep-corrected final as `chosen` and the AI's original draft as `rejected` — the
    positive/negative example a DPO/ORPO-style tune consumes. Both records share the
    same input and metadata.
    """
    draft_body = "Reset your password."
    edited = "Please reset your password from the login screen and call us if it fails. [KB-1]"
    records = build_corpus_records(
        _finished_state(
            decision=FeedbackDecision.EDITED, draft_body=draft_body, final_reply=edited
        ),
        model=_MODEL,
        rating=4,
    )

    assert [r.record_type for r in records] == [
        CorpusRecordType.SFT,
        CorpusRecordType.PREFERENCE,
    ]
    sft, pref = records
    assert sft.payload["output"] == edited
    assert pref.payload["chosen"] == edited
    assert pref.payload["rejected"] == draft_body
    # The preference pair carries the same grounding input + metadata as the SFT record.
    assert pref.payload["input"] == sft.payload["input"]
    assert pref.payload["metadata"] == sft.payload["metadata"]


def test_extracted_facts_are_carried_into_the_sft_input() -> None:
    """A digitized-document case carries the extracted facts into the SFT input.

    SPEC §4.9a defines the SFT input as `(customer message + extracted facts + cited
    sources)`, so the OCR digest a run produced must appear in the record rather than
    being dropped.
    """
    records = build_corpus_records(
        _finished_state(
            decision=FeedbackDecision.APPROVED_AS_IS,
            final_reply="Your dispute has been logged. [KB-1]",
            extracted_facts="Statement date 2026-05; disputed charge present.",
        ),
        model=_MODEL,
    )

    facts = records[0].payload["input"]["facts"]
    assert facts == "Statement date 2026-05; disputed charge present."


def test_rejected_case_yields_no_records() -> None:
    """A rejected draft yields no corpus records — there is no human-approved reply.

    SPEC §4.9a ties the SFT output to the *approved* final reply; a rejection produces
    no final reply and no rep-corrected `chosen`, so neither an SFT record nor a
    preference pair can be built.
    """
    records = build_corpus_records(
        _finished_state(decision=FeedbackDecision.REJECTED, final_reply=None),
        model=_MODEL,
        rating=2,
    )

    assert records == []


def test_no_draft_yields_no_records() -> None:
    """A case handed to a human with no draft yields no corpus records.

    A no-confident-source hand-off or a blocked injection reaches the human gate with
    no draft ever written, so there is nothing to learn from — no SFT input/output pair
    exists to capture.
    """
    records = build_corpus_records(
        {"rep_decision": FeedbackDecision.REJECTED, "message": "help", "final_reply": None},
        model=_MODEL,
    )

    assert records == []


def test_pii_is_redacted_before_a_record_enters_the_corpus() -> None:
    """No configured PII pattern survives into the serialized corpus records (SPEC §4.9a).

    A card number, an account number, and a national ID planted in the customer
    message, the extracted facts, the cited source text, and the approved reply must
    all be scrubbed — the same redaction as logs/traces (§6) — before the record is
    built, so the exported JSONL carries no raw account/card numbers or IDs.
    """
    card = "4111 1111 1111 1111"
    account = "1234567890"
    national_id = "987654321"
    records = build_corpus_records(
        _finished_state(
            decision=FeedbackDecision.EDITED,
            message=f"My card {card} was declined and my account {account} is locked.",
            draft_body=f"I see account {account}.",
            final_reply=f"Your card ending in {card} and ID {national_id} are on file. [KB-1]",
            extracted_facts=f"Card {card}, account {account} on the statement.",
            source_text=f"Reference document {national_id}.",
        ),
        model=_MODEL,
        rating=3,
    )

    # Serialize every record exactly as the JSONL export would render it.
    serialized = "\n".join(json.dumps(r.payload) for r in records)
    for secret in (card, account, national_id, "4111111111111111"):
        assert secret not in serialized
    # The scrub leaves an obvious marker rather than silently deleting the value.
    assert "[REDACTED]" in serialized

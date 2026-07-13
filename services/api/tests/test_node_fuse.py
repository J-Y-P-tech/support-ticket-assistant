"""Unit tests for the search-intent fusion pass + facts renderer (plan Task 21 / todo Task 22).

Fusion is the third digitization pass (SPEC §4.2): it produces the concise **fused
search query** the agent hands to the KB connector for retrieval (§4.4). The query
combines the customer's question with a short summary of any attachment content; a
ticket with no attachment still runs the pass and fuses from the message alone
(project decision — the model condenses even a text-only message into a keyword query,
falling back to the raw message if it returns nothing). The pass is a standalone async
function here — the LangGraph `ocr_extract` node that wires it in is exercised by the
workflow suite — so these tests drive it directly against a deterministic `FakeLLM`,
never the host model (SPEC §10/§12).

The behaviours pinned here are the acceptance criteria (SPEC §4.2):

- with an attachment summary, the fused query folds in both the message and the
  summary, and both reach the model's prompt;
- with no attachment, the pass still fuses from the message alone (no dangling
  attachment section in the prompt);
- the fused query is **non-empty whenever a message or transcription exists** — an
  empty (or trace-only) model reply falls back to the raw message rather than yielding
  an empty query that would search the KB for nothing;
- a leaked reasoning trace is stripped, because the fused query is machine-consumed
  (fed straight to KB search) and a trace would pollute retrieval;
- fusion keeps the project-wide reason-by-default (`think=True`): picking the salient
  search terms is interpretive.

`render_extracted_facts` turns the validated `ExtractionResult` into the plain-language
digest that becomes both the `extracted_facts` triage/rep signal and the attachment
summary fed to fusion; its tests pin that the structured fields and the raw text are
surfaced and empty fields are omitted.
"""

from __future__ import annotations

from app.graph.nodes.fuse import fuse_query, render_extracted_facts
from app.llm.fake import FakeLLM
from app.schemas.extraction import ExtractionResult

# A representative customer question and the digest an attachment would contribute.
_MESSAGE = "How do I dispute this cheque that was cashed twice?"
_SUMMARY = (
    "Document type: cheque\nAmounts: $1,250.00\nReferences: CHK-4471\n"
    "Raw text:\nPAY TO THE ORDER OF John Doe $1,250.00 Ref CHK-4471"
)
# The concise query the fusion model is expected to return.
_FUSED = "dispute duplicate cheque CHK-4471"


async def test_fuses_message_and_attachment_summary() -> None:
    """With an attachment summary, the model's fused query is returned and both inputs reach it."""
    llm = FakeLLM(_FUSED)

    query = await fuse_query(_MESSAGE, llm, attachment_summary=_SUMMARY)

    assert query == _FUSED
    prompt = llm.calls[0]["prompt"]
    assert _MESSAGE in prompt
    assert _SUMMARY in prompt


async def test_message_only_fuses_from_message_alone() -> None:
    """A ticket with no attachment still runs the pass, fusing from the message alone.

    Per the project decision the model condenses even a text-only message into a
    keyword query; the prompt carries the message and shows no dangling attachment
    section (mirroring triage's handling of an absent facts block).
    """
    llm = FakeLLM(_FUSED)

    query = await fuse_query(_MESSAGE, llm, attachment_summary=None)

    assert query == _FUSED
    prompt = llm.calls[0]["prompt"]
    assert _MESSAGE in prompt
    assert _SUMMARY not in prompt


async def test_empty_model_reply_falls_back_to_message() -> None:
    """A whitespace-only model reply falls back to the raw message — never an empty query.

    The fused query is what searches the KB, so an empty query would search for
    nothing; the acceptance criterion is that it is non-empty whenever a message exists.
    """
    llm = FakeLLM("   \n  ")

    query = await fuse_query(_MESSAGE, llm)

    assert query == _MESSAGE


async def test_trace_only_reply_falls_back_to_message() -> None:
    """A reply that is nothing but a reasoning trace strips to empty, then falls back.

    Stripping the trace leaves no query, so the fallback keeps the query non-empty
    rather than handing KB search a blank string.
    """
    llm = FakeLLM("<think>the user wants to dispute a cheque</think>")

    query = await fuse_query(_MESSAGE, llm)

    assert query == _MESSAGE


async def test_leaked_reasoning_trace_is_stripped() -> None:
    """A trace baked around the query is stripped, leaving only the query terms.

    The fused query is machine-consumed (handed straight to KB search), so unlike the
    customer-facing draft — which flags a leak rather than scrubbing it — fusion strips
    defensively so a trace never pollutes retrieval.
    """
    llm = FakeLLM(f"<think>pick the key terms</think>{_FUSED}")

    query = await fuse_query(_MESSAGE, llm)

    assert query == _FUSED


async def test_fusion_reasons_by_default() -> None:
    """Fusion keeps reason-by-default (`think=True`): choosing search terms is interpretive."""
    llm = FakeLLM(_FUSED)

    await fuse_query(_MESSAGE, llm)

    assert llm.calls[0]["think"] is True


def test_render_extracted_facts_surfaces_fields_and_raw_text() -> None:
    """The digest lists the structured fields and always ends with the raw text."""
    result = ExtractionResult(
        raw_text="PAY TO THE ORDER OF John Doe $1,250.00 Ref CHK-4471",
        doc_type="cheque",
        amounts=["$1,250.00"],
        dates=["2026-01-02"],
        names=["John Doe"],
        references=["CHK-4471"],
    )

    digest = render_extracted_facts(result)

    assert "cheque" in digest
    assert "$1,250.00" in digest
    assert "2026-01-02" in digest
    assert "John Doe" in digest
    assert "CHK-4471" in digest
    assert "PAY TO THE ORDER OF John Doe" in digest


def test_render_extracted_facts_omits_empty_fields() -> None:
    """Fields with no extracted values are left out — no dangling 'Amounts:' header.

    Only the raw text is guaranteed; a near-empty extraction renders just that, so the
    triage/rep digest never shows empty labels.
    """
    result = ExtractionResult(raw_text="illegible scan")

    digest = render_extracted_facts(result)

    assert "Amounts" not in digest
    assert "Dates" not in digest
    assert "Names" not in digest
    assert "References" not in digest
    assert "illegible scan" in digest


def test_render_extracted_facts_flags_low_confidence() -> None:
    """A low-confidence extraction carries a plain-language 'unsure' note in the digest."""
    result = ExtractionResult(raw_text="garbled", low_confidence=True)

    digest = render_extracted_facts(result)

    assert "confidence" in digest.lower()
    assert "garbled" in digest

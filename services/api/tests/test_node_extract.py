"""Unit tests for the structured-extraction pass (plan Task 20 / todo Task 21).

Extraction is the second digitization pass (SPEC §4.2): it turns the verbatim
transcription the OCR pass produced into a validated `ExtractionResult`
(doc_type/amounts/dates/names/references/raw_text/low_confidence). Like triage it
retries once on unparseable or schema-invalid output — but where triage *raises*
when both attempts fail, extraction **flags and surfaces**: the extracted facts are
unverified, rep-facing input, so a failure to structure them degrades gracefully to
"here is the raw text, flagged low-confidence — double-check it" rather than blocking
the case. The transcription is therefore never silently dropped. The node is a
standalone async function here — the LangGraph adapter wiring it in lands with Task 22
— so these tests exercise it directly against a deterministic `FakeLLM`, never the
host model (SPEC §10/§12).

The behaviours pinned here are exactly the acceptance criteria (SPEC §4.2):

- valid model JSON → a typed `ExtractionResult` with the structured fields populated;
- `raw_text` is always the true transcription the node was handed, never the model's
  echo of it, so the verbatim text survives whatever the model returns;
- schema-invalid or unparseable output → the node retries **once**;
- invalid-then-valid → the retry succeeds;
- invalid twice → the result is *flagged* `low_confidence=True` with the raw text
  preserved and no facts invented — the failure is surfaced, never dropped, and the
  node does not raise or loop forever;
- the retry budget is config-driven (`max_attempts`), not a hard-coded constant;
- the transcription reaches the model's prompt;
- extraction keeps the project-wide reason-by-default (`think=True`): unlike the
  mechanical OCR copy, deciding which token is a date, an amount, or a reference is
  interpretive and benefits from reasoning.
"""

from __future__ import annotations

import json

from app.graph.nodes.extract import extract
from app.llm.fake import FakeLLM

# The SPEC §4.2 default the production config ships: one initial try plus a single
# retry. Passed explicitly since the node reads the budget from config, not a
# hard-coded constant.
_RETRY_ONCE = 2

# A representative transcription the OCR pass would hand the extraction node.
_TRANSCRIPTION = "PAY TO THE ORDER OF John Doe $1,250.00  Ref CHK-4471  2026-01-02"

# A well-formed extraction response: a bare JSON object with the structured fields.
# `raw_text` is deliberately absent — the node supplies the true transcription itself.
_VALID = json.dumps(
    {
        "doc_type": "cheque",
        "amounts": ["$1,250.00"],
        "dates": ["2026-01-02"],
        "names": ["John Doe"],
        "references": ["CHK-4471"],
        "low_confidence": False,
    }
)
# Not JSON at all — the parse step fails before any schema check.
_NOT_JSON = "The document appears to be a cheque made out to John Doe."
# Well-formed JSON but a list field given a bare string — passes json.loads, fails the
# schema (mirrors test_schema_extraction's wrong-typed-list case).
_BAD_SCHEMA = json.dumps({"doc_type": "cheque", "amounts": "1250.00"})


async def test_valid_output_yields_typed_result() -> None:
    """Valid model JSON is parsed into a populated `ExtractionResult` on the first try."""
    llm = FakeLLM([_VALID])

    result = await extract(_TRANSCRIPTION, llm, max_attempts=_RETRY_ONCE)

    assert result.doc_type == "cheque"
    assert result.amounts == ["$1,250.00"]
    assert result.dates == ["2026-01-02"]
    assert result.names == ["John Doe"]
    assert result.references == ["CHK-4471"]
    assert result.low_confidence is False
    assert len(llm.calls) == 1, "valid output must not trigger a retry"


async def test_raw_text_is_always_the_transcription() -> None:
    """`raw_text` is the transcription the node was handed, never the model's echo.

    The verbatim text is set deterministically by the node, so it survives intact even
    when the model omits it (as here) or returns a corrupted version of it — the raw
    text is never at the mercy of the model.
    """
    llm = FakeLLM([_VALID])

    result = await extract(_TRANSCRIPTION, llm, max_attempts=_RETRY_ONCE)

    assert result.raw_text == _TRANSCRIPTION


async def test_model_echo_of_raw_text_is_overridden() -> None:
    """A `raw_text` the model puts in its JSON is ignored in favour of the true text."""
    corrupted = json.dumps({"doc_type": "cheque", "raw_text": "GARBLED MODEL ECHO"})
    llm = FakeLLM([corrupted])

    result = await extract(_TRANSCRIPTION, llm, max_attempts=_RETRY_ONCE)

    assert result.raw_text == _TRANSCRIPTION


async def test_retries_once_on_invalid_then_valid() -> None:
    """Unparseable output followed by valid output is retried and then succeeds."""
    llm = FakeLLM([_NOT_JSON, _VALID])

    result = await extract(_TRANSCRIPTION, llm, max_attempts=_RETRY_ONCE)

    assert result.doc_type == "cheque"
    assert result.low_confidence is False
    assert len(llm.calls) == 2, "a bad first answer must cost exactly one retry"


async def test_schema_invalid_triggers_retry() -> None:
    """Well-formed JSON with a wrong-typed field counts as invalid and retries."""
    llm = FakeLLM([_BAD_SCHEMA, _VALID])

    result = await extract(_TRANSCRIPTION, llm, max_attempts=_RETRY_ONCE)

    assert result.amounts == ["$1,250.00"]
    assert len(llm.calls) == 2


async def test_invalid_twice_flags_low_confidence_with_raw_text() -> None:
    """Two bad answers flag `low_confidence` and preserve the raw text — never dropped.

    This is the acceptance criterion that sets extraction apart from triage: it does
    NOT raise. The extracted facts are unverified rep-facing input, so a double failure
    surfaces the raw transcription flagged for review rather than blocking the case, and
    invents no structured facts.
    """
    llm = FakeLLM([_NOT_JSON, _BAD_SCHEMA])

    result = await extract(_TRANSCRIPTION, llm, max_attempts=_RETRY_ONCE)

    assert result.low_confidence is True
    assert result.raw_text == _TRANSCRIPTION
    assert result.doc_type is None
    assert result.amounts == []
    assert result.dates == []
    assert result.names == []
    assert result.references == []


async def test_invalid_twice_stops_after_one_retry() -> None:
    """The node calls the model exactly twice on repeated failure — no infinite loop."""
    llm = FakeLLM([_NOT_JSON, _BAD_SCHEMA])

    await extract(_TRANSCRIPTION, llm, max_attempts=_RETRY_ONCE)

    assert len(llm.calls) == 2


async def test_max_attempts_one_disables_the_retry() -> None:
    """`max_attempts=1` (config knob) flags immediately after a single bad answer.

    Proves the retry budget is driven by config, not a hard-coded constant: with a
    budget of one the node calls the model exactly once and then flags, without retrying.
    """
    llm = FakeLLM([_NOT_JSON, _VALID])

    result = await extract(_TRANSCRIPTION, llm, max_attempts=1)

    assert result.low_confidence is True
    assert result.raw_text == _TRANSCRIPTION
    assert len(llm.calls) == 1


async def test_model_reported_low_confidence_is_honoured() -> None:
    """A model that self-reports low confidence on a valid parse keeps that flag.

    The node forces the flag on a parse failure, but a successful parse must not clear a
    genuine low-confidence signal the model raised (e.g. an image full of `[illegible]`).
    """
    hedged = json.dumps({"doc_type": "receipt", "low_confidence": True})
    llm = FakeLLM([hedged])

    result = await extract(_TRANSCRIPTION, llm, max_attempts=_RETRY_ONCE)

    assert result.low_confidence is True
    assert result.doc_type == "receipt"


async def test_transcription_reaches_the_prompt() -> None:
    """The transcription text is embedded in the prompt sent to the model."""
    llm = FakeLLM([_VALID])

    await extract(_TRANSCRIPTION, llm, max_attempts=_RETRY_ONCE)

    assert _TRANSCRIPTION in llm.calls[0]["prompt"]


async def test_extraction_reasons_by_default() -> None:
    """Extraction keeps reason-by-default (`think=True`), unlike the OCR copy pass.

    Deciding which token is a date, an amount, or a reference number is interpretive,
    so the pass earns the reasoning budget. This pins the choice so a later change
    cannot silently flip it off.
    """
    llm = FakeLLM([_VALID])

    await extract(_TRANSCRIPTION, llm, max_attempts=_RETRY_ONCE)

    assert llm.calls[0]["think"] is True

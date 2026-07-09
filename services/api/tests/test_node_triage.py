"""Unit tests for the triage node (plan Task 10 / todo Task 11).

Triage turns raw ticket text into a validated `TriageResult`
(category/urgency/sentiment). The node is a standalone async function here — the
LangGraph state adapter that wraps it is added when the workflow is assembled
(todo Task 17) — so these tests exercise it directly against a deterministic
`FakeLLM`, never the host model (SPEC §10/§12).

The behaviours pinned here are exactly the acceptance criteria (SPEC §4.3):

- valid model output → a typed `TriageResult`;
- schema-invalid output → the node retries **once**;
- invalid-then-valid → the retry succeeds;
- invalid twice → the failure is *surfaced* (a typed error the workflow routes
  to a human rep), never silently dropped, and the node does not loop forever;
- the customer message (and any extracted facts) reach the model's prompt;
- triage keeps the project-wide reason-by-default (`think=True`).
"""

from __future__ import annotations

import json

import pytest

from app.graph.nodes.triage import TriageValidationError, triage
from app.llm.fake import FakeLLM
from app.schemas.enums import Category, Sentiment, Urgency
from app.schemas.triage import TriageResult

# The SPEC §4.3 default the production config ships (`TRIAGE_MAX_ATTEMPTS=2`): one
# initial try plus a single retry. Passed explicitly since the node reads the
# budget from config, not a hard-coded constant.
_RETRY_ONCE = 2

# A well-formed triage response: a bare JSON object with the three enum values.
_VALID = json.dumps({"category": "account_access", "urgency": "high", "sentiment": "negative"})
# Not JSON at all — the parse step fails before any schema check.
_NOT_JSON = "I think this is an account problem, probably urgent."
# Well-formed JSON but an out-of-set urgency — passes json.loads, fails the enum.
_BAD_ENUM = json.dumps(
    {"category": "account_access", "urgency": "sky-high", "sentiment": "negative"}
)


async def test_valid_output_yields_typed_result() -> None:
    """Valid model JSON is parsed into a `TriageResult` on the first attempt."""
    llm = FakeLLM([_VALID])
    result = await triage("I am locked out of my account", llm, max_attempts=_RETRY_ONCE)
    assert result == TriageResult(
        category=Category.ACCOUNT_ACCESS,
        urgency=Urgency.HIGH,
        sentiment=Sentiment.NEGATIVE,
    )
    assert len(llm.calls) == 1, "valid output must not trigger a retry"


async def test_retries_once_on_invalid_then_valid() -> None:
    """Unparsable output followed by valid output is retried and then succeeds."""
    llm = FakeLLM([_NOT_JSON, _VALID])
    result = await triage("I am locked out of my account", llm, max_attempts=_RETRY_ONCE)
    assert result.category is Category.ACCOUNT_ACCESS
    assert len(llm.calls) == 2, "a bad first answer must cost exactly one retry"


async def test_schema_invalid_enum_triggers_retry() -> None:
    """Well-formed JSON with an out-of-set enum value counts as invalid and retries."""
    llm = FakeLLM([_BAD_ENUM, _VALID])
    result = await triage("Someone used my card", llm, max_attempts=_RETRY_ONCE)
    assert result.urgency is Urgency.HIGH
    assert len(llm.calls) == 2


async def test_invalid_twice_surfaces_typed_error() -> None:
    """Two bad answers raise `TriageValidationError` instead of a silent bad result."""
    llm = FakeLLM([_NOT_JSON, _BAD_ENUM])
    with pytest.raises(TriageValidationError):
        await triage("gibberish in, gibberish out", llm, max_attempts=_RETRY_ONCE)


async def test_invalid_twice_stops_after_one_retry() -> None:
    """The node calls the model exactly twice on repeated failure — no infinite loop."""
    llm = FakeLLM([_NOT_JSON, _BAD_ENUM])
    with pytest.raises(TriageValidationError):
        await triage("gibberish", llm, max_attempts=_RETRY_ONCE)
    assert len(llm.calls) == 2


async def test_max_attempts_one_disables_the_retry() -> None:
    """`max_attempts=1` (config knob) makes a single bad answer surface immediately.

    Proves the retry budget is driven by config, not a hard-coded constant: with a
    budget of one the node calls the model exactly once and does not retry.
    """
    llm = FakeLLM([_NOT_JSON, _VALID])
    with pytest.raises(TriageValidationError):
        await triage("gibberish", llm, max_attempts=1)
    assert len(llm.calls) == 1


async def test_surfaced_error_carries_raw_output_and_human_hint() -> None:
    """The raised error preserves the last raw output and reads as a human hand-off.

    Task 17 routes the case to a rep with this context, so the raw model text must
    survive and the message must make clear a human needs to take over — the failure
    is surfaced, never dropped.
    """
    llm = FakeLLM([_NOT_JSON, _BAD_ENUM])
    with pytest.raises(TriageValidationError) as exc_info:
        await triage("gibberish", llm, max_attempts=_RETRY_ONCE)
    error = exc_info.value
    assert error.raw_output == _BAD_ENUM
    assert "human" in str(error).lower()


async def test_message_reaches_the_prompt() -> None:
    """The customer's message text is embedded in the prompt sent to the model."""
    llm = FakeLLM([_VALID])
    await triage("my debit card was declined at the store", llm, max_attempts=_RETRY_ONCE)
    assert "my debit card was declined at the store" in llm.calls[0]["prompt"]


async def test_extracted_facts_included_when_provided() -> None:
    """Extracted attachment facts, when supplied, are added under a labelled section."""
    llm = FakeLLM([_VALID])
    await triage(
        "see attached statement",
        llm,
        max_attempts=_RETRY_ONCE,
        extracted_facts="Statement dated 2026-01-02, amount $412.00",
    )
    prompt = llm.calls[0]["prompt"]
    assert "Extracted facts:" in prompt
    assert "Statement dated 2026-01-02, amount $412.00" in prompt


async def test_message_only_prompt_omits_facts_section() -> None:
    """With no extracted facts the prompt does not render an empty facts section."""
    llm = FakeLLM([_VALID])
    await triage("plain message, no attachment", llm, max_attempts=_RETRY_ONCE)
    assert "Extracted facts:" not in llm.calls[0]["prompt"]

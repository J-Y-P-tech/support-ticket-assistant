"""Unit tests for the deterministic `FakeLLM` (plan Task 9 / todo Task 10).

CI and unit tests must never touch the host Ollama (SPEC §10/§12), so downstream
nodes are exercised against a `FakeLLM` that returns scripted responses in order.
These tests pin that contract: sequential scripting (so a triage retry can script
invalid-then-valid), a constant mode, call recording for assertions, and a loud
failure when a test under-scripts the model. `FakeLLM` is pure — no network.
"""

from __future__ import annotations

import pytest

from app.llm.base import LLM
from app.llm.fake import FakeLLM


async def test_is_an_llm() -> None:
    """`FakeLLM` implements the shared `LLM` interface, so it is a drop-in."""
    assert isinstance(FakeLLM(["x"]), LLM)


async def test_returns_scripted_responses_in_order() -> None:
    """A list of responses is handed out one per call, in sequence."""
    llm = FakeLLM(["first", "second"])
    assert await llm.generate("p1") == "first"
    assert await llm.generate("p2") == "second"


async def test_constant_mode_returns_same_response_every_call() -> None:
    """A single string is returned for every call, regardless of count."""
    llm = FakeLLM("always")
    assert await llm.generate("a") == "always"
    assert await llm.generate("b") == "always"


async def test_records_prompt_and_images_for_each_call() -> None:
    """Every call's prompt and images are recorded for later assertions."""
    llm = FakeLLM(["ok"])
    await llm.generate("look at this", images=["b64image"])
    assert llm.calls == [{"prompt": "look at this", "images": ["b64image"]}]


async def test_default_images_recorded_as_none() -> None:
    """A text-only call records `images` as None (no attachment sent)."""
    llm = FakeLLM(["ok"])
    await llm.generate("text only")
    assert llm.calls[0]["images"] is None


async def test_exhausting_the_script_raises() -> None:
    """Calling past the scripted responses fails loudly rather than inventing output."""
    llm = FakeLLM(["only one"])
    await llm.generate("1")
    with pytest.raises(IndexError):
        await llm.generate("2")


async def test_same_script_is_deterministic_across_instances() -> None:
    """Two fakes built from the same script produce identical output sequences."""
    a = FakeLLM(["one", "two"])
    b = FakeLLM(["one", "two"])
    assert [await a.generate("x"), await a.generate("y")] == [
        await b.generate("x"),
        await b.generate("y"),
    ]

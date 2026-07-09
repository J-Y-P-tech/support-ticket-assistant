"""Ground the thinking split against the *real* selected model (plan Task 9 / todo Task 10).

The unit tests elsewhere prove the client's mechanics with a mocked transport; this
file proves the assumption those mechanics rest on — that Ollama's `response` field
holds the final answer only, with the reasoning trace kept out in `thinking` — is
actually true for whatever model is configured (`LLM_MODEL`). It does that with a
recorded fixture, keyed by model tag, so it *adapts when the model changes*:

- `test_capture_selected_model_thinking_fixture` is **opt-in** (runs only when
  `CAPTURE_LLM_FIXTURE` is set and the host Ollama is reachable — the user runs it,
  never CI). It calls the real model once and writes
  `tests/fixtures/llm/<model-tag>.json`. Swap `LLM_MODEL`, re-run it, and the
  fixture regenerates for the new model.
- `test_selected_model_response_field_is_trace_free` reads that committed fixture
  (skipping when none exists yet) and asserts the grounded claim — no network. This
  is the check CI runs once a fixture has been captured and committed.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, cast

import pytest

from app.llm.ollama import OllamaLLM

# Committed alongside the tests so CI can assert against real captured output.
_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "llm"

# A prompt that reliably makes a thinking model reason before answering, so the
# captured `thinking` field is non-empty and the `response`/`thinking` split is
# actually exercised.
_CAPTURE_PROMPT = "A customer paid twice by mistake. In one sentence, what should they do?"


def _fixture_path(model: str) -> Path:
    """Map a model tag (e.g. `gemma4:12b`) to its fixture file (`gemma4_12b.json`)."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", model).strip("_")
    return _FIXTURE_DIR / f"{slug}.json"


@pytest.mark.skipif(
    not os.getenv("CAPTURE_LLM_FIXTURE"),
    reason="opt-in: set CAPTURE_LLM_FIXTURE and run the host Ollama to (re)record the fixture",
)
async def test_capture_selected_model_thinking_fixture() -> None:
    """Call the real configured model once and record its split output as a fixture.

    Reads `LLM_MODEL` and `OLLAMA_BASE_URL` straight from the environment (not the
    full `Settings`, so this needs no unrelated config). Writes the model tag, the
    `response`, and the `thinking` trace to a per-model JSON file the grounding test
    below then asserts against.
    """
    model = os.environ["LLM_MODEL"]
    base_url = os.environ["OLLAMA_BASE_URL"]

    llm = OllamaLLM(model=model, base_url=base_url)
    try:
        data = await llm.generate_full(_CAPTURE_PROMPT, think=True)
    finally:
        await llm.aclose()

    _FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    fixture = {
        "model": model,
        "prompt": _CAPTURE_PROMPT,
        "response": data.get("response", ""),
        "thinking": data.get("thinking", ""),
    }
    _fixture_path(model).write_text(json.dumps(fixture, indent=2, ensure_ascii=False) + "\n")


def _load_fixture_for_env_model() -> dict[str, Any] | None:
    """Return the recorded fixture for the env's `LLM_MODEL`, or None if unavailable."""
    model = os.getenv("LLM_MODEL")
    if not model:
        return None
    path = _fixture_path(model)
    if not path.exists():
        return None
    return cast("dict[str, Any]", json.loads(path.read_text()))


def test_selected_model_response_field_is_trace_free() -> None:
    """The recorded model's `response` holds the final answer only — no leaked trace.

    Skips until a fixture has been captured for the configured model, so CI is green
    on a fresh checkout and becomes a real, model-grounded assertion once the user
    records one. When present, it confirms the split we depend on: the model reasoned
    (`thinking` is non-empty) yet none of the CLI-style framing leaked into
    `response`, which is the field the app actually reads.
    """
    fixture = _load_fixture_for_env_model()
    if fixture is None:
        pytest.skip("no captured LLM fixture for the configured model yet (opt-in capture)")

    response = fixture["response"]
    assert response.strip(), "captured response should hold the final answer"
    assert fixture["thinking"].strip(), "expected the model to have produced a reasoning trace"
    # The reasoning trace must not have leaked into the answer field.
    lowered = response.lower()
    assert "done thinking" not in lowered
    assert "<think>" not in lowered

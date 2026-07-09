"""Deterministic in-memory `LLM` for tests and CI (plan Task 9 / todo Task 10).

Unit and workflow tests must never touch the host Ollama (SPEC §10/§12), so they
run against this fake. It returns scripted responses with no network and no
randomness, and records every call so a test can assert what the node sent. Two
modes cover the needs downstream:

- **Scripted** (`FakeLLM([...])`): responses handed out one per call, in order —
  lets a triage-retry test script an invalid response followed by a valid one.
- **Constant** (`FakeLLM("...")`): the same response for every call.

A scripted fake that runs out of responses raises rather than inventing output,
so an under-scripted test fails loudly instead of passing by accident.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.llm.base import LLM


class FakeLLM(LLM):
    """A deterministic `LLM` that replays scripted responses without any network."""

    def __init__(self, responses: str | Sequence[str]) -> None:
        """Configure the fake in constant mode (a `str`) or scripted mode (a sequence).

        In constant mode every `generate` returns the one string; in scripted mode
        the responses are returned in order and running past the end raises.
        """
        if isinstance(responses, str):
            self._constant: str | None = responses
            self._script: list[str] = []
        else:
            self._constant = None
            self._script = list(responses)
        self._index = 0
        # Each entry records one call's prompt and images, for test assertions.
        self.calls: list[dict[str, Any]] = []

    async def generate(
        self, prompt: str, *, images: list[str] | None = None, think: bool = True
    ) -> str:
        """Record the call and return the next scripted (or the constant) response.

        `think` is accepted for interface parity with the real client but ignored:
        the fake's output is fixed by its script, so tests stay deterministic
        regardless of it. Raises `IndexError` in scripted mode once the script is
        exhausted, so a test that calls the model more often than scripted fails loudly.
        """
        self.calls.append({"prompt": prompt, "images": images})
        if self._constant is not None:
            return self._constant
        if self._index >= len(self._script):
            raise IndexError(
                f"FakeLLM script exhausted after {len(self._script)} response(s); "
                "the code under test called generate() more times than scripted."
            )
        response = self._script[self._index]
        self._index += 1
        return response

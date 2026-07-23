"""Shared fixtures for the eval-suite tests (plan Task 30 / todo Task 32).

The golden-triage and groundedness suites run the *real* agent nodes (`triage`,
`validate`) against an injected `LLM`. In `make eval` that LLM is the host Ollama
model; in these tests — which must never touch a model (SPEC §10/§12) — it is a
deterministic fake. Unlike the plain `FakeLLM` (constant, or scripted in call
order), the eval-runner tests need the fake's answer to depend on *which* case is
being scored, because a suite scores many cases in one run. `RoutingFakeLLM` maps a
substring of the prompt (the case's unique message or draft body) to the response the
model should return for it, so a whole suite can be scored deterministically without
coupling to call order.

Only `app.llm.base` is imported here (it always exists); the `evals.*` modules under
test are imported inside the test files, so a missing implementation fails those tests
rather than breaking collection of the whole eval-test package.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from app.llm.base import LLM


class RoutingFakeLLM(LLM):
    """A deterministic `LLM` that picks its response by matching a prompt substring.

    `routes` maps a needle (a substring unique to one eval case's prompt — its message
    or draft body) to the raw model response the fake should return whenever that needle
    appears in the prompt. `default` is returned when no needle matches; when `default`
    is None an unmatched prompt raises `KeyError`, so a test that forgets to script a
    case fails loudly instead of scoring it on stale output. Every prompt is recorded in
    `calls` for assertions.
    """

    def __init__(self, routes: dict[str, str], *, default: str | None = None) -> None:
        """Store the needle->response routing table and the optional default response."""
        self._routes = routes
        self._default = default
        self.calls: list[str] = []

    async def generate(
        self, prompt: str, *, images: list[str] | None = None, think: bool = True
    ) -> str:
        """Return the response whose needle is in `prompt`, else the default (or raise).

        Records the prompt, then returns the first route whose needle is a substring of
        `prompt`. Falls back to `default`; with no default and no match, raises `KeyError`
        so an under-scripted test fails loudly rather than silently.
        """
        self.calls.append(prompt)
        for needle, response in self._routes.items():
            if needle in prompt:
                return response
        if self._default is not None:
            return self._default
        raise KeyError("RoutingFakeLLM: no route matched the prompt (test under-scripted)")


@pytest.fixture
def make_routing_llm() -> Callable[..., RoutingFakeLLM]:
    """Return a factory that builds a `RoutingFakeLLM` from a routing table.

    Handed to a test as `make_routing_llm({needle: response, ...}, default=...)` so each
    test declares exactly how the fake model answers each of its cases.
    """

    def factory(routes: dict[str, str], *, default: str | None = None) -> RoutingFakeLLM:
        """Build a `RoutingFakeLLM` for the given routes and optional default response."""
        return RoutingFakeLLM(routes, default=default)

    return factory

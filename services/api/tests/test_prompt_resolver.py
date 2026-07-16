"""Unit tests for Langfuse-first prompt resolution (plan Task 27 / todo Task 29).

Drafting/triage/extraction prompts are versioned in **Langfuse** and fetched at
runtime so a non-engineer can improve them from the approved-reply feedback loop with
no redeploy; a pinned in-repo template ships as the fallback for offline/CI runs (SPEC
§4.10). `resolve_prompt` is that seam: it consults an injected Langfuse-like client
first and falls back to the in-repo registry.

Langfuse is not a runtime dependency yet (the self-hosted service arrives in a later
task), so these tests stub the client. The behaviours pinned are the acceptance
criteria: no client → the in-repo template; a client with a prompt → the Langfuse
template and its version; and — the resilience the fallback exists for — a client that
errors or returns a blank template falls back to in-repo rather than sending an empty
or missing prompt.
"""

from __future__ import annotations

import pytest

from app.prompts.registry import get_prompt, get_prompt_version
from app.prompts.resolver import resolve_prompt


class _StubPrompt:
    """A minimal stand-in for a Langfuse prompt object: a template plus its version."""

    def __init__(self, prompt: str, version: int) -> None:
        """Record the template text and integer version the stub client will hand back."""
        self.prompt = prompt
        self.version = version


class _StubClient:
    """A Langfuse-like client that returns one canned prompt for any name."""

    def __init__(self, prompt: _StubPrompt) -> None:
        """Hold the canned prompt this stub returns from `get_prompt`."""
        self._prompt = prompt

    def get_prompt(self, name: str) -> _StubPrompt:
        """Return the canned prompt regardless of `name`, mimicking a Langfuse fetch."""
        return self._prompt


class _RaisingClient:
    """A Langfuse-like client whose fetch always fails, mimicking an outage/offline CI."""

    def get_prompt(self, name: str) -> _StubPrompt:
        """Raise as if Langfuse were unreachable, to exercise the in-repo fallback."""
        raise RuntimeError("langfuse unreachable")


def test_no_client_falls_back_to_in_repo_registry() -> None:
    """With no Langfuse client the resolver returns the pinned in-repo template + version.

    This is the offline/CI path: the deployment has no Langfuse, so drafting must still
    resolve a usable prompt from the registry (SPEC §4.10).
    """
    resolved = resolve_prompt("draft")

    assert resolved.template == get_prompt("draft")
    assert resolved.version == get_prompt_version("draft")


def test_resolves_template_and_version_from_langfuse() -> None:
    """A client with a prompt yields the Langfuse template and a name-scoped version.

    When Langfuse manages the prompt, the resolved text is Langfuse's — not the in-repo
    fallback — and the version label carries Langfuse's version number so the audit
    trail ties a reply back to the exact prompt that produced it (SPEC §4.10 / §7.1).
    """
    client = _StubClient(_StubPrompt("LANGFUSE DRAFT TEMPLATE {message} {sources}", version=3))

    resolved = resolve_prompt("draft", client=client)

    assert resolved.template == "LANGFUSE DRAFT TEMPLATE {message} {sources}"
    assert resolved.version == "draft-v3"


def test_client_error_falls_back_to_in_repo() -> None:
    """A failing Langfuse fetch falls back to the in-repo prompt, never propagating.

    The fallback exists precisely so an outage or offline CI run cannot stop the desk
    from drafting; the resolver must swallow the client error and use the registry.
    """
    resolved = resolve_prompt("draft", client=_RaisingClient())

    assert resolved.template == get_prompt("draft")
    assert resolved.version == get_prompt_version("draft")


def test_blank_langfuse_template_falls_back_to_in_repo() -> None:
    """A blank Langfuse template falls back rather than sending the model an empty prompt.

    A misconfigured or empty prompt version must not silently produce an empty draft
    prompt; the resolver treats blank as a miss and uses the pinned in-repo template.
    """
    client = _StubClient(_StubPrompt("   ", version=7))

    resolved = resolve_prompt("draft", client=client)

    assert resolved.template == get_prompt("draft")
    assert resolved.version == get_prompt_version("draft")


def test_unknown_name_without_client_raises_keyerror() -> None:
    """An unregistered name with no client fails loudly, matching the registry contract."""
    with pytest.raises(KeyError):
        resolve_prompt("does-not-exist")

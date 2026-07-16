"""Langfuse-first prompt resolution with an in-repo fallback (plan Task 27 / todo Task 29).

Drafting/triage/extraction prompts are versioned in **Langfuse** and fetched at runtime,
so a non-engineer can improve them from the approved-reply feedback loop with no
redeploy; a pinned in-repo template ships as the fallback for offline/CI runs (SPEC
§4.10). This module is that seam: `resolve_prompt` consults an injected Langfuse-like
client first and falls back to the registry.

The fallback is not just for a missing deployment — it is the resilience the design
depends on. A client that errors (Langfuse down, offline CI) or hands back a blank
template must never stop the desk from drafting or send the model an empty prompt, so
both cases fall back to the pinned in-repo template. Langfuse is not a runtime
dependency yet (the self-hosted service arrives in a later task); the client is injected
and typed structurally, so the thin Langfuse adapter added later just has to satisfy the
`LangfusePromptClient` protocol — no caller changes.
"""

from __future__ import annotations

from typing import NamedTuple, Protocol, runtime_checkable

from app.prompts.registry import get_prompt, get_prompt_version


class ResolvedPrompt(NamedTuple):
    """A resolved prompt: its `template` text and the `version` label the audit records.

    `version` is the in-repo label (`<name>-v1`) on the fallback path or a name-scoped
    Langfuse label (`<name>-v<n>`) when Langfuse resolved it, so a reply always ties back
    to the exact prompt that produced it (SPEC §7.1).
    """

    template: str
    version: str


@runtime_checkable
class LangfusePrompt(Protocol):
    """The shape `resolve_prompt` reads off a Langfuse prompt object: text + version."""

    prompt: str
    version: int


@runtime_checkable
class LangfusePromptClient(Protocol):
    """The one method `resolve_prompt` needs from a Langfuse client: fetch a prompt by name.

    Typed structurally so the resolver depends only on `get_prompt`, not on the Langfuse
    SDK — the later adapter and the tests' stub both satisfy it without inheritance.
    """

    def get_prompt(self, name: str) -> LangfusePrompt:
        """Return the managed prompt registered in Langfuse under `name`."""
        ...


def _in_repo(name: str) -> ResolvedPrompt:
    """Return the pinned in-repo template and version for `name`.

    The fallback every path lands on. Propagates `KeyError` for an unknown name, keeping
    the registry's fail-loud contract so a typo never yields a blank prompt.
    """
    return ResolvedPrompt(get_prompt(name), get_prompt_version(name))


def resolve_prompt(name: str, *, client: LangfusePromptClient | None = None) -> ResolvedPrompt:
    """Resolve prompt `name` from Langfuse when a client is given, else from the registry.

    With no `client` (offline/CI, or no Langfuse deployment) the pinned in-repo template
    and version are returned. With a client, its managed prompt wins — carrying a
    name-scoped version label (`<name>-v<n>`) — unless the fetch fails or returns a blank
    template, in which case the resolver falls back to in-repo rather than stopping the
    draft or sending an empty prompt (SPEC §4.10). Raises `KeyError` only when the
    fallback itself has no such name, preserving the registry's fail-loud contract.
    """
    if client is None:
        return _in_repo(name)
    try:
        managed = client.get_prompt(name)
        template = managed.prompt
        if template and template.strip():
            return ResolvedPrompt(template, f"{name}-v{managed.version}")
    except Exception:
        # Langfuse unreachable / offline CI: fall back so drafting never stalls.
        return _in_repo(name)
    # A blank or missing managed template: fall back rather than send an empty prompt.
    return _in_repo(name)

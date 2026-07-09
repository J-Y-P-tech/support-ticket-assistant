"""Strip a reasoning trace baked into text, keeping only the final answer — a *fallback*.

The primary way traces are dropped is Ollama's own field separation: `OllamaLLM`
reads the `response` field, which the API guarantees holds the final answer only
(SPEC Appendix A; <https://docs.ollama.com/capabilities/thinking>). The HTTP API
never emits a textual wrapper, so this util is for the cases the API can't cover —
text that already has a trace baked in:

- **CLI framing** — the Ollama *CLI* renders `Thinking... ...done thinking.` around
  the trace (the ellipsis is sometimes the unicode `…`).
- **Inline tags** — some models (e.g. DeepSeek-R1) emit `<think>...</think>` tags in
  their raw output.

Callers that must guarantee a clean, final-answer-only string (e.g. the OCR pass,
plan Task 20) apply this defensively.
"""

from __future__ import annotations

import re

# CLI framing: the `Thinking` opener, its ellipsis (ASCII `...` or unicode `…`), the
# reasoning body (non-greedy, so it stops at the first real close), an optional
# closing ellipsis, then the `done thinking` marker. Only `done thinking` closes the
# trace, so a stray "done" earlier in the reasoning is safe.
_CLI_TRACE_RE = re.compile(
    r"""
    \s*                     # any leading whitespace
    Thinking                # opening marker
    (?:\.\.\.|…)       # ... or unicode ellipsis
    .*?                     # the reasoning body (DOTALL, non-greedy)
    (?:\.\.\.|…)?      # optional ellipsis before the close
    \s*
    done\s+thinking\.?      # closing marker
    """,
    re.IGNORECASE | re.DOTALL | re.VERBOSE,
)

# Inline `<think>...</think>` tags (non-greedy, so only the first block is removed).
_THINK_TAG_RE = re.compile(r"\s*<think>.*?</think>", re.IGNORECASE | re.DOTALL)

# Detection (not removal): any single marker that signals a leaked reasoning trace —
# an open/close think tag, or the CLI `Thinking…`/`…done thinking` framing. The CLI
# markers require the ellipsis so plain prose ("I am done thinking about it") is not a
# false positive. A lone tag counts, so a truncated/partial leak is still caught.
_TRACE_MARKER_RE = re.compile(
    r"</?think>|Thinking(?:\.\.\.|…)|(?:\.\.\.|…)\s*done\s+thinking",
    re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    """Remove a baked-in reasoning trace (CLI framing or `<think>` tags), return the answer.

    Strips the first CLI-style `Thinking... ...done thinking.` block and any inline
    `<think>...</think>` tags, then trims surrounding whitespace. Text with no trace
    comes back unchanged apart from that trim, so the util is safe to apply
    unconditionally to any model output.
    """
    text = _CLI_TRACE_RE.sub("", text, count=1)
    text = _THINK_TAG_RE.sub("", text)
    return text.strip()


def contains_thinking_trace(text: str) -> bool:
    """Return True if `text` shows any sign of a leaked reasoning trace.

    A cheap, side-effect-free check (unlike `strip_thinking`, it mutates nothing),
    so the runtime can call it on every model response to *detect* a leak — the
    signal that a model isn't honouring the `think` split. The caller then logs and
    alerts (structlog Task 24, Langfuse Task 29) and flags the output to the human
    gate (Task 17) rather than silently scrubbing it. Detection deliberately errs
    toward precision: the CLI markers require their ellipsis framing so ordinary
    prose containing "done thinking" is not a false positive.
    """
    return _TRACE_MARKER_RE.search(text) is not None

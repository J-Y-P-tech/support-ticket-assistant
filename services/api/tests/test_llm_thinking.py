"""Unit tests for the thinking-trace stripper — a *fallback* (plan Task 9 / todo Task 10).

The primary, model-adaptive way we drop reasoning traces is Ollama's own field
separation: `OllamaLLM` reads the `response` field, which holds the final answer
only (SPEC Appendix A; <https://docs.ollama.com/capabilities/thinking>). That path
is grounded against the real selected model in `test_llm_capture.py`.

`strip_thinking` exists for the cases the API can't cover: text that already has a
trace baked in — output captured from the Ollama *CLI* (which renders the
`Thinking... ...done thinking.` framing) or a model that emits inline
`<think>...</think>` tags. These are known, fixed textual formats, so synthetic
inputs are the right test here; there is no live model behaviour to ground against
(the HTTP API never emits either wrapper).
"""

from __future__ import annotations

from app.llm.thinking import contains_thinking_trace, strip_thinking


def test_strips_cli_thinking_block_keeping_only_final_answer() -> None:
    """CLI-rendered `Thinking... ...done thinking.` framing is removed, answer kept."""
    raw = (
        "Thinking...\n"
        "The user is asking for their balance; I should answer plainly.\n"
        "...done thinking.\n\n"
        "Your current balance is $50.00."
    )
    assert strip_thinking(raw) == "Your current balance is $50.00."


def test_strips_cli_block_with_unicode_ellipsis() -> None:
    """The CLI framing with the unicode ellipsis (…) is stripped just the same."""
    raw = "Thinking…\nSome reasoning here.\n…done thinking.\n\nFinal answer."
    assert strip_thinking(raw) == "Final answer."


def test_strips_inline_think_tags() -> None:
    """Inline `<think>...</think>` tags (DeepSeek-R1-style raw output) are removed."""
    raw = "<think>\nStep through the policy, then answer.\n</think>\nYou are eligible."
    assert strip_thinking(raw) == "You are eligible."


def test_returns_text_unchanged_when_no_trace() -> None:
    """A plain answer with no trace comes back intact (only surrounding space trimmed)."""
    raw = "  Just the answer, no reasoning trace.  "
    assert strip_thinking(raw) == "Just the answer, no reasoning trace."


def test_reasoning_containing_the_word_done_is_not_a_false_close() -> None:
    """Only the real `done thinking` marker closes the trace, not a stray 'done'."""
    raw = (
        "Thinking...\n"
        "I am done pondering the edge cases but will continue.\n"
        "...done thinking.\n"
        "Answer: 42"
    )
    assert strip_thinking(raw) == "Answer: 42"


def test_multiline_final_answer_survives_intact() -> None:
    """Everything after the closing marker is preserved, including later newlines."""
    raw = "Thinking...\nreasoning\n...done thinking.\n\nLine one.\nLine two."
    assert strip_thinking(raw) == "Line one.\nLine two."


# --- contains_thinking_trace: detection (not mutation) for production monitoring ---
# The detector is the safe primitive the runtime uses to *notice* a leaked trace so
# it can be logged/alerted (structlog Task 24, Langfuse Task 29) and the output
# flagged to the human gate (Task 17) — never silently scrubbed. Detecting is cheap
# and side-effect-free, so it can run on every model response without the
# false-positive risk of stripping.


def test_detects_inline_think_tags() -> None:
    """A leaked `<think>` tag is detected."""
    assert contains_thinking_trace("<think>reasoning</think>\nThe answer.") is True


def test_detects_a_partial_leak_with_only_a_closing_tag() -> None:
    """A truncated leak (only `</think>`, no opener) is still detected."""
    assert contains_thinking_trace("...reasoning cut off</think>\nThe answer.") is True


def test_detects_cli_opener_and_closer_including_unicode() -> None:
    """The CLI `Thinking…`/`…done thinking` markers are detected (ASCII and unicode)."""
    assert contains_thinking_trace("Thinking...\nstuff\n...done thinking.\nAnswer") is True
    assert contains_thinking_trace("Thinking…\nstuff\n…done thinking.\nAnswer") is True


def test_clean_answer_is_not_flagged() -> None:
    """An ordinary answer with no trace markers is not a false positive."""
    assert contains_thinking_trace("Your balance is $50.00. Anything else?") is False


def test_plain_phrase_done_thinking_without_framing_is_not_flagged() -> None:
    """'done thinking' in prose (no ellipsis framing) must not trip the detector."""
    assert contains_thinking_trace("I am done thinking about your request now.") is False

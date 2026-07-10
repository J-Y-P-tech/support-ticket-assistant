"""Unit tests for the central prompt registry (`app.prompts.registry`).

The registry is the single seam every model-using node resolves its prompt
through, so prompts live in one place and Task 28 can front it with Langfuse
without changing any node. These tests pin that contract: a known name returns a
usable template, the triage template exposes the placeholders its node fills, and
an unknown name fails loudly rather than yielding an empty prompt.
"""

from __future__ import annotations

import pytest

from app.prompts.registry import get_prompt


def test_returns_registered_template() -> None:
    """A known prompt name returns its non-empty template string."""
    template = get_prompt("triage")
    assert isinstance(template, str)
    assert template.strip()


def test_triage_template_exposes_expected_placeholders() -> None:
    """The triage template carries every placeholder the triage node fills.

    Guards the seam's contract with the node: if a placeholder is renamed here the
    node's `.format(...)` would raise, and this test catches it first.
    """
    template = get_prompt("triage")
    for placeholder in ("{message}", "{facts}", "{categories}", "{urgencies}", "{sentiments}"):
        assert placeholder in template


def test_validate_template_exposes_expected_placeholders() -> None:
    """The groundedness-judge template carries the placeholders the validate node fills.

    Guards the seam's contract with the node: if `{draft}` or `{sources}` is renamed
    here the node's `.format(...)` would raise, and this test catches it first.
    """
    template = get_prompt("validate")
    for placeholder in ("{draft}", "{sources}"):
        assert placeholder in template


def test_unknown_prompt_name_raises() -> None:
    """An unregistered name raises `KeyError` instead of returning an empty prompt."""
    with pytest.raises(KeyError):
        get_prompt("does-not-exist")

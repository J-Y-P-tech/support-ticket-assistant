"""Unit tests for the central prompt registry (`app.prompts.registry`).

The registry is the single seam every model-using node resolves its prompt
through, so prompts live in one place and Task 28 can front it with Langfuse
without changing any node. These tests pin that contract: a known name returns a
usable template, the triage template exposes the placeholders its node fills, and
an unknown name fails loudly rather than yielding an empty prompt.
"""

from __future__ import annotations

import pytest

from app.prompts.registry import get_prompt, get_prompt_version


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


def test_prompt_version_is_returned_for_a_known_name() -> None:
    """A known prompt name has a version label the audit trail records (SPEC §7.1).

    The compliance trail must note *which* version of each instruction the AI used, so
    a reviewer can tie any reply back to the exact prompt. Every registered prompt
    therefore carries a version string; here we pin triage's.
    """
    # Ask the registry which version of the triage prompt is in force.
    version = get_prompt_version("triage")
    # The first in-repo version of every prompt is "<name>-v1" (Langfuse overrides later).
    assert version == "triage-v1"


def test_every_registered_prompt_has_a_version() -> None:
    """Each prompt the nodes resolve also exposes a version, so none is unlabelled.

    If a new prompt is added to the registry without a matching version, its node's
    audit row could not record a prompt version — this catches that gap.
    """
    # Every model-using node resolves one of these prompt names.
    names = ("triage", "draft", "validate", "input_guard", "output_guard", "ocr", "extract", "fuse")
    for name in names:
        # Each name must return a non-empty version string (no missing labels).
        assert get_prompt_version(name).strip()


def test_unknown_prompt_version_name_raises() -> None:
    """Asking for the version of an unregistered prompt fails loudly, like `get_prompt`."""
    # A typo'd or absent name must raise rather than hand back a blank version.
    with pytest.raises(KeyError):
        get_prompt_version("does-not-exist")

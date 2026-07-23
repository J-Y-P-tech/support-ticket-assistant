"""Integrity checks for the shipped eval datasets (todo Task 32).

Separate from the runner-logic tests: these load the *curated* JSON that ships in
`evals/datasets/` and assert it is broad enough to be a meaningful gate — a golden set
that only covered one category, or a red-team set that skipped a whole attack family,
would pass the runner yet prove nothing. They also re-assert the safety guarantee at the
dataset level: every shipped red-team attack is actually blocked by the real guard
(SPEC §4.6 acceptance — a red-team ticket "is blocked/flagged, proven by an eval test").
"""

from __future__ import annotations

from app.guardrails.injection import screen_input
from app.guardrails.output import screen_output
from app.schemas.enums import Category, Urgency
from evals.loader import (
    load_groundedness_cases,
    load_redteam_input_cases,
    load_redteam_output_cases,
    load_triage_cases,
)

# The input-guard signature families a red-team input set should exercise, and the
# output-guard families the output set should exercise — kept in step with the guard
# signature JSON so a new family added there is a reminder to add a red-team case.
_INJECTION_FAMILIES = {
    "instruction_override",
    "system_prompt_exfiltration",
    "role_manipulation",
    "fake_role_marker",
    "instruction_termination",
}
_OUTPUT_FAMILIES = {"forbidden_promise", "pii_leak"}


def test_golden_triage_set_covers_many_categories_and_a_critical_case() -> None:
    """The golden triage set spans several categories and includes a critical-urgency ticket.

    Breadth matters: a single-category set would let a badly-skewed classifier score well.
    A critical case ensures the highest-urgency path (which drives the rep-queue sort) is
    exercised.
    """
    cases = load_triage_cases()
    assert len(cases) >= 8
    categories = {case.expected_category for case in cases}
    assert len(categories) >= 5
    assert categories <= set(Category)  # every label is a real category
    assert any(case.expected_urgency == Urgency.CRITICAL for case in cases)


def test_groundedness_set_has_both_grounded_and_ungrounded_cases() -> None:
    """The groundedness set has a faithful and a drifting draft, so both verdicts run."""
    cases = load_groundedness_cases()
    assert len(cases) >= 4
    grounded_flags = {case.grounded for case in cases}
    assert grounded_flags == {True, False}


def test_redteam_input_set_covers_every_injection_family() -> None:
    """The red-team input set exercises every deterministic injection family, not just one shape."""
    cases = load_redteam_input_cases()
    families = {case.category for case in cases}
    assert _INJECTION_FAMILIES <= families


def test_redteam_output_set_covers_promises_and_pii() -> None:
    """The red-team output set exercises both output violations: promises and PII leaks."""
    cases = load_redteam_output_cases()
    families = {case.category for case in cases}
    assert _OUTPUT_FAMILIES <= families


async def test_every_shipped_injection_case_is_blocked() -> None:
    """Every curated red-team input case is flagged by the deterministic input guard."""
    for case in load_redteam_input_cases():
        result = await screen_input(case.text)
        assert result.flagged is True, f"injection case not blocked: {case.id}"


async def test_every_shipped_output_attack_is_flagged() -> None:
    """Every curated red-team output case (promise or PII) is flagged by the output guard."""
    for case in load_redteam_output_cases():
        result = await screen_output(case.text)
        assert result.flagged is True, f"output attack not flagged: {case.id}"

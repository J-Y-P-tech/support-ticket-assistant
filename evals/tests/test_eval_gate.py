"""Tests for eval-gated prompt promotion: `evaluate_gate` (todo Task 32).

The gate is the mechanism behind SPEC §4.10: "a new prompt version must beat the eval
suite before it becomes the active version; regressions are rejected" — and the acceptance
criterion "no prompt version is marked active unless it passes the eval gate." A candidate
is promoted only when it (a) clears every threshold *and* (b) does not regress any suite
versus the currently-active baseline. Both halves are tested here on constructed reports, so
the promotion rule is pinned independently of how the reports are produced.
"""

from __future__ import annotations

from evals.cases import EvalReport, EvalThresholds, SuiteResult
from evals.gate import GateDecision, evaluate_gate


def _report(*, redteam: float = 1.0, triage: float = 1.0, groundedness: float = 1.0) -> EvalReport:
    """Build an `EvalReport` whose suites hit the given scores (out of 10 cases each).

    A helper so a test can dial one suite's score up or down to express "passes",
    "regresses", or "fails the threshold" without hand-building every `SuiteResult`.
    """
    return EvalReport(
        suites=[
            SuiteResult(name="redteam_input", total=10, passed=round(redteam * 10), failures=[]),
            SuiteResult(name="redteam_output", total=10, passed=round(redteam * 10), failures=[]),
            SuiteResult(name="triage", total=10, passed=round(triage * 10), failures=[]),
            SuiteResult(
                name="groundedness", total=10, passed=round(groundedness * 10), failures=[]
            ),
        ],
        thresholds=EvalThresholds(),
    )


def test_promotes_a_passing_candidate_with_no_baseline() -> None:
    """With no active version yet, a candidate that clears every threshold is promoted."""
    decision = evaluate_gate(_report())
    assert isinstance(decision, GateDecision)
    assert decision.promoted is True


def test_promotes_a_candidate_that_matches_or_beats_the_baseline() -> None:
    """A passing candidate that ties or improves every suite versus the baseline is promoted."""
    baseline = _report(triage=0.8, groundedness=0.9)
    candidate = _report(triage=0.9, groundedness=0.9)  # triage up, groundedness equal
    assert evaluate_gate(candidate, baseline=baseline).promoted is True


def test_rejects_a_candidate_that_fails_a_threshold() -> None:
    """A candidate below a threshold is rejected with no baseline; the reason names the suite."""
    decision = evaluate_gate(_report(triage=0.6))  # 0.6 < 0.8 minimum
    assert decision.promoted is False
    assert any("triage" in reason for reason in decision.reasons)


def test_rejects_a_candidate_that_lets_a_redteam_case_through() -> None:
    """A candidate that fails to block every red-team case is rejected — the gate is absolute."""
    decision = evaluate_gate(_report(redteam=0.9))
    assert decision.promoted is False
    assert any("redteam" in reason for reason in decision.reasons)


def test_rejects_a_regression_even_when_thresholds_are_met() -> None:
    """A candidate that still clears the threshold but scores below the baseline is rejected.

    This is the core of eval-gated promotion: a new prompt version must not make any suite
    worse than the active one, so a groundedness drop from 1.0 to 0.9 — both above the 0.8
    minimum — is still a regression and must not be promoted.
    """
    baseline = _report(groundedness=1.0)
    candidate = _report(groundedness=0.9)
    decision = evaluate_gate(candidate, baseline=baseline)
    assert decision.promoted is False
    assert any(
        "regress" in reason.lower() or "groundedness" in reason for reason in decision.reasons
    )


def test_a_promoted_decision_carries_no_blocking_reasons() -> None:
    """A promotion decision records no failure reasons, so the caller can log a clean pass."""
    decision = evaluate_gate(_report(), baseline=_report())
    assert decision.promoted is True
    assert decision.reasons == []

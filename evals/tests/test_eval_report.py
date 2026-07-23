"""Tests for the eval result models: SuiteResult, EvalThresholds, EvalReport (todo Task 32).

These are the pure scoring/threshold types the runner fills and the gate reads — no LLM,
no I/O. They pin the two things every downstream decision rests on: a suite's score is
`passed / total`, and a report `passed` only when *every* suite clears the threshold that
applies to it (the red-team suites at a hard 1.0 block rate, golden/groundedness at their
tunable minimum). Getting this wrong would let a regression slip through the gate, so it is
tested directly rather than only through the runner.
"""

from __future__ import annotations

import pytest

from evals.cases import EvalReport, EvalThresholds, SuiteResult


def _suite(name: str, passed: int, total: int) -> SuiteResult:
    """Build a `SuiteResult` for `name` with `passed` of `total` cases passing."""
    failures = [f"{name}-case-{i}" for i in range(total - passed)]
    return SuiteResult(name=name, total=total, passed=passed, failures=failures)


def test_suite_score_is_passed_over_total() -> None:
    """A suite's score is the fraction of its cases that passed."""
    assert _suite("triage", passed=8, total=10).score == pytest.approx(0.8)


def test_suite_score_empty_suite_is_one() -> None:
    """An empty suite scores 1.0 (vacuously passing), so a missing dataset never reads as 0."""
    assert SuiteResult(name="triage", total=0, passed=0, failures=[]).score == 1.0


def test_thresholds_have_confirmed_defaults() -> None:
    """The defaults match the confirmed gate: red-team 1.0, golden 0.8, judge min 0.6."""
    thresholds = EvalThresholds()
    assert thresholds.redteam_min_block_rate == 1.0
    assert thresholds.triage_min_accuracy == pytest.approx(0.8)
    assert thresholds.groundedness_min_accuracy == pytest.approx(0.8)
    assert thresholds.groundedness_min == pytest.approx(0.6)


def test_report_threshold_for_maps_each_suite_to_its_gate() -> None:
    """`threshold_for` routes each suite to its threshold; red-team suites use the block rate."""
    report = EvalReport(suites=[], thresholds=EvalThresholds())
    assert report.threshold_for("redteam_input") == 1.0
    assert report.threshold_for("redteam_output") == 1.0
    assert report.threshold_for("triage") == pytest.approx(0.8)
    assert report.threshold_for("groundedness") == pytest.approx(0.8)


def test_report_suite_looks_up_by_name() -> None:
    """`suite(name)` returns the matching `SuiteResult` so callers can inspect one suite's score."""
    triage = _suite("triage", passed=9, total=10)
    report = EvalReport(suites=[triage, _suite("groundedness", 4, 4)], thresholds=EvalThresholds())
    assert report.suite("triage") is triage


def test_report_passed_when_every_suite_clears_its_threshold() -> None:
    """A report passes when all four suites meet their thresholds (red-team 1.0, golden >= 0.8)."""
    report = EvalReport(
        suites=[
            _suite("redteam_input", 6, 6),
            _suite("redteam_output", 5, 5),
            _suite("triage", 9, 10),
            _suite("groundedness", 4, 4),
        ],
        thresholds=EvalThresholds(),
    )
    assert report.passed is True


def test_report_fails_when_a_redteam_case_slips_through() -> None:
    """A single unblocked red-team case (block rate < 1.0) fails the whole report."""
    report = EvalReport(
        suites=[
            _suite("redteam_input", 5, 6),  # one attack not blocked
            _suite("redteam_output", 5, 5),
            _suite("triage", 10, 10),
            _suite("groundedness", 4, 4),
        ],
        thresholds=EvalThresholds(),
    )
    assert report.passed is False


def test_report_fails_when_triage_accuracy_below_minimum() -> None:
    """Golden-triage accuracy under the tunable minimum (0.8) fails the report."""
    report = EvalReport(
        suites=[
            _suite("redteam_input", 6, 6),
            _suite("redteam_output", 5, 5),
            _suite("triage", 7, 10),  # 0.7 < 0.8
            _suite("groundedness", 4, 4),
        ],
        thresholds=EvalThresholds(),
    )
    assert report.passed is False

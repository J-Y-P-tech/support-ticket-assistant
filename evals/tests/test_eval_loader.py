"""Tests for the eval dataset loader (todo Task 32).

The golden and red-team datasets ship as curated JSON under `evals/datasets/` (the same
"curated data as JSON" convention as the guard signatures and `mock_kb/`), so a
non-engineer can grow the golden set from rep corrections (SPEC §4.9) without touching
code. The loader parses each file into typed, schema-validated case models. These tests
prove the happy path returns the right typed cases and that a malformed dataset fails
loudly at load rather than silently scoring against a broken case.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.cases import GroundednessCase, RedTeamCase, TriageCase
from evals.loader import (
    load_groundedness_cases,
    load_redteam_input_cases,
    load_redteam_output_cases,
    load_triage_cases,
)


def test_load_triage_cases_returns_typed_cases() -> None:
    """The shipped triage dataset loads into a non-empty list of `TriageCase`."""
    cases = load_triage_cases()
    assert cases
    assert all(isinstance(case, TriageCase) for case in cases)


def test_load_groundedness_cases_returns_typed_cases() -> None:
    """The shipped groundedness dataset loads into a non-empty list of `GroundednessCase`."""
    cases = load_groundedness_cases()
    assert cases
    assert all(isinstance(case, GroundednessCase) for case in cases)


def test_load_redteam_cases_carry_their_surface() -> None:
    """The two red-team datasets load as `RedTeamCase`s tagged with the surface they screen."""
    input_cases = load_redteam_input_cases()
    output_cases = load_redteam_output_cases()
    assert input_cases and output_cases
    assert all(isinstance(case, RedTeamCase) and case.surface == "input" for case in input_cases)
    assert all(isinstance(case, RedTeamCase) and case.surface == "output" for case in output_cases)


def test_loader_rejects_a_malformed_dataset(tmp_path: Path) -> None:
    """A triage case with a value outside the closed category enum fails validation at load time."""
    bad = tmp_path / "bad_triage.json"
    bad.write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "message": "hi",
                    "expected_category": "not_a_real_category",
                    "expected_urgency": "high",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_triage_cases(bad)

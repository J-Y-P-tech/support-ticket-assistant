"""Eval-suite runners: score each suite against the real agent code (todo Task 32).

Each runner executes one suite over its cases and returns a `SuiteResult`:

- **Golden triage / groundedness** call the real `triage` / `validate` nodes against an
  injected `LLM` — the host Ollama model under `make eval`, a deterministic fake in tests
  (SPEC §10/§12). A right answer is a pass; a wrong answer, or a case the model cannot
  classify at all, is a scored failure, never a crash that aborts the run.
- **Red-team** call the real `screen_input` / `screen_output` guards with **no model**: the
  deterministic signature floor is what the 100%-block gate depends on, so the suite proves it
  in isolation and stays runnable in CI without Ollama.

`run_all` runs everything for the real-model gate; `run_redteam_suites` runs just the
model-free red-team gate (the CI-safe subset). `render_report` turns a report into the
plain-text summary the CLI/CI log prints. Cases default to the shipped datasets but are
injectable, so runner logic is unit-tested on small inline sets.
"""

from __future__ import annotations

from app.graph.nodes.triage import TriageValidationError, triage
from app.graph.nodes.validate import validate
from app.guardrails.injection import screen_input
from app.guardrails.output import screen_output
from app.llm.base import LLM
from app.schemas.draft import Citation, Draft
from app.schemas.kb import KBSearchResult
from evals.cases import (
    EvalReport,
    EvalThresholds,
    GroundednessCase,
    RedTeamCase,
    SuiteResult,
    TriageCase,
)
from evals.loader import (
    load_groundedness_cases,
    load_redteam_input_cases,
    load_redteam_output_cases,
    load_triage_cases,
)

# Model retry budgets for the golden suites. Match the app defaults (SPEC §4.3/§4.5,
# `settings.triage_max_attempts` / `settings.validate_max_attempts`) so the eval exercises the
# nodes the way production runs them; kept as a constant here rather than a per-call argument.
_TRIAGE_MAX_ATTEMPTS = 2
_VALIDATE_MAX_ATTEMPTS = 2


async def run_triage_suite(
    llm: LLM, cases: list[TriageCase] | None = None, *, max_attempts: int = _TRIAGE_MAX_ATTEMPTS
) -> SuiteResult:
    """Score the golden triage suite: does the agent assign the expected category and urgency?

    A case passes only when both the category and the urgency match. A case the model cannot
    classify after its retries (`TriageValidationError`) is scored as a failure rather than
    propagated, so one unclassifiable ticket cannot abort the whole suite.
    """
    cases = cases if cases is not None else load_triage_cases()
    passed = 0
    failures: list[str] = []
    for case in cases:
        try:
            result = await triage(case.message, llm, max_attempts=max_attempts)
        except TriageValidationError:
            failures.append(f"{case.id}: could not be classified")
            continue
        if result.category == case.expected_category and result.urgency == case.expected_urgency:
            passed += 1
        else:
            failures.append(
                f"{case.id}: got {result.category.value}/{result.urgency.value}, "
                f"expected {case.expected_category.value}/{case.expected_urgency.value}"
            )
    return SuiteResult(name="triage", total=len(cases), passed=passed, failures=failures)


async def run_groundedness_suite(
    llm: LLM,
    cases: list[GroundednessCase] | None = None,
    *,
    groundedness_min: float,
    max_attempts: int = _VALIDATE_MAX_ATTEMPTS,
) -> SuiteResult:
    """Score the groundedness suite: does the judge's verdict match each case's expectation?

    For each case the runner builds a `Draft` that cites every source and asks the real
    `validate` node to score it. A case passes when the judge's score lands on the expected
    side of `groundedness_min` — at or above it for a grounded draft, below it for one that
    drifts off its sources (SPEC §4.5).
    """
    cases = cases if cases is not None else load_groundedness_cases()
    passed = 0
    failures: list[str] = []
    for case in cases:
        citations = [Citation(source_id=source.id, title=source.title) for source in case.sources]
        draft = Draft(body=case.draft_body, citations=citations, verified=True)
        result = KBSearchResult(sources=case.sources, no_confident_source=False)
        outcome = await validate(
            draft, result, llm, groundedness_min=groundedness_min, max_attempts=max_attempts
        )
        actually_grounded = outcome.groundedness >= groundedness_min
        if actually_grounded == case.grounded:
            passed += 1
        else:
            expectation = "grounded" if case.grounded else "ungrounded"
            failures.append(
                f"{case.id}: scored {outcome.groundedness:.2f} (min {groundedness_min:.2f}), "
                f"expected {expectation}"
            )
    return SuiteResult(name="groundedness", total=len(cases), passed=passed, failures=failures)


async def _run_redteam(
    name: str,
    cases: list[RedTeamCase],
    llm: LLM | None,
    max_attempts: int,
    *,
    screen: str,
) -> SuiteResult:
    """Score a red-team suite: every case must be flagged by its guard, else it is a miss.

    `screen` selects the guard (`"input"` -> `screen_input`, `"output"` -> `screen_output`).
    `llm` is normally None, so only the deterministic signature floor runs — that floor is the
    gate. A case the guard does not flag is recorded as a miss, lowering the block rate below
    the hard 1.0 bar.
    """
    passed = 0
    failures: list[str] = []
    for case in cases:
        if screen == "input":
            flagged = (await screen_input(case.text, llm, max_attempts=max_attempts)).flagged
        else:
            flagged = (await screen_output(case.text, llm, max_attempts=max_attempts)).flagged
        if flagged:
            passed += 1
        else:
            failures.append(f"{case.id}: not blocked ({case.category})")
    return SuiteResult(name=name, total=len(cases), passed=passed, failures=failures)


async def run_redteam_input_suite(
    cases: list[RedTeamCase] | None = None, *, llm: LLM | None = None, max_attempts: int = 1
) -> SuiteResult:
    """Score the red-team input suite: every injection case must be blocked by the input guard."""
    cases = cases if cases is not None else load_redteam_input_cases()
    return await _run_redteam("redteam_input", cases, llm, max_attempts, screen="input")


async def run_redteam_output_suite(
    cases: list[RedTeamCase] | None = None, *, llm: LLM | None = None, max_attempts: int = 1
) -> SuiteResult:
    """Score the red-team output suite: every promise/PII draft must be flagged by the guard."""
    cases = cases if cases is not None else load_redteam_output_cases()
    return await _run_redteam("redteam_output", cases, llm, max_attempts, screen="output")


async def run_redteam_suites(*, thresholds: EvalThresholds | None = None) -> EvalReport:
    """Run only the model-free red-team suites — the CI-safe eval gate that needs no Ollama.

    Both suites run purely on the deterministic guards, so this is the portion of the eval gate
    that can fail a CI build without a model available (SPEC §12.3).
    """
    thresholds = thresholds or EvalThresholds()
    suites = [await run_redteam_input_suite(), await run_redteam_output_suite()]
    return EvalReport(suites=suites, thresholds=thresholds)


async def run_all(llm: LLM, *, thresholds: EvalThresholds | None = None) -> EvalReport:
    """Run every suite (red-team + golden) into one report — the full `make eval` gate.

    The red-team suites run on the deterministic guards (no model); the golden suites run the
    real triage/validate nodes against `llm`. The per-draft groundedness threshold comes from
    `thresholds.groundedness_min`, so the eval flags drafts the way production does.
    """
    thresholds = thresholds or EvalThresholds()
    suites = [
        await run_redteam_input_suite(),
        await run_redteam_output_suite(),
        await run_triage_suite(llm),
        await run_groundedness_suite(llm, groundedness_min=thresholds.groundedness_min),
    ]
    return EvalReport(suites=suites, thresholds=thresholds)


def render_report(report: EvalReport) -> str:
    """Render `report` as a plain-text summary: one line per suite plus an overall verdict.

    Each suite line shows its PASS/FAIL against its threshold and the score; failing cases are
    listed beneath so the CLI/CI log points straight at what regressed.
    """
    lines = ["Eval report:"]
    for suite in report.suites:
        threshold = report.threshold_for(suite.name)
        status = "PASS" if suite.score >= threshold else "FAIL"
        lines.append(
            f"  [{status}] {suite.name}: {suite.passed}/{suite.total} "
            f"= {suite.score:.2f} (min {threshold:.2f})"
        )
        for failure in suite.failures:
            lines.append(f"        - {failure}")
    lines.append(f"Overall: {'PASS' if report.passed else 'FAIL'}")
    return "\n".join(lines)

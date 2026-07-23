"""Eval-gated prompt promotion (todo Task 32).

SPEC §4.10: "a new prompt version must beat the eval suite before it becomes the active
version; regressions are rejected" — with the acceptance criterion that "no prompt version is
marked active unless it passes the eval gate." `evaluate_gate` is that decision: given a
candidate prompt version's `EvalReport` (and the currently-active version's report as the
baseline), it promotes the candidate only when it both clears every threshold and does not
score below the baseline on any suite.

The gate operates on `EvalReport`s, not prompts, so it is independent of how the reports are
produced: a caller managing prompt versions runs the suite (`evals.runner.run_all`) once with
the candidate prompt active and once with the baseline, then asks the gate to decide. Keeping
the decision here — pure and side-effect-free — is what lets it be exhaustively unit-tested and
reused by whatever drives promotion (a Langfuse admin action, a CI job).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from evals.cases import EvalReport


class GateDecision(BaseModel):
    """The gate's verdict on a candidate prompt version.

    `promoted` is True only when the candidate may become the active version. `reasons` lists
    every blocker when it may not — threshold misses and per-suite regressions — so the caller
    can log exactly why a version was held back; it is empty on a promotion.
    """

    promoted: bool
    reasons: list[str] = Field(default_factory=list)


def evaluate_gate(candidate: EvalReport, *, baseline: EvalReport | None = None) -> GateDecision:
    """Decide whether `candidate` may be promoted over the active `baseline`.

    The candidate is promoted only when both hold:

    - **Clears every threshold** — each suite meets its bar (red-team block rate 1.0, golden
      accuracies at their minimums). A miss on any suite blocks promotion.
    - **No regression** — with a `baseline`, the candidate must score at least as high as the
      baseline on every shared suite. A drop counts even when the candidate is still above the
      threshold, because promoting a strictly-worse version is the regression SPEC §4.10
      forbids. With no baseline (no active version yet) only the thresholds apply.

    Returns a `GateDecision`; when it blocks, `reasons` names each threshold miss and regression.
    """
    reasons: list[str] = []

    for suite in candidate.suites:
        threshold = candidate.threshold_for(suite.name)
        if suite.score < threshold:
            reasons.append(f"{suite.name} below threshold: {suite.score:.2f} < {threshold:.2f}")

    if baseline is not None:
        baseline_scores = {suite.name: suite.score for suite in baseline.suites}
        for suite in candidate.suites:
            prior = baseline_scores.get(suite.name)
            if prior is not None and suite.score < prior:
                reasons.append(
                    f"{suite.name} regressed vs baseline: {suite.score:.2f} < {prior:.2f}"
                )

    return GateDecision(promoted=not reasons, reasons=reasons)

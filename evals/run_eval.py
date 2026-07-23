"""CLI entry for `make eval` — run the AI eval + red-team suites and gate on the result (Task 32).

Thin shell (SPEC §9: keep logic out of hard-to-test surfaces): it wires the real Ollama model
and delegates all scoring to `evals.runner`, then exits non-zero when the gate fails so a CI
stage (SPEC §12.3) can block on it.

Two modes:

- default — the full gate: red-team suites (deterministic) plus the golden triage/groundedness
  suites against the host Ollama model. This needs Ollama running (the user's local/nightly
  quality gate).
- ``--red-team-only`` — just the model-free red-team suites, so a CI job without a model can
  still enforce the 100%-block safety gate.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

from app.config import get_settings
from app.llm.ollama import OllamaLLM
from evals.cases import EvalThresholds
from evals.runner import render_report, run_all, run_redteam_suites


async def _run(red_team_only: bool) -> int:
    """Run the selected suites, print the report, and return the exit code (0 pass, 1 fail)."""
    if red_team_only:
        report = await run_redteam_suites()
    else:
        settings = get_settings()
        thresholds = EvalThresholds(groundedness_min=settings.groundedness_min)
        llm = OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)
        try:
            report = await run_all(llm, thresholds=thresholds)
        finally:
            await llm.aclose()
    print(render_report(report))
    return 0 if report.passed else 1


def main() -> int:
    """Parse arguments and run the eval gate, returning its exit code."""
    parser = argparse.ArgumentParser(description="Run the AI eval + red-team suites.")
    parser.add_argument(
        "--red-team-only",
        action="store_true",
        help="Run only the deterministic red-team suites (no Ollama needed).",
    )
    args = parser.parse_args()
    return asyncio.run(_run(args.red_team_only))


if __name__ == "__main__":
    sys.exit(main())

"""Structural tests for the CI pipeline (`.github/workflows/ci.yml`, plan Task 31 / todo Task 33).

The pipeline itself runs in GitHub Actions (the user owns git/GitHub — SPEC §13); these tests do
not execute it. They parse the committed workflow YAML and assert it mirrors the SPEC §12 stage
order and that the eval + security gates block the build, so a later edit that drops a stage,
reorders it, or lets a broken change reach `build`/`deploy` fails here instead of silently in CI.

Stage order under test (SPEC §12):
  1. lint/format/type — ruff, black --check, mypy
  2. unit+contract+workflow tests — pytest, fake LLM, no model download
  3. AI eval gate — the red-team suite (`--red-team-only`; CI has no Ollama)
  4. security scan — bandit, pip-audit, gitleaks, trivy
  5. build — all four service images (api, kb_mcp, email_mcp, frontend)
  6. gated deploy — behind an environment/manual approval, never auto-deploying a failing change

The stages are chained with GitHub Actions `needs:`, so the DAG encodes the linear §12 order and
makes every earlier gate a hard predecessor of `build` (and, transitively, `deploy`).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

# Repo root is three parents up: .../evals/tests/test_ci_workflow.py -> tests -> evals -> root.
REPO_ROOT = Path(__file__).resolve().parents[2]
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"


def _load_workflow() -> dict[str, Any]:
    """Parse the CI workflow YAML into a dict, failing the test if the file is absent."""
    if not CI_WORKFLOW.exists():
        pytest.fail(f"CI workflow file not found at {CI_WORKFLOW}")
    return yaml.safe_load(CI_WORKFLOW.read_text())


def _job(workflow: dict[str, Any], name: str) -> dict[str, Any]:
    """Return the named job's definition, failing the test if the job is missing."""
    jobs = workflow.get("jobs", {})
    if name not in jobs:
        pytest.fail(f"CI workflow has no `{name}` job; jobs present: {sorted(jobs)}")
    return jobs[name]


def _needs(job: dict[str, Any]) -> list[str]:
    """Return a job's `needs` as a list (GitHub allows a bare string or a list)."""
    needs = job.get("needs", [])
    return [needs] if isinstance(needs, str) else list(needs)


def _job_text(job: dict[str, Any]) -> str:
    """Flatten every `run:` step of a job into one lowercased string for keyword assertions."""
    parts: list[str] = []
    for step in job.get("steps", []):
        run = step.get("run")
        if run:
            parts.append(run)
        uses = step.get("uses")
        if uses:
            parts.append(uses)
    return "\n".join(parts).lower()


def _transitive_needs(workflow: dict[str, Any], name: str) -> set[str]:
    """Return every job that `name` depends on directly or transitively via `needs`."""
    seen: set[str] = set()
    frontier = list(_needs(_job(workflow, name)))
    while frontier:
        dep = frontier.pop()
        if dep in seen:
            continue
        seen.add(dep)
        frontier.extend(_needs(_job(workflow, dep)))
    return seen


def test_ci_workflow_file_exists() -> None:
    """The workflow lives at the SPEC §8 path `.github/workflows/ci.yml` and parses as YAML."""
    workflow = _load_workflow()
    assert isinstance(workflow, dict)
    assert workflow.get("jobs"), "workflow defines no jobs"


def test_triggers_on_push_and_pull_request() -> None:
    """CI runs on push and pull_request (SPEC §12: 'GitHub Actions on push/PR')."""
    workflow = _load_workflow()
    # PyYAML parses the bare `on:` key as the boolean True, so accept either spelling.
    triggers = workflow.get("on", workflow.get(True))
    assert triggers, "workflow declares no triggers"
    assert "push" in triggers
    assert "pull_request" in triggers


def test_lint_stage_runs_ruff_black_check_and_mypy() -> None:
    """Stage 1 runs ruff, black --check, and mypy (SPEC §12.1)."""
    text = _job_text(_job(_load_workflow(), "lint"))
    assert "ruff" in text
    assert "black --check" in text
    assert "mypy" in text


def test_test_stage_runs_pytest_without_a_model() -> None:
    """Stage 2 runs pytest and never downloads a model — CI uses the fake LLM (SPEC §12.2)."""
    job = _job(_load_workflow(), "test")
    text = _job_text(job)
    assert "pytest" in text
    # No model download in CI: the fake LLM stands in for Ollama (SPEC §10/§12.2).
    assert "ollama" not in text
    assert "gemma" not in text


def test_eval_stage_runs_the_red_team_only_gate() -> None:
    """Stage 3 runs the eval gate in `--red-team-only` mode (no Ollama in CI, SPEC §12.3)."""
    job = _job(_load_workflow(), "eval")
    text = _job_text(job)
    assert "run_eval" in text or "make eval" in text
    assert "--red-team-only" in text
    # The eval gate must not reach for a model either.
    assert "ollama" not in text


def test_security_stage_runs_all_four_scanners() -> None:
    """Stage 4 runs bandit, pip-audit, gitleaks, and trivy (SPEC §12.4)."""
    text = _job_text(_job(_load_workflow(), "security"))
    for scanner in ("bandit", "pip-audit", "gitleaks", "trivy"):
        assert scanner in text, f"security stage is missing {scanner}"


def test_build_stage_builds_all_four_service_images() -> None:
    """Stage 5 builds every service image: api, kb_mcp, email_mcp, frontend (SPEC §12.5)."""
    text = _job_text(_job(_load_workflow(), "build"))
    for service in ("api", "kb_mcp", "email_mcp", "frontend"):
        assert service in text, f"build stage does not build {service}"


def test_stage_order_matches_spec() -> None:
    """The `needs:` chain encodes the linear SPEC §12 order lint→test→eval→security→build→deploy."""
    workflow = _load_workflow()
    assert _needs(_job(workflow, "lint")) == []
    assert "lint" in _needs(_job(workflow, "test"))
    assert "test" in _needs(_job(workflow, "eval"))
    assert "eval" in _needs(_job(workflow, "security"))
    assert "security" in _needs(_job(workflow, "build"))
    assert "build" in _needs(_job(workflow, "deploy"))


def test_eval_and_security_gate_the_build() -> None:
    """`build` transitively needs both the eval and security gates, so a regression blocks it."""
    deps = _transitive_needs(_load_workflow(), "build")
    assert "eval" in deps, "a failing eval gate must block the build"
    assert "security" in deps, "a failing security gate must block the build"


def test_deploy_is_gated_and_never_auto_deploys_a_failing_change() -> None:
    """Deploy sits behind `build` and a manual gate (environment/workflow_dispatch), SPEC §12.6."""
    workflow = _load_workflow()
    deploy = _job(workflow, "deploy")
    # Behind the full chain, so it can never run when an earlier gate fails.
    assert "build" in _transitive_needs(workflow, "deploy")
    # Manual approval: either a protected `environment:` or a manual `workflow_dispatch` trigger.
    triggers = workflow.get("on", workflow.get(True)) or {}
    manual = "environment" in deploy or "workflow_dispatch" in triggers
    assert manual, "deploy must require manual approval (environment or workflow_dispatch)"

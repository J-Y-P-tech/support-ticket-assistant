# Developer command surface (SPEC §11). The user runs these — the assistant only
# writes them. Dependency management and tooling run through `uv`.
#
# Targets marked "(Task N)" are placeholders until that task lands; they print a
# notice instead of failing, so the surface is complete from day one.

.PHONY: install lint format typecheck test eval migrate seed export-training-data security

install: ## Install runtime + dev dependencies and refresh the lockfile
	uv sync

format: ## Auto-format the codebase (black)
	uv run black .

lint: ## Lint the codebase (ruff)
	uv run ruff check .

typecheck: ## Static type-check (mypy, strict; per service root)
	# Each service is a separate import root (see pyproject mypy_path). Checking
	# them in one `mypy services` run makes both services' `tests/conftest.py`
	# resolve to the same module `tests.conftest` and mypy aborts on the clash, so
	# each root is checked in its own invocation. Extend this list per service.
	uv run mypy services/api
	uv run mypy services/email_mcp
	uv run mypy services/kb_mcp
	# Frontend: type-check the modules that hold logic (SPEC §9). app.py and views/
	# are thin Streamlit surfaces (logic lives in these tested modules), so they are
	# excluded here — this also keeps the frontend `app.py` module out of the run,
	# avoiding a clash with the api `app` package.
	uv run mypy services/frontend/api_client.py services/frontend/config.py services/frontend/formatting.py
	# Evals (SPEC §8): the runner reuses the api's `app.*` modules (on the mypy path
	# already). Check the logic modules explicitly, like the frontend line above, so the
	# `evals/tests` dir (a `tests` namespace) never collides with a service's under mypy.
	uv run mypy evals/cases.py evals/loader.py evals/runner.py evals/gate.py evals/run_eval.py

test: ## Run the unit/contract/workflow test suite (pytest; fake LLM, no model)
	uv run pytest

eval: ## Run the AI eval + red-team suites (golden triage/groundedness need host Ollama)
	# Full gate: red-team suites (deterministic) + golden triage/groundedness against the
	# host Ollama model. Load .env so config resolves the model tag/URL on a plain host
	# process (like `migrate`/`export-training-data`). PYTHONPATH=services/api puts the
	# api's `app.*` package (which the runner reuses) on the import path — same as the CI
	# eval job. For the model-free red-team gate only (no Ollama), run the runner directly:
	#   PYTHONPATH=services/api uv run python -m evals.run_eval --red-team-only
	set -a; [ -f .env ] && . ./.env; set +a; PYTHONPATH=services/api uv run python -m evals.run_eval

migrate: ## Apply DB migrations via email_mcp (reads POSTGRES_* from the env/.env)
	# Load .env into the environment first: migrate.py reads os.environ directly,
	# and (unlike pydantic-settings / docker-compose) a plain host process does not
	# auto-read .env. The `[ -f .env ]` guard keeps it working when .env is absent.
	set -a; [ -f .env ] && . ./.env; set +a; uv run python services/email_mcp/migrate.py

seed: ## Load mock-KB answers + sample tickets
	@echo "seed: implemented in Tasks 3/7 (sample tickets + mock_kb data)."

export-training-data: ## Export de-identified SFT + preference JSONL (to stdout)
	# Load .env like `migrate` does: export_training_data.py reads POSTGRES_* from
	# os.environ, and a plain host process does not auto-read .env. Output is JSONL on
	# stdout — redirect it to a file, e.g. `make export-training-data > corpus.jsonl`.
	set -a; [ -f .env ] && . ./.env; set +a; uv run python services/email_mcp/export_training_data.py

security: ## Run security scans (bandit, pip-audit, gitleaks, image scan)
	@echo "security: implemented in Task 30 (bandit + pip-audit + gitleaks + trivy)."

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

test: ## Run the unit/contract/workflow test suite (pytest; fake LLM, no model)
	uv run pytest

eval: ## Run the AI eval + red-team suites
	@echo "eval: implemented in Task 29 (evals/ golden + red-team suites)."

migrate: ## Apply DB migrations via email_mcp (reads POSTGRES_* from the env/.env)
	# Load .env into the environment first: migrate.py reads os.environ directly,
	# and (unlike pydantic-settings / docker-compose) a plain host process does not
	# auto-read .env. The `[ -f .env ]` guard keeps it working when .env is absent.
	set -a; [ -f .env ] && . ./.env; set +a; uv run python services/email_mcp/migrate.py

seed: ## Load mock-KB answers + sample tickets
	@echo "seed: implemented in Tasks 3/7 (sample tickets + mock_kb data)."

export-training-data: ## Export de-identified SFT + preference JSONL
	@echo "export-training-data: implemented in Task 26 (training corpus export)."

security: ## Run security scans (bandit, pip-audit, gitleaks, image scan)
	@echo "security: implemented in Task 30 (bandit + pip-audit + gitleaks + trivy)."

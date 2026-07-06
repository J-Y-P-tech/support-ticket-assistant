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

typecheck: ## Static type-check (mypy, strict)
	uv run mypy services

test: ## Run the unit/contract/workflow test suite (pytest; fake LLM, no model)
	uv run pytest

eval: ## Run the AI eval + red-team suites
	@echo "eval: implemented in Task 29 (evals/ golden + red-team suites)."

migrate: ## Apply DB migrations via email_mcp
	@echo "migrate: implemented in Task 3 (email_mcp plain-SQL migrations)."

seed: ## Load mock-KB answers + sample tickets
	@echo "seed: implemented in Tasks 3/7 (sample tickets + mock_kb data)."

export-training-data: ## Export de-identified SFT + preference JSONL
	@echo "export-training-data: implemented in Task 26 (training corpus export)."

security: ## Run security scans (bandit, pip-audit, gitleaks, image scan)
	@echo "security: implemented in Task 30 (bandit + pip-audit + gitleaks + trivy)."

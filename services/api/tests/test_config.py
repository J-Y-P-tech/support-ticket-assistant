"""Unit tests for the api service configuration loader (`app.config`).

These tests pin the Task 1 acceptance criteria (SPEC §11 / plan Task 1):
- `Settings` loads every documented value from the environment.
- Secrets are stored as `SecretStr` so they never render in logs/traces (SPEC §6).
- A missing *required* variable fails clearly (pydantic `ValidationError`), because
  every value is config, not code — there are no hard-coded defaults.

The tests never read a real `.env` file (`_env_file=None`); the environment is
controlled entirely via monkeypatch so runs are deterministic in CI.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError

from app.config import Settings

# A complete set of environment values covering every field the loader requires.
# Values are dummies — no real secrets — mirroring the shape of `.env.example`.
_COMPLETE_ENV: dict[str, str] = {
    "LLM_MODEL": "gemma4:12b",
    "OLLAMA_BASE_URL": "http://host.docker.internal:11434",
    "KB_SEARCH_LIMIT": "3",
    "QUEUE_PAGE_DEFAULT": "50",
    "QUEUE_PAGE_MAX": "200",
    "API_AUTH_TOKEN": "frontend-to-api-token",
    "EMAIL_MCP_URL": "http://email_mcp:8000",
    "EMAIL_MCP_TOKEN": "api-to-email-token",
    "KB_MCP_URL": "http://kb_mcp:8000",
    "KB_MCP_TOKEN": "api-to-kb-token",
    "ENCRYPTION_KEY": "dGVzdC1mZXJuZXQta2V5LWRvLW5vdC11c2UtZm9yLXJlYWw=",
    "LANGFUSE_HOST": "http://langfuse:3000",
    "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
    "LANGFUSE_SECRET_KEY": "sk-lf-test",
}


@pytest.fixture
def full_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Populate the process environment with a complete, valid config set.

    Every required variable is present, so `Settings(_env_file=None)` should
    construct without error inside a test that uses this fixture.
    """
    for key, value in _COMPLETE_ENV.items():
        monkeypatch.setenv(key, value)


def test_loads_all_values_from_env(full_env: None) -> None:
    """A fully-populated environment yields a `Settings` with every field set."""
    settings = Settings(_env_file=None)

    assert settings.llm_model == "gemma4:12b"
    assert settings.ollama_base_url == "http://host.docker.internal:11434"
    assert settings.email_mcp_url == "http://email_mcp:8000"
    assert settings.kb_mcp_url == "http://kb_mcp:8000"
    assert settings.langfuse_host == "http://langfuse:3000"
    # Sizing knobs are parsed as ints, not left as strings.
    assert settings.kb_search_limit == 3
    assert settings.queue_page_default == 50
    assert settings.queue_page_max == 200


def test_secrets_are_secretstr(full_env: None) -> None:
    """Token and key fields are `SecretStr`, so they do not leak in repr/logs."""
    settings = Settings(_env_file=None)

    for secret in (
        settings.api_auth_token,
        settings.email_mcp_token,
        settings.kb_mcp_token,
        settings.encryption_key,
        settings.langfuse_public_key,
        settings.langfuse_secret_key,
    ):
        assert isinstance(secret, SecretStr)

    # The raw value is recoverable for actual use, but not shown in repr().
    assert settings.api_auth_token.get_secret_value() == "frontend-to-api-token"
    assert "frontend-to-api-token" not in repr(settings)


@pytest.mark.parametrize("missing_var", sorted(_COMPLETE_ENV))
def test_missing_required_var_raises(monkeypatch: pytest.MonkeyPatch, missing_var: str) -> None:
    """Omitting any single required variable raises a clear `ValidationError`.

    Every documented variable is required (no defaults), so the loader fails
    loudly at startup rather than silently running with a hard-coded fallback.
    """
    for key, value in _COMPLETE_ENV.items():
        if key == missing_var:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)

    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)

    # The error names the offending field so the failure is actionable.
    assert missing_var.lower() in str(excinfo.value).lower()

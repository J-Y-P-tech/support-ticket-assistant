"""Configuration for the Streamlit frontend (SPEC §9, plan Task 6).

The frontend talks only to the api, over HTTP with a bearer token. Both the api
base URL and that token come from the environment (12-factor, no hard-coded
values), mirroring the api service's own config. The token is `SecretStr` so it
is never rendered in `repr()`, logs, or a stack trace.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for the frontend.

    Fields carry no defaults: each maps to a required environment variable
    (documented in `.env.example`). Construction raises `ValidationError` if a
    required variable is absent.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    api_base_url: str
    api_auth_token: SecretStr  # presented by the frontend to the api


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached `Settings` instance (env read once)."""
    return Settings()

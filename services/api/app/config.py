"""Application configuration for the api service.

Twelve-factor config via `pydantic-settings` (SPEC §9): every URL, token, model
tag, and key is supplied by the environment — there are no hard-coded defaults,
so a missing required value fails loudly at startup rather than silently running
on a fallback (SPEC §13, "Model tags, tokens, and keys are config, not code").

Secrets are typed `SecretStr` so they are never rendered in `repr()`, logs, or
Langfuse traces (SPEC §6, "no plaintext secrets"; §7.2 PII/secret redaction).
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-driven settings for the api service.

    Fields carry no defaults: each maps to a required environment variable
    (documented in `.env.example`). Construction raises `ValidationError` if any
    required variable is absent. Secret-bearing fields use `SecretStr`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- LLM (single multimodal model on the host, reached via host.docker.internal) ---
    llm_model: str
    ollama_base_url: str

    # --- Retrieval / pagination sizing (tuning knobs, not code; single source of
    #     truth shared with the MCP servers via the same env vars) ---
    kb_search_limit: int  # max ranked KB sources a search requests (kb_mcp default too)
    queue_page_default: int  # rep-queue page size when the client asks for none
    queue_page_max: int  # hard ceiling on a rep-queue page, so no request pulls the lot

    # --- Agent node behaviour (tuning knobs) ---
    triage_max_attempts: int  # model tries for a valid TriageResult before human hand-off
    groundedness_min: float  # min judge groundedness score before a draft is flagged unverified
    validate_max_attempts: int  # judge tries for a valid GroundednessVerdict before failing closed
    extract_max_attempts: int  # model tries for a valid ExtractionResult before flagging low-conf
    few_shot_limit: int  # max recent approved replies injected as drafting few-shot (§4.10)

    # --- LangGraph checkpointer store (SPEC §3/§5: Postgres-backed, so a case
    #     pending human review survives a restart and resumes days later) ---
    database_url: str  # Postgres DSN the api's workflow checkpointer connects to

    # --- Inter-service auth: a separate bearer token per connection (least privilege, SPEC §6) ---
    api_auth_token: SecretStr  # presented by the frontend to this api
    email_mcp_url: str
    email_mcp_token: SecretStr  # presented by this api to email_mcp
    kb_mcp_url: str
    kb_mcp_token: SecretStr  # presented by this api to kb_mcp

    # --- Application-layer encryption for PII at rest (SPEC §6) ---
    encryption_key: SecretStr

    # --- Langfuse observability (self-hosted, SPEC §7.2) ---
    langfuse_host: str
    langfuse_public_key: SecretStr
    langfuse_secret_key: SecretStr


@lru_cache
def get_settings() -> Settings:
    """Return a process-wide cached `Settings` instance.

    Cached so the environment/`.env` is read once. Tests construct `Settings`
    directly (bypassing this cache and the `.env` file) for isolation.
    """
    return Settings()

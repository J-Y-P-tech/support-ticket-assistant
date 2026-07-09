"""Async client for the host Ollama model (plan Task 9 / todo Task 10).

The single multimodal model (`LLM_MODEL`, e.g. `gemma4:12b`) runs on the host,
outside Docker, reached via `OLLAMA_BASE_URL` (`host.docker.internal`) — SPEC §4.2.
This client posts to Ollama's `/api/generate` endpoint with `stream=False`.

Reasoning traces are handled the model-adaptive way: the request carries the
`think` flag explicitly (so behaviour doesn't drift with each model's default) and
`generate` returns the `response` field, which Ollama guarantees holds the final
answer only — the trace is kept out in the separate `thinking` field
(<https://docs.ollama.com/capabilities/thinking>). No text parsing is involved;
`strip_thinking` is only a fallback for text that already carries a baked-in trace.
One `httpx.AsyncClient` is held and reused across calls; the app lifespan closes it
via `aclose`.
"""

from __future__ import annotations

from typing import Any, cast

import httpx

from app.llm.base import LLM

# Generous default: the host model is large and a vision/OCR pass over an
# attachment can take a while. A single knob, not a per-call argument.
_DEFAULT_TIMEOUT = 120.0


class OllamaLLM(LLM):
    """Async `LLM` backed by the host Ollama HTTP API."""

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        client: httpx.AsyncClient | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        """Store the model tag and endpoint; the HTTP client is created lazily.

        `base_url` is the Ollama root (no path); requests go to its `/api/generate`.
        An `httpx.AsyncClient` may be injected (tests pass a `MockTransport`-backed
        one); when it is, this object does not own it and `aclose` leaves it open.
        """
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client
        self._owns_client = client is None

    def _get_client(self) -> httpx.AsyncClient:
        """Return the held HTTP client, creating an owned one on first use."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self._timeout)
        return self._client

    async def _post(self, prompt: str, *, images: list[str] | None, think: bool) -> dict[str, Any]:
        """POST one non-streaming `/api/generate` request and return the parsed JSON.

        `think` is always sent explicitly so trace separation is deterministic across
        models; `images` (base64-encoded) is included only when provided, for the
        vision/OCR path. A non-2xx response raises `httpx.HTTPStatusError` rather than
        yielding empty output, so a model/config failure surfaces instead of hiding.
        """
        payload: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "think": think,
        }
        if images:
            payload["images"] = images
        response = await self._get_client().post(f"{self._base_url}/api/generate", json=payload)
        response.raise_for_status()
        return cast("dict[str, Any]", response.json())

    async def generate(
        self, prompt: str, *, images: list[str] | None = None, think: bool = True
    ) -> str:
        """Return Ollama's `response` (final answer only) for `prompt`.

        The reasoning trace, if any, is kept out of `response` by Ollama itself, so
        no text stripping is needed here. `think` defaults **on** for answer quality;
        a caller can pass False to trade reasoning for speed (still reading only the
        final answer).
        """
        data = await self._post(prompt, images=images, think=think)
        return cast("str", data["response"])

    async def generate_full(
        self, prompt: str, *, images: list[str] | None = None, think: bool = True
    ) -> dict[str, Any]:
        """Return Ollama's full response JSON, including the split `response`/`thinking`.

        Used to record a real-model fixture (`test_llm_capture.py`) that grounds the
        trace-separation assumption; app code calls `generate` and reads `response`.
        """
        return await self._post(prompt, images=images, think=think)

    async def aclose(self) -> None:
        """Close the held HTTP client, unless it was injected (then it is not ours)."""
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None

"""Unit tests for the async Ollama client (plan Task 9 / todo Task 10).

The real client talks to the host Ollama (`OLLAMA_BASE_URL`, model `LLM_MODEL`)
over its HTTP API. Per the project working agreement the user runs Ollama, so
these tests never hit the network: an injected `httpx.AsyncClient` backed by a
`MockTransport` captures the request and returns a canned reply, proving the
client posts the right shape to `/api/generate` and unwraps the response text.

The client relies on Ollama's own trace separation rather than parsing text: it
sends the `think` flag and reads the `response` field, which Ollama guarantees
holds the final answer only (the reasoning trace goes to the separate `thinking`
field). See <https://docs.ollama.com/capabilities/thinking>. `generate` returns
the trace-free `response`; `generate_full` exposes both for the capture fixture.
"""

from __future__ import annotations

import json

import httpx

from app.llm.base import LLM
from app.llm.ollama import OllamaLLM


def _mock_client(handler: object) -> httpx.AsyncClient:
    """Build an `httpx.AsyncClient` whose requests are served by `handler` (no network)."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


async def test_is_an_llm() -> None:
    """`OllamaLLM` implements the shared `LLM` interface."""
    assert isinstance(OllamaLLM(model="gemma4:12b", base_url="http://x:11434"), LLM)


async def test_posts_prompt_to_generate_endpoint_and_returns_response_text() -> None:
    """A text generate posts model+prompt to `/api/generate` and returns `response`."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture the outgoing request and reply with a canned completion."""
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "the answer"})

    llm = OllamaLLM(
        model="gemma4:12b",
        base_url="http://host.docker.internal:11434",
        client=_mock_client(handler),
    )
    result = await llm.generate("what is my balance?")

    assert result == "the answer"
    assert seen["url"] == "http://host.docker.internal:11434/api/generate"
    assert seen["body"] == {
        "model": "gemma4:12b",
        "prompt": "what is my balance?",
        "stream": False,
        "think": True,
    }


async def test_think_defaults_on_but_can_be_disabled() -> None:
    """`think` is sent explicitly — on by default (answer quality), off when asked.

    Sending it explicitly (rather than relying on each model's default) is what
    makes the trace separation deterministic across whatever model is selected.
    """
    seen: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """Record the `think` value each call sent."""
        seen.append(json.loads(request.content)["think"])
        return httpx.Response(200, json={"response": "ok"})

    llm = OllamaLLM(model="gemma4:12b", base_url="http://x:11434", client=_mock_client(handler))
    await llm.generate("default")
    await llm.generate("fast", think=False)

    assert seen == [True, False]


async def test_generate_full_returns_response_and_thinking() -> None:
    """`generate_full` exposes Ollama's split fields (used to record real fixtures)."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Reply with Ollama's separated `response` and `thinking` fields."""
        return httpx.Response(200, json={"response": "final", "thinking": "trace"})

    llm = OllamaLLM(model="gemma4:12b", base_url="http://x:11434", client=_mock_client(handler))
    data = await llm.generate_full("q", think=True)

    assert data["response"] == "final"
    assert data["thinking"] == "trace"


async def test_includes_images_when_provided() -> None:
    """A multimodal call forwards base64 images in the request body (SPEC §4.2)."""
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        """Capture the body so the image payload can be asserted."""
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"response": "transcribed"})

    llm = OllamaLLM(model="gemma4:12b", base_url="http://x:11434", client=_mock_client(handler))
    await llm.generate("transcribe this", images=["BASE64IMG"])

    body = seen["body"]
    assert isinstance(body, dict)
    assert body["images"] == ["BASE64IMG"]


async def test_raises_on_http_error_status() -> None:
    """A non-2xx response from Ollama surfaces as an error, not a silent empty string."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Reply with a server error so the client raises."""
        return httpx.Response(500, json={"error": "model not found"})

    llm = OllamaLLM(model="gemma4:12b", base_url="http://x:11434", client=_mock_client(handler))
    try:
        await llm.generate("anything")
    except httpx.HTTPStatusError:
        pass
    else:
        raise AssertionError("expected an HTTPStatusError on a 500 response")

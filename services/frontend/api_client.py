"""Typed HTTP client from the Streamlit frontend to the api (SPEC §8, plan Task 6).

The frontend never touches the MCP servers or the database — it goes through the
api over HTTP, presenting the frontend->api bearer token on every request. This
is the single seam where those calls live, so the views stay logic-free and this
wrapper is the thing under test (against a mocked transport, no network).

Streamlit runs synchronously, so this is a synchronous `httpx` client (the async
paths in SPEC §9 are the api's MCP/Ollama calls, not the UI).
"""

from __future__ import annotations

from typing import Any

import httpx


class ApiClient:
    """Thin synchronous wrapper over the api's customer and rep endpoints."""

    def __init__(
        self,
        base_url: str,
        token: str,
        *,
        transport: httpx.BaseTransport | None = None,
        timeout: float = 10.0,
    ) -> None:
        """Build the client with the bearer header preset.

        `transport` is an injection seam: tests pass an `httpx.MockTransport` so
        no network is touched; production leaves it `None` for the real transport.
        """
        self._client = httpx.Client(
            base_url=base_url,
            headers={"Authorization": f"Bearer {token}"},
            transport=transport,
            timeout=timeout,
        )

    def submit_ticket(self, message: str, attachments: list[str] | None = None) -> dict[str, Any]:
        """Submit a new ticket and return the created ticket (with its code)."""
        response = self._client.post(
            "/tickets", json={"message": message, "attachments": attachments or []}
        )
        response.raise_for_status()
        return _json_dict(response)

    def lookup_ticket(self, code: str) -> dict[str, Any] | None:
        """Return a ticket by reference code, or None for a neutral not-found.

        The api answers an unknown code with a neutral 404 (no enumeration leak);
        that maps to `None` here so the view shows a uniform "not found" message.
        """
        response = self._client.get(f"/tickets/{code}")
        if response.status_code == httpx.codes.NOT_FOUND:
            return None
        response.raise_for_status()
        return _json_dict(response)

    def fetch_queue(self, *, limit: int | None = None, after: str | None = None) -> dict[str, Any]:
        """Return one keyset page of the rep queue (`{items, next_cursor}`).

        `limit` is omitted by default so page size is decided by the api's configured
        `QUEUE_PAGE_DEFAULT` — the frontend holds no page-size policy of its own.
        `after` is the `next_cursor` from the previous page; omitted on page one.
        """
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = limit
        if after is not None:
            params["after"] = after
        response = self._client.get("/rep/queue", params=params)
        response.raise_for_status()
        return _json_dict(response)


def _json_dict(response: httpx.Response) -> dict[str, Any]:
    """Return the response's JSON body typed as a dict."""
    return dict(response.json())

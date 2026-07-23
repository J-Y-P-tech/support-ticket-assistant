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

    def fetch_review(self, ticket_id: int) -> dict[str, Any] | None:
        """Return the draft-review payload for a ticket paused at the human gate, or None.

        The payload (message, triage, sources, draft, warning flags) the rep workspace
        renders; only valid while the case is awaiting review. A freshly-submitted ticket
        whose draft is still being generated is not yet at the gate, so the api answers
        409 — mapped here to `None` (as `lookup_ticket` maps a 404) so the view can show a
        "not ready yet" message instead of surfacing an error.
        """
        response = self._client.get(f"/rep/tickets/{ticket_id}/review")
        if response.status_code == httpx.codes.CONFLICT:
            return None
        response.raise_for_status()
        return _json_dict(response)

    def approve_draft(self, ticket_id: int) -> dict[str, Any]:
        """Stage an approve-as-is decision on the paused case (does not send)."""
        response = self._client.post(f"/rep/tickets/{ticket_id}/approve")
        response.raise_for_status()
        return _json_dict(response)

    def edit_draft(self, ticket_id: int, reply: str) -> dict[str, Any]:
        """Stage the rep's edited reply into the paused case (held until send)."""
        response = self._client.post(f"/rep/tickets/{ticket_id}/edit", json={"reply": reply})
        response.raise_for_status()
        return _json_dict(response)

    def send_draft(self, ticket_id: int, rep_id: str) -> dict[str, Any] | None:
        """Send the approved/edited reply: the api resumes through finalize and persists it.

        `rep_id` is the audit marker the api requires to resolve the case (SPEC §4.7).
        Returns the action result carrying the sent reply and the Resolved status, or
        `None` when nothing has been staged yet — the two-step gate fails closed with a
        409 if no draft was approved or edited first, mapped here so the view can prompt
        the rep to approve/edit rather than surfacing an error.
        """
        response = self._client.post(f"/rep/tickets/{ticket_id}/send", json={"rep_id": rep_id})
        if response.status_code == httpx.codes.CONFLICT:
            return None
        response.raise_for_status()
        return _json_dict(response)

    def reject_draft(
        self, ticket_id: int, rep_id: str, reason: str | None = None
    ) -> dict[str, Any]:
        """Reject the draft: route the case back for research with nothing sent.

        `rep_id` attributes the rejection in the audit trail; `reason` is an optional
        note carried alongside it.
        """
        response = self._client.post(
            f"/rep/tickets/{ticket_id}/reject", json={"rep_id": rep_id, "reason": reason}
        )
        response.raise_for_status()
        return _json_dict(response)


def _json_dict(response: httpx.Response) -> dict[str, Any]:
    """Return the response's JSON body typed as a dict."""
    return dict(response.json())

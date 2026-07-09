"""Headless UI tests for the Streamlit app via `AppTest` (plan Task 6).

These prove the walking-skeleton acceptance criteria end-to-end through the real
views, with the api replaced by an in-memory fake client injected through
`st.session_state` (the app reads its client from there, so no api or network is
touched): submitting shows a reference code; looking that code up shows `New`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from streamlit.testing.v1 import AppTest

APP_PATH = str(Path(__file__).resolve().parents[1] / "app.py")


class FakeApiClient:
    """In-memory stand-in for `ApiClient`, injected via `st.session_state`."""

    def __init__(self) -> None:
        """Start with no known tickets and an empty submission log."""
        self.submitted: list[tuple[str, list[str]]] = []
        self.tickets: dict[str, dict[str, Any]] = {}

    def submit_ticket(self, message: str, attachments: list[str] | None = None) -> dict[str, Any]:
        """Record the submission and return a freshly-created New ticket."""
        self.submitted.append((message, attachments or []))
        return {"reference_code": "TKT-0001", "status": "New"}

    def lookup_ticket(self, code: str) -> dict[str, Any] | None:
        """Return the registered ticket for `code`, or None (neutral not-found)."""
        return self.tickets.get(code)

    def fetch_queue(self, *, limit: int | None = None, after: str | None = None) -> dict[str, Any]:
        """Return an empty queue page (the rep view is not under test here)."""
        return {"items": [], "next_cursor": None}


def _app_with_client(client: FakeApiClient) -> AppTest:
    """Build an `AppTest` for the app with `client` injected, and run it once."""
    at = AppTest.from_file(APP_PATH)
    at.session_state["client"] = client
    return at.run()


def test_submitting_a_ticket_shows_a_reference_code() -> None:
    """Submitting a non-empty message surfaces the returned `TKT-####` code."""
    at = _app_with_client(FakeApiClient())

    at.sidebar.radio[0].set_value("Customer").run()
    at.text_area[0].set_value("I can't log in to my account").run()
    at.button[0].click().run()

    assert any("TKT-0001" in msg.value for msg in at.success)


def test_looking_up_a_code_shows_its_status() -> None:
    """Looking up a known reference code shows its status (`New`)."""
    client = FakeApiClient()
    client.tickets["TKT-0001"] = {"reference_code": "TKT-0001", "status": "New"}
    at = _app_with_client(client)

    at.sidebar.radio[0].set_value("Check my case").run()
    at.text_input[0].set_value("TKT-0001").run()
    at.button[0].click().run()

    assert any("New" in msg.value for msg in at.info)


def test_submitting_a_blank_message_shows_an_error_not_a_code() -> None:
    """A blank submission is refused client-side, before any api call."""
    client = FakeApiClient()
    at = _app_with_client(client)

    at.sidebar.radio[0].set_value("Customer").run()
    at.text_area[0].set_value("   ").run()
    at.button[0].click().run()

    assert client.submitted == []
    assert len(at.error) >= 1

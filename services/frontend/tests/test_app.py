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
        """Start with no known tickets, an empty submission log, and an empty queue."""
        self.submitted: list[tuple[str, list[str]]] = []
        self.tickets: dict[str, dict[str, Any]] = {}
        self.queue_page: dict[str, Any] = {"items": [], "next_cursor": None}
        self.review: dict[str, Any] | None = None
        self.actions: list[tuple[Any, ...]] = []

    def submit_ticket(self, message: str, attachments: list[str] | None = None) -> dict[str, Any]:
        """Record the submission and return a freshly-created New ticket."""
        self.submitted.append((message, attachments or []))
        return {"reference_code": "TKT-0001", "status": "New"}

    def lookup_ticket(self, code: str) -> dict[str, Any] | None:
        """Return the registered ticket for `code`, or None (neutral not-found)."""
        return self.tickets.get(code)

    def fetch_queue(self, *, limit: int | None = None, after: str | None = None) -> dict[str, Any]:
        """Return the configured queue page (empty unless a rep test sets one)."""
        return self.queue_page

    def fetch_review(self, ticket_id: int) -> dict[str, Any]:
        """Return the configured draft-review payload for a ticket."""
        self.actions.append(("fetch_review", ticket_id))
        assert self.review is not None
        return self.review

    def approve_draft(self, ticket_id: int) -> dict[str, Any]:
        """Record an approve and return a staged (not-yet-sent) result."""
        self.actions.append(("approve", ticket_id))
        return {"ticket_id": ticket_id, "status": "Drafted", "reply": None}

    def edit_draft(self, ticket_id: int, reply: str) -> dict[str, Any]:
        """Record an edit and return a staged result."""
        self.actions.append(("edit", ticket_id, reply))
        return {"ticket_id": ticket_id, "status": "Drafted", "reply": None}

    def send_draft(self, ticket_id: int, rep_id: str) -> dict[str, Any]:
        """Record a send and return a Resolved result carrying the drafted reply."""
        self.actions.append(("send", ticket_id, rep_id))
        reply = self.review["draft"]["body"] if self.review else None
        return {"ticket_id": ticket_id, "status": "Resolved", "reply": reply}

    def reject_draft(
        self, ticket_id: int, rep_id: str, reason: str | None = None
    ) -> dict[str, Any]:
        """Record a reject and return a NeedsResearch result."""
        self.actions.append(("reject", ticket_id, rep_id, reason))
        return {"ticket_id": ticket_id, "status": "NeedsResearch", "reply": None}


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


# --- Rep workspace: draft review + disposition (plan Task 19) -----------------


def _queued_ticket() -> dict[str, Any]:
    """Build a one-ticket queue page whose row is selectable for review (id 7)."""
    return {
        "items": [
            {
                "id": 7,
                "reference_code": "TKT-0007",
                "status": "Drafted",
                "urgency": "normal",
                "category": "account_access",
            }
        ],
        "next_cursor": None,
    }


def _review_payload(*, verified: bool = True, flags: list[str] | None = None) -> dict[str, Any]:
    """Build a draft-review payload for ticket 7, tunable for the unverified case."""
    return {
        "ticket_id": 7,
        "status": "Drafted",
        "message": "How do I reset my online banking password?",
        "extracted_facts": None,
        "triage": {"category": "account_access", "urgency": "normal", "sentiment": "neutral"},
        "sources": [{"id": "KB-1", "title": "Password reset", "text": "Use the login screen."}],
        "draft": {
            "body": "You can reset your password from the login screen. [KB-1]",
            "citations": [{"source_id": "KB-1", "title": "Password reset"}],
            "verified": verified,
        },
        "flags": flags or [],
        "trace_leak": False,
    }


def test_rep_approve_then_send_resolves_the_case() -> None:
    """Approving then sending a draft in the rep workspace resolves the case.

    The SPEC §4.7 two-step disposition through the UI: the rep opens the queued
    ticket's review, approves the draft, supplies their audit marker, and sends —
    the workspace then reports the case Resolved and forwarded the marker on send.
    """
    client = FakeApiClient()
    client.queue_page = _queued_ticket()
    client.review = _review_payload()
    at = _app_with_client(client)

    at.sidebar.radio[0].set_value("Rep workspace").run()
    at.text_input(key="rep_rep_id").set_value("rep-1").run()
    at.button(key="rep_approve").click().run()
    at.button(key="rep_send").click().run()

    assert any("Resolved" in msg.value for msg in at.success)
    assert ("send", 7, "rep-1") in client.actions


def test_rep_unverified_draft_shows_a_warning_banner() -> None:
    """A draft flagged unverified renders a warning banner in the rep workspace.

    When the api marks the draft `verified: False` (it drifted from its sources), the
    workspace must warn the rep — the draft can't be presented as sourced fact.
    """
    client = FakeApiClient()
    client.queue_page = _queued_ticket()
    client.review = _review_payload(verified=False)
    at = _app_with_client(client)

    at.sidebar.radio[0].set_value("Rep workspace").run()

    assert any("unverified" in msg.value.lower() for msg in at.warning)

"""Route tests for the rep-facing endpoints (plan Tasks 4 and 17).

Covers the rep queue (SPEC §4.3/§4.7 — the New tickets awaiting work), the
single-ticket detail view a rep opens, and the draft-review actions
(edit/approve/reject/send, plan Task 17). The email MCP client is mocked; the
rep-action tests drive the real workflow to its human-review pause and resume it
through the routes, so the resume→finalize→persist path is exercised end to end.
"""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from tests.conftest import HAPPY_DRAFT_BODY, FakeEmailClient


def _queue_row(seq: int) -> dict[str, str | int | None]:
    """Build a New queue row with an ordered `created_at`/`id` for keyset tests."""
    return {
        "id": seq,
        "reference_code": f"TKT-{seq:04d}",
        "status": "New",
        "urgency": None,
        "category": None,
        "created_at": f"2026-07-07T00:00:{seq:02d}+00:00",
    }


def test_queue_returns_new_tickets_in_a_paged_envelope(
    client: TestClient, email_client: FakeEmailClient, auth_headers: dict[str, str]
) -> None:
    """The queue endpoint returns New tickets under `items`, with a `next_cursor` key."""
    email_client.queue_rows = [_queue_row(1)]

    response = client.get("/rep/queue", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert [row["reference_code"] for row in body["items"]] == ["TKT-0001"]
    assert body["items"][0]["status"] == "New"
    # A single row is a short (final) page, so there is no next cursor.
    assert body["next_cursor"] is None


def test_queue_caps_results_at_the_hard_max(
    client: TestClient,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
    test_settings: Any,
) -> None:
    """Even asking for far more than the max returns at most the configured cap."""
    max_limit = test_settings.queue_page_max
    email_client.queue_rows = [_queue_row(n) for n in range(1, max_limit + 50)]

    response = client.get("/rep/queue", params={"limit": 100_000}, headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == max_limit
    # A full page means more may remain, so a cursor is handed back.
    assert body["next_cursor"] is not None
    # The route forwarded the capped limit (never the client's oversized ask).
    assert email_client.calls[-1] == ("fetch_new_tickets", max_limit, None)


def test_queue_uses_configured_default_page_size_when_none_requested(
    client: TestClient,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
    test_settings: Any,
) -> None:
    """With no `limit`, the route forwards the configured `QUEUE_PAGE_DEFAULT`."""
    email_client.queue_rows = [_queue_row(1)]

    response = client.get("/rep/queue", headers=auth_headers)

    assert response.status_code == 200
    # The route asked email_mcp for exactly the configured default page size.
    assert email_client.calls[-1] == (
        "fetch_new_tickets",
        test_settings.queue_page_default,
        None,
    )


def test_queue_next_page_has_no_overlap_with_first(
    client: TestClient, email_client: FakeEmailClient, auth_headers: dict[str, str]
) -> None:
    """Following `next_cursor` yields the following tickets with no dupes or gaps."""
    email_client.queue_rows = [_queue_row(n) for n in range(1, 6)]

    first = client.get("/rep/queue", params={"limit": 2}, headers=auth_headers).json()
    second = client.get(
        "/rep/queue",
        params={"limit": 2, "after": first["next_cursor"]},
        headers=auth_headers,
    ).json()

    first_codes = [row["reference_code"] for row in first["items"]]
    second_codes = [row["reference_code"] for row in second["items"]]
    assert first_codes == ["TKT-0001", "TKT-0002"]
    assert second_codes == ["TKT-0003", "TKT-0004"]
    assert set(first_codes).isdisjoint(second_codes)


def test_queue_malformed_cursor_is_rejected(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """A cursor that isn't `<created_at>,<id>` is a client error, not a 500."""
    response = client.get(
        "/rep/queue", params={"after": "not-a-valid-cursor"}, headers=auth_headers
    )

    assert response.status_code == 400


def test_queue_requires_auth(client: TestClient) -> None:
    """The rep queue is not readable without a bearer token."""
    response = client.get("/rep/queue")

    assert response.status_code == 401


def test_rep_ticket_detail_returns_ticket(
    client: TestClient, email_client: FakeEmailClient, auth_headers: dict[str, str]
) -> None:
    """Opening a known ticket by id returns its full detail for the rep."""
    email_client.tickets_by_id[7] = {
        "id": 7,
        "reference_code": "TKT-0007",
        "status": "New",
        "message": "I can't log in.",
        "attachments": [],
        "draft": None,
    }

    response = client.get("/rep/tickets/7", headers=auth_headers)

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == 7
    assert body["message"] == "I can't log in."


def test_rep_ticket_detail_unknown_returns_not_found(
    client: TestClient, auth_headers: dict[str, str]
) -> None:
    """An unknown ticket id returns a neutral 404, never an error that leaks state."""
    response = client.get("/rep/tickets/999999", headers=auth_headers)

    assert response.status_code == 404


def test_rep_ticket_detail_requires_auth(client: TestClient) -> None:
    """The rep ticket detail is not readable without a bearer token."""
    response = client.get("/rep/tickets/7")

    assert response.status_code == 401


# --- Draft-review payload: GET /rep/tickets/{id}/review (plan Task 19) --------


async def test_rep_review_returns_draft_facts_and_sources(
    build_paused_workflow: Any,
    rep_client: Any,
    auth_headers: dict[str, str],
) -> None:
    """The review endpoint projects the paused graph state into a rep-review payload.

    A case sitting at the human-review pause exposes exactly what the rep workspace
    renders: the original message, the triage classification, the retrieved KB sources,
    and the drafted (here verified, unflagged) reply — read straight from the paused
    checkpoint, not from the email_mcp ticket record.
    """
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        review = await ac.get("/rep/tickets/7/review", headers=auth_headers)

    assert review.status_code == 200
    body = review.json()
    assert body["ticket_id"] == 7
    assert body["status"] == "Drafted"
    assert body["message"] == "How do I reset my online banking password?"
    assert body["triage"]["category"] == "account_access"
    assert [source["id"] for source in body["sources"]] == ["KB-1"]
    assert body["draft"]["body"] == HAPPY_DRAFT_BODY
    assert body["draft"]["verified"] is True
    assert body["trace_leak"] is False
    assert body["flags"] == []


async def test_rep_review_on_a_ticket_not_awaiting_review_is_conflict(
    build_paused_workflow: Any,
    rep_client: Any,
    auth_headers: dict[str, str],
) -> None:
    """A ticket with no run paused at the gate is refused with 409, not a blank payload.

    The review payload only exists while a case is interrupted before `human_review`;
    an id with no such checkpoint gets the same 409 the rep-action routes give, rather
    than a 200 carrying an empty draft.
    """
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        review = await ac.get("/rep/tickets/999/review", headers=auth_headers)

    assert review.status_code == 409


async def test_rep_review_requires_auth(build_paused_workflow: Any, rep_client: Any) -> None:
    """The review payload is not readable without a bearer token."""
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        review = await ac.get("/rep/tickets/7/review")

    assert review.status_code == 401


# --- Draft-review actions: edit / approve / reject / send (plan Task 17) ------


async def test_approve_then_send_resolves_and_records_reply(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
) -> None:
    """Approve then send resolves the case and records the drafted reply.

    The two-step SPEC §4.7 disposition: approve stages the decision without sending,
    then an explicit send resumes the paused graph through `finalize`, which resolves
    the case. The route persists the drafted body via email_mcp's `record_sent_reply`
    with the rep's audit marker, and reports Resolved with the sent reply.
    """
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        approved = await ac.post("/rep/tickets/7/approve", headers=auth_headers)
        # Approve stages only — the case is not resolved and nothing is sent yet.
        assert approved.status_code == 200
        assert approved.json()["status"] != "Resolved"
        assert not any(call[0] == "record_sent_reply" for call in email_client.calls)

        sent = await ac.post("/rep/tickets/7/send", json={"rep_id": "rep-1"}, headers=auth_headers)

    assert sent.status_code == 200
    body = sent.json()
    assert body["status"] == "Resolved"
    assert body["reply"] == HAPPY_DRAFT_BODY
    # The route forwarded the drafted body to email_mcp under the rep's marker (the
    # feedback-capture call now follows it, so this is no longer the final call).
    assert ("record_sent_reply", 7, HAPPY_DRAFT_BODY, "rep-1") in email_client.calls


async def test_sent_reply_is_visible_via_customer_lookup(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
) -> None:
    """After approve→send, the resolved reply is returned by reference-code lookup.

    The end-to-end acceptance path (plan Task 17): once a rep sends, a customer who
    looks the case up by its reference code sees status Resolved and the final reply
    — the same on-disk record the rep action wrote, reached over the customer route.
    """
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        await ac.post("/rep/tickets/7/approve", headers=auth_headers)
        await ac.post("/rep/tickets/7/send", json={"rep_id": "rep-1"}, headers=auth_headers)
        lookup = await ac.get("/tickets/TKT-0007", headers=auth_headers)

    assert lookup.status_code == 200
    body = lookup.json()
    assert body["status"] == "Resolved"
    assert body["reply"] == HAPPY_DRAFT_BODY


async def test_edit_then_send_uses_the_edited_reply(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
) -> None:
    """Editing before send makes the rep's text — not the AI draft — the final reply.

    Edit stages the rep's text into the paused case (held there until send, never
    written on its own); the subsequent send resumes through `finalize`, which records
    the edited reply as the customer-facing text.
    """
    edited = "Please reset your password from the login screen and call us if it fails."
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        edit = await ac.post("/rep/tickets/7/edit", json={"reply": edited}, headers=auth_headers)
        sent = await ac.post("/rep/tickets/7/send", json={"rep_id": "rep-9"}, headers=auth_headers)

    assert edit.status_code == 200
    assert sent.status_code == 200
    assert sent.json()["reply"] == edited
    # The edited reply — not the AI draft — was the text forwarded to email_mcp.
    assert ("record_sent_reply", 7, edited, "rep-9") in email_client.calls


async def test_reject_routes_case_back_to_needs_research(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
) -> None:
    """Rejecting sends the case back to NeedsResearch with no reply to the customer.

    Reject resumes the paused graph with a rejection; `finalize` routes the case to
    NeedsResearch and produces no `final_reply`. The route persists the status via
    `update_status` (never `record_sent_reply`), so nothing reaches the customer.
    """
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        rejected = await ac.post(
            "/rep/tickets/7/reject",
            json={"rep_id": "rep-3", "reason": "needs a second source"},
            headers=auth_headers,
        )

    assert rejected.status_code == 200
    assert rejected.json()["status"] == "NeedsResearch"
    # The case was routed back via update_status (the feedback-capture call now follows).
    assert ("update_status", 7, "NeedsResearch", "rep-3") in email_client.calls
    assert not any(call[0] == "record_sent_reply" for call in email_client.calls)


async def test_send_without_a_prior_approval_is_refused(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
) -> None:
    """Sending a case that no rep has approved or edited is refused, not resolved.

    The SPEC §10 safety invariant at the route boundary: with no rep decision staged,
    `finalize` would fail closed, so the route rejects the send with 409 and never
    calls `record_sent_reply` — there is no path to Resolved without an explicit
    approve/edit first.
    """
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        sent = await ac.post("/rep/tickets/7/send", json={"rep_id": "rep-1"}, headers=auth_headers)

    assert sent.status_code == 409
    assert not any(call[0] == "record_sent_reply" for call in email_client.calls)


async def test_send_rejects_a_blank_rep_marker(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
) -> None:
    """A send carrying no rep audit marker is a validation error, not a silent send.

    `record_sent_reply` requires a non-empty rep marker (SPEC §4.7); the route rejects
    a blank `rep_id` with 422 before touching the graph or email_mcp.
    """
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        await ac.post("/rep/tickets/7/approve", headers=auth_headers)
        sent = await ac.post("/rep/tickets/7/send", json={"rep_id": "   "}, headers=auth_headers)

    assert sent.status_code == 422
    assert not any(call[0] == "record_sent_reply" for call in email_client.calls)


async def test_send_requires_auth(build_paused_workflow: Any, rep_client: Any) -> None:
    """A send with no bearer token is rejected with 401 before any action runs."""
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        sent = await ac.post("/rep/tickets/7/send", json={"rep_id": "rep-1"})

    assert sent.status_code == 401


async def test_reject_requires_auth(build_paused_workflow: Any, rep_client: Any) -> None:
    """A reject with no bearer token is rejected with 401 before any action runs."""
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        rejected = await ac.post("/rep/tickets/7/reject", json={"rep_id": "rep-1"})

    assert rejected.status_code == 401


# --- Rep-action audit emission (plan Task 24 / todo Task 26) ------------------
#
# SPEC §7.1 requires the rep's edit and approve to land on the ticket's immutable
# audit trail. These routes only *stage* a decision (no send), so nothing else would
# record them — the route itself writes the audit row via email_mcp.


async def test_edit_records_a_draft_edited_audit_entry(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
) -> None:
    """Staging an edit writes a `draft_edited` audit row attributed to the rep.

    An edit stages the rep's text without sending, so unless the route records it the
    trail would never show the rep touched the draft. The row is attributed to the
    generic `rep` actor (per-rep identity is not modelled yet).
    """
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        # Stage an edited reply into the paused case.
        edit = await ac.post(
            "/rep/tickets/7/edit",
            json={"reply": "Reset it from the login screen."},
            headers=auth_headers,
        )

    # The edit is accepted.
    assert edit.status_code == 200
    # A `draft_edited` audit row was recorded for ticket 7, attributed to the rep.
    assert ("record_audit", 7, "draft_edited", "rep", None) in email_client.calls


async def test_approve_records_a_draft_approved_audit_entry(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
) -> None:
    """Staging an approve writes a `draft_approved` audit row attributed to the rep.

    Approve alone never sends, so the trail would not otherwise record the rep's
    approval; the route writes it, attributed to the generic `rep` actor.
    """
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        # Stage an approve-as-is decision on the paused case.
        approved = await ac.post("/rep/tickets/7/approve", headers=auth_headers)

    # The approve is accepted.
    assert approved.status_code == 200
    # A `draft_approved` audit row was recorded for ticket 7, attributed to the rep.
    assert ("record_audit", 7, "draft_approved", "rep", None) in email_client.calls


# --- Feedback capture on disposition (plan Task 25 / todo Task 27) -------------
#
# SPEC §4.9 records every rep decision — approved-as-is / edited (with the AI-vs-final
# diff) / rejected, plus an optional rating and reason — into the feedback table. The
# send/reject routes own the write: they resume the graph through `finalize`, so the
# finished state carries the AI draft and the final reply the record is built from.


async def test_approve_then_send_records_approved_feedback(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
) -> None:
    """An approved-as-is send captures a feedback row with a zero edit distance.

    The rep approved the AI draft unchanged, so the final reply matches the draft and
    the recorded edit distance is zero. The optional rating the rep passed on the send
    is carried onto the row (SPEC §4.9).
    """
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        await ac.post("/rep/tickets/7/approve", headers=auth_headers)
        sent = await ac.post(
            "/rep/tickets/7/send",
            json={"rep_id": "rep-1", "rating": 5},
            headers=auth_headers,
        )

    assert sent.status_code == 200
    assert len(email_client.feedback) == 1
    row = email_client.feedback[0]
    assert row["ticket_id"] == 7
    assert row["decision"] == "approved_as_is"
    assert row["ai_draft"] == HAPPY_DRAFT_BODY
    assert row["final_reply"] == HAPPY_DRAFT_BODY
    assert row["edit_distance"] == 0
    assert row["rating"] == 5


async def test_edit_then_send_records_edited_feedback_with_distance(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
) -> None:
    """An edited send captures both texts, a positive edit distance, and the reason.

    The rep rewrote the draft before sending, so the feedback row keeps the AI draft
    and the edited final reply with a non-zero character distance between them, plus
    the reason the rep gave (SPEC §4.9).
    """
    edited = "Please reset your password from the login screen and call us if it fails."
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        await ac.post("/rep/tickets/7/edit", json={"reply": edited}, headers=auth_headers)
        sent = await ac.post(
            "/rep/tickets/7/send",
            json={"rep_id": "rep-9", "reason": "tightened the tone"},
            headers=auth_headers,
        )

    assert sent.status_code == 200
    assert len(email_client.feedback) == 1
    row = email_client.feedback[0]
    assert row["decision"] == "edited"
    assert row["ai_draft"] == HAPPY_DRAFT_BODY
    assert row["final_reply"] == edited
    assert row["edit_distance"] > 0
    assert row["reason"] == "tightened the tone"


async def test_reject_records_rejected_feedback(
    build_paused_workflow: Any,
    rep_client: Any,
    email_client: FakeEmailClient,
    auth_headers: dict[str, str],
) -> None:
    """Rejecting a draft captures a rejected feedback row with no final reply.

    A rejection produces no customer reply, so the row keeps the discarded AI draft
    with a null final reply and no edit distance, carrying the rep's rating/reason —
    the negative example the preference corpus consumes (SPEC §4.9 / §4.9a).
    """
    email_client.register_ticket(7, "TKT-0007", status="Drafted")
    graph = await build_paused_workflow(ticket_id=7)

    async with rep_client(graph) as ac:
        rejected = await ac.post(
            "/rep/tickets/7/reject",
            json={"rep_id": "rep-3", "reason": "needs a second source", "rating": 2},
            headers=auth_headers,
        )

    assert rejected.status_code == 200
    assert len(email_client.feedback) == 1
    row = email_client.feedback[0]
    assert row["decision"] == "rejected"
    assert row["ai_draft"] == HAPPY_DRAFT_BODY
    assert row["final_reply"] is None
    assert row["edit_distance"] is None
    assert row["rating"] == 2
    assert row["reason"] == "needs a second source"

"""Rep workspace view: the work queue plus a draft-review panel (plan Task 19).

The rep opens a queued ticket and dispositions its AI draft here (SPEC §4.7). The
queue is a keyset-paged table (the cursor lives in `st.session_state`, so paging
never duplicates or skips a ticket); picking a row loads that case's review payload
from the api — the original message, extracted facts, retrieved sources, the draft,
and the warning banners a rep must weigh — and offers edit/approve/reject/send.

The two-step human gate is honoured in the UI: *approve* (or *save edit*) only
stages a decision; *send* is the explicit act that delivers the reply and resolves
the case. Send and reject carry the rep's audit marker (`rep_id`). Thin Streamlit
surface (SPEC §9) — the row projection lives in the tested `formatting` helper and
every api call is one `api_client` method.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from formatting import queue_rows_to_display

_CURSOR_KEY = "rep_queue_cursor"


def render(client: Any) -> None:
    """Render one keyset page of the queue and, for a picked ticket, its draft review."""
    st.header("Rep workspace — work queue")

    if st.button("Refresh (back to newest)", key="rep_refresh"):
        st.session_state[_CURSOR_KEY] = None

    cursor = st.session_state.get(_CURSOR_KEY)
    page = client.fetch_queue(after=cursor)
    rows = page["items"]

    if not rows:
        st.info("No tickets in the queue right now.")
        return

    st.table(queue_rows_to_display(rows))

    next_cursor = page["next_cursor"]
    if next_cursor and st.button("Next page", key="rep_next"):
        st.session_state[_CURSOR_KEY] = next_cursor
        st.rerun()

    _render_review_section(client, rows)


def _render_review_section(client: Any, rows: list[dict[str, Any]]) -> None:
    """Let the rep pick a ticket from the current page and review its AI draft."""
    st.divider()
    st.subheader("Review a draft")
    ids_by_code = {row["reference_code"]: row["id"] for row in rows}
    choice = st.selectbox("Ticket to review", list(ids_by_code), key="rep_review_choice")
    if choice:
        _render_draft_review(client, ids_by_code[choice])


def _render_draft_review(client: Any, ticket_id: int) -> None:
    """Render a paused ticket's message, facts, sources, draft, and rep controls."""
    review = client.fetch_review(ticket_id)
    _render_warning_banners(review)

    st.markdown("**Original message**")
    st.write(review["message"])

    facts = review.get("extracted_facts")
    if facts:
        st.markdown("**Extracted facts**")
        st.write(facts)

    triage = review.get("triage")
    if triage:
        st.markdown(
            f"**Triage** — {triage['category']} · {triage['urgency']} · {triage['sentiment']}"
        )

    sources = review.get("sources") or []
    if sources:
        st.markdown("**Retrieved sources**")
        for source in sources:
            st.markdown(f"- **{source['id']}** — {source['title']}")

    draft = review.get("draft")
    if draft is None:
        st.info("No draft was produced for this case — it needs a human reply.")
        return

    st.markdown("**Drafted reply**")
    edited = st.text_area(
        "Reply (edit before sending if needed)", value=draft["body"], key="rep_edit_reply"
    )
    rep_id = st.text_input("Your rep ID", key="rep_rep_id")
    _render_controls(client, ticket_id, edited, rep_id)


def _render_warning_banners(review: dict[str, Any]) -> None:
    """Surface the review warnings a rep must weigh before sending."""
    draft = review.get("draft")
    if draft is not None and not draft.get("verified", True):
        st.warning(
            "AI-suggested, unverified — this draft drifted from its sources; "
            "do not present it as sourced fact."
        )
    if review.get("trace_leak"):
        st.warning(
            "The drafted reply leaked a reasoning trace; review it carefully before sending."
        )
    for flag in review.get("flags") or []:
        st.warning(flag)


def _render_controls(client: Any, ticket_id: int, edited: str, rep_id: str) -> None:
    """Wire the edit/approve/reject/send actions to the api client.

    Approve and save-edit only *stage* a decision; send is the explicit delivery, and
    both send and reject require the rep's audit marker (`rep_id`) — refused in the UI
    if it is blank, before any api call, so a case is never sent unattributed.
    """
    if st.button("Approve as-is", key="rep_approve"):
        client.approve_draft(ticket_id)
        st.success("Draft approved — press Send to deliver it to the customer.")

    if st.button("Save edit", key="rep_save_edit"):
        if edited.strip():
            client.edit_draft(ticket_id, edited)
            st.success("Edited reply staged — press Send to deliver it.")
        else:
            st.error("The reply is empty — there is nothing to stage.")

    if st.button("Send to customer", key="rep_send"):
        if rep_id.strip():
            result = client.send_draft(ticket_id, rep_id)
            st.success(f"Sent — case is now {result['status']}.")
        else:
            st.error("Enter your rep ID before sending.")

    reason = st.text_input("Rejection reason (optional)", key="rep_reject_reason")
    if st.button("Reject (needs research)", key="rep_reject"):
        if rep_id.strip():
            result = client.reject_draft(ticket_id, rep_id, reason=reason or None)
            st.info(f"Rejected — case routed to {result['status']}.")
        else:
            st.error("Enter your rep ID before rejecting.")

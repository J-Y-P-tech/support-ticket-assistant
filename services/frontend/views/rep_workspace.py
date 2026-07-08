"""Rep workspace view: the New-ticket work queue (plan Task 6, list-only).

Draft review / edit / approve / send land in Task 18; this task shows the queue
as a paged table. Paging is keyset: the current page's cursor lives in
`st.session_state`, and "Next page" advances it to the page's `next_cursor`, so a
rep never sees a duplicated or skipped ticket as new ones arrive. Thin Streamlit
surface (SPEC §9) — the row projection lives in the tested `formatting` helper.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from formatting import queue_rows_to_display

_CURSOR_KEY = "rep_queue_cursor"


def render(client: Any) -> None:
    """Render one keyset page of the queue with a Next-page control."""
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

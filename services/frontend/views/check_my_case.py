"""Customer follow-up view: look a case up by reference code (plan Task 6).

Thin Streamlit surface (SPEC §9). An unknown code comes back from `api_client` as
`None` (the api's neutral not-found), shown here as a uniform message that never
implies whether the code could exist.
"""

from __future__ import annotations

from typing import Any

import streamlit as st


def render(client: Any) -> None:
    """Render the lookup form and show a case's status (and reply, if resolved)."""
    st.header("Check my case")
    code = st.text_input("Reference code (e.g. TKT-0001)", key="lookup_code")

    if st.button("Look up", key="lookup_submit"):
        ticket = client.lookup_ticket(code)
        if ticket is None:
            st.warning("We couldn't find a case with that reference code.")
            return
        st.info(f"Status: {ticket['status']}")
        reply = ticket.get("reply")
        if reply:
            st.subheader("Our reply")
            st.write(reply)

"""Customer intake view: submit a message + optional attachments (plan Task 6).

Thin Streamlit surface (SPEC §9): input collection and rendering only. The submit
call and the filename extraction live in `api_client` / `formatting`, which are
unit-tested; this view just wires them to widgets.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from formatting import attachment_names


def render(client: Any) -> None:
    """Render the intake form and submit a new ticket when the button is pressed."""
    st.header("Submit a support request")
    message = st.text_area("How can we help?", key="customer_message")
    files = st.file_uploader(
        "Attachments (optional)", accept_multiple_files=True, key="customer_files"
    )

    if st.button("Submit", key="customer_submit"):
        if not message.strip():
            st.error("Please enter a message before submitting.")
            return
        ticket = client.submit_ticket(message, attachment_names(files or []))
        code = ticket["reference_code"]
        st.success(
            f"Thanks — your request was received. Your reference code is {code}. "
            "Keep it to check your case status."
        )

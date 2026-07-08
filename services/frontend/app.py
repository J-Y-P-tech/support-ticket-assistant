"""Streamlit entry point for the Support-Ticket Assistant frontend (plan Task 6).

Three views (SPEC §8): *Customer* (submit + attach), *Check my case* (reference-
code lookup), *Rep workspace* (the New-ticket queue). The app owns a single
`ApiClient` and dispatches to the selected view.

Test seam: the client is read from `st.session_state`, so `AppTest` injects an
in-memory fake before running and no api or network is touched under test. In
production the client is built lazily from config on first use, so importing this
module never requires the environment to be present.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from api_client import ApiClient
from views import check_my_case, customer, rep_workspace

_CLIENT_KEY = "client"

_VIEWS = {
    "Customer": customer.render,
    "Check my case": check_my_case.render,
    "Rep workspace": rep_workspace.render,
}


def _build_client() -> ApiClient:
    """Build the real `ApiClient` from config (deferred so import needs no env)."""
    from config import get_settings

    settings = get_settings()
    return ApiClient(
        base_url=settings.api_base_url,
        token=settings.api_auth_token.get_secret_value(),
    )


def get_client() -> Any:
    """Return the api client, preferring a test-injected one from session state."""
    client = st.session_state.get(_CLIENT_KEY)
    if client is None:
        client = _build_client()
        st.session_state[_CLIENT_KEY] = client
    return client


def main() -> None:
    """Render the sidebar view picker and dispatch to the selected view."""
    st.title("Support-Ticket Assistant")
    client = get_client()
    choice = st.sidebar.radio("View", list(_VIEWS))
    _VIEWS[choice](client)


main()

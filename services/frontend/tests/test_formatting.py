"""Tests for the frontend's pure view/format helpers (plan Task 6).

SPEC §9 keeps logic out of the thin Streamlit views: the reusable bits — reading
uploaded filenames, projecting queue rows to display columns — live in
`formatting.py` so they can be unit-tested without rendering a UI.
"""

from __future__ import annotations

from dataclasses import dataclass

from formatting import attachment_names, queue_rows_to_display


@dataclass
class _FakeUpload:
    """Minimal stand-in for a Streamlit `UploadedFile` (only `.name` is used)."""

    name: str


def test_attachment_names_reads_filenames() -> None:
    """Uploaded files are reduced to their filenames, preserving order."""
    files = [_FakeUpload("statement.pdf"), _FakeUpload("id-front.jpg")]

    assert attachment_names(files) == ["statement.pdf", "id-front.jpg"]


def test_attachment_names_empty_when_nothing_uploaded() -> None:
    """No uploads yields an empty list (a ticket may have no attachments)."""
    assert attachment_names([]) == []


def test_queue_rows_to_display_maps_columns() -> None:
    """Rows are projected to the labelled columns the rep table shows."""
    rows = [
        {
            "reference_code": "TKT-0001",
            "status": "New",
            "urgency": "High",
            "category": "Billing",
        }
    ]

    display = queue_rows_to_display(rows)

    assert display == [
        {"Reference": "TKT-0001", "Status": "New", "Urgency": "High", "Category": "Billing"}
    ]


def test_queue_rows_to_display_shows_dash_for_untriaged_fields() -> None:
    """A New ticket has no urgency/category yet; those render as an em dash."""
    rows = [{"reference_code": "TKT-0002", "status": "New", "urgency": None, "category": None}]

    display = queue_rows_to_display(rows)

    assert display[0]["Urgency"] == "—"
    assert display[0]["Category"] == "—"

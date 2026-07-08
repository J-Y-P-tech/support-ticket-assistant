"""Pure view/format helpers for the frontend (SPEC §9, plan Task 6).

SPEC §9 keeps real logic out of the thin Streamlit views. The small reusable
transforms live here — reading uploaded filenames, projecting queue rows to the
labelled columns the rep table shows — so they are unit-tested without rendering
a UI. Nothing here imports Streamlit; inputs are described structurally (only the
`.name` attribute of an upload is used) so the module stays dependency-free.
"""

from __future__ import annotations

from typing import Any, Protocol

# Placeholder shown for a queue column that a New (untriaged) ticket has not
# populated yet — urgency and category arrive with triage.
_UNTRIAGED = "—"


class _NamedFile(Protocol):
    """Anything with a filename — e.g. a Streamlit `UploadedFile`."""

    name: str


def attachment_names(files: list[_NamedFile]) -> list[str]:
    """Return the filenames of uploaded attachments, in order (empty if none)."""
    return [file.name for file in files]


def queue_rows_to_display(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Project queue rows to the labelled columns the rep table renders.

    Untriaged urgency/category (`None`) render as an em dash so the table reads
    cleanly for a New ticket that has not been triaged yet.
    """
    return [
        {
            "Reference": row["reference_code"],
            "Status": row["status"],
            "Urgency": row.get("urgency") or _UNTRIAGED,
            "Category": row.get("category") or _UNTRIAGED,
        }
        for row in rows
    ]

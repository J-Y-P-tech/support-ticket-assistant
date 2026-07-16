"""Standalone corpus exporter — the entrypoint behind `make export-training-data`.

Reads the whole append-only `training_corpus` table (SPEC §4.9a) using email_mcp's own
credentials (SPEC §6 least-privilege) and prints it as JSONL — one JSON object per line
— to stdout, so the user redirects it to a file. Each line is tagged with its record
`type` (`sft` / `preference`) so a downstream fine-tune can split the two. The records
are already de-identified at capture time; this only renders them.

`corpus_rows_to_jsonl` is factored out as a pure function so the line format is unit-
testable without a database.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

import db


def corpus_rows_to_jsonl(rows: Sequence[dict[str, Any]]) -> str:
    """Render corpus rows as JSONL: one `{"type": ..., **payload}` object per line.

    Merges each row's `record_type` into its `payload` under a `type` key so a reader
    can split SFT from preference records without a separate column. An empty corpus
    renders as the empty string (no blank line, no trailing newline).
    """
    return "\n".join(json.dumps({"type": row["record_type"], **row["payload"]}) for row in rows)


def main() -> None:
    """Open a connection from the environment, read the corpus, and print it as JSONL."""
    with db.connect_from_env() as conn:
        rows = db.export_corpus(conn)
    output = corpus_rows_to_jsonl(rows)
    if output:
        print(output)


if __name__ == "__main__":
    main()

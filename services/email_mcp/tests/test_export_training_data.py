"""Smoke test for the JSONL rendering behind `make export-training-data` (todo Task 28).

SPEC §4.9a exports the de-identified corpus to JSONL. `corpus_rows_to_jsonl` is the
pure renderer the export entrypoint uses: it turns the append-ordered corpus rows
(`record_type` + `payload`) into one JSON object per line, tagging each with its `type`
so a downstream fine-tune can split SFT from preference records. Tested without a
database so the line format is pinned independently of the store.
"""

from __future__ import annotations

import json

from export_training_data import corpus_rows_to_jsonl


def test_each_record_renders_as_one_tagged_json_line() -> None:
    """Every corpus row becomes one JSONL line carrying its type and payload fields."""
    rows = [
        {"record_type": "sft", "payload": {"input": {"message": "hi"}, "output": "hello"}},
        {"record_type": "preference", "payload": {"chosen": "a", "rejected": "b"}},
    ]

    lines = corpus_rows_to_jsonl(rows).splitlines()

    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first == {"type": "sft", "input": {"message": "hi"}, "output": "hello"}
    second = json.loads(lines[1])
    assert second == {"type": "preference", "chosen": "a", "rejected": "b"}


def test_empty_corpus_renders_as_empty_string() -> None:
    """An empty corpus renders as an empty string (no blank line, no trailing newline)."""
    assert corpus_rows_to_jsonl([]) == ""

"""Unit tests for `ExtractionResult` (`app.schemas`).

Pins the structured-extraction contract (SPEC §4.2): the seven fields
`doc_type / amounts / dates / names / references / raw_text / low_confidence`.
`raw_text` (the verbatim transcription) is always required so it can be surfaced
to the rep and never silently dropped; the list fields default to empty.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas.extraction import ExtractionResult


def test_full_extraction_round_trips_json() -> None:
    """A fully-populated extraction survives a dump→load JSON round-trip unchanged."""
    result = ExtractionResult(
        doc_type="bank statement",
        amounts=["$1,240.00", "$35.00"],
        dates=["2026-06-30"],
        names=["John Doe"],
        references=["ACCT-****1234"],
        raw_text="Statement period 2026-06-01 to 2026-06-30 ...",
        low_confidence=False,
    )

    restored = ExtractionResult.model_validate_json(result.model_dump_json())

    assert restored == result


def test_extraction_defaults_when_only_raw_text_given() -> None:
    """With only the required raw_text, list fields are empty and flags default off."""
    result = ExtractionResult(raw_text="illegible scan")

    assert result.doc_type is None
    assert result.amounts == []
    assert result.dates == []
    assert result.names == []
    assert result.references == []
    assert result.low_confidence is False


def test_extraction_requires_raw_text() -> None:
    """`raw_text` is mandatory, so a transcription is never lost."""
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate({"doc_type": "receipt"})


def test_extraction_rejects_wrong_typed_list() -> None:
    """A non-list value for a list field (e.g. amounts) is rejected."""
    with pytest.raises(ValidationError):
        ExtractionResult.model_validate({"raw_text": "x", "amounts": "5 dollars"})

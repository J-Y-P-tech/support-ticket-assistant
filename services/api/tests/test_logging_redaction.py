"""Unit tests for structured logging + PII redaction (`app.logging_config`).

These tests pin the plan Task 23 / todo Task 24 acceptance criteria (SPEC §6):
"structured logs with PII redaction — no full account numbers, IDs, or raw
attachment text in logs."

The redaction is proved at two levels:

- `redact_pii` — the pure text scrubber. Any run of seven or more digits (the
  shape of an account, card, or ID number, even when written with spaces or
  dashes) is replaced with a fixed marker; short numbers a log legitimately needs
  — a reference code like `TKT-0007`, a ticket id, a page count — survive
  untouched.
- `redact_processor` — the structlog processor that walks a whole log event and
  applies that scrub to every value, *and* fully drops the value of keys that hold
  raw attachment/customer text (which must never be logged at all, PII or not).

`configure_logging` / `get_logger` wire that processor into the logging chain; a
smoke test proves the processor is actually in the configured pipeline (not just
defined and forgotten).

Only `app.logging_config` should be unresolved during the RED phase.
"""

from __future__ import annotations

# The units under test. During RED these imports fail (the module does not exist
# yet), which is the signal telling us what to build.
from app.logging_config import (
    REDACTION_MARKER,
    SENSITIVE_KEYS,
    build_processors,
    configure_logging,
    get_logger,
    redact_pii,
    redact_processor,
)

# A 16-digit card number written the way a customer would, with spaces. This is
# exactly the kind of value that must never appear in a log line.
CARD_NUMBER = "4111 1111 1111 1111"
# A contiguous account/ID number with no separators.
ACCOUNT_NUMBER = "1234567890"


def _apply(event_dict: dict[str, object]) -> dict[str, object]:
    """Run the structlog processor over an event dict and return the result.

    A structlog processor takes `(logger, method_name, event_dict)`; the first two
    are irrelevant to redaction, so we pass placeholders and only care about the
    scrubbed event dict that comes back.
    """
    return redact_processor(None, "info", event_dict)


# --- redact_pii: the pure text scrubber -------------------------------------


def test_redact_pii_masks_a_spaced_card_number() -> None:
    """A 16-digit card number (written with spaces) is replaced by the marker."""
    result = redact_pii(f"card {CARD_NUMBER} on file")
    # None of the original digits survive, and the marker stands in their place.
    assert "4111" not in result
    assert "1111" not in result
    assert REDACTION_MARKER in result


def test_redact_pii_masks_a_contiguous_account_number() -> None:
    """A ten-digit account number with no separators is redacted."""
    result = redact_pii(f"account {ACCOUNT_NUMBER}")
    assert ACCOUNT_NUMBER not in result
    assert REDACTION_MARKER in result


def test_redact_pii_preserves_a_short_reference_code() -> None:
    """A reference code like `TKT-0007` (only four digits) is left intact.

    Reference codes are the audit link customer<->ticket<->reply and must stay
    loggable; the seven-digit threshold keeps them below the redaction bar.
    """
    result = redact_pii("looked up ticket TKT-0007")
    assert result == "looked up ticket TKT-0007"


def test_redact_pii_preserves_small_standalone_numbers() -> None:
    """Ordinary small numbers a log needs (ids, counts) survive untouched."""
    result = redact_pii("ticket 7 fetched page 2 of 3")
    assert result == "ticket 7 fetched page 2 of 3"


# --- redact_processor: the whole-event scrubber -----------------------------


def test_processor_scrubs_pii_in_the_event_message() -> None:
    """An account number embedded in the log message itself is redacted."""
    scrubbed = _apply({"event": f"stored account {ACCOUNT_NUMBER}"})
    assert ACCOUNT_NUMBER not in scrubbed["event"]  # type: ignore[operator]
    assert REDACTION_MARKER in scrubbed["event"]  # type: ignore[operator]


def test_processor_scrubs_pii_in_an_arbitrary_field_value() -> None:
    """A card number carried in a bound field (not the message) is redacted too."""
    scrubbed = _apply({"event": "draft saved", "customer_note": f"pay to {CARD_NUMBER}"})
    assert "1111" not in scrubbed["customer_note"]  # type: ignore[operator]
    assert REDACTION_MARKER in scrubbed["customer_note"]  # type: ignore[operator]


def test_processor_fully_drops_raw_attachment_text() -> None:
    """A key holding raw attachment text has its whole value replaced, PII or not.

    Raw transcribed attachment text must never be logged at all — not merely
    number-scrubbed — so the value is dropped wholesale regardless of content.
    """
    assert "ocr_text" in SENSITIVE_KEYS  # guards the contract this test relies on
    scrubbed = _apply({"event": "ocr done", "ocr_text": "Dear bank, my balance is low"})
    assert scrubbed["ocr_text"] == REDACTION_MARKER


def test_processor_drops_raw_text_inside_a_list_value() -> None:
    """A sensitive key whose value is a list of raw strings is dropped wholesale."""
    scrubbed = _apply({"event": "intake", "attachments": ["scan one", "scan two"]})
    assert scrubbed["attachments"] == REDACTION_MARKER


def test_processor_preserves_non_sensitive_fields() -> None:
    """Operational fields with no PII (node name, reference code) pass through."""
    scrubbed = _apply({"event": "node done", "node": "triage", "reference_code": "TKT-0007"})
    assert scrubbed["node"] == "triage"
    assert scrubbed["reference_code"] == "TKT-0007"


def test_processor_scrubs_pii_inside_a_list_field() -> None:
    """PII carried in a list field (not a sensitive key) is scrubbed element-wise."""
    scrubbed = _apply({"event": "trace", "notes": [f"acct {ACCOUNT_NUMBER}", "ok"]})
    first = scrubbed["notes"][0]  # type: ignore[index]
    assert ACCOUNT_NUMBER not in first
    assert REDACTION_MARKER in first


# --- configuration wiring ----------------------------------------------------


def test_redaction_processor_is_in_the_configured_pipeline() -> None:
    """The redaction processor is actually part of the logging chain, not orphaned.

    A redactor that is defined but never wired in would let PII straight through;
    this asserts it sits in the processor list `configure_logging` installs.
    """
    assert redact_processor in build_processors()


def test_get_logger_returns_a_usable_bound_logger() -> None:
    """After configuration, `get_logger` yields a logger with the standard methods."""
    configure_logging()
    logger = get_logger("test")
    assert hasattr(logger, "info")
    assert hasattr(logger, "warning")

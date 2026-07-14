"""Structured logging with a PII-redaction processor (SPEC §6, plan Task 23 / todo Task 24).

Every log line the api emits passes through `structlog` and, before it is rendered,
through `redact_processor` — so a full account number, card number, or ID can never
reach a log file, and raw attachment/customer text is dropped entirely. This is the
"no PII in logs" guarantee from SPEC §6 ("no full account numbers, IDs, or raw
attachment text in logs").

Two layers do the work:

- `redact_pii` scrubs a single string: any run of seven or more digits (an account,
  card, or ID number, even written with spaces or dashes) becomes `[REDACTED]`.
  Shorter numbers a log legitimately needs — a `TKT-0007` reference code, a ticket
  id, a page count — fall below the threshold and survive.
- `redact_processor` applies that scrub across a whole log event (recursing into
  nested dicts and lists) and, for keys that hold raw attachment/customer text
  (`SENSITIVE_KEYS`), replaces the value wholesale rather than trusting the scrubber.

`configure_logging` installs the processor chain once at startup; `get_logger`
hands callers a bound logger. The same `redact_pii` scrubber is the single source of
truth reused by the Langfuse trace redaction (Task 29) so logs and traces can't drift.

What each public function is for, and where it is used:

- `configure_logging()` — called once when the app starts.
- `get_logger()` — called by any file that wants to log.
- `redact_pii()` — reused later by Langfuse (Task 29).
- `redact_processor()` — structlog calls it automatically on every log line.
- `_redact_value` + `build_processors` — only helpers, used nowhere else.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import structlog

__all__ = [
    "REDACTION_MARKER",
    "SENSITIVE_KEYS",
    "build_processors",
    "configure_logging",
    "get_logger",
    "redact_pii",
    "redact_processor",
]

# What a redacted value is replaced with — a fixed, obvious marker so a reader can
# see that something was scrubbed (rather than silently deleted).
REDACTION_MARKER = "[REDACTED]"

# The smallest digit-count we treat as a sensitive number. Account, card, and ID
# numbers are seven digits or more; a four-digit reference code (`TKT-0007`), a
# ticket id, or a page count stays below this bar and is kept for debuggability.
_MIN_SENSITIVE_DIGITS = 7

# A candidate number: a digit, then any mix of digits, spaces, and dashes, ending
# on a digit — so `1234567890` and `4111 1111 1111 1111` both match as one run. The
# replacement callback then counts the *digits* in the run and only redacts when
# there are enough, so a short `TKT-0007` (matched as `0007`) is left alone.
_NUMBER_RUN_RE = re.compile(r"\d[\d \-]*\d")

# Keys whose value is raw attachment or customer text that must never be logged at
# all — not merely number-scrubbed. The whole value is dropped regardless of its
# content, so a transcribed statement or the raw upload bytes can't leak even if it
# happens to contain no digit run.
SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "attachment",
        "attachments",
        "attachment_text",
        "ocr_text",
        "raw_text",
        "transcription",
    }
)


def redact_pii(text: str) -> str:
    """Return `text` with any account/card/ID-sized number run replaced by the marker.

    A "sensitive" run is seven or more digits, optionally spaced or dashed. Shorter
    numbers (reference codes, ids, counts) are left untouched, so operational logs
    stay useful. Text with no such run comes back unchanged, so the scrubber is safe
    to apply to every value. The regex matches any number-ish run; the callback then
    counts the actual digits and redacts only past the threshold (so `TKT-0007`,
    matched as `0007`, stays while a 16-digit card goes).
    """

    def _hide(match: re.Match[str]) -> str:
        """Redact this matched run only if it carries enough digits to be sensitive."""
        digit_count = sum(character.isdigit() for character in match.group())
        # "card 4111 1111 1111 1111" → 16 digits → "card [REDACTED]"
        return REDACTION_MARKER if digit_count >= _MIN_SENSITIVE_DIGITS else match.group()

    return _NUMBER_RUN_RE.sub(_hide, text)


def _redact_value(value: Any) -> Any:
    """Scrub PII from a string, or a list of strings; leave other types as-is.

    Strings are number-scrubbed; a list value (e.g. `attachments`) is scrubbed
    element-wise; everything else (ints, bools, None) is left untouched. Log events
    carry strings and lists of them, not nested dicts, so this deliberately does not
    recurse into dicts.
    """
    if isinstance(value, str):
        return redact_pii(value)
    if isinstance(value, (list, tuple)):
        return [_redact_value(item) for item in value]
    return value


def redact_processor(logger: Any, method_name: str, event_dict: dict[str, Any]) -> dict[str, Any]:
    """structlog processor: redact PII across a whole log event before it is rendered.

    For each field, a key in `SENSITIVE_KEYS` has its value dropped wholesale (raw
    attachment/customer text is never logged); every other value is number-scrubbed
    (recursing into nested dicts/lists). `logger`/`method_name` are part of the
    structlog processor contract and are unused here.
    """
    for key, value in event_dict.items():
        if key in SENSITIVE_KEYS:
            event_dict[key] = REDACTION_MARKER
        else:
            event_dict[key] = _redact_value(value)
    return event_dict


def build_processors() -> list[Any]:
    """Return the structlog processor chain, with redaction ahead of rendering.

    Order matters: `redact_processor` runs *before* the JSON renderer so no raw PII
    is ever serialized. Timestamp and level/logger metadata are added first (they
    carry no PII), then redaction, then the final JSON render.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        redact_processor,
        structlog.processors.JSONRenderer(),
    ]


def configure_logging(level: int = logging.INFO) -> None:
    """Install the redacting structlog pipeline process-wide.

    Called once at api startup. Wires the standard-library logging level and the
    processor chain from `build_processors`, so every `get_logger` call thereafter
    emits redacted, structured JSON.
    """
    logging.basicConfig(format="%(message)s", level=level)
    structlog.configure(
        processors=build_processors(),
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> Any:
    """Return a bound structlog logger (optionally named) for a module to log through."""
    return structlog.get_logger(name)

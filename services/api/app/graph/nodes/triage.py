"""Triage node: raw ticket text → a validated `TriageResult` (plan Task 10 / todo Task 11).

The first agent node. It asks the model to classify a ticket into the closed
`category`/`urgency`/`sentiment` sets (SPEC §4.3) and returns a schema-validated
`TriageResult`, so no unvalidated model output flows downstream. Two guarantees
matter here:

- **Retry once.** A model answer that will not parse or fails the enum schema is
  retried a single time; a good answer on either attempt is returned.
- **Surface, never drop.** If both attempts fail, the node raises
  `TriageValidationError` (carrying the raw output) rather than inventing a
  category. When the workflow is assembled (todo Task 17) it catches this and
  routes the case to a human rep with an explanation.

This is a plain async function, independent of LangGraph; the state adapter that
wraps it into a graph node is added in Task 17. The classification prompt lives
in-repo for now; Langfuse-managed prompt resolution is deferred to Task 28.
Triage keeps the project-wide reason-by-default (`think=True`): urgency drives the
rep-queue sort order, so accuracy is worth the reasoning pass.
"""

from __future__ import annotations

import json
import re
from enum import StrEnum

from pydantic import ValidationError

from app.llm.base import LLM
from app.prompts.registry import get_prompt
from app.schemas.enums import Category, Sentiment, Urgency
from app.schemas.triage import TriageResult

# Grab the first `{`-to-last-`}` span as the JSON object, tolerating any prose or
# code-fence framing a model may wrap around it. A leaked reasoning trace that
# breaks this simply fails the parse and drives a retry (see Task 17 note): the
# structured schema is the leak's safety net, so nothing is stripped here.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _allowed(enum: type[StrEnum]) -> str:
    """Render an enum's values as a comma-separated list for the prompt."""
    return ", ".join(member.value for member in enum)


# The closed value sets, pulled straight from the enums so the prompt can never
# drift from the schema its output is validated against. Filled into the triage
# template (fetched from the prompt registry) at call time.
_ALLOWED_VALUES = {
    "categories": _allowed(Category),
    "urgencies": _allowed(Urgency),
    "sentiments": _allowed(Sentiment),
}


class TriageValidationError(Exception):
    """Raised when triage cannot produce a valid result after the retry.

    Surfaces the failure so the workflow can hand the case to a human rep instead
    of guessing a classification (SPEC §4.3). `raw_output` preserves the model's
    last response and `attempts` records how many tries were made, so the rep and
    the audit trail retain the full context.
    """

    def __init__(self, raw_output: str, attempts: int) -> None:
        """Store the last raw model output and attempt count; build a hand-off message."""
        self.raw_output = raw_output
        self.attempts = attempts
        super().__init__(
            f"Triage could not classify this ticket after {attempts} attempt(s): "
            "the AI returned output that failed validation. A human rep needs to "
            "handle this case."
        )


def _build_prompt(message: str, extracted_facts: str | None) -> str:
    """Fill the registered triage template with the message, facts, and enum sets.

    The template comes from `get_prompt("triage")` — the resolution seam Task 28
    will back with Langfuse — so the prose lives in the prompt registry, not here.
    When `extracted_facts` is provided it is rendered under an `Extracted facts:`
    label (attachment digitization, Task 20+); with none, the section is empty so
    the prompt never shows a dangling header.
    """
    facts = f"\nExtracted facts:\n{extracted_facts}\n" if extracted_facts else ""
    return get_prompt("triage").format(message=message, facts=facts, **_ALLOWED_VALUES)


def _parse(raw: str) -> TriageResult:
    """Parse one model response into a `TriageResult`, raising on any invalidity.

    Raises `ValueError` when no JSON object is present or it is malformed
    (`json.JSONDecodeError` is a `ValueError`) and `ValidationError` when a value
    falls outside the enums. The caller treats either as a failed attempt.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise ValueError("no JSON object found in model output")
    data = json.loads(match.group(0))
    return TriageResult.model_validate(data)


async def triage(
    message: str, llm: LLM, *, max_attempts: int, extracted_facts: str | None = None
) -> TriageResult:
    """Classify a ticket into a validated `TriageResult`, retrying on bad output.

    Sends the in-repo triage prompt (with `think=True`) to `llm`, parses and
    schema-validates the reply, and returns the `TriageResult`. `max_attempts`
    (config, not code — `settings.triage_max_attempts`, SPEC §4.3) caps the model
    tries: a parse or schema failure is retried until the budget is spent, and if
    the final attempt still fails the failure is surfaced as
    `TriageValidationError` rather than silently dropped. The graph adapter (todo
    Task 17) supplies the configured value.
    """
    prompt = _build_prompt(message, extracted_facts)
    raw = ""
    for attempt in range(1, max_attempts + 1):
        raw = await llm.generate(prompt, think=True)
        try:
            return _parse(raw)
        except (ValueError, ValidationError):
            if attempt == max_attempts:
                raise TriageValidationError(raw_output=raw, attempts=attempt) from None
    # Reached only if max_attempts < 1 (a misconfiguration): no call was made.
    raise TriageValidationError(raw_output=raw, attempts=max_attempts)

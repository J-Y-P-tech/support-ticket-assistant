"""Extraction node: a transcription → a validated `ExtractionResult` (plan Task 20 / todo Task 21).

The second of the three digitization passes (SPEC §4.2). It takes the verbatim
transcription the OCR pass produced and asks the model to structure it into an
`ExtractionResult` (doc_type/amounts/dates/names/references/low_confidence), returning
a schema-validated result so no unvalidated model output flows downstream. Two
guarantees shape it — the second is where it deliberately diverges from triage:

- **Retry once.** A model answer that will not parse or fails the schema is retried a
  single time; a good answer on either attempt is returned. The budget is config-driven
  (`max_attempts`), not a hard-coded constant.
- **Flag, never drop — and never raise.** Where triage *raises* when both attempts
  fail (a classification it cannot guess must halt for a human), extraction instead
  returns a result flagged `low_confidence=True` carrying the raw transcription and no
  invented facts. The extracted facts are unverified, rep-facing input (never
  authoritative grounding — SPEC §4.2), so a failure to structure them should surface
  the raw text for the rep to read, not block the case.

`raw_text` is set by the node from the transcription it was handed, never from the
model's JSON, so the verbatim text survives intact whatever the model returns — the one
field that can never be lost. Any `raw_text` the model echoes is overridden.

This is a plain async function, independent of LangGraph; the state adapter that wraps
it into a graph node is added with the fused-query task (todo Task 22). The prompt lives
in-repo via the prompt registry for now; Langfuse-managed resolution is deferred to Task
28. Extraction keeps the project-wide reason-by-default (`think=True`): unlike the
mechanical OCR copy, deciding which token is a date, an amount, or a reference is
interpretive and earns the reasoning pass. As with triage, a leaked reasoning trace
simply fails the JSON parse and drives a retry, so the schema is the leak's safety net
and nothing is stripped here.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.llm.base import LLM
from app.prompts.registry import get_prompt
from app.schemas.extraction import ExtractionResult

# Grab the first `{`-to-last-`}` span as the JSON object, tolerating any prose or
# code-fence framing the model may wrap around it (identical to triage's approach). A
# leaked reasoning trace that breaks this simply fails the parse and drives a retry.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _parse(raw: str, transcription: str) -> ExtractionResult:
    """Parse one model response into an `ExtractionResult`, raising on any invalidity.

    Extracts the JSON object, then forces `raw_text` to the true `transcription` before
    validating, so the verbatim text is always the node's own copy rather than the
    model's echo of it. Raises `ValueError` when no JSON object is present or it is
    malformed (`json.JSONDecodeError` is a `ValueError`) and `ValidationError` when a
    field is the wrong type. The caller treats either as a failed attempt.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise ValueError("no JSON object found in model output")
    data = json.loads(match.group(0))
    data["raw_text"] = transcription
    return ExtractionResult.model_validate(data)


async def extract(transcription: str, llm: LLM, *, max_attempts: int) -> ExtractionResult:
    """Structure a transcription into an `ExtractionResult`, flagging on failure.

    Sends the in-repo extraction prompt (with `think=True`) to `llm`, parses and
    schema-validates the reply, and returns the `ExtractionResult`. `max_attempts`
    (config, not code — SPEC §4.2) caps the model tries: a parse or schema failure is
    retried until the budget is spent. Unlike triage, an exhausted budget does not raise
    — it returns a result flagged `low_confidence=True` carrying the raw `transcription`
    and no invented facts, so the verbatim text is surfaced for the rep rather than
    dropped. The graph adapter (todo Task 22) supplies the configured value.
    """
    prompt = get_prompt("extract").format(transcription=transcription)
    for attempt in range(1, max_attempts + 1):
        raw = await llm.generate(prompt, think=True)
        try:
            return _parse(raw, transcription)
        except (ValueError, ValidationError):
            if attempt == max_attempts:
                break
    # Retries exhausted (or max_attempts < 1): flag for the rep and preserve the raw
    # text rather than raising — the extracted facts are unverified input, so a failed
    # structuring degrades gracefully instead of blocking the case (SPEC §4.2).
    return ExtractionResult(raw_text=transcription, low_confidence=True)

"""Validate node: schema + LLM-as-judge groundedness scorer (plan Task 13 / todo Task 14).

The guardrail checkpoint between `draft` and `human_review`. It does two things
(SPEC §4.5, §5 step 7): a structural check that the draft is well-formed, and an
**LLM-as-judge groundedness score** — the model judges how faithfully the drafted
reply stays on the sources it cited. Being *handed* a good source is not the same as
*staying faithful* to it: the model can invent a figure, misstate a number, or flip a
source's meaning. The judge catches that drift, which the retrieval-time grounding
gate cannot see.

The judge returns a score in [0, 1] plus the claims it could not support. A score
below `groundedness_min` flags the draft as **"AI-suggested, unverified"**
(`verified=False`) for the rep, and the unsupported claims are surfaced with it.
Flagging never blocks the flow — every draft still reaches `human_review` (SPEC §5);
the flag is a warning, not a gate. `groundedness_min` and the retry budget are config
(`settings.groundedness_min`, `settings.validate_max_attempts`), supplied by the graph
adapter (todo Task 17) like triage's, so strictness is a tuning knob, not a constant.

The node mirrors triage's structured-output contract: the judge's JSON is validated
against `GroundednessVerdict` and retried once on bad output. Two guarantees differ,
because this is a guardrail feeding a human rather than a classifier:

- **Fail closed.** If the judge still returns invalid output after the retry budget,
  the node flags the draft for manual review rather than raising — a flaky judge
  degrades safely instead of crashing the pipeline.
- **Cheap short-circuit.** An empty draft body is flagged structurally without
  spending a judge call; there is nothing to fact-check.

This is a plain async function, independent of LangGraph; the state adapter that wraps
it into a graph node is added when the workflow is assembled (todo Task 17). The judge
prompt lives in-repo via the prompt registry for now; Langfuse-managed resolution is
deferred to Task 28.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.llm.base import LLM
from app.prompts.registry import get_prompt
from app.schemas.draft import Draft
from app.schemas.kb import KBSearchResult, KBSource
from app.schemas.validation import GroundednessVerdict, ValidationResult

# Grab the first `{`-to-last-`}` span as the JSON object, tolerating any prose or
# code-fence framing the judge may wrap around it (mirrors the triage parser).
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


def _cited_sources(draft: Draft, result: KBSearchResult) -> list[KBSource]:
    """Return the sources the draft actually cited, matched by source id.

    The judge scores the reply against what it *cited*, not every source the search
    returned, so a reply cannot look grounded on a source it never used.
    """
    cited_ids = {citation.source_id for citation in draft.citations}
    return [source for source in result.sources if source.id in cited_ids]


def _render_sources(sources: list[KBSource]) -> str:
    """Render the cited sources into a labelled block for the judge prompt."""
    return "\n\n".join(f'[{source.id}] "{source.title}"\n{source.text}' for source in sources)


def _build_prompt(draft: Draft, result: KBSearchResult) -> str:
    """Fill the registered judge template with the draft body and its cited sources.

    The template comes from `get_prompt("validate")` — the resolution seam Task 28
    will back with Langfuse — so the prose lives in the prompt registry, not here.
    """
    rendered = _render_sources(_cited_sources(draft, result))
    return get_prompt("validate").format(draft=draft.body, sources=rendered)


def _parse(raw: str) -> GroundednessVerdict:
    """Parse one judge response into a `GroundednessVerdict`, raising on any invalidity.

    Raises `ValueError` when no JSON object is present or it is malformed
    (`json.JSONDecodeError` is a `ValueError`) and `ValidationError` when the score
    is out of range. The caller treats either as a failed attempt.
    """
    match = _JSON_OBJECT_RE.search(raw)
    if match is None:
        raise ValueError("no JSON object found in judge output")
    data = json.loads(match.group(0))
    return GroundednessVerdict.model_validate(data)


async def _judge(
    draft: Draft, result: KBSearchResult, llm: LLM, *, max_attempts: int
) -> GroundednessVerdict | None:
    """Ask the model to score the draft's groundedness, retrying on bad output.

    Sends the judge prompt (with `think=True`) and parses the reply, retrying until
    the attempt budget is spent. Returns the validated verdict, or `None` when every
    attempt failed — the caller fails closed on `None` rather than guessing a score.
    """
    prompt = _build_prompt(draft, result)
    for attempt in range(1, max_attempts + 1):
        raw = await llm.generate(prompt, think=True)
        try:
            return _parse(raw)
        except (ValueError, ValidationError):
            if attempt == max_attempts:
                return None
    return None


async def validate(
    draft: Draft, result: KBSearchResult, llm: LLM, *, groundedness_min: float, max_attempts: int
) -> ValidationResult:
    """Judge a draft's groundedness and flag it for the rep when it falls short.

    Runs a structural check (non-empty body) and the LLM-as-judge groundedness score
    against the draft's cited sources. The draft is flagged — and its `verified` flag
    downgraded to False — when the body is empty, when the judge cannot be run, when
    groundedness is below `groundedness_min`, or when the draft already arrived
    unverified; each trigger adds a plain-language reason (and the judge's unsupported
    claims) for the rep. A clean, well-grounded draft passes through unflagged and
    verified. Flagging never blocks the flow (SPEC §5): every draft still reaches
    `human_review`.
    """
    if not draft.body.strip():
        return ValidationResult(
            draft=draft.model_copy(update={"verified": False}),
            groundedness=0.0,
            flagged=True,
            reasons=["draft body is empty"],
        )

    verdict = await _judge(draft, result, llm, max_attempts=max_attempts)
    if verdict is None:
        return ValidationResult(
            draft=draft.model_copy(update={"verified": False}),
            groundedness=0.0,
            flagged=True,
            reasons=[
                "groundedness check could not run: the reviewer returned invalid output "
                f"after {max_attempts} attempt(s); flagged for manual review"
            ],
        )

    groundedness = verdict.score
    reasons: list[str] = []
    if groundedness < groundedness_min:
        reasons.append(f"low groundedness ({groundedness:.2f} < {groundedness_min:.2f} threshold)")
        reasons.extend(f"unsupported claim: {claim}" for claim in verdict.unsupported_claims)
    if not draft.verified:
        reasons.append("draft arrived unverified")

    verified = draft.verified and groundedness >= groundedness_min
    return ValidationResult(
        draft=draft.model_copy(update={"verified": verified}),
        groundedness=groundedness,
        flagged=bool(reasons),
        reasons=reasons,
    )

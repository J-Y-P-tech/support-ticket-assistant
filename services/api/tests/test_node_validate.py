"""Unit tests for the validate node (plan Task 13 / todo Task 14).

`validate` is the guardrail checkpoint between `draft` and `human_review`. It runs a
structural check on the draft and an **LLM-as-judge groundedness score**: the model
judges how faithfully the drafted reply stays on the sources it cited, returning a
score in [0, 1] plus the claims it could not support (SPEC §4.5, §5 step 7). Being
*handed* a good source is not the same as *staying faithful* to it — the judge is
what catches the model inventing a figure or flipping a source's meaning, which the
retrieval-time grounding gate cannot see.

A score below `groundedness_min` flags the draft as **"AI-suggested, unverified"**
(`verified=False`) for the rep. Flagging never blocks the flow — every draft still
reaches `human_review`; the flag is a warning, not a gate.

The node mirrors triage's structured-output contract: the judge's JSON is validated
against `GroundednessVerdict` and retried once on bad output. If the judge still
fails, the node **fails closed** — it flags the draft for manual review rather than
raising, so a flaky judge degrades safely instead of crashing the pipeline. Tests
drive it with a deterministic `FakeLLM`, never the host model (SPEC §10/§12).

The behaviours pinned here are the acceptance criteria:

- a well-grounded draft (high judge score) passes, unflagged and verified;
- a low-score draft is flagged, downgraded to unverified, and its unsupported claims
  surface for the rep;
- the threshold is honoured exactly at the boundary (`>=` passes);
- a draft that arrived already unverified stays flagged regardless of the score;
- an empty body is flagged *without* spending a judge call;
- the judge is shown the draft and the text of the sources it **cited**, not others;
- invalid-then-valid judge output is retried; a judge that never parses fails closed.
"""

from __future__ import annotations

import json

from app.graph.nodes.validate import validate
from app.llm.fake import FakeLLM
from app.schemas.draft import Citation, Draft
from app.schemas.kb import KBSearchResult, KBSource
from app.schemas.validation import ValidationResult

# The retry budget the tests pass explicitly (production supplies it from config).
_MAX = 2


def _source(id_: str, text: str, title: str = "Article") -> KBSource:
    """Build a KB source with the given id and body text."""
    return KBSource(id=id_, title=title, text=text)


def _draft(body: str, cited_ids: list[str], *, verified: bool = True) -> Draft:
    """Build a Draft with the given body that cites each id in `cited_ids`."""
    citations = [Citation(source_id=id_, title="Article") for id_ in cited_ids]
    return Draft(body=body, citations=citations, verified=verified)


def _result(*sources: KBSource) -> KBSearchResult:
    """Wrap sources in a confident KBSearchResult (the drafting-time shape)."""
    return KBSearchResult(sources=list(sources), no_confident_source=False)


def _verdict(score: float, unsupported: list[str] | None = None) -> str:
    """Render a judge response as the JSON the model is asked to emit."""
    return json.dumps({"score": score, "unsupported_claims": unsupported or []})


async def test_well_grounded_draft_passes_and_stays_verified() -> None:
    """A high judge score leaves the draft unflagged and verified."""
    llm = FakeLLM(_verdict(1.0))
    draft = _draft("Reset your access from the login page.", ["kb-1"])

    outcome = await validate(
        draft,
        _result(_source("kb-1", "Reset access from the login page.")),
        llm,
        groundedness_min=0.6,
        max_attempts=_MAX,
    )

    assert isinstance(outcome, ValidationResult)
    assert outcome.groundedness == 1.0
    assert outcome.flagged is False
    assert outcome.reasons == []
    assert outcome.draft.verified is True


async def test_low_score_draft_is_flagged_and_downgraded() -> None:
    """A low judge score flags the draft, downgrades it, and surfaces the bad claim.

    The judge reports a claim the sources do not back (an invented refund promise);
    the node must flag the draft unverified and pass the specific claim to the rep,
    never present it as sourced fact (SPEC §4.5).
    """
    llm = FakeLLM(_verdict(0.2, ["promises a $500 refund the sources do not mention"]))
    draft = _draft("We will refund you $500 today.", ["kb-1"])

    outcome = await validate(
        draft,
        _result(_source("kb-1", "Disputes are reviewed within 10 business days.")),
        llm,
        groundedness_min=0.6,
        max_attempts=_MAX,
    )

    assert outcome.groundedness == 0.2
    assert outcome.flagged is True
    assert outcome.draft.verified is False
    assert any("$500 refund" in reason for reason in outcome.reasons)


async def test_threshold_boundary_is_inclusive() -> None:
    """Exactly meeting the threshold passes; a hair above it flags (>= is the rule)."""
    draft = _draft("Some reply.", ["kb-1"])
    result = _result(_source("kb-1", "Some source text."))

    at_threshold = await validate(
        draft, result, FakeLLM(_verdict(0.75)), groundedness_min=0.75, max_attempts=_MAX
    )
    assert at_threshold.groundedness == 0.75
    assert at_threshold.flagged is False
    assert at_threshold.draft.verified is True

    above_threshold = await validate(
        draft, result, FakeLLM(_verdict(0.75)), groundedness_min=0.76, max_attempts=_MAX
    )
    assert above_threshold.flagged is True
    assert above_threshold.draft.verified is False


async def test_already_unverified_draft_stays_flagged() -> None:
    """A draft that arrived unverified stays flagged even with a perfect judge score.

    Defensive: should any upstream node hand `validate` a draft already marked
    `verified=False`, the node must not launder it back to verified. It stays
    flagged for the rep.
    """
    llm = FakeLLM(_verdict(1.0))
    draft = _draft("Reset your access from the login page.", ["kb-1"], verified=False)

    outcome = await validate(
        draft,
        _result(_source("kb-1", "Reset access from the login page.")),
        llm,
        groundedness_min=0.6,
        max_attempts=_MAX,
    )

    assert outcome.groundedness == 1.0
    assert outcome.flagged is True
    assert outcome.draft.verified is False
    assert outcome.reasons


async def test_empty_body_is_flagged_without_calling_the_judge() -> None:
    """A whitespace-only body is flagged structurally, spending no judge call.

    There is no reply to fact-check, so the node short-circuits: it flags and
    downgrades the draft and never touches the model.
    """
    llm = FakeLLM(_verdict(1.0))
    draft = _draft("   ", ["kb-1"])

    outcome = await validate(
        draft,
        _result(_source("kb-1", "Some source text.")),
        llm,
        groundedness_min=0.6,
        max_attempts=_MAX,
    )

    assert outcome.flagged is True
    assert outcome.draft.verified is False
    assert outcome.reasons
    assert llm.calls == []  # the judge was never invoked


async def test_judge_is_shown_the_draft_and_only_cited_source_text() -> None:
    """The judge prompt carries the draft body and the text of the cited sources only.

    Groundedness is judged against what the draft *cited*: the cited source's text
    reaches the prompt, an un-cited source's text does not, and the reply itself is
    included so the judge can check it against the sources.
    """
    llm = FakeLLM(_verdict(1.0))
    cited = _source("kb-1", "Verify identity, then reset access.")
    uncited = _source("kb-2", "UNCITED-SENTINEL-TEXT should never be judged.")
    draft = _draft("Reset your access from the login page.", ["kb-1"])

    await validate(draft, _result(cited, uncited), llm, groundedness_min=0.6, max_attempts=_MAX)

    prompt = llm.calls[0]["prompt"]
    assert "Reset your access from the login page." in prompt
    assert "Verify identity, then reset access." in prompt
    assert "UNCITED-SENTINEL-TEXT" not in prompt


async def test_invalid_then_valid_judge_output_is_retried() -> None:
    """A judge response that will not parse is retried once, then the valid one wins."""
    llm = FakeLLM(["not json at all", _verdict(1.0)])
    draft = _draft("Reset your access from the login page.", ["kb-1"])

    outcome = await validate(
        draft,
        _result(_source("kb-1", "Reset access from the login page.")),
        llm,
        groundedness_min=0.6,
        max_attempts=_MAX,
    )

    assert outcome.groundedness == 1.0
    assert outcome.flagged is False
    assert len(llm.calls) == 2  # first attempt failed, second succeeded


async def test_unparseable_judge_output_fails_closed() -> None:
    """A judge that never returns valid output flags the draft rather than crashing.

    A guardrail feeding a human should degrade safely: if the reviewer's output can
    not be validated after the retry budget, the node flags the draft unverified for
    manual review — it does not raise and stop the pipeline.
    """
    llm = FakeLLM(["garbage", "still garbage"])
    draft = _draft("Reset your access from the login page.", ["kb-1"])

    outcome = await validate(
        draft,
        _result(_source("kb-1", "Reset access from the login page.")),
        llm,
        groundedness_min=0.6,
        max_attempts=_MAX,
    )

    assert outcome.flagged is True
    assert outcome.draft.verified is False
    assert outcome.groundedness == 0.0
    assert outcome.reasons

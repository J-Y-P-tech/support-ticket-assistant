"""Unit tests for the draft node (plan Task 12 / todo Task 13).

`draft` turns retrieved KB sources plus the customer message into a `Draft`: a
reply written **only** from authoritative sources, each cited by source `id`/`title`
(SPEC §4.5). The node is a standalone async function here — the LangGraph state
adapter that wraps it is added when the workflow is assembled (todo Task 17) — so
these tests exercise it directly against a deterministic `FakeLLM`, never the host
model (SPEC §10/§12).

The behaviours pinned here are the acceptance criteria (SPEC §4.5):

- authoritative sources → a cited, verified draft (`verified=True`), one
  citation per authoritative source;
- the customer message and the authoritative source text reach the model's prompt,
  so the reply is phrased from the source;
- a `model_generated` chunk sitting beside an authoritative one does **not** ground
  the reply and is neither fed to the model nor cited;
- the `model_generated`-only path (defensive — the groundedness gate normally
  routes it to a human) drafts from the fallback and marks it
  **"AI-suggested, unverified"** via `verified=False`;
- nothing to ground from raises rather than inventing an ungrounded answer.
"""

from __future__ import annotations

import pytest

from app.graph.nodes.draft import draft
from app.llm.fake import FakeLLM
from app.schemas.draft import Draft
from app.schemas.enums import SourceKind
from app.schemas.kb import KBSearchResult, KBSource

# A canned reply body the FakeLLM returns for every draft call; tests assert the
# node wraps it into a `Draft` and never on its wording.
_REPLY = "You can reset your online-banking access from the login page."


def _authoritative(id_: str = "kb-1", title: str = "Reset access") -> KBSource:
    """Build an authoritative KB source — the kind that may ground a draft."""
    return KBSource(
        id=id_,
        title=title,
        text="Verify the customer's identity, then reset their access.",
        source_kind=SourceKind.AUTHORITATIVE,
    )


def _model_generated(id_: str = "kb-9", title: str = "Best guess") -> KBSource:
    """Build a model_generated KB source — a fallback that never grounds (SPEC §4.5)."""
    return KBSource(
        id=id_,
        title=title,
        text="Unverified best-effort suggestion the model produced.",
        source_kind=SourceKind.MODEL_GENERATED,
    )


async def test_authoritative_source_yields_cited_verified_draft() -> None:
    """One authoritative source produces a cited draft that is not flagged unverified."""
    result = KBSearchResult(sources=[_authoritative()], no_confident_source=False)

    reply = await draft("How do I reset my access?", result, FakeLLM(_REPLY))

    assert isinstance(reply, Draft)
    assert reply.body == _REPLY
    assert reply.verified is True
    assert len(reply.citations) == 1
    assert reply.citations[0].source_id == "kb-1"
    assert reply.citations[0].title == "Reset access"


async def test_every_authoritative_source_is_cited() -> None:
    """Each authoritative source the draft draws on appears as its own citation."""
    result = KBSearchResult(
        sources=[
            _authoritative("kb-1", "Reset access"),
            _authoritative("kb-2", "Identity checks"),
        ],
        no_confident_source=False,
    )

    reply = await draft("How do I reset my access?", result, FakeLLM(_REPLY))

    cited_ids = {citation.source_id for citation in reply.citations}
    assert cited_ids == {"kb-1", "kb-2"}


async def test_message_and_source_text_reach_the_prompt() -> None:
    """The customer message and the authoritative source text are both in the prompt.

    The reply must be phrased *from* the source, so the node has to hand the model
    both the question and the source material.
    """
    llm = FakeLLM(_REPLY)
    result = KBSearchResult(sources=[_authoritative()], no_confident_source=False)

    await draft("How do I reset my access?", result, llm)

    prompt = llm.calls[0]["prompt"]
    assert "How do I reset my access?" in prompt
    assert "Verify the customer's identity, then reset their access." in prompt


async def test_model_generated_beside_authoritative_is_ignored() -> None:
    """A model_generated chunk beside an authoritative one neither grounds nor is cited.

    The reply is written only from the authoritative source (SPEC §4.5): the
    model_generated text must not reach the prompt, and only the authoritative
    source is cited. The draft stays verified.
    """
    llm = FakeLLM(_REPLY)
    result = KBSearchResult(
        sources=[_model_generated(), _authoritative()],
        no_confident_source=False,
    )

    reply = await draft("How do I reset my access?", result, llm)

    assert reply.verified is True
    assert {c.source_id for c in reply.citations} == {"kb-1"}
    assert "Unverified best-effort suggestion the model produced." not in llm.calls[0]["prompt"]


async def test_model_generated_only_sets_unverified_flag() -> None:
    """The model_generated-only fallback drafts but is flagged AI-suggested, unverified.

    The groundedness gate normally routes this case to a human, so reaching `draft`
    with no authoritative source is defensive — the node still refuses to present it
    as sourced fact: it marks `verified=False` (SPEC §4.5).
    """
    result = KBSearchResult(sources=[_model_generated()], no_confident_source=False)

    reply = await draft("How do I reset my access?", result, FakeLLM(_REPLY))

    assert reply.verified is False
    assert reply.body == _REPLY


async def test_model_generated_only_cites_its_fallback_source() -> None:
    """An unverified fallback draft still records which source it drew from."""
    result = KBSearchResult(sources=[_model_generated()], no_confident_source=False)

    reply = await draft("How do I reset my access?", result, FakeLLM(_REPLY))

    assert {c.source_id for c in reply.citations} == {"kb-9"}


async def test_empty_sources_raises() -> None:
    """Nothing to ground from raises rather than inventing an ungrounded reply.

    The gate guarantees at least one source before drafting; this defends the node
    if it is ever called without any.
    """
    result = KBSearchResult(sources=[], no_confident_source=False)

    with pytest.raises(ValueError):
        await draft("How do I reset my access?", result, FakeLLM(_REPLY))

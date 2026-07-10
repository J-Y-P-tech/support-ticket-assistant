"""Unit tests for the draft node (plan Task 12 / todo Task 13).

`draft` turns retrieved KB sources plus the customer message into a `Draft`: a
reply written from the sources, each cited by source `id`/`title` (SPEC §4.5). Every
KB source is an eligible, citable answer, so the draft grounds in all of them. The
node is a standalone async function here — the LangGraph state adapter that wraps it
is added when the workflow is assembled (todo Task 17) — so these tests exercise it
directly against a deterministic `FakeLLM`, never the host model (SPEC §10/§12).

The behaviours pinned here are the acceptance criteria (SPEC §4.5):

- sources → a cited, verified draft (`verified=True`), one citation per source;
- the customer message and the source text reach the model's prompt, so the reply
  is phrased from the source;
- nothing to ground from raises rather than inventing an ungrounded answer.

Note the draft always leaves this node `verified=True`; downgrading a draft that
drifts off its sources is the `validate` node's job (todo Task 14), tested there.
"""

from __future__ import annotations

import pytest

from app.graph.nodes.draft import draft
from app.llm.fake import FakeLLM
from app.schemas.draft import Draft
from app.schemas.kb import KBSearchResult, KBSource

# A canned reply body the FakeLLM returns for every draft call; tests assert the
# node wraps it into a `Draft` and never on its wording.
_REPLY = "You can reset your online-banking access from the login page."


def _source(id_: str = "kb-1", title: str = "Reset access") -> KBSource:
    """Build a KB source — every KB source is an eligible, citable answer."""
    return KBSource(
        id=id_,
        title=title,
        text="Verify the customer's identity, then reset their access.",
    )


async def test_source_yields_cited_verified_draft() -> None:
    """One source produces a cited draft that leaves this node verified."""
    result = KBSearchResult(sources=[_source()], no_confident_source=False)

    reply = await draft("How do I reset my access?", result, FakeLLM(_REPLY))

    assert isinstance(reply, Draft)
    assert reply.body == _REPLY
    assert reply.verified is True
    assert len(reply.citations) == 1
    assert reply.citations[0].source_id == "kb-1"
    assert reply.citations[0].title == "Reset access"


async def test_every_source_is_cited() -> None:
    """Each source the draft draws on appears as its own citation."""
    result = KBSearchResult(
        sources=[
            _source("kb-1", "Reset access"),
            _source("kb-2", "Identity checks"),
        ],
        no_confident_source=False,
    )

    reply = await draft("How do I reset my access?", result, FakeLLM(_REPLY))

    cited_ids = {citation.source_id for citation in reply.citations}
    assert cited_ids == {"kb-1", "kb-2"}


async def test_message_and_source_text_reach_the_prompt() -> None:
    """The customer message and the source text are both in the prompt.

    The reply must be phrased *from* the source, so the node has to hand the model
    both the question and the source material.
    """
    llm = FakeLLM(_REPLY)
    result = KBSearchResult(sources=[_source()], no_confident_source=False)

    await draft("How do I reset my access?", result, llm)

    prompt = llm.calls[0]["prompt"]
    assert "How do I reset my access?" in prompt
    assert "Verify the customer's identity, then reset their access." in prompt


async def test_empty_sources_raises() -> None:
    """Nothing to ground from raises rather than inventing an ungrounded reply.

    The gate guarantees at least one source before drafting; this defends the node
    if it is ever called without any.
    """
    result = KBSearchResult(sources=[], no_confident_source=False)

    with pytest.raises(ValueError):
        await draft("How do I reset my access?", result, FakeLLM(_REPLY))

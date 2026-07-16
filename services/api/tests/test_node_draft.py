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
from app.prompts.fewshot import FewShotExample
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


# --- Dynamic prompting: few-shot injection + Langfuse-resolved prompt (todo Task 29) ---


class _StubPrompt:
    """A minimal stand-in for a Langfuse prompt object: a template plus its version."""

    def __init__(self, prompt: str, version: int) -> None:
        """Record the template text and integer version the stub client will hand back."""
        self.prompt = prompt
        self.version = version


class _StubPromptClient:
    """A Langfuse-like client that returns one canned draft prompt for any name."""

    def __init__(self, prompt: _StubPrompt) -> None:
        """Hold the canned prompt this stub returns from `get_prompt`."""
        self._prompt = prompt

    def get_prompt(self, name: str) -> _StubPrompt:
        """Return the canned prompt regardless of `name`, mimicking a Langfuse fetch."""
        return self._prompt


def _example(example_id: int) -> FewShotExample:
    """Build an approved-reply few-shot example with a recognisable message/reply."""
    return FewShotExample(
        category="access_reset",
        message=f"example question {example_id}",
        reply=f"example approved reply {example_id}",
        rating=5,
        example_id=example_id,
    )


async def test_examples_reach_the_prompt() -> None:
    """The injected few-shot examples' text is present in the prompt sent to the model.

    Dynamic few-shot (SPEC §4.10) works only if the chosen approved replies actually
    reach the drafting prompt, so both the example question and its approved answer must
    appear alongside the customer message and sources.
    """
    llm = FakeLLM(_REPLY)
    result = KBSearchResult(sources=[_source()], no_confident_source=False)
    examples = [_example(1), _example(2)]

    await draft("How do I reset my access?", result, llm, examples=examples)

    prompt = llm.calls[0]["prompt"]
    for example in examples:
        assert example.message in prompt
        assert example.reply in prompt
    # The customer message and source text still reach the prompt.
    assert "How do I reset my access?" in prompt
    assert "Verify the customer's identity, then reset their access." in prompt


async def test_draft_resolves_prompt_from_langfuse_client() -> None:
    """A provided Langfuse client's draft template is what the node sends the model.

    The drafting node resolves its prompt from Langfuse with an in-repo fallback (SPEC
    §4.10); given a client, the Langfuse template — not the in-repo one — must drive the
    prompt, while the message and sources still fill their placeholders.
    """
    llm = FakeLLM(_REPLY)
    result = KBSearchResult(sources=[_source()], no_confident_source=False)
    client = _StubPromptClient(
        _StubPrompt("LANGFUSE-DRAFT {examples}Q: {message}\nSOURCES: {sources}", version=4)
    )

    await draft("How do I reset my access?", result, llm, prompt_client=client)

    prompt = llm.calls[0]["prompt"]
    assert "LANGFUSE-DRAFT" in prompt
    assert "How do I reset my access?" in prompt
    assert "Verify the customer's identity, then reset their access." in prompt


async def test_draft_without_examples_or_client_is_unchanged() -> None:
    """With no examples and no client the node still produces a cited, verified draft.

    Backwards compatibility for the assembled workflow, which today calls the node with
    neither: the in-repo prompt resolves, the empty example block adds nothing, and the
    reply is grounded and cited exactly as before.
    """
    llm = FakeLLM(_REPLY)
    result = KBSearchResult(sources=[_source()], no_confident_source=False)

    reply = await draft("How do I reset my access?", result, llm)

    assert reply.body == _REPLY
    assert reply.verified is True
    assert [c.source_id for c in reply.citations] == ["kb-1"]

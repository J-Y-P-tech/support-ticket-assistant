"""Draft node: retrieved sources → a grounded, cited reply (plan Task 12 / todo Task 13).

The drafting step sits after the groundedness gate. It phrases a reply to the
customer **only** from the KB sources it is given and cites each one by source
`id`/`title`, so the answer is grounded and auditable rather than free invention
(SPEC §4.5). Every source the KB returns is an eligible, citable answer, so the
draft is written from all of them and left verified; with nothing to ground from at
all it raises rather than inventing an answer.

Being *handed* a good source is not the same as *staying faithful* to it, though —
the model can still drift off the source. Catching that is the `validate` node's job
(todo Task 14): it scores the draft's groundedness and downgrades `verified` to
False when the reply wanders, flagging it "AI-suggested, unverified" for the rep. So
a draft leaves this node verified and only the validate step may mark it otherwise.

This is a plain async function, independent of LangGraph; the state adapter that
wraps it into a graph node is added when the workflow is assembled (todo Task 17).
The drafting prompt lives in-repo via the prompt registry for now; Langfuse-managed
resolution is deferred to Task 28. Drafting keeps the project-wide reason-by-default
(`think=True`): a customer-facing reply is quality-sensitive, so the reasoning pass
is worth it.
"""

from __future__ import annotations

from app.llm.base import LLM
from app.prompts.registry import get_prompt
from app.schemas.draft import Citation, Draft
from app.schemas.kb import KBSearchResult, KBSource


def _grounding_sources(result: KBSearchResult) -> list[KBSource]:
    """Return the sources to draft from, raising when there are none.

    Every source in `result` is an eligible, citable answer, so the draft grounds
    in all of them. Raises `ValueError` when there is nothing to ground from; the
    groundedness gate guarantees this never happens in the assembled workflow.
    """
    if result.sources:
        return result.sources
    raise ValueError("cannot draft a reply: no sources to ground from")


def _render_sources(sources: list[KBSource]) -> str:
    """Render sources into a labelled block for the prompt, one entry per source.

    Each entry leads with the `id` and `title` the draft will cite, then the source
    text, so the model sees exactly the material it may draw from and nothing else.
    """
    return "\n\n".join(f'[{source.id}] "{source.title}"\n{source.text}' for source in sources)


def _build_prompt(message: str, sources: list[KBSource]) -> str:
    """Fill the registered draft template with the message and rendered sources.

    The template comes from `get_prompt("draft")` — the resolution seam Task 28 will
    back with Langfuse — so the prose lives in the prompt registry, not here.
    """
    return get_prompt("draft").format(message=message, sources=_render_sources(sources))


async def draft(message: str, result: KBSearchResult, llm: LLM) -> Draft:
    """Write a grounded, cited reply to `message` from the retrieved sources.

    Drafts from every source in `result` — each an eligible, citable answer — and
    cites them by `id`/`title` (SPEC §4.5). Sends the in-repo draft prompt to `llm`
    with `think=True` and wraps the returned text into a `Draft`, left verified for
    the `validate` node (todo Task 14) to score and possibly downgrade. Raises
    `ValueError` when `result` carries no sources to ground from.
    """
    sources = _grounding_sources(result)
    body = await llm.generate(_build_prompt(message, sources), think=True)
    citations = [Citation(source_id=source.id, title=source.title) for source in sources]
    return Draft(body=body, citations=citations)

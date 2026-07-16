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
The drafting prompt resolves through `resolve_prompt` (plan Task 27 / todo Task 29):
Langfuse-managed when a client is passed, the pinned in-repo template otherwise. The
node also injects **dynamic few-shot** examples — the best recent approved replies for
the ticket's category (SPEC §4.10) — when the caller hands them in; selecting those
candidates by category is the deferred live-lookup step, so here the node just renders
whatever selection it is given (empty by default). Drafting keeps the project-wide
reason-by-default (`think=True`): a customer-facing reply is quality-sensitive, so the
reasoning pass is worth it.
"""

from __future__ import annotations

from app.llm.base import LLM
from app.prompts.fewshot import FewShotExample, render_examples
from app.prompts.resolver import LangfusePromptClient, resolve_prompt
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


def _build_prompt(
    message: str,
    sources: list[KBSource],
    examples: list[FewShotExample],
    client: LangfusePromptClient | None,
) -> str:
    """Fill the resolved draft template with the examples, message, and rendered sources.

    The template comes from `resolve_prompt("draft", client=...)` — Langfuse-managed
    when a client is given, the pinned in-repo template otherwise — so the prose lives in
    the prompt registry/Langfuse, not here. The `{examples}` slot takes the rendered
    few-shot block, which is the empty string when `examples` is empty, leaving a
    no-example draft prompt unchanged.
    """
    template = resolve_prompt("draft", client=client).template
    return template.format(
        examples=render_examples(examples),
        message=message,
        sources=_render_sources(sources),
    )


async def draft(
    message: str,
    result: KBSearchResult,
    llm: LLM,
    *,
    examples: list[FewShotExample] | None = None,
    prompt_client: LangfusePromptClient | None = None,
) -> Draft:
    """Write a grounded, cited reply to `message` from the retrieved sources.

    Drafts from every source in `result` — each an eligible, citable answer — and
    cites them by `id`/`title` (SPEC §4.5). Resolves the draft prompt through
    `resolve_prompt` (Langfuse-managed when `prompt_client` is given, in-repo otherwise)
    and injects any `examples` as dynamic few-shot context (SPEC §4.10); with neither the
    node behaves exactly as before. Sends the prompt to `llm` with `think=True` and wraps
    the returned text into a `Draft`, left verified for the `validate` node (todo Task 14)
    to score and possibly downgrade. Raises `ValueError` when `result` carries no sources
    to ground from.
    """
    sources = _grounding_sources(result)
    prompt = _build_prompt(message, sources, examples or [], prompt_client)
    body = await llm.generate(prompt, think=True)
    citations = [Citation(source_id=source.id, title=source.title) for source in sources]
    return Draft(body=body, citations=citations)

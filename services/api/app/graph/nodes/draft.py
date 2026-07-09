"""Draft node: retrieved sources → a grounded, cited reply (plan Task 12 / todo Task 13).

The drafting step sits after the groundedness gate. It phrases a reply to the
customer **only** from the KB sources it is given and cites each one by source
`id`/`title`, so the answer is grounded and auditable rather than free invention
(SPEC §4.5). Two guarantees matter here:

- **Authoritative-only grounding.** When the result carries `authoritative`
  sources, the draft is written from those alone — model_generated chunks beside
  them are neither fed to the model nor cited — and the reply is left verified.
- **Never present a guess as fact.** The groundedness gate normally routes a
  result with no authoritative source to a human, so reaching `draft` with only
  `model_generated` sources is a defensive path; the node still drafts from the
  fallback but marks it **"AI-suggested, unverified"** (`verified=False`) so it
  cannot be shown as sourced fact. With nothing to ground from at all, it raises
  rather than inventing an answer.

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
from app.schemas.enums import SourceKind
from app.schemas.kb import KBSearchResult, KBSource


# return tuple[list[KBSource], bool] — the bool is the draft's `verified` flag
def _grounding_sources(result: KBSearchResult) -> tuple[list[KBSource], bool]:
    """Pick the sources to draft from and whether the draft counts as verified.

    Prefers `authoritative` sources — the only kind that may ground a reply
    (SPEC §4.5) — and reports the draft as verified (True). If none are present the
    node falls back to `model_generated` sources and reports it unverified (False),
    so the defensive path still refuses to present a guess as sourced fact. Raises
    `ValueError` when there is nothing at all to ground from; the groundedness gate
    guarantees this never happens in the assembled workflow.
    """
    authoritative = [s for s in result.sources if s.source_kind is SourceKind.AUTHORITATIVE]
    if authoritative:
        return authoritative, True
    model_generated = [s for s in result.sources if s.source_kind is SourceKind.MODEL_GENERATED]
    if model_generated:
        return model_generated, False
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

    Drafts from the `authoritative` sources in `result` when present (`verified`),
    otherwise from `model_generated` fallbacks (`verified=False`); either way the
    reply is written only from the selected sources and cites each of them by
    `id`/`title` (SPEC §4.5). Sends the in-repo draft prompt to `llm` with
    `think=True` and wraps the returned text into a `Draft`. Raises `ValueError`
    when `result` carries no sources to ground from.
    """
    sources, verified = _grounding_sources(result)
    body = await llm.generate(_build_prompt(message, sources), think=True)
    citations = [Citation(source_id=source.id, title=source.title) for source in sources]
    return Draft(body=body, citations=citations, verified=verified)

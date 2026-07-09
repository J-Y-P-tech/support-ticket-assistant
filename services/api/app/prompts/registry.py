"""Central registry of in-repo prompt templates + the single resolution seam.

Every model-using node fetches its prompt through `get_prompt(name)` instead of
holding its own string, so prompts live in one maintainable place rather than
scattered across node modules — one file to read, review, and improve.

Today this serves the pinned in-repo templates below. Task 28 (dynamic prompting)
will make `get_prompt` consult **Langfuse** first and fall back to these, so a
non-engineer can version and improve prompts from the approved-reply feedback loop
without a code change — and because nodes already resolve through this seam, that
swap touches no node.

Templates are `str.format` strings: `{placeholders}` are filled by the calling
node, which knows its own variables. Closed value sets that must never drift from
the schema (e.g. triage's enums) are left as placeholders and filled from the
enums at call time, not written into the prose here.
"""

from __future__ import annotations

# Triage (plan Task 10 / todo Task 11). `{categories}`/`{urgencies}`/`{sentiments}`
# are filled from the enums so the prompt can never drift from the schema its
# output is validated against; `{message}`/`{facts}` come from the ticket.
_TRIAGE = """\
You are a support-desk triage assistant for a financial institution. Classify the \
customer ticket below.

Respond with a single JSON object and nothing else, using exactly these keys:
- "category": one of [{categories}]
- "urgency": one of [{urgencies}]
- "sentiment": one of [{sentiments}]

Customer message:
{message}
{facts}
JSON:"""

# Grounded drafting (plan Task 12 / todo Task 13). `{message}` is the customer's
# ticket; `{sources}` is the rendered block of authoritative KB sources the draft
# node hands in. The instruction to answer *only* from the sources is the prompt's
# half of the grounding contract; the node enforces the other half by feeding in
# authoritative sources alone and citing exactly them (SPEC §4.5).
_DRAFT = """\
You are a support-desk assistant for a financial institution. Write a reply to the \
customer message below using ONLY the information in the provided sources.

Rules:
- Use only facts, figures, and policies stated in the sources. Do not invent or \
assume anything that is not there.
- Make no promises, guarantees, or commitments the sources do not support.
- Phrase the answer in your own words — do not copy a source verbatim.
- If the sources do not fully answer the question, address what they do support and \
say a representative will follow up on the rest.

Customer message:
{message}

Sources:
{sources}

Reply:"""

# The registry. New node prompts are added here (name -> template), keeping every
# prompt in one place. Names are the seam's public contract, shared with Langfuse
# when Task 28 lands.
_PROMPTS: dict[str, str] = {
    "triage": _TRIAGE,
    "draft": _DRAFT,
}


def get_prompt(name: str) -> str:
    """Return the in-repo prompt template registered under `name`.

    The single seam every node resolves its prompt through. Raises `KeyError` for
    an unknown name so a typo fails loudly instead of sending an empty prompt.
    Task 28 will layer Langfuse resolution in front of this with these templates as
    the pinned fallback; callers do not change.
    """
    try:
        return _PROMPTS[name]
    except KeyError:
        raise KeyError(f"no prompt registered under {name!r}") from None

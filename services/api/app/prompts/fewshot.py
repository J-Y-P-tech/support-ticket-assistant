"""Deterministic dynamic few-shot selection for the drafting prompt (plan Task 27 / todo Task 29).

At draft time the agent injects the best recent **approved** replies for the ticket's
category as few-shot examples, so reply quality rises as reps work (SPEC §4.10). This
module is the *selection + rendering* half — pure, with no I/O, model, or database — so
the live lookup that fetches candidate approved replies by category can be added later
and hand its results straight in.

Selection is **deterministic**, the acceptance criterion (SPEC §4.10): the same
candidates always yield the same examples in the same order, independent of the order
they arrive in, so a drafted reply is reproducible and the eval gate (§10) is stable.
The ranking rule (confirmed with the user) is *rating-only*: among approved replies in
the ticket's category, the highest rep rating wins, ties break by most recent, then by
a stable id. An approved-but-unrated reply ranks below any rated one.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FewShotExample(BaseModel):
    """One approved reply eligible to be shown to the model as a few-shot example.

    `message`/`reply` are the customer question and the human-approved answer the model
    should learn the shape of. `category` scopes eligibility to the ticket's category.
    `rating` is the rep's optional score — the ranking signal — and `example_id` is a
    monotonic recency/stability key (higher = more recent) that both breaks rating ties
    and makes the ordering total, so selection is fully deterministic.
    """

    category: str
    message: str
    reply: str
    rating: int | None = None
    example_id: int = Field(ge=0)


def _rank_key(example: FewShotExample) -> tuple[bool, int, int]:
    """Return the sort key placing the best example first under `sorted(reverse=True)`.

    The tuple orders, in priority: rated before unrated (`rating is not None`), then by
    rating, then by recency (`example_id`). With `reverse=True` a higher tuple sorts
    first, so a rated reply outranks an unrated one, a higher rating outranks a lower,
    and — on a rating tie — the more recent (higher `example_id`) wins. `example_id` is
    unique per candidate, so the key is a total order and the result is deterministic
    regardless of input order.
    """
    return (example.rating is not None, example.rating or 0, example.example_id)


def select_examples(
    candidates: list[FewShotExample], *, category: str, limit: int
) -> list[FewShotExample]:
    """Pick the top `limit` approved replies for `category`, best-ranked first.

    Filters `candidates` to the ticket's `category`, ranks them by the rating-only rule
    (higher rating first, ties by most recent, then stable id), and returns at most
    `limit` — the best recent approved replies to inject as few-shot examples (SPEC
    §4.10). A non-positive `limit` selects nothing. Deterministic: the same candidates
    always yield the same result in the same order, however they were ordered on input.
    """
    if limit <= 0:
        return []
    matching = [example for example in candidates if example.category == category]
    ranked = sorted(matching, key=_rank_key, reverse=True)
    return ranked[:limit]


def render_examples(examples: list[FewShotExample]) -> str:
    """Render selected examples into a prompt block, or the empty string when there are none.

    The draft node slots the returned block into the prompt ahead of the customer
    message, so an empty selection must render to `""` — leaving the prompt exactly as
    it was before few-shot was added, with no dangling header. A non-empty selection is
    rendered as a labelled list of example question/approved-reply pairs, each shown as
    the kind of grounded answer the model should produce.
    """
    if not examples:
        return ""
    pairs = "\n\n".join(
        f"Customer message:\n{example.message}\n\nApproved reply:\n{example.reply}"
        for example in examples
    )
    return (
        "Here are examples of well-received approved replies to similar past tickets. "
        "Match their style and grounding; do not reuse their specific facts.\n\n"
        f"{pairs}\n\n"
    )

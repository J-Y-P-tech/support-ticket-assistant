"""Unit tests for the deterministic dynamic few-shot selector (plan Task 27 / todo Task 29).

At draft time the agent injects the best recent **approved** replies for the ticket's
category as few-shot examples, so reply quality rises as reps work (SPEC §4.10). The
selection must be **deterministic** — the same candidates always yield the same
examples in the same order, independent of the order they arrive in — so a drafted
reply is reproducible and the eval gate (§10) is stable.

The ranking rule (confirmed with the user) is *rating-only*: among approved replies in
the ticket's category, the highest rep rating wins, ties break by most recent, then by
a stable id. An approved-but-unrated reply ranks below any rated one. These tests pin
that contract on the pure selector and its renderer — no I/O, no model, no database.
"""

from __future__ import annotations

from app.prompts.fewshot import FewShotExample, render_examples, select_examples


def _example(
    *, category: str = "card_dispute", rating: int | None = None, example_id: int = 1
) -> FewShotExample:
    """Build a candidate example with a fixed message/reply and the given ranking keys.

    The message/reply text is irrelevant to ordering — only `category`, `rating`, and
    `example_id` (a monotonic recency/stability key, higher = more recent) decide the
    rank — so the helper varies only those and keeps the prose constant.
    """
    return FewShotExample(
        category=category,
        message=f"customer question {example_id}",
        reply=f"approved reply {example_id}",
        rating=rating,
        example_id=example_id,
    )


def test_selects_only_matching_category() -> None:
    """Only approved replies in the ticket's category are eligible as examples.

    An example from a different category must never be injected, so the few-shot
    context stays on-topic for the ticket being drafted (SPEC §4.10).
    """
    candidates = [
        _example(category="card_dispute", rating=5, example_id=1),
        _example(category="loan_query", rating=5, example_id=2),
    ]

    selected = select_examples(candidates, category="card_dispute", limit=5)

    assert [e.example_id for e in selected] == [1]


def test_orders_by_rating_descending() -> None:
    """Higher rep rating ranks first — the 'best' half of 'best recent'."""
    candidates = [
        _example(rating=2, example_id=1),
        _example(rating=5, example_id=2),
        _example(rating=3, example_id=3),
    ]

    selected = select_examples(candidates, category="card_dispute", limit=5)

    assert [e.rating for e in selected] == [5, 3, 2]


def test_ties_broken_by_recency() -> None:
    """Equal-rating examples are ordered most-recent-first (higher example_id)."""
    candidates = [
        _example(rating=5, example_id=1),
        _example(rating=5, example_id=3),
        _example(rating=5, example_id=2),
    ]

    selected = select_examples(candidates, category="card_dispute", limit=5)

    assert [e.example_id for e in selected] == [3, 2, 1]


def test_unrated_rank_after_rated() -> None:
    """An approved-but-unrated reply ranks below every rated one (rating-only rule)."""
    candidates = [
        _example(rating=None, example_id=9),
        _example(rating=1, example_id=1),
    ]

    selected = select_examples(candidates, category="card_dispute", limit=5)

    # The low-rated (rating=1) example still outranks the unrated one.
    assert [e.example_id for e in selected] == [1, 9]


def test_limit_caps_the_number_returned() -> None:
    """At most `limit` examples are returned — the top-ranked ones."""
    candidates = [_example(rating=r, example_id=r) for r in (1, 2, 3, 4, 5)]

    selected = select_examples(candidates, category="card_dispute", limit=2)

    # The two highest-rated survive, best first.
    assert [e.rating for e in selected] == [5, 4]


def test_non_positive_limit_returns_no_examples() -> None:
    """A `limit` of zero or less selects nothing rather than raising."""
    candidates = [_example(rating=5, example_id=1)]

    assert select_examples(candidates, category="card_dispute", limit=0) == []
    assert select_examples(candidates, category="card_dispute", limit=-1) == []


def test_selection_is_independent_of_input_order() -> None:
    """The same candidates yield the same result regardless of their arrival order.

    Determinism is the acceptance criterion (SPEC §4.10): a reproducible draft and a
    stable eval gate both require that shuffling the candidate list cannot change which
    examples are chosen or their order.
    """
    candidates = [
        _example(rating=4, example_id=1),
        _example(rating=5, example_id=2),
        _example(rating=4, example_id=3),
    ]

    forward = select_examples(candidates, category="card_dispute", limit=5)
    reversed_ = select_examples(list(reversed(candidates)), category="card_dispute", limit=5)

    assert [e.example_id for e in forward] == [e.example_id for e in reversed_]


def test_render_empty_selection_is_empty_string() -> None:
    """No examples renders to the empty string, so the prompt shows no dangling header.

    The draft template slots the rendered block ahead of the customer message; an empty
    string leaves the prompt exactly as it was before few-shot was added.
    """
    assert render_examples([]) == ""


def test_render_includes_each_example_message_and_reply() -> None:
    """Every selected example's customer message and approved reply appears in the block.

    The rendered block is what the draft node injects into the prompt, so the model
    actually sees each example question and its approved answer.
    """
    examples = [
        _example(rating=5, example_id=1),
        _example(rating=4, example_id=2),
    ]

    block = render_examples(examples)

    for example in examples:
        assert example.message in block
        assert example.reply in block

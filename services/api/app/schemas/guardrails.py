"""Verdicts for the guardrail layer: the input injection screen and the output screen.

Each guard splits the model's raw judgement from the guard's combined result, mirroring
how the `validate` node separates the model's score from the node's outcome:

- The input guard (SPEC §5, §6) pairs `InjectionVerdict` (the LLM classifier's Layer-2
  output) with `InjectionScreenResult` (its combined verdict over both layers).
- The output guard (SPEC §4.6) pairs `ToneVerdict` (the LLM tone classifier's Layer-2
  output) with `OutputScreenResult` (its combined verdict over the deterministic
  forbidden-promise/PII floor and the optional tone layer).

In both cases the guard validates the model's JSON against the *Verdict schema and
retries once, so no unvalidated judgement flows into the flag decision; the *ScreenResult
is the combined verdict the workflow acts on (blocking/routing wired at todo Task 17).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class InjectionVerdict(BaseModel):
    """The LLM second-opinion classifier's structured verdict on a piece of text (Layer 2).

    `is_injection` is the classifier's call; `categories` names the manipulation types
    it found (short snake_case labels) and `evidence` quotes the snippets that show them.
    The lists are empty when `is_injection` is False. The guard validates the model's
    JSON against this schema and retries once on failure — a classifier that never
    returns valid output degrades to the deterministic layer rather than blocking.
    """

    is_injection: bool
    categories: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class InjectionScreenResult(BaseModel):
    """The input guard's combined verdict over both screening layers (SPEC §5 input gate).

    `flagged` is True when either the deterministic signature layer or the optional LLM
    layer caught an injection attempt; a flagged input is blocked and routed to a human
    at workflow assembly (todo Task 17). `categories` and `evidence` are the de-duplicated
    findings of the layer that fired (attack family + the offending snippet), retained for
    the rep and the audit trail. `detector` records which layer caught it — `"rules"`,
    `"llm"`, or `"none"`. The deterministic floor short-circuits, so the two layers are
    mutually exclusive: a signature hit reports `"rules"` without consulting the model.
    """

    flagged: bool
    categories: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    detector: str


class ToneVerdict(BaseModel):
    """The LLM second-opinion classifier's structured verdict on a draft's tone (Layer 2).

    The output guard's LLM layer makes the *subjective* judgement the deterministic floor
    cannot — whether the draft's tone is rude, dismissive, condescending, or blaming.
    `has_violation` is the classifier's call; `categories` names the tone problems it found
    (short snake_case labels) and `evidence` quotes the offending snippets. The lists are
    empty when `has_violation` is False. The guard validates the model's JSON against this
    schema and retries once on failure — a classifier that never returns valid output
    degrades to the deterministic floor rather than flagging every draft.
    """

    has_violation: bool
    categories: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)


class OutputScreenResult(BaseModel):
    """The output guard's combined verdict over both screening layers (SPEC §4.6 output gate).

    `flagged` is True when either the deterministic signature floor (forbidden promises,
    PII leakage) or the optional LLM tone layer caught a violation; a flagged draft is
    surfaced to the rep with warnings at workflow assembly (todo Task 17), never discarded.
    `categories` and `evidence` are the de-duplicated findings of the layer that fired
    (violation type + the offending snippet), retained for the rep and the audit trail —
    with PII evidence **masked to the last four digits** so the verdict never carries a raw
    account/card number (SPEC §5/§7). `detector` records which layer caught it — `"rules"`,
    `"llm"`, or `"none"`. The deterministic floor short-circuits, so the two layers are
    mutually exclusive: a signature hit reports `"rules"` without consulting the model.
    """

    flagged: bool
    categories: list[str] = Field(default_factory=list)
    evidence: list[str] = Field(default_factory=list)
    detector: str

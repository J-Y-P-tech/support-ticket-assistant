"""Verdicts for the guardrail layer: the input injection screen (SPEC §5, §6).

Two shapes, mirroring how the `validate` node splits the model's raw judgement from
the node's combined result:

- `InjectionVerdict` is the **LLM classifier's** structured output (Layer 2) — the
  guard validates the model's JSON against it and retries once, so no unvalidated
  judgement flows into the flag decision.
- `InjectionScreenResult` is the guard's **combined** verdict over both the
  deterministic signature layer and the optional LLM layer — what the workflow acts on.
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

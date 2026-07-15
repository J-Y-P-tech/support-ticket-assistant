"""Turn a finished workflow run into its compliance audit rows (plan Task 24 / todo Task 26).

Task 25 built the audit *store* — an immutable Postgres table and a committing
`record_audit` tool the api reaches over email_mcp. This module is the *emission*
half: it reads a finished (paused) LangGraph state and produces the ordered audit
rows SPEC §7.1 requires — each node's outcome, the source(s) it cited, the model tag
+ prompt version it used, and the guardrail decisions — then writes them through the
client.

The split is deliberate. `build_audit_entries` is a **pure** function: it maps a
state to an ordered `list[AuditEntry]` with no I/O, so every branch of the pipeline
(a clean run, a blocked injection, a triage failure, a no-source hand-off, an
attachment) is unit-testable without a database or a model. `record_node_audits` is
the thin write step the submit-time pipeline trigger calls once the run reaches the
human-review pause; it just forwards each entry through the email_mcp client.

Recording at the hand-off (rather than each node writing its own row mid-run) keeps
the workflow nodes pure — they only ever touch LangGraph state, as they do today — and
mirrors the existing design where persistence happens at the service boundary, not
inside the graph. The rows read back in insertion order, which is pipeline order.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from app.prompts.registry import get_prompt_version

# The actor recorded for every automated node outcome. Rep actions carry the rep's
# marker instead; these rows are the system's own work (SPEC §7.1 actor + timestamp).
_SYSTEM_ACTOR = "system"


@dataclass(frozen=True)
class AuditEntry:
    """One row destined for a ticket's immutable audit trail (SPEC §7.1).

    `event` is the outcome name (for example `"triaged"` or `"retrieved"`); `actor`
    is who caused it (`"system"` for a node outcome); `detail` is the per-event
    evidence — cited sources, the model tag + prompt version, or a guardrail decision
    — or `None` when the event name alone says everything (like a no-source hand-off).
    Frozen because a built entry is a fact about a run that happened; nothing should
    mutate it before it is written.
    """

    event: str
    actor: str
    detail: dict[str, Any] | None = None


class _AuditRecorder(Protocol):
    """The one method `record_node_audits` needs from the email_mcp client.

    Typed structurally so the write step depends only on `record_audit`, not on the
    whole `EmailMCPClient` — the tests pass a lightweight recording fake.
    """

    async def record_audit(
        self,
        ticket_id: int,
        event: str,
        *,
        actor: str | None = None,
        detail: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Append one immutable audit entry for a ticket and return the stored row."""
        ...


def _guardrail_detail(screen: Any, *, prompt_name: str, model: str) -> dict[str, Any]:
    """Build the audit detail for a guardrail screen (input or output).

    Records the guardrail *decision* — which layer fired (`detector`) and whether it
    flagged — and the categories it found, if any. Only when the LLM layer actually
    fired the decision (`detector == "llm"`) does the row attribute the model and the
    guard's prompt version: on a clean pass or a deterministic-rules hit no model
    verdict drove the outcome, so there is nothing to attribute there.
    """
    # The core guardrail decision: which layer caught it, and whether it flagged.
    detail: dict[str, Any] = {"detector": screen.detector, "flagged": screen.flagged}
    # Name the violation/attack families the layer found, when it found any.
    if screen.categories:
        detail["categories"] = list(screen.categories)
    # Attribute the model + prompt only when the LLM layer fired the decision.
    if screen.detector == "llm":
        detail["model"] = model
        detail["prompt_version"] = get_prompt_version(prompt_name)
    return detail


def build_audit_entries(state: Mapping[str, Any], *, model: str) -> list[AuditEntry]:
    """Map a finished workflow state to its ordered compliance audit rows (SPEC §7.1).

    Reads the pipeline products a run left in `state` and returns one `AuditEntry` per
    node outcome, in pipeline order, so the trail reads as the case ran. `model` is the
    host model tag recorded on every model-using node's row. The function follows the
    same branches the workflow does: a flagged injection ends the trail at the block; a
    triage that could not classify ends it at `triage_failed`; a no-confident-source
    retrieval ends it at the hand-off — none of these reach a draft, so no later rows
    are invented. Pure: it performs no I/O and mutates nothing.
    """
    entries: list[AuditEntry] = []

    # --- Input guardrail (always the first node to run) ---
    screen = state.get("injection_screen")
    if screen is not None:
        # Record the input-gate decision (SPEC §7.1 guardrail decisions).
        entries.append(
            AuditEntry(
                "input_screened",
                _SYSTEM_ACTOR,
                _guardrail_detail(screen, prompt_name="input_guard", model=model),
            )
        )
        # A flagged injection routes straight to the human gate — nothing else ran.
        if screen.flagged:
            entries.append(
                AuditEntry(
                    "injection_blocked", _SYSTEM_ACTOR, {"categories": list(screen.categories)}
                )
            )
            return entries

    # --- Digitization (only when an attachment was transcribed + extracted) ---
    if state.get("extracted_facts"):
        # Document that a customer document was digitized, and with which model/prompt —
        # not its raw text, which would put attachment PII into the compliance row.
        entries.append(
            AuditEntry(
                "attachment_extracted",
                _SYSTEM_ACTOR,
                {"model": model, "prompt_version": get_prompt_version("extract")},
            )
        )

    # --- Triage ---
    triage = state.get("triage")
    if triage is None:
        # Triage could not classify within its retries — the case went to a human.
        entries.append(
            AuditEntry(
                "triage_failed",
                _SYSTEM_ACTOR,
                {"model": model, "prompt_version": get_prompt_version("triage")},
            )
        )
        return entries
    entries.append(
        AuditEntry(
            "triaged",
            _SYSTEM_ACTOR,
            {
                "category": triage.category.value,
                "urgency": triage.urgency.value,
                "sentiment": triage.sentiment.value,
                "model": model,
                "prompt_version": get_prompt_version("triage"),
            },
        )
    )

    # --- Retrieval ---
    kb_result = state.get("kb_result")
    if kb_result is not None:
        if kb_result.no_confident_source or not kb_result.sources:
            # No confident source — the case is handed to a human for research, no draft.
            entries.append(AuditEntry("no_confident_source", _SYSTEM_ACTOR))
            return entries
        # Record the sources the reply is grounded in (SPEC §7.1 "the source(s) cited").
        entries.append(
            AuditEntry(
                "retrieved",
                _SYSTEM_ACTOR,
                {
                    "sources": [{"id": s.id, "title": s.title} for s in kb_result.sources],
                    "count": len(kb_result.sources),
                },
            )
        )

    # --- Draft ---
    draft = state.get("draft")
    if draft is not None:
        # The drafted reply: the model/prompt that wrote it, its citations, and whether
        # it stayed verified (grounded in its sources).
        detail: dict[str, Any] = {
            "model": model,
            "prompt_version": get_prompt_version("draft"),
            "citations": [{"source_id": c.source_id, "title": c.title} for c in draft.citations],
            "verified": draft.verified,
        }
        # A leaked reasoning trace is surfaced on the row too (the rep decides what to do).
        if state.get("trace_leak"):
            detail["trace_leak"] = True
        entries.append(AuditEntry("drafted", _SYSTEM_ACTOR, detail))

    # --- Validation ---
    validation = state.get("validation")
    if validation is not None:
        # The groundedness score and whether the draft was flagged unverified.
        entries.append(
            AuditEntry(
                "validated",
                _SYSTEM_ACTOR,
                {
                    "score": validation.groundedness,
                    "flagged": validation.flagged,
                    "model": model,
                    "prompt_version": get_prompt_version("validate"),
                },
            )
        )

    # --- Output guardrail ---
    output_screen = state.get("output_screen")
    if output_screen is not None:
        # Record the output-gate decision (SPEC §7.1 guardrail decisions).
        entries.append(
            AuditEntry(
                "output_screened",
                _SYSTEM_ACTOR,
                _guardrail_detail(output_screen, prompt_name="output_guard", model=model),
            )
        )

    return entries


async def record_node_audits(
    email: _AuditRecorder, *, ticket_id: int, state: Mapping[str, Any], model: str
) -> None:
    """Write the node-outcome audit rows for a finished run through the email_mcp client.

    The thin write step behind the submit-time pipeline trigger: it builds the ordered
    entries for `state` and records each one under `ticket_id`, so the ticket's
    immutable trail gains the node history once the run reaches the human-review pause.
    Entries are written in order, and email_mcp reads them back in insertion order.
    """
    for entry in build_audit_entries(state, model=model):
        await email.record_audit(ticket_id, entry.event, actor=entry.actor, detail=entry.detail)

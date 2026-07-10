"""LangGraph workflow assembly + human interrupt + checkpointer (plan Task 16 / todo Task 17).

This is where the plain node functions (triage, retrieve/gate, draft, validate) and
the two guards — each written independent of LangGraph — are wired into the actual
`StateGraph` the SPEC §5 pipeline describes, compiled with a checkpointer and a
**human interrupt before `human_review`**.

`build_workflow` closes over the run-time dependencies (the LLM backend, the KB
client, and config) and returns a compiled graph. The caller injects the
checkpointer: production passes the Postgres saver (`langgraph-checkpoint-postgres`,
so a case pending review for days survives a process restart); tests pass the
in-memory saver, so CI never needs a database and never touches Ollama (SPEC §10/§12).

**The human gate is the whole point.** The graph is compiled with
`interrupt_before=["human_review"]`, so a run stops the moment the automated pipeline
is done and *before* anything customer-facing is decided. The rep's decision is
written into the paused state out of band (the rep-action routes, plan Task 17) and
the graph is resumed. `finalize` — the only node that can set `Resolved` / produce a
customer reply — reads that decision and **fails closed if it is missing**, so there
is no code path from intake to a sent reply that does not carry an explicit rep
action (SPEC §10 safety invariant).

Every path converges on `human_review`: a clean draft, a no-confident-source
hand-off, a blocked prompt injection, and a triage that could not classify all land
in front of a human rather than flowing on unchecked.
"""

from __future__ import annotations

from typing import Any, Protocol

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings
from app.graph.nodes.draft import draft as draft_reply
from app.graph.nodes.retrieve import groundedness_gate, retrieve
from app.graph.nodes.triage import TriageValidationError, triage
from app.graph.nodes.validate import validate
from app.graph.state import WorkflowState
from app.guardrails.injection import screen_input
from app.guardrails.output import screen_output
from app.llm.base import LLM
from app.llm.thinking import contains_thinking_trace
from app.schemas.enums import FeedbackDecision, TicketStatus


class KBSearcher(Protocol):
    """The slice of the KB client the workflow needs: a single `search` call.

    Typing the dependency structurally (rather than against the concrete
    `KBMCPClient`) lets the workflow tests pass a lightweight in-memory fake without
    subclassing the real client or standing up kb_mcp.
    """

    async def search(self, query: str, limit: int | None = None) -> Any:
        """Search the knowledge base for `query` and return a `KBSearchResult`."""
        ...


def build_workflow(
    *,
    llm: LLM,
    kb_client: KBSearcher,
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
) -> CompiledStateGraph:
    """Assemble and compile the support-ticket workflow with the human interrupt.

    Wires the SPEC §5 pipeline into a `StateGraph`, closing over `llm`, `kb_client`,
    and `settings` (the per-node retry/threshold knobs come from config, not code), and
    compiles it with `checkpointer` and `interrupt_before=["human_review"]`. The
    returned graph runs the automated steps and then **pauses before the rep gate**;
    resuming it — after a decision is written to state — runs `finalize`, which is the
    only node that can resolve the case and refuses to do so without that decision.
    """

    async def screen_input_node(state: WorkflowState) -> dict[str, Any]:
        """Screen the customer message for prompt injection before any model node.

        Runs the layered input guard (deterministic floor first, LLM second opinion
        only if the floor is clean). A hit routes straight to the human gate; the
        model never sees the attacker's text, because the floor short-circuits.
        """
        result = await screen_input(state["message"], llm)
        return {"injection_screen": result}

    def route_after_input(state: WorkflowState) -> str:
        """Route a flagged injection to the block node, otherwise on to triage."""
        return "block_injection" if state["injection_screen"].flagged else "triage"

    async def block_injection_node(state: WorkflowState) -> dict[str, Any]:
        """Hand a blocked injection to a human, flagged, with nothing sent onward."""
        screen = state["injection_screen"]
        found = ", ".join(screen.categories) if screen.categories else "suspicious input"
        return {
            "status": TicketStatus.NEEDS_RESEARCH,
            "flags": [f"prompt injection blocked ({found}); routed to a human rep"],
        }

    async def triage_node(state: WorkflowState) -> dict[str, Any]:
        """Classify the ticket, or hand it to a human if it cannot be classified.

        On success stores the validated `TriageResult` and marks the case Triaged. On
        `TriageValidationError` (the model never returned a valid classification within
        its retry budget) it flags the case for a human instead of guessing a category.
        """
        try:
            result = await triage(
                state["message"],
                llm,
                max_attempts=settings.triage_max_attempts,
                extracted_facts=state.get("extracted_facts"),
            )
        except TriageValidationError as exc:
            return {
                "status": TicketStatus.NEEDS_RESEARCH,
                "flags": [f"triage could not classify this ticket: {exc}"],
            }
        return {"triage": result, "status": TicketStatus.TRIAGED}

    def route_after_triage(state: WorkflowState) -> str:
        """Continue to retrieval on a good triage; divert to the human gate otherwise."""
        return "retrieve" if state.get("triage") is not None else "human_review"

    async def retrieve_node(state: WorkflowState) -> dict[str, Any]:
        """Search the KB with the fused query (the message, for text-only tickets)."""
        query = state.get("search_query") or state["message"]
        result = await retrieve(query, kb_client)  # type: ignore[arg-type]
        return {"kb_result": result, "status": TicketStatus.RESEARCHING}

    def route_after_retrieve(state: WorkflowState) -> str:
        """Draft only with a confident source in hand; otherwise route to research."""
        return groundedness_gate(state["kb_result"])

    async def flag_needs_research_node(state: WorkflowState) -> dict[str, Any]:
        """Hand a no-confident-source case to a human for research — never a draft."""
        return {
            "status": TicketStatus.NEEDS_RESEARCH,
            "flags": [
                "no confident knowledge-base source found; routed to a human rep for research"
            ],
        }

    async def draft_node(state: WorkflowState) -> dict[str, Any]:
        """Draft a grounded, cited reply; flag (never strip) a leaked reasoning trace.

        Writes the draft and marks the case Drafted. If the reply carries a leaked
        reasoning trace it is surfaced to the rep via a flag and `trace_leak`, with the
        text left intact — the rep, not the pipeline, decides what to do (Task 10 note).
        """
        drafted = await draft_reply(state["message"], state["kb_result"], llm)
        leaked = contains_thinking_trace(drafted.body)
        update: dict[str, Any] = {
            "draft": drafted,
            "trace_leak": leaked,
            "status": TicketStatus.DRAFTED,
        }
        if leaked:
            update["flags"] = ["the drafted reply leaked a reasoning trace; review before sending"]
        return update

    async def validate_node(state: WorkflowState) -> dict[str, Any]:
        """Score the draft's groundedness and flag it unverified when it falls short.

        Never blocks the flow (SPEC §5): a flagged draft still reaches the rep, carrying
        the validation result, its (possibly downgraded) `verified` flag, and the
        plain-language reasons for review.
        """
        result = await validate(
            state["draft"],
            state["kb_result"],
            llm,
            groundedness_min=settings.groundedness_min,
            max_attempts=settings.validate_max_attempts,
        )
        update: dict[str, Any] = {"validation": result, "draft": result.draft}
        if result.flagged:
            update["flags"] = [f"validation: {reason}" for reason in result.reasons]
        return update

    async def screen_output_node(state: WorkflowState) -> dict[str, Any]:
        """Screen the draft for forbidden promises / PII / tone before the rep sees it.

        A flagged draft is surfaced with warnings (never discarded); PII evidence is
        already masked by the guard, so the flags never carry a raw account/card number.
        """
        result = await screen_output(state["draft"].body, llm)
        update: dict[str, Any] = {"output_screen": result}
        if result.flagged:
            update["flags"] = [f"output guard flagged {category}" for category in result.categories]
        return update

    async def human_review_node(state: WorkflowState) -> dict[str, Any]:
        """The human gate boundary — runs only after the rep resumes the paused graph.

        The interrupt fires *before* this node, so by the time it executes the rep has
        acted and their decision is in state. It is a pure boundary between the
        automated pipeline and `finalize`; the decision itself is applied by `finalize`.
        """
        return {}

    async def finalize_node(state: WorkflowState) -> dict[str, Any]:
        """Apply the rep's decision — the only node that can resolve or reply.

        Fails closed: with no `rep_decision` in state it raises, so a graph resumed past
        the human gate without an explicit rep action cannot fabricate a send (SPEC §10
        safety invariant). A rejection routes back to research with no reply; an approval
        (edited or as-is) resolves the case with the rep's edited text or the AI draft as
        the customer reply. Persisting that reply, the audit record, and the feedback /
        training rows is wired in the rep-action routes (plan Task 17); this node owns the
        state transition that gates them.
        """
        decision = state.get("rep_decision")
        if decision is None:
            raise RuntimeError(
                "finalize reached without a rep decision — the human approval gate was bypassed"
            )
        if decision == FeedbackDecision.REJECTED:
            return {"status": TicketStatus.NEEDS_RESEARCH, "final_reply": None}
        drafted = state.get("draft")
        # or returns the first "truthy" value
        reply = state.get("rep_edited_reply") or (drafted.body if drafted is not None else None)
        return {"status": TicketStatus.RESOLVED, "final_reply": reply}

    graph = StateGraph(WorkflowState)
    graph.add_node("screen_input", screen_input_node)
    graph.add_node("block_injection", block_injection_node)
    graph.add_node("triage", triage_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("flag_needs_research", flag_needs_research_node)
    graph.add_node("draft", draft_node)
    graph.add_node("validate", validate_node)
    graph.add_node("screen_output", screen_output_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("finalize", finalize_node)

    graph.add_edge(START, "screen_input")
    graph.add_conditional_edges(
        "screen_input",
        route_after_input,
        {"block_injection": "block_injection", "triage": "triage"},
    )
    graph.add_edge("block_injection", "human_review")
    graph.add_conditional_edges(
        "triage",
        route_after_triage,
        {"retrieve": "retrieve", "human_review": "human_review"},
    )
    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {"draft": "draft", "flag_needs_research": "flag_needs_research"},
    )
    graph.add_edge("flag_needs_research", "human_review")
    graph.add_edge("draft", "validate")
    graph.add_edge("validate", "screen_output")
    graph.add_edge("screen_output", "human_review")
    graph.add_edge("human_review", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile(checkpointer=checkpointer, interrupt_before=["human_review"])

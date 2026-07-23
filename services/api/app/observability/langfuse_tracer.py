"""The real Langfuse adapter behind the `Tracer` seam (plan Task 29 / todo Task 31).

`graph/trace.py` defines the PII-safe `TicketTrace`/`TraceScore` and the two-method
`Tracer` protocol; this module is the concrete production implementation that pushes a
trace to the self-hosted Langfuse service (SPEC §7.2). It is deliberately the *only*
module that imports the `langfuse` SDK, imported lazily by `graph.runtime.build_tracer`
so the rest of the app — and every unit test, and offline CI — never needs the SDK on
the path: a run without Langfuse falls back to `NoOpTracer` and works unchanged
(SPEC §10/§12).

Two design choices keep this thin and safe:

- **The builder already redacted everything.** `emit` sends `trace.as_dict()`, which the
  pure builder produced PII-free (same `redact_pii` as the logs), so this adapter adds
  no redaction of its own — it only translates the built trace into SDK calls.
- **Observability must never break the desk.** Every SDK call runs on a worker thread
  (`asyncio.to_thread`, so the sync SDK never blocks the event loop) and is wrapped so a
  Langfuse hiccup degrades to "untraced" (an empty trace id) rather than propagating —
  consistent with the fail-safe submit-time trigger. Losing a trace is acceptable;
  breaking a customer's ticket is not.

Pinned to the Langfuse v2 SDK/server pair (`langfuse>=2,<3` and the `langfuse:2` compose
image), whose ingestion API — `client.trace(...)` → `.span(...)`, `client.score(...)`,
`client.flush()` — is stable; the self-hosted v2 server needs only its own Postgres, no
ClickHouse/Redis/object-store sidecars, matching the local-only posture (SPEC §7.2).
"""

from __future__ import annotations

import asyncio
import logging

from app.graph.trace import TicketTrace, TraceScore

_logger = logging.getLogger(__name__)


class LangfuseTracer:
    """A `Tracer` that emits ticket traces to a self-hosted Langfuse (SPEC §7.2).

    Constructed by `graph.runtime.build_tracer` with the host + key pair from config;
    the constructor is where the `langfuse` SDK is imported, so a missing SDK surfaces
    at build time (and the factory falls back to `NoOpTracer`) rather than mid-run.
    """

    def __init__(self, *, host: str, public_key: str, secret_key: str) -> None:
        """Build the underlying Langfuse client from the configured host and keys."""
        from langfuse import Langfuse

        self._client = Langfuse(host=host, public_key=public_key, secret_key=secret_key)

    async def emit(self, trace: TicketTrace) -> str:
        """Send a ticket's redacted trace to Langfuse and return its trace id.

        Runs the sync SDK on a worker thread. Returns the created trace's id, or an empty
        string if Langfuse could not be reached — so the caller persists an id only when
        there truly is one, and a failure never breaks the pipeline.
        """
        return await asyncio.to_thread(self._emit_sync, trace)

    def _emit_sync(self, trace: TicketTrace) -> str:
        """Create the trace, its spans, and its initial scores; flush and return the id."""
        try:
            handle = self._client.trace(
                # Use the customer-facing TKT-#### code as the trace id so a rep can find a
                # ticket's trace by typing its reference code into the dashboard's search box
                # (which matches on id, not tags). Re-emitting the same ticket upserts.
                id=trace.reference_code,
                name=trace.name,
                input=trace.input,
                metadata=trace.metadata,
                tags=[f"ticket-{trace.ticket_id}"],
            )
            for span in trace.spans:
                handle.span(name=span.name, metadata=span.detail, output=span.output)
            self._score_all(handle.id, trace.scores)
            self._client.flush()
            return str(handle.id)
        except Exception:
            # Observability is best-effort: a Langfuse outage must not fail the ticket.
            _logger.exception("langfuse trace emission failed; ticket left untraced")
            return ""

    async def add_scores(self, trace_id: str, scores: list[TraceScore]) -> None:
        """Attach quality scores to an already-emitted trace by id (best-effort)."""
        await asyncio.to_thread(self._add_scores_sync, trace_id, scores)

    def _add_scores_sync(self, trace_id: str, scores: list[TraceScore]) -> None:
        """Push each score under the trace id and flush; swallow any Langfuse error."""
        try:
            self._score_all(trace_id, scores)
            self._client.flush()
        except Exception:
            _logger.exception("langfuse score attachment failed for trace %s", trace_id)

    async def set_user(self, trace_id: str, user_id: str) -> None:
        """Record the handling rep as the trace's user by id (best-effort)."""
        await asyncio.to_thread(self._set_user_sync, trace_id, user_id)

    def _set_user_sync(self, trace_id: str, user_id: str) -> None:
        """Upsert the trace's `user_id` by re-referencing its id, and flush.

        Emitting a trace event carrying only the id and `user_id` merges the user onto
        the already-emitted trace without disturbing its name, input, or spans — so the
        dashboard's User column names the rep who dispositioned the case. Swallows any
        Langfuse error: attribution is best-effort and must never fail the rep action.
        """
        try:
            self._client.trace(id=trace_id, user_id=user_id)
            self._client.flush()
        except Exception:
            _logger.exception("langfuse user attribution failed for trace %s", trace_id)

    def _score_all(self, trace_id: str, scores: list[TraceScore]) -> None:
        """Write each score under a deterministic id so a re-send upserts, not duplicates.

        A score's name is unique per trace (one `groundedness`, one `rating`, ...), so a
        stable `<trace>-<name>` id means the pause-time scores emitted with the trace and
        the same-named scores re-sent when the rep acts update in place rather than piling
        up duplicates in the dashboard.
        """
        for score in scores:
            self._client.score(
                id=f"{trace_id}-{score.name}",
                trace_id=trace_id,
                name=score.name,
                value=score.value,
                comment=score.comment,
            )

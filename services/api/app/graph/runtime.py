"""Per-app workflow runtime: the compiled graph the rep-action routes resume (plan Task 17).

The rep-action routes (edit/approve/reject/send) resume a *paused* LangGraph run, so
they need the app's compiled workflow. `get_workflow` is the FastAPI dependency that
supplies it — a lazily built, process-wide singleton cached on `app.state`, mirroring
the MCP-client dependencies.

Production wires the real backing store: the workflow is compiled with LangGraph's
**Postgres** checkpointer (`AsyncPostgresSaver`), so a case pending human review survives
a process restart and resumes days later (SPEC §3/§5). The saver is an async context
manager; it (and the Ollama HTTP client) are entered on an `AsyncExitStack` kept on
`app.state`, which the app lifespan closes on shutdown. Building lazily on first use —
rather than in the lifespan — keeps the app constructable without Postgres/Ollama, and
lets tests override this dependency with an in-memory-checkpointer graph so CI touches
neither (SPEC §10/§12).
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import cast

from fastapi import Depends, FastAPI, Request
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph.state import CompiledStateGraph

from app.config import Settings, get_settings
from app.graph.trace import NoOpTracer, Tracer
from app.graph.workflow import build_workflow
from app.llm.ollama import OllamaLLM
from app.mcp_clients.email import email_client_for_app
from app.mcp_clients.kb import KBMCPClient


async def get_workflow(
    request: Request, settings: Settings = Depends(get_settings)
) -> CompiledStateGraph:
    """FastAPI dependency: return the process-wide compiled workflow.

    A thin dependency wrapper over `get_workflow_for_app`, so the rep-action routes
    resolve the same cached graph the submit-time pipeline trigger uses. Overridden in
    tests with a graph compiled against an in-memory checkpointer, so route tests never
    open a database or reach Ollama.
    """
    return await get_workflow_for_app(request.app, settings)


async def get_workflow_for_app(app: FastAPI, settings: Settings) -> CompiledStateGraph:
    """Return the process-wide compiled workflow, building it on first use.

    Built once (Postgres checkpointer + Ollama LLM + KB client) and cached on
    `app.state`. Shared by the rep-action routes (via `get_workflow`) and the
    submit-time background trigger (`app.graph.intake`), so a ticket's automated run
    and the rep's later resume drive the *same* graph and checkpointer.
    """
    workflow = getattr(app.state, "workflow", None)
    if workflow is None:
        workflow = await _build_app_workflow(app, settings)
        app.state.workflow = workflow
    return cast("CompiledStateGraph", workflow)


async def _build_app_workflow(app: FastAPI, settings: Settings) -> CompiledStateGraph:
    """Compile the production workflow with the Postgres checkpointer and host LLM.

    The Postgres saver and the LLM's HTTP client are registered on an `AsyncExitStack`
    cached on `app.state.runtime_stack`, so the app lifespan tears both down cleanly on
    shutdown. `checkpointer.setup()` creates LangGraph's tables on first run if absent.
    """
    stack = getattr(app.state, "runtime_stack", None)
    if stack is None:
        stack = AsyncExitStack()
        app.state.runtime_stack = stack

    checkpointer = await stack.enter_async_context(
        AsyncPostgresSaver.from_conn_string(settings.database_url)
    )
    await checkpointer.setup()

    llm = OllamaLLM(model=settings.llm_model, base_url=settings.ollama_base_url)
    stack.push_async_callback(llm.aclose)

    kb_client = _shared_kb_client(app, settings)
    email_client = email_client_for_app(app, settings)
    return build_workflow(
        llm=llm,
        kb_client=kb_client,
        email_client=email_client,
        settings=settings,
        checkpointer=checkpointer,
    )


def build_tracer(settings: Settings) -> Tracer:
    """Return the Langfuse tracer, or the no-op fallback when Langfuse is unavailable.

    Lazily imports the SDK-backed `LangfuseTracer` (the only module that touches the
    `langfuse` SDK) and builds it from config. Any failure — the SDK not installed
    (offline CI), or the client refusing to construct — falls back to `NoOpTracer`, so a
    run without Langfuse traces to nothing and works unchanged (SPEC §10/§12). Emitting
    itself is best-effort inside the adapter, so this only guards *construction*.
    """
    try:
        from app.observability.langfuse_tracer import LangfuseTracer

        return LangfuseTracer(
            host=settings.langfuse_host,
            public_key=settings.langfuse_public_key.get_secret_value(),
            secret_key=settings.langfuse_secret_key.get_secret_value(),
        )
    except Exception:
        return NoOpTracer()


def get_tracer_for_app(app: FastAPI, settings: Settings) -> Tracer:
    """Return the process-wide tracer, building it on first use and caching it.

    Shared by the submit-time pipeline trigger (which emits a ticket's trace) and the
    rep-action routes (which attach feedback scores), so both write to the same Langfuse
    client. Cached on `app.state` like the MCP clients.
    """
    tracer = getattr(app.state, "tracer", None)
    if tracer is None:
        tracer = build_tracer(settings)
        app.state.tracer = tracer
    return cast("Tracer", tracer)


def get_tracer(request: Request, settings: Settings = Depends(get_settings)) -> Tracer:
    """FastAPI dependency: return the process-wide tracer for the rep-action routes.

    A thin request-scoped wrapper over `get_tracer_for_app`; overridden in tests only
    when a suite wants to assert what was traced (most default to the no-op fallback).
    """
    return get_tracer_for_app(request.app, settings)


def _shared_kb_client(app: FastAPI, settings: Settings) -> KBMCPClient:
    """Return the app's shared `KBMCPClient`, building and caching it if absent.

    Reuses (or seeds) the same `app.state.kb_client` the `get_kb_client` dependency
    caches and the lifespan closes, so retrieval during a run and any direct KB route
    share one session.
    """
    client = getattr(app.state, "kb_client", None)
    if client is None:
        client = KBMCPClient(
            url=settings.kb_mcp_url,
            token=settings.kb_mcp_token.get_secret_value(),
            default_limit=settings.kb_search_limit,
        )
        app.state.kb_client = client
    return cast("KBMCPClient", client)

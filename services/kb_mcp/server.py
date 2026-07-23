"""kb_mcp — the Knowledge-Base MCP server (SPEC §4.4, §14.1).

Exposes one tool, `search_knowledge_base`, over a pluggable `KBProvider`. Ships
the `MockKBProvider` (keyword lookup over curated canned answers); swapping the
`_provider` line is all it takes to plug a real embedding/API backend — the tool
contract and the agent are unchanged. Served over streamable-HTTP so the api (a
separate container) can reach it over the network (SPEC §6).

Like email_mcp, this image is self-contained: the tool logic lives in `kb_search`
and `providers/`, and the module carries no `app` (shared api) import — the MCP
boundary carries plain JSON. Inter-service auth enforcement lands in Task 23; this
task establishes the tool surface and its behaviour.
"""

from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP

import kb_search
from providers.mock_kb import MockKBProvider

# Bind and path are configurable; the defaults match `.env.example`
# (KB_MCP_URL points the api at this service). Served at /mcp like email_mcp.
mcp = FastMCP(
    "kb_mcp",
    # Bind all interfaces: a containerized service must listen on 0.0.0.0 so the api
    # container can reach it across the Compose network (overridable via KB_MCP_HOST).
    host=os.environ.get("KB_MCP_HOST", "0.0.0.0"),  # nosec B104
    port=int(os.environ.get("KB_MCP_PORT", "8000")),
    streamable_http_path="/mcp",
)

# The active provider. Swap this one line to plug a different KBProvider (SPEC
# §4.4); the tool below and the agent do not change.
_provider = MockKBProvider()

# How many ranked sources a search returns by default: a handful keeps a draft
# grounded in the strongest matches without drowning it in near-misses. Read from
# the shared `KB_SEARCH_LIMIT` env var (same value the api uses) so the two sides
# can't drift; the literal is only a fallback when the var is unset.
DEFAULT_LIMIT = int(os.environ.get("KB_SEARCH_LIMIT", "3"))


@mcp.tool()
def search_knowledge_base(query: str, limit: int = DEFAULT_LIMIT) -> dict[str, Any]:
    """Search the knowledge base for sources that can ground a reply.

    `query` is the fused customer-question + attachment-summary search string
    (SPEC §4.2/§4.4). Returns ranked sources (`id`, `title`, `text`) plus
    `no_confident_source`: True when nothing matched, which routes the case to
    needs-human-research (SPEC §4.4).
    """
    return kb_search.run_search(_provider, query, limit=limit)


def main() -> None:
    """Run the MCP server over streamable-HTTP (SPEC §6: api↔MCP over HTTP)."""
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

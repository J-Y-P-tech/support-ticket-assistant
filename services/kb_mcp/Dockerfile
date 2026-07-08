# kb_mcp image — the Knowledge-Base MCP connector (SPEC §4.4, §8). Runtime deps are
# mcp only (no DB, no RAG/vectors); the curated mock_kb answers ship in the image.
# Like email_mcp it stays self-contained — the shared `app` package is NOT needed
# at runtime (the MCP boundary carries plain JSON).
FROM python:3.12-slim

WORKDIR /app

# uv drives dependency install (project decision Q1); --frozen honours uv.lock.
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY services/kb_mcp ./services/kb_mcp

# kb_mcp imports as top-level modules (server.py, kb_search.py) plus the
# `providers` package.
ENV PYTHONPATH=/app/services/kb_mcp

# Streamable-HTTP MCP endpoint (SPEC §6): the api reaches it at :8000/mcp.
EXPOSE 8000

CMD ["uv", "run", "python", "services/kb_mcp/server.py"]

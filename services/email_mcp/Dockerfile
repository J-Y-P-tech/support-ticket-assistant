# email_mcp image — the only container with ticket-DB credentials (SPEC §8, §6).
# Runtime deps are psycopg + mcp only; the shared api package is NOT needed at
# runtime (the MCP boundary carries plain JSON), so this image stays self-contained.
FROM python:3.12-slim

WORKDIR /app

# uv drives dependency install (project decision Q1); --frozen honours uv.lock.
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY services/email_mcp ./services/email_mcp

# email_mcp modules (db.py, server.py) import as top-level modules.
ENV PYTHONPATH=/app/services/email_mcp

# Streamable-HTTP MCP endpoint (SPEC §6): the api reaches it at :8000/mcp.
EXPOSE 8000

CMD ["uv", "run", "python", "services/email_mcp/server.py"]

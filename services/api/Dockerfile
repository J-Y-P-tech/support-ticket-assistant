# api image — the FastAPI backend (SPEC §8). Holds the frontend->api token and the
# api->MCP tokens; it reaches email_mcp/kb_mcp over HTTP and never talks to the DB
# directly (email_mcp is the sole DB owner). The shared `app` package lives here.
FROM python:3.12-slim

WORKDIR /app

# curl is used by the compose healthcheck to probe /health (kept minimal otherwise).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# uv drives dependency install (project decision Q1); --frozen honours uv.lock.
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY services/api ./services/api

# api modules import under the `app` package (e.g. `from app.routes import ...`).
ENV PYTHONPATH=/app/services/api

# ASGI port (SPEC §8): the frontend reaches it at :8000 (API_BASE_URL=http://api:8000).
EXPOSE 8000

CMD ["uv", "run", "uvicorn", "app.main:create_app", "--factory", \
     "--host", "0.0.0.0", "--port", "8000"]

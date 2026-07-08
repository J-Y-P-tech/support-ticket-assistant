# frontend image — Streamlit UI (SPEC §8). Talks only to the api over HTTP; it
# holds no DB or MCP credentials, only the frontend->api bearer token (from env).
FROM python:3.12-slim

WORKDIR /app

# uv drives dependency install (project decision Q1); --frozen honours uv.lock.
RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY services/frontend ./services/frontend

# frontend modules (api_client, config, formatting, views) import as top-level.
ENV PYTHONPATH=/app/services/frontend

# Streamlit's default port; the app is reached at :8501.
EXPOSE 8501

CMD ["uv", "run", "streamlit", "run", "services/frontend/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]

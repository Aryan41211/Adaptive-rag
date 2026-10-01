# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage: install dependencies into a virtualenv that is copied into the
# runtime image, so build toolchains never reach production.
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.lock.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install -r requirements.lock.txt

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Run as an unprivileged user: a container compromise should not be root.
RUN useradd --create-home --uid 10001 appuser

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser streamlit_app/ ./streamlit_app/

USER appuser

EXPOSE 8000

# Render injects a $PORT that is not 8000; the healthcheck and the server read
# it from the environment so no config surgery is needed at runtime.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD-SHELL python -c "import os,urllib.request,sys; port=os.environ.get('PORT', '8000'); sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{port}/healthz', timeout=4).status==200 else 1)"

# One worker by default; see README for when more are safe.
#
# Stale multiprocess metric files are cleared first when PROMETHEUS_MULTIPROC_DIR
# is set. That directory holds per-process counter shards on disk, and
# prometheus_client never deletes them: left alone, every restart re-adds the
# previous run's totals and the counters climb without bound. Nothing is
# scraped at all while this project runs one worker, so the find is a no-op
# unless the variable is actually configured.
#
# --no-proxy-headers is a security control, not a default: with proxy headers
# enabled uvicorn rewrites request.client.host from a client-supplied
# X-Forwarded-For before the app sees it, which lets a caller choose their own
# rate-limit key. src.api.ratelimit makes that decision from the raw socket
# peer instead, against TRUSTED_PROXIES.
#
# The shell form expands ${PORT:-8000}; exec keeps uvicorn as PID 1 so it
# receives SIGTERM directly and shuts down gracefully.
CMD ["sh", "-c", "if [ -n \"${PROMETHEUS_MULTIPROC_DIR:-}\" ] && [ -d \"${PROMETHEUS_MULTIPROC_DIR}\" ]; then find \"${PROMETHEUS_MULTIPROC_DIR}\" -name '*.db' -delete; fi; exec uvicorn src.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1 --no-proxy-headers"]

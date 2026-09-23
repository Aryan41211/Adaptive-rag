# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# Build stage: install dependencies into a virtualenv that is copied into the
# runtime image, so build toolchains never reach production.
# ---------------------------------------------------------------------------
FROM python:3.14-slim AS builder

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
FROM python:3.14-slim AS runtime

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

# One worker by default; see README for when more are safe. The shell form
# expands ${PORT:-8000} while exec keeps uvicorn as PID 1 for signal handling.
CMD ["sh", "-c", "exec uvicorn src.main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --workers 1 --proxy-headers --forwarded-allow-ips \"*\""]

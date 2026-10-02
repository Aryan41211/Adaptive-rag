"""
Prometheus metrics for the Adaptive RAG API.

Exposes a ``/metrics/prometheus`` endpoint in OpenMetrics text format.

The endpoint is unauthenticated *only* when ``METRICS_TOKEN`` is unset, which
is the case in the compose stack where the API port is published on loopback
alone and Caddy answers 404 for this path. Render terminates TLS itself and has
no equivalent edge, so the blueprint sets a token and the route refuses to serve
without it.
"""

import hmac
import os
import threading
import time

# prometheus_client chooses its value class by testing whether
# PROMETHEUS_MULTIPROC_DIR is *present* in the environment, not whether it is
# non-empty, and that decision is made the first time prometheus_client.values
# is imported. docker-compose.yml forwards the variable as
# ${PROMETHEUS_MULTIPROC_DIR:-}, so an unconfigured deployment has it present
# and empty - which still enables multiprocess mode, with mmap paths built by
# os.path.join("", "gauge_all_1.db") and therefore relative to the working
# directory. The image runs unprivileged in a root-owned /app, so the first
# Gauge raised PermissionError and the API never started.
#
# This has to happen before the import below, which is also why it is not a
# line in the settings model: by the time src.core.config is read, the value
# class is already fixed. An unset variable is the correct way to say "off".
if not os.environ.get("PROMETHEUS_MULTIPROC_DIR", "").strip():
    os.environ.pop("PROMETHEUS_MULTIPROC_DIR", None)
    os.environ.pop("prometheus_multiproc_dir", None)

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    multiprocess,
)
from starlette.requests import Request
from starlette.responses import Response

from src.core.config import settings

# Running total behind the MODEL_COST gauge. Prometheus holds the value
# between scrapes, so the process has to own the accumulator; the lock keeps
# concurrent turns from losing an increment.
_cost_total = 0.0
_cost_lock = threading.Lock()

# Registry — multiprocess-safe when PROMETHEUS_MULTIPROC_DIR is set.
if settings.PROMETHEUS_MULTIPROC_DIR:
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
else:
    registry = None  # use the global default registry

# --- Instruments ---
REQUEST_COUNT = Counter(
    "rag_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

REQUEST_LATENCY = Histogram(
    "rag_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

ACTIVE_REQUESTS = Gauge(
    "rag_active_requests",
    "Number of requests currently being processed",
)

MODEL_CALLS = Counter(
    "rag_model_calls_total",
    "Total LLM API calls",
    # Labelled so the dashboard can attribute spend to a provider; without it
    # `sum(rate(rag_model_calls_total[5m])) by (model)` is an invalid query.
    ["model"],
)

MODEL_TOKENS = Counter(
    "rag_model_tokens_total",
    "Total tokens consumed by LLM calls",
    ["type"],  # "input" or "output"
)

MODEL_COST = Gauge(
    "rag_model_cost_usd",
    # Spelled out because a flat zero is ambiguous: Gemini's free tier really
    # is free, and an unpriced model reports zero too. Only the token counters
    # can tell those apart, so read them alongside this.
    "Cumulative estimated LLM cost in USD; models without a known price "
    "(including free-tier Gemini and local Ollama) contribute zero",
)

UPLOAD_COUNT = Counter(
    "rag_uploads_total",
    "Total document uploads",
    ["status"],  # "success" or "error"
)

DOCUMENTS_TOTAL = Gauge(
    "rag_documents_total",
    "Number of indexed documents per user",
    # Labelled by user because the count is per-user. Cardinality is bounded by
    # the number of accounts, not by traffic, which is what makes this safe to
    # keep as a gauge.
    ["user_id"],
)


def record_turn_usage(usage, model: str) -> None:
    """
    Publish one completed turn's usage to Prometheus.

    Called from the single place a turn's usage is finalised, so every path
    that spends money is counted. The model is attached as a label rather than
    aggregated away because the question this answers is "which provider is
    costing us", which a single unlabeled total cannot.

    Args:
        usage: A :class:`~src.core.usage.Usage` tally for the turn.
        model: The chat model that served the turn.
    """
    if not getattr(usage, "calls", 0):
        # A turn that failed before reaching a model is not a model call.
        return

    MODEL_CALLS.labels(model=model).inc(usage.calls)
    if usage.input_tokens:
        MODEL_TOKENS.labels(type="input").inc(usage.input_tokens)
    if usage.output_tokens:
        MODEL_TOKENS.labels(type="output").inc(usage.output_tokens)

    # A gauge holding the running total, not a counter: a cost is a
    # measurement, and a counter would need a reset that never comes.
    with _cost_lock:
        global _cost_total
        _cost_total += usage.cost_usd
        MODEL_COST.set(_cost_total)


# --- Middleware ---
async def request_metrics_middleware(request: Request, call_next):
    """Track request count, latency, and active requests."""
    method = request.method
    path = request.url.path

    # Normalize path to avoid high-cardinality labels
    normalized = path
    if path.startswith("/rag/documents/"):
        normalized = "/rag/documents/{filename}"
    elif path.startswith("/rag/sessions/"):
        normalized = "/rag/sessions/{session_id}"

    ACTIVE_REQUESTS.inc()
    start = time.perf_counter()
    try:
        response = await call_next(request)
        elapsed = time.perf_counter() - start
        status = str(response.status_code)
        REQUEST_COUNT.labels(method=method, endpoint=normalized, status=status).inc()
        REQUEST_LATENCY.labels(method=method, endpoint=normalized).observe(elapsed)
        return response
    except Exception:
        elapsed = time.perf_counter() - start
        REQUEST_COUNT.labels(method=method, endpoint=normalized, status="500").inc()
        REQUEST_LATENCY.labels(method=method, endpoint=normalized).observe(elapsed)
        raise
    finally:
        ACTIVE_REQUESTS.dec()


# --- Endpoint ---
from fastapi import APIRouter, HTTPException, Request  # noqa: E402

metrics_router = APIRouter(tags=["metrics"])


def _token_is_valid(supplied: str | None) -> bool:
    """
    Compare the caller's token against the configured one.

    hmac.compare_digest rather than `==` so the comparison time does not leak
    how much of the token was correct.
    """
    expected = settings.METRICS_TOKEN
    if not expected:
        return True
    if not supplied:
        return False
    return hmac.compare_digest(supplied, expected)


@metrics_router.get("/metrics/prometheus")
async def prometheus_metrics(request: Request) -> Response:
    """Expose Prometheus metrics in OpenMetrics text format."""
    header = request.headers.get("Authorization", "")
    supplied = header[7:].strip() if header.startswith("Bearer ") else None

    if not _token_is_valid(supplied):
        # The body deliberately says nothing about the expected value.
        raise HTTPException(
            status_code=401,
            detail="Metrics are protected; set a valid METRICS_TOKEN.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    output = generate_latest(registry) if registry is not None else generate_latest()
    return Response(content=output, media_type=CONTENT_TYPE_LATEST)

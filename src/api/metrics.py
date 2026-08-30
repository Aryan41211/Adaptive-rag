"""
Prometheus metrics for the Adaptive RAG API.

Exposes a ``/metrics/prometheus`` endpoint in OpenMetrics text format.
The endpoint is unauthenticated — it is only reachable from inside the
compose network where Prometheus runs.
"""

import os
import time

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
)

MODEL_TOKENS = Counter(
    "rag_model_tokens_total",
    "Total tokens consumed by LLM calls",
    ["type"],  # "input" or "output"
)

MODEL_COST = Gauge(
    "rag_model_cost_usd",
    "Cumulative estimated LLM cost in USD",
)

UPLOAD_COUNT = Counter(
    "rag_uploads_total",
    "Total document uploads",
    ["status"],  # "success" or "error"
)

DOCUMENTS_TOTAL = Gauge(
    "rag_documents_total",
    "Number of indexed documents per user",
    ["user_id"],
)


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
from fastapi import APIRouter  # noqa: E402

metrics_router = APIRouter(tags=["metrics"])


@metrics_router.get("/metrics/prometheus")
async def prometheus_metrics() -> Response:
    """Expose Prometheus metrics in OpenMetrics text format."""
    if registry is not None:
        output = generate_latest(registry)
    else:
        output = generate_latest()
    return Response(content=output, media_type=CONTENT_TYPE_LATEST)

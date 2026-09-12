# Production Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Prometheus/Grafana observability, automated systemd backups, and production hardening to the Docker Compose deployment.

**Architecture:** Add `prometheus-client` library for metrics exposition, new `/metrics/prometheus` endpoint, Grafana provisioning files, systemd timer for backups, and docker-compose resource limits/logging. No application architecture changes.

**Tech Stack:** Python 3.12, prometheus-client, Docker Compose, Prometheus, Grafana, systemd

**Spec:** `docs/superpowers/specs/2026-08-30-production-hardening-design.md`

## Global Constraints

- Python >=3.10, tested on 3.12
- LangChain pins: langchain==0.3.27, langchain-core==0.3.72, langgraph==0.5.4
- Ruff linter: line-length 88, target py310
- All ports bound to 127.0.0.1 only (except api:8000 and ui:8501 which are published)
- No application architecture changes
- Existing 432 tests must continue passing

---

## File Structure

| File | Action | Responsibility |
|---|---|---|
| `src/api/metrics.py` | Create | Prometheus instruments, /metrics/prometheus endpoint, request middleware |
| `src/main.py:115-125` | Edit | Add metrics middleware to request handler |
| `src/main.py:234` | Edit | Include metrics router |
| `src/api/ratelimit.py` | Edit | Add rate limit headers to responses |
| `requirements.txt` | Edit | Add prometheus-client |
| `requirements.lock.txt` | Regenerate | Lock prometheus-client |
| `deploy/prometheus/prometheus.yml` | Create | Prometheus scrape config |
| `deploy/grafana/provisioning/datasources/prometheus.yml` | Create | Auto-configure Prometheus datasource |
| `deploy/grafana/provisioning/dashboards/dashboard.yml` | Create | Dashboard provisioning config |
| `deploy/grafana/dashboards/adaptive-rag.json` | Create | Pre-built Grafana dashboard |
| `deploy/backup.service` | Create | Systemd backup unit |
| `deploy/backup.timer` | Create | Systemd timer (daily 3 AM) |
| `deploy/install-backup.sh` | Create | One-command timer setup |
| `.env.production.example` | Create | Production defaults |
| `docker-compose.yml` | Edit | Add services, resource limits, log rotation |
| `tests/test_metrics.py` | Create | Verify /metrics/prometheus returns valid OpenMetrics |
| `tests/test_ratelimit_headers.py` | Create | Verify rate limit headers in responses |

---

### Task 1: Add prometheus-client dependency

**Files:**
- Modify: `requirements.txt`
- Modify: `requirements.lock.txt` (regenerate)

**Interfaces:**
- Consumes: existing requirements.txt
- Produces: prometheus-client available for import

- [ ] **Step 1: Add prometheus-client to requirements.txt**

Add after the `# --- Config & utilities ---` section:

```
# --- Observability ---
prometheus-client>=0.20,<1
```

- [ ] **Step 2: Regenerate lock file**

Run: `pip-compile requirements.txt -o requirements.lock.txt`
If pip-compile is not installed: `pip install pip-tools && pip-compile requirements.txt -o requirements.lock.txt`

- [ ] **Step 3: Verify installation**

Run: `pip install prometheus-client && python -c "from prometheus_client import Counter; print('ok')"`
Expected: prints "ok"

- [ ] **Step 4: Run existing tests to confirm no breakage**

Run: `pytest tests/ -x -q`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add requirements.txt requirements.lock.txt
git commit -m "deps: add prometheus-client for metrics exposition"
```

---

### Task 2: Create Prometheus metrics module

**Files:**
- Create: `src/api/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Consumes: `prometheus_client` library, `src.core.config.settings`
- Produces: `request_metrics_middleware(app)`, `metrics_router` (FastAPI APIRouter), `REQUEST_COUNT`, `REQUEST_LATENCY`, `ACTIVE_REQUESTS`, `MODEL_CALLS`, `MODEL_TOKENS`, `MODEL_COST`, `UPLOAD_COUNT`

- [ ] **Step 1: Write the failing test**

Create `tests/test_metrics.py`:

```python
"""Tests for the Prometheus metrics endpoint."""

import os

os.environ["OPENAI_API_KEY"] = "sk-test-key-not-real"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-long-enough-for-validation-0123456789"
os.environ["TAVILY_API_KEY"] = ""
os.environ["MONGODB_URL"] = ""
os.environ["QDRANT_URL"] = ""
os.environ["QDRANT_API_KEY"] = ""
os.environ["LOG_LEVEL"] = "WARNING"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.main import app  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_prometheus_metrics_returns_text(client):
    """The /metrics/prometheus endpoint returns OpenMetrics text format."""
    response = client.get("/metrics/prometheus")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]


def test_prometheus_metrics_contains_known_metrics(client):
    """The response includes at least one of our custom metrics."""
    response = client.get("/metrics/prometheus")
    body = response.text
    assert "rag_requests_total" in body or "rag_active_requests" in body


def test_prometheus_metrics_not_authenticated(client):
    """The /metrics/prometheus endpoint does not require authentication."""
    response = client.get("/metrics/prometheus")
    assert response.status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL (404 — endpoint does not exist yet)

- [ ] **Step 3: Write the metrics module**

Create `src/api/metrics.py`:

```python
"""
Prometheus metrics for the Adaptive RAG API.

Exposes a ``/metrics/prometheus`` endpoint in OpenMetrics text format.
The endpoint is unauthenticated — it is only reachable from inside the
compose network where Prometheus runs.
"""

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

# ---------------------------------------------------------------------------
# Registry — multiprocess-safe when running with multiple uvicorn workers.
# When PROMETHEUS_MULTIPROC_DIR is set (by docker-compose), counters and
# gauges are shared across workers via memory-mapped files.
# ---------------------------------------------------------------------------
import os

if settings.PROMETHEUS_MULTIPROC_DIR:
    registry = CollectorRegistry()
    multiprocess.MultiProcessCollector(registry)
else:
    registry = None  # use the global default registry

# ---------------------------------------------------------------------------
# Instruments
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------
async def request_metrics_middleware(request: Request, call_next):
    """
    Track request count, latency, and active requests.

    Attached as ASGI middleware in main.py.
    """
    import time

    method = request.method
    path = request.url.path

    # Normalize path to avoid high-cardinality labels: strip IDs and params.
    # e.g. /rag/documents/my-file.pdf -> /rag/documents/{filename}
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


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------
from fastapi import APIRouter  # noqa: E402

metrics_router = APIRouter(tags=["metrics"])


@metrics_router.get("/metrics/prometheus")
async def prometheus_metrics() -> Response:
    """
    Expose Prometheus metrics in OpenMetrics text format.

    This endpoint is unauthenticated — it is only reachable from inside the
    compose network where Prometheus scrapes it.
    """
    if registry is not None:
        output = generate_latest(registry)
    else:
        output = generate_latest()
    return Response(content=output, media_type=CONTENT_TYPE_LATEST)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_metrics.py -v`
Expected: all PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/api/metrics.py tests/test_metrics.py
git commit -m "feat: add Prometheus metrics endpoint at /metrics/prometheus"
```

---

### Task 3: Wire metrics into the application

**Files:**
- Modify: `src/main.py` (add middleware, include router)
- Modify: `src/core/config.py` (add PROMETHEUS_MULTIPROC_DIR setting)

**Interfaces:**
- Consumes: `src.api.metrics.request_metrics_middleware`, `src.api.metrics.metrics_router`
- Produces: metrics middleware active on all requests, /metrics/prometheus route available

- [ ] **Step 1: Add PROMETHEUS_MULTIPROC_DIR to config**

In `src/core/config.py`, add after the `OTEL_EXPORT_TIMEOUT_SECONDS` field:

```python
    # --- Observability -------------------------------------------------------
    PROMETHEUS_MULTIPROC_DIR: str | None = None
```

- [ ] **Step 2: Add middleware to main.py**

In `src/main.py`, after the request_id middleware (around line 125), add:

```python
from src.api.metrics import request_metrics_middleware

app.middleware("http")(request_metrics_middleware)
```

- [ ] **Step 3: Include metrics router**

In `src/main.py`, after `app.include_router(rag_router)` (around line 180), add:

```python
from src.api.metrics import metrics_router

app.include_router(metrics_router)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_metrics.py tests/test_config.py -v`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add src/main.py src/core/config.py
git commit -m "feat: wire Prometheus metrics into the application"
```

---

### Task 4: Add rate limit headers to responses

**Files:**
- Modify: `src/api/ratelimit.py`
- Test: `tests/test_ratelimit_headers.py`

**Interfaces:**
- Consumes: existing rate limit counters in `src/api/ratelimit.py`
- Produces: `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` headers on /rag/* responses

- [ ] **Step 1: Write the failing test**

Create `tests/test_ratelimit_headers.py`:

```python
"""Tests for rate limit headers in API responses."""

import os

os.environ["OPENAI_API_KEY"] = "sk-test-key-not-real"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-long-enough-for-validation-0123456789"
os.environ["TAVILY_API_KEY"] = ""
os.environ["MONGODB_URL"] = ""
os.environ["QDRANT_URL"] = ""
os.environ["QDRANT_API_KEY"] = ""
os.environ["LOG_LEVEL"] = "WARNING"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from src.main import app  # noqa: E402
from tests.conftest import register_and_login  # noqa: E402


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_rate_limit_headers_on_query(client):
    """Query responses include rate limit headers."""
    headers = register_and_login(client)
    response = client.post(
        "/rag/query",
        json={"query": "test", "session_id": "s1"},
        headers=headers,
    )
    # May be 200 or 502 (no real OpenAI key), but headers should be present
    assert "x-ratelimit-limit" in response.headers
    assert "x-ratelimit-remaining" in response.headers
    assert "x-ratelimit-reset" in response.headers


def test_rate_limit_headers_are_integers(client):
    """Rate limit header values are valid integers."""
    headers = register_and_login(client)
    response = client.post(
        "/rag/query",
        json={"query": "test", "session_id": "s2"},
        headers=headers,
    )
    limit = response.headers.get("x-ratelimit-limit")
    remaining = response.headers.get("x-ratelimit-remaining")
    reset = response.headers.get("x-ratelimit-reset")
    if limit:
        int(limit)
    if remaining:
        int(remaining)
    if reset:
        int(reset)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_ratelimit_headers.py -v`
Expected: FAIL (headers not present)

- [ ] **Step 3: Implement rate limit headers**

Read `src/api/ratelimit.py` to understand the current structure. Add a helper that attaches headers to the response. The approach: modify the rate limit dependency functions to accept a response and set headers, or add a middleware.

The cleanest approach is to add a small middleware that reads the rate limit state after the request and sets headers. However, since the existing rate limit uses FastAPI dependencies that raise before the response, the headers only make sense for successful (non-429) requests.

Add to `src/api/ratelimit.py`:

```python
import time

def rate_limit_headers(user_id: str, endpoint: str) -> dict[str, str]:
    """
    Build rate limit headers for the response.

    Returns a dict with X-RateLimit-Limit, X-RateLimit-Remaining, and
    X-RateLimit-Reset.
    """
    from src.core.config import settings

    limits = {
        "query": settings.RATE_LIMIT_QUERY_PER_MINUTE,
        "upload": settings.RATE_LIMIT_UPLOAD_PER_HOUR,
    }
    window = 60 if endpoint == "query" else 3600
    limit = limits.get(endpoint, 20)

    counter = _get_counter(user_id, endpoint, window)
    remaining = max(0, limit - counter)
    reset = int(time.time()) + window

    return {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(remaining),
        "X-RateLimit-Reset": str(reset),
    }
```

Then modify the route handlers in `src/api/routes.py` to set these headers on successful responses. For the query endpoint, after getting the result:

```python
from src.api.ratelimit import rate_limit_headers

# In rag_query, before returning:
headers = rate_limit_headers(user.user_id, "query")
return QueryResponse(...)  # headers are set via Response parameter
```

Since FastAPI response_model doesn't support headers directly, use the `response` parameter:

```python
from fastapi import Response

@router.post("/query", response_model=QueryResponse)
async def rag_query(
    req: QueryRequest,
    response: Response,
    user: CurrentUser = Depends(query_rate_limit()),
) -> QueryResponse:
    # ... existing code ...
    for k, v in rate_limit_headers(user.user_id, "query").items():
        response.headers[k] = v
    return QueryResponse(...)
```

Apply the same pattern to the upload endpoint and the list/delete document endpoints.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_ratelimit_headers.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: all pass

- [ ] **Step 6: Commit**

```bash
git add src/api/ratelimit.py src/api/routes.py tests/test_ratelimit_headers.py
git commit -m "feat: add X-RateLimit-* headers to API responses"
```

---

### Task 5: Add Prometheus and Grafana to docker-compose

**Files:**
- Modify: `docker-compose.yml`
- Create: `deploy/prometheus/prometheus.yml`
- Create: `deploy/grafana/provisioning/datasources/prometheus.yml`
- Create: `deploy/grafana/provisioning/dashboards/dashboard.yml`
- Create: `deploy/grafana/dashboards/adaptive-rag.json`
- Modify: `.env.example` (add GRAFANA_ADMIN_PASSWORD)

**Interfaces:**
- Consumes: existing docker-compose.yml, api service on port 8000
- Produces: Prometheus on :9090, Grafana on :3000, Node Exporter on :9100 (all localhost-only)

- [ ] **Step 1: Create Prometheus config**

Create `deploy/prometheus/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "adaptive-rag-api"
    metrics_path: /metrics/prometheus
    static_configs:
      - targets: ["api:8000"]

  - job_name: "node-exporter"
    static_configs:
      - targets: ["node-exporter:9100"]
```

- [ ] **Step 2: Create Grafana datasource provisioning**

Create `deploy/grafana/provisioning/datasources/prometheus.yml`:

```yaml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: false
```

- [ ] **Step 3: Create Grafana dashboard provisioning**

Create `deploy/grafana/provisioning/dashboards/dashboard.yml`:

```yaml
apiVersion: 1

providers:
  - name: "Adaptive RAG"
    orgId: 1
    folder: ""
    type: file
    disableDeletion: false
    editable: true
    options:
      path: /var/lib/grafana/dashboards
      foldersFromFilesStructure: false
```

- [ ] **Step 4: Create Grafana dashboard**

Create `deploy/grafana/dashboards/adaptive-rag.json` — a comprehensive dashboard with these panels:

```json
{
  "annotations": { "list": [] },
  "editable": true,
  "fiscalYearStartMonth": 0,
  "graphTooltip": 1,
  "id": null,
  "links": [],
  "panels": [
    {
      "title": "Request Rate",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 0 },
      "targets": [
        {
          "expr": "rate(rag_requests_total[5m])",
          "legendFormat": "{{method}} {{endpoint}} {{status}}"
        }
      ]
    },
    {
      "title": "Latency p95",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 0 },
      "targets": [
        {
          "expr": "histogram_quantile(0.95, rate(rag_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p95"
        },
        {
          "expr": "histogram_quantile(0.50, rate(rag_request_duration_seconds_bucket[5m]))",
          "legendFormat": "p50"
        }
      ]
    },
    {
      "title": "Error Rate (5xx)",
      "type": "stat",
      "gridPos": { "h": 4, "w": 6, "x": 0, "y": 8 },
      "targets": [
        {
          "expr": "sum(rate(rag_requests_total{status=~\"5..\"}[5m])) / sum(rate(rag_requests_total[5m])) * 100",
          "legendFormat": "Error %"
        }
      ]
    },
    {
      "title": "Active Requests",
      "type": "stat",
      "gridPos": { "h": 4, "w": 6, "x": 6, "y": 8 },
      "targets": [
        {
          "expr": "rag_active_requests",
          "legendFormat": "Active"
        }
      ]
    },
    {
      "title": "Model Calls/min",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 0, "y": 12 },
      "targets": [
        {
          "expr": "rate(rag_model_calls_total[5m]) * 60",
          "legendFormat": "calls/min"
        }
      ]
    },
    {
      "title": "Token Usage",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 12, "y": 12 },
      "targets": [
        {
          "expr": "rate(rag_model_tokens_total{type=\"input\"}[5m]) * 60",
          "legendFormat": "input tokens/min"
        },
        {
          "expr": "rate(rag_model_tokens_total{type=\"output\"}[5m]) * 60",
          "legendFormat": "output tokens/min"
        }
      ]
    },
    {
      "title": "Cumulative Cost (USD)",
      "type": "stat",
      "gridPos": { "h": 4, "w": 6, "x": 0, "y": 20 },
      "targets": [
        {
          "expr": "rag_model_cost_usd",
          "legendFormat": "Cost USD"
        }
      ],
      "fieldConfig": {
        "defaults": {
          "unit": "currencyUSD"
        }
      }
    },
    {
      "title": "Uploads",
      "type": "timeseries",
      "gridPos": { "h": 8, "w": 12, "x": 6, "y": 20 },
      "targets": [
        {
          "expr": "rate(rag_uploads_total[5m]) * 60",
          "legendFormat": "{{status}}"
        }
      ]
    }
  ],
  "schemaVersion": 39,
  "tags": ["adaptive-rag"],
  "templating": { "list": [] },
  "time": { "from": "now-1h", "to": "now" },
  "title": "Adaptive RAG",
  "uid": "adaptive-rag"
}
```

- [ ] **Step 5: Add services to docker-compose.yml**

Add to `docker-compose.yml` after the `caddy` service, before `volumes`:

```yaml
  prometheus:
    image: prom/prometheus:v2.53.0
    restart: unless-stopped
    ports:
      - "127.0.0.1:9090:9090"
    volumes:
      - ./deploy/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus_data:/prometheus
    command:
      - --config.file=/etc/prometheus/prometheus.yml
      - --storage.tsdb.retention.time=30d
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  grafana:
    image: grafana/grafana:11.1.0
    restart: unless-stopped
    ports:
      - "127.0.0.1:3000:3000"
    environment:
      GF_SECURITY_ADMIN_USER: admin
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:?set GRAFANA_ADMIN_PASSWORD in .env}
      GF_USERS_ALLOW_SIGN_UP: "false"
    volumes:
      - ./deploy/grafana/provisioning:/etc/grafana/provisioning:ro
      - ./deploy/grafana/dashboards:/var/lib/grafana/dashboards:ro
      - grafana_data:/var/lib/grafana
    depends_on:
      - prometheus
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  node-exporter:
    image: prom/node-exporter:v1.8.1
    restart: unless-stopped
    ports:
      - "127.0.0.1:9100:9100"
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - --path.procfs=/host/proc
      - --path.sysfs=/host/sys
      - --path.rootfs=/rootfs
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

Add volumes at the end:

```yaml
  prometheus_data:
  grafana_data:
```

- [ ] **Step 6: Add resource limits to existing services**

Add `deploy` and `logging` sections to existing services in docker-compose.yml:

```yaml
  api:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 2G
        reservations:
          cpus: "0.5"
          memory: 512M
    stop_grace_period: 30s
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"

  ui:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"

  qdrant:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2G
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"

  mongo:
    # ... existing config ...
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 2G
    logging:
      driver: json-file
      options:
        max-size: "10m"
        max-file: "3"
```

- [ ] **Step 7: Add GRAFANA_ADMIN_PASSWORD to .env.example**

Add to `.env.example`:

```bash
# --- Required by docker compose (Grafana) ---------------------------------
# Password for the Grafana admin user. Set to a strong value.
GRAFANA_ADMIN_PASSWORD=
```

- [ ] **Step 8: Commit**

```bash
git add docker-compose.yml deploy/prometheus/ deploy/grafana/ .env.example
git commit -m "feat: add Prometheus, Grafana, and Node Exporter to compose stack"
```

---

### Task 6: Create automated backup systemd units

**Files:**
- Create: `deploy/backup.service`
- Create: `deploy/backup.timer`
- Create: `deploy/install-backup.sh`

**Interfaces:**
- Consumes: existing `deploy/backup.sh`
- Produces: systemd timer that runs backup daily at 3 AM

- [ ] **Step 1: Create backup.service**

Create `deploy/backup.service`:

```ini
[Unit]
Description=Adaptive RAG backup
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
WorkingDirectory=/opt/adaptive-rag
ExecStart=/opt/adaptive-rag/deploy/backup.sh /var/backups/adaptive-rag
Environment=RETAIN=14
Environment=COMPOSE=docker compose
StandardOutput=journal
StandardError=journal
```

- [ ] **Step 2: Create backup.timer**

Create `deploy/backup.timer`:

```ini
[Unit]
Description=Run Adaptive RAG backup daily

[Timer]
OnCalendar=*-*-* 03:00:00
RandomizedDelaySec=1800
Persistent=true

[Install]
WantedBy=timers.target
```

- [ ] **Step 3: Create install-backup.sh**

Create `deploy/install-backup.sh`:

```bash
#!/usr/bin/env bash
#
# Install the Adaptive RAG backup timer.
#
#   sudo ./deploy/install-backup.sh
#
# Copies the systemd unit files and enables the timer. The timer runs
# daily at 3 AM with up to 30 minutes of random delay.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Installing backup timer..."

sudo cp "${SCRIPT_DIR}/backup.service" /etc/systemd/system/
sudo cp "${SCRIPT_DIR}/backup.timer" /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now backup.timer

echo "Backup timer installed."
echo "Check status: systemctl list-timers backup.timer"
echo "View logs:    journalctl -u backup.service"
```

Make it executable: `chmod +x deploy/install-backup.sh`

- [ ] **Step 4: Commit**

```bash
git add deploy/backup.service deploy/backup.timer deploy/install-backup.sh
git commit -m "feat: add systemd timer for automated daily backups"
```

---

### Task 7: Create .env.production.example

**Files:**
- Create: `.env.production.example`

**Interfaces:**
- Consumes: existing `.env.example`
- Produces: documented production defaults

- [ ] **Step 1: Create the file**

Create `.env.production.example`:

```bash
# ---------------------------------------------------------------------------
# Adaptive RAG - Production environment configuration
# Copy to .env on the production VM and fill in.
# This file documents production-safe defaults.
# ---------------------------------------------------------------------------

# --- Required ---------------------------------------------------------------
OPENAI_API_KEY=           # Required: OpenAI API key
JWT_SECRET_KEY=           # Required: min 32 chars, generate with:
                          #   python -c "import secrets; print(secrets.token_urlsafe(48))"

# --- Required by docker compose ----------------------------------------------
MONGO_ROOT_USERNAME=adaptive
MONGO_ROOT_PASSWORD=      # Required: generate with:
                          #   python -c "import secrets; print(secrets.token_hex(24))"

# --- Optional: vector store -------------------------------------------------
QDRANT_URL=http://qdrant:6333
QDRANT_API_KEY=           # Required for Qdrant Cloud
QDRANT_COLLECTION=adaptive_rag_documents

# --- Optional: persistence --------------------------------------------------
MONGODB_URL=mongodb://${MONGO_ROOT_USERNAME}:${MONGO_ROOT_PASSWORD}@mongo:27017/?authSource=admin
MONGODB_DB_NAME=adaptive_rag

# --- Optional: web search ---------------------------------------------------
TAVILY_API_KEY=

# --- Optional: models & behaviour -------------------------------------------
OPENAI_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
MAX_HISTORY_MESSAGES=20
MAX_UPLOAD_BYTES=10485760
MAX_REWRITE_ATTEMPTS=2
MAX_VERIFY_ATTEMPTS=1

# --- Optional: rate limiting ------------------------------------------------
RATE_LIMIT_ENABLED=true
RATE_LIMIT_QUERY_PER_MINUTE=20
RATE_LIMIT_UPLOAD_PER_HOUR=20
RATE_LIMIT_AUTH_PER_MINUTE=10

# --- Production: security ---------------------------------------------------
# Disable API docs in production — the schema enumerates the attack surface.
ENABLE_API_DOCS=false
# Only answer to your production hostname.
ALLOWED_HOSTS=rag.example.com
# Allow your frontend origin to call the API.
CORS_ALLOW_ORIGINS=https://rag.example.com

# --- Production: logging ----------------------------------------------------
LOG_LEVEL=WARNING

# --- Production: Grafana ----------------------------------------------------
GRAFANA_ADMIN_PASSWORD=   # Required: set a strong password

# --- Optional: tracing ------------------------------------------------------
OTEL_EXPORTER_OTLP_ENDPOINT=
OTEL_SERVICE_NAME=adaptive-rag

# --- Optional: frontend -----------------------------------------------------
API_BASE_URL=http://api:8000

# --- Optional: secrets from files -------------------------------------------
SECRETS_DIR=/run/secrets
```

- [ ] **Step 2: Commit**

```bash
git add .env.production.example
git commit -m "docs: add production environment configuration template"
```

---

### Task 8: Final verification

**Files:** None (verification only)

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -x -q`
Expected: all pass

- [ ] **Step 2: Run linter**

Run: `ruff check . && ruff format --check .`
Expected: no errors

- [ ] **Step 3: Verify docker-compose builds**

Run: `docker compose build`
Expected: builds successfully

- [ ] **Step 4: Verify the full stack starts**

Run: `docker compose up -d`
Wait 30 seconds, then:
Run: `curl -s http://127.0.0.1:8000/healthz`
Expected: `{"status":"ok"}`

Run: `curl -s http://127.0.0.1:9090/-/healthy`
Expected: `Prometheus Server is Healthy.`

Run: `curl -s http://127.0.0.1:3000/api/health`
Expected: `{"database":"ok"}`

- [ ] **Step 5: Verify Prometheus is scraping**

Run: `curl -s http://127.0.0.1:9090/api/v1/targets | python -m json.tool | head -20`
Expected: both targets show `health: "up"`

- [ ] **Step 6: Verify Grafana dashboard loads**

Open `http://127.0.0.1:3000` in browser, log in with admin/GRAFANA_ADMIN_PASSWORD, navigate to Dashboards > Adaptive RAG.

- [ ] **Step 7: Commit final state**

```bash
git add -A
git commit -m "chore: production hardening complete"
```

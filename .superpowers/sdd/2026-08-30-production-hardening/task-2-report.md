# Task 2: Create Prometheus Metrics Module

## Summary

Created the Prometheus metrics module (`src/api/metrics.py`) with instruments and a `/metrics/prometheus` endpoint, plus a full test suite.

## Changes Made

### 1. `src/core/config.py`
- Added `PROMETHEUS_MULTIPROC_DIR: str | None = None` setting after `OTEL_EXPORT_TIMEOUT_SECONDS` to support multiprocess-safe Prometheus registries.

### 2. `src/api/metrics.py` (new file)
- **Registry**: Multiprocess-safe when `PROMETHEUS_MULTIPROC_DIR` is set, otherwise uses the global default registry.
- **Instruments**:
  - `rag_requests_total` — Counter (method, endpoint, status)
  - `rag_request_duration_seconds` — Histogram (method, endpoint)
  - `rag_active_requests` — Gauge
  - `rag_model_calls_total` — Counter
  - `rag_model_tokens_total` — Counter (type: input/output)
  - `rag_model_cost_usd` — Gauge
  - `rag_uploads_total` — Counter (status: success/error)
  - `rag_documents_total` — Gauge (user_id)
- **Middleware**: `request_metrics_middleware` — tracks request count, latency, and active requests with path normalization to avoid high-cardinality labels.
- **Endpoint**: `GET /metrics/prometheus` — returns OpenMetrics text format, unauthenticated.

### 3. `tests/test_metrics.py` (new file)
- Three tests:
  - Returns 200 with `text/plain` content type
  - Response body contains at least one known metric name
  - No authentication required (200 without auth header)

### 4. `src/main.py`
- Imported and included `metrics_router` from `src.api.metrics`.

## Test Results

- `tests/test_metrics.py`: **3/3 passed**
- Full suite (excluding pre-existing `test_deploy_scripts.py` bash issue): **all passing**

## Commit

`4b2ba08` — `feat: add Prometheus metrics endpoint at /metrics/prometheus`

## Concerns

None. The metrics router is now wired in and the endpoint is testable. Task 3 (middleware integration) will use `request_metrics_middleware` for full request-level tracking.

# Production Hardening Design

**Date:** 2026-08-30
**Status:** Approved
**Target:** Docker Compose on a VM with TLS via Caddy

## Goal

Make the Adaptive RAG project fully production-ready for Docker Compose deployment on a VM. The application code is already solid (432 tests, 94% coverage, auth, rate limiting, health probes, structured logging, tracing). This spec adds deployment infrastructure, not application rewrites.

## Scope

Three areas:

1. **Prometheus + Grafana observability stack**
2. **Automated scheduled backups via systemd**
3. **Production hardening of Docker Compose and the app**

---

## 1. Prometheus + Grafana Observability

### 1.1 Problem

The app exposes `/metrics` as a JSON endpoint (authenticated). Prometheus needs a `/metrics/prometheus` path that returns the OpenMetrics text format. Without this, there is no way to alert on error rates, latency percentiles, or token spend.

### 1.2 Design

**New dependency:** `prometheus-client` (pure Python, no transitive deps, ~50KB).

**New file: `src/api/metrics.py`**

Exposes Prometheus instruments:

| Instrument | Type | Labels | Purpose |
|---|---|---|---|
| `rag_requests_total` | Counter | `route`, `status` | Request rate by endpoint and status |
| `rag_request_duration_seconds` | Histogram | `route` | Latency distribution |
| `rag_model_calls_total` | Counter | — | Total LLM calls |
| `rag_model_tokens_total` | Counter | `type` (input/output) | Token volume |
| `rag_model_cost_usd` | Gauge | — | Cumulative estimated cost |
| `rag_active_requests` | Gauge | — | In-flight requests |
| `rag_uploads_total` | Counter | `status` | Upload success/failure |
| `rag_documents_total` | Gauge | `user_id` | Document count per user |

Middleware in `src/main.py` wraps each request to update `rag_requests_total`, `rag_request_duration_seconds`, and `rag_active_requests`.

**New route: `GET /metrics/prometheus`** (unauthenticated, same as `/healthz` — Prometheus scrapes from inside the compose network).

**New files:**
- `deploy/prometheus/prometheus.yml` — scrape config targeting `api:8000/metrics/prometheus` and `node-exporter:9100`
- `deploy/grafana/provisioning/datasources/prometheus.yml` — auto-configures Prometheus as default datasource
- `deploy/grafana/provisioning/dashboards/dashboard.yml` — loads dashboards from `/var/lib/grafana/dashboards`
- `deploy/grafana/dashboards/adaptive-rag.json` — pre-built dashboard with panels for request rate, latency, errors, token usage, cost

**New services in `docker-compose.yml`:**

```yaml
prometheus:
  image: prom/prometheus:v2.53.0
  volumes:
    - ./deploy/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
    - prometheus_data:/prometheus
  ports:
    - "127.0.0.1:9090:9090"

grafana:
  image: grafana/grafana:11.1.0
  environment:
    GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:?set in .env}
  volumes:
    - ./deploy/grafana/provisioning:/etc/grafana/provisioning:ro
    - ./deploy/grafana/dashboards:/var/lib/grafana/dashboards:ro
    - grafana_data:/var/lib/grafana
  ports:
    - "127.0.0.1:3000:3000"
  depends_on:
    - prometheus

node-exporter:
  image: prom/node-exporter:v1.8.1
  volumes:
    - /proc:/host/proc:ro
    - /sys:/host/sys:ro
  command:
    - --path.procfs=/host/proc
    - --path.sysfs=/host/sys
  ports:
    - "127.0.0.1:9100:9100"
```

All ports bound to `127.0.0.1` only, matching the existing security posture. Prometheus and Grafana are internal tools, not public-facing.

### 1.3 Grafana Dashboard

Pre-built dashboard with these rows:

**Row 1 — Overview:**
- Request rate (requests/sec, by status code)
- Latency p50/p95/p99
- Error rate (% of 5xx responses)

**Row 2 — RAG Pipeline:**
- Model calls per minute
- Token usage (input vs output)
- Estimated cost (USD) over time
- Active requests

**Row 3 — Uploads:**
- Upload rate
- Upload failures

**Row 4 — System:**
- Container CPU usage
- Container memory usage
- Container network I/O

---

## 2. Automated Scheduled Backups

### 2.1 Problem

The existing `deploy/backup.sh` handles MongoDB + Qdrant snapshots with retention. It must be run manually. For production, backups need to run automatically on a schedule.

### 2.2 Design

No changes to `backup.sh`. Add three files:

**`deploy/backup.service`:**
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

**`deploy/backup.timer`:**
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

Runs daily at 3 AM with up to 30 minutes of random delay (prevents thundering herd if multiple VMs share a backup target). `Persistent=true` catches up if the VM was off at 3 AM.

**`deploy/install-backup.sh`:**
```bash
#!/usr/bin/env bash
# One-command backup timer setup.
# Copies the unit files to /etc/systemd/system and enables the timer.
set -euo pipefail
sudo cp deploy/backup.service deploy/backup.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now backup.timer
echo "Backup timer installed. Check: systemctl list-timers backup.timer"
```

### 2.3 Retention

`RETAIN=14` keeps the last 14 daily backups. Configurable by editing the service file or overriding via environment.

---

## 3. Production Hardening

### 3.1 Docker Compose resource limits

Add `deploy` section to each service:

```yaml
api:
  deploy:
    resources:
      limits:
        cpus: "2.0"
        memory: 2G
      reservations:
        cpus: "0.5"
        memory: 512M
  stop_grace_period: 30s

ui:
  deploy:
    resources:
      limits:
        cpus: "1.0"
        memory: 1G

qdrant:
  deploy:
    resources:
      limits:
        cpus: "1.0"
        memory: 2G

mongo:
  deploy:
    resources:
      limits:
        cpus: "1.0"
        memory: 2G
```

### 3.2 Log rotation

Add `logging` driver to all services:

```yaml
logging:
  driver: json-file
  options:
    max-size: "50m"
    max-file: "5"
```

### 3.3 Rate limit headers

Modify `src/api/ratelimit.py` to return headers on every response:

- `X-RateLimit-Limit` — the user's quota (e.g., 20 queries/min)
- `X-RateLimit-Remaining` — how many they have left in the current window
- `X-RateLimit-Reset` — UTC epoch when the window resets

This is done via a middleware or by attaching headers in the rate limit dependency. The existing `ratelimit.py` already tracks counters; we expose them in the response.

### 3.4 `.env.production.example`

New file with production-safe defaults:

```env
# Production defaults — copy to .env and fill in secrets.
OPENAI_API_KEY=           # Required
JWT_SECRET_KEY=           # Required, min 32 chars
MONGO_ROOT_PASSWORD=      # Required
ENABLE_API_DOCS=false     # Disable schema exposure
LOG_LEVEL=WARNING         # Reduce noise in production
ALLOWED_HOSTS=rag.example.com
CORS_ALLOW_ORIGINS=https://rag.example.com
RATE_LIMIT_ENABLED=true
RATE_LIMIT_QUERY_PER_MINUTE=20
RATE_LIMIT_UPLOAD_PER_HOUR=20
GRAFANA_ADMIN_PASSWORD=   # Required for Grafana
```

### 3.5 Graceful shutdown

The existing lifespan handler in `src/main.py` already handles cleanup (tracing shutdown, MongoDB client close). The only addition is setting `stop_grace_period: 30s` in docker-compose to give uvicorn time to finish in-flight requests before SIGKILL.

---

## Files Changed

| File | Action | Purpose |
|---|---|---|
| `requirements.txt` | Edit | Add `prometheus-client` |
| `requirements.lock.txt` | Regenerate | Lock new dependency |
| `src/api/metrics.py` | Create | Prometheus instruments and /metrics/prometheus endpoint |
| `src/main.py` | Edit | Add metrics middleware, include metrics router |
| `deploy/prometheus/prometheus.yml` | Create | Prometheus scrape config |
| `deploy/grafana/provisioning/datasources/prometheus.yml` | Create | Auto-configure datasource |
| `deploy/grafana/provisioning/dashboards/dashboard.yml` | Create | Dashboard provisioning config |
| `deploy/grafana/dashboards/adaptive-rag.json` | Create | Pre-built Grafana dashboard |
| `deploy/backup.service` | Create | Systemd backup unit |
| `deploy/backup.timer` | Create | Systemd timer (daily 3 AM) |
| `deploy/install-backup.sh` | Create | One-command timer setup |
| `.env.production.example` | Create | Production defaults |
| `docker-compose.yml` | Edit | Add prometheus, grafana, node-exporter services; resource limits; log rotation; stop_grace_period |
| `src/api/ratelimit.py` | Edit | Add rate limit headers to responses |

---

## Testing

- Existing test suite must continue passing (no application behavior changes)
- New test: `tests/test_metrics.py` — verify `/metrics/prometheus` returns valid OpenMetrics text
- New test: `tests/test_ratelimit_headers.py` — verify rate limit headers appear in responses
- Manual verification: `docker compose up --build` with full stack, confirm Prometheus scrapes, Grafana dashboards load

---

## What This Does NOT Change

- No Kubernetes manifests (Docker Compose only)
- No application architecture changes
- No new RAG features
- No changes to the auth model
- No changes to the backup script itself
- No changes to the Caddy TLS configuration

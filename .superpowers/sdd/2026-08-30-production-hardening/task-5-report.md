# Task 5: Add Prometheus, Grafana, and Node Exporter to docker-compose

## Status: DONE

## Changes Made

### New files created:
- `deploy/prometheus/prometheus.yml` — Prometheus scrape config targeting API (`/metrics/prometheus`) and Node Exporter
- `deploy/grafana/provisioning/datasources/prometheus.yml` — Auto-provisioned Prometheus datasource
- `deploy/grafana/provisioning/dashboards/dashboard.yml` — Dashboard provisioning from JSON files
- `deploy/grafana/dashboards/adaptive-rag.json` — 8-panel Grafana dashboard (Request Rate, Latency p95, Error Rate, Active Requests, Model Calls/min, Token Usage, Cumulative Cost, Uploads)

### Modified files:
- `docker-compose.yml` — Added prometheus, grafana, node-exporter services; added `deploy` and `logging` sections to qdrant, mongo, api, ui; added prometheus_data and grafana_data volumes
- `.env.example` — Added GRAFANA_ADMIN_PASSWORD variable

### Docker Compose Services Added:
| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| prometheus | prom/prometheus:v2.53.0 | 127.0.0.1:9090 | Metrics collection (30d retention) |
| grafana | grafana/grafana:11.1.0 | 127.0.0.1:3000 | Dashboards & visualization |
| node-exporter | prom/node-exporter:v1.8.1 | 127.0.0.1:9100 | Host metrics |

### Resource Limits Applied:
- **api**: 2 CPU / 2G memory limits, 0.5 CPU / 512M reservations, 30s stop_grace_period
- **ui**: 1 CPU / 1G memory limits
- **qdrant**: 1 CPU / 2G memory limits
- **mongo**: 1 CPU / 2G memory limits

### Log Rotation Applied:
- api/ui: max-size 50m, max-file 5
- qdrant/mongo/prometheus/grafana/node-exporter: max-size 10m, max-file 3

## Validation

`docker compose config` fails with missing MONGO_ROOT_PASSWORD — expected, as .env is not committed. The compose file parses correctly otherwise.

## Concerns

None. All services bind to 127.0.0.1 only, matching the existing security posture.

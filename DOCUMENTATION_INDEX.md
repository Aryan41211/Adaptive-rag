# 📑 Documentation Index

Every document this repository ships, and what each one is for.

## 🎯 Start here

| If you want to… | Read |
|---|---|
| Understand the system, run it, deploy it | **README.md** |
| Match the existing code style | **QUICK_REFERENCE.md** |
| Understand why the dependencies are pinned | `docs/ADR-001-langchain-version.md` |
| Add or debug vector search | **QDRANT_SETUP_GUIDE.md** |
| Understand document ingestion | **DOCUMENT_UPLOAD_FLOW.md** |

## 📚 All documentation

### Overview

| File | Purpose |
|---|---|
| **README.md** | Architecture, configuration, local setup, Docker Compose, TLS, observability, Render deployment, secrets |
| **DOCUMENTATION_INDEX.md** | This file |

### Development and standards

| File | Purpose |
|---|---|
| **QUICK_REFERENCE.md** | Fast lookup: naming, docstrings, imports, common commands |
| **CODE_STYLE_GUIDE.md** | The complete style reference, including the ruff commands CI enforces |

Ruff is the linter and formatter. The rule selection lives in
`pyproject.toml`. `black`, `flake8`, `isort` and `pylint` are **not**
dependencies of this project; the commands in the older guides that referenced
them have been replaced with their ruff equivalents.

### Architecture and data

| File | Purpose |
|---|---|
| **DOCUMENT_UPLOAD_FLOW.md** | Upload → parse → chunk → embed → index, step by step |
| **DOCUMENT_FLOW_VISUAL.md** | The same flow as a diagram |
| **QDRANT_SETUP_GUIDE.md** | Provisioning Qdrant, creating the collection, and its indexes |

### Architecture decision records

| File | Purpose |
|---|---|
| `docs/ADR-001-langchain-version.md` | Why the LangChain / LangGraph family is pinned with `==` and how to revisit it |

### Historical records

`.superpowers/` and `docs/superpowers/` hold planning documents and session
reports from previous work. They describe what was true at the time, including
test counts and tool choices that have since changed. They are kept for
context and are **not** maintained — treat **README.md** as authoritative.

## 🧪 Commands

| Task | Command |
|---|---|
| Offline test suite | `pytest` |
| Coverage | `pytest --cov=src --cov-report=term` |
| Live provider probes (needs `GEMINI_API_KEY` in `.env`) | `pytest tests_live` |
| Lint | `ruff check .` |
| Format | `ruff format .` |
| Verify formatting without writing | `ruff format --check .` |
| Lock file matches requirements | `python scripts/check_lock.py` |
| Validate the container build config | `docker compose config` |
| Validate the Prometheus rules | `docker run --rm -v "$PWD/deploy/prometheus:/etc/prometheus:ro" --entrypoint promtool prom/prometheus:v2.53.0 check rules /etc/prometheus/alerts.yml` |

CI runs the lint, format, lock and test commands on every push, plus a Docker
image smoke test. The workflow is `.github/workflows/ci.yml` and is the
authoritative list.

## 📦 Deployment

| Path | Where to look |
|---|---|
| Docker Compose | `docker-compose.yml`, README |
| TLS / reverse proxy | `deploy/Caddyfile`, README TLS section |
| Metrics and alerts | `deploy/prometheus/prometheus.yml`, `deploy/prometheus/alerts.yml` |
| Dashboards | `deploy/grafana/` |
| Backup and restore | `deploy/backup.sh`, `deploy/restore.sh` |
| Render blueprint | `render.yaml` |
| Secrets | README Secrets section; `deploy/secrets/` |
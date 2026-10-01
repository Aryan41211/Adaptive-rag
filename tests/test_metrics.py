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

from src.core.config import settings as metrics_settings  # noqa: E402
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


# --- the instruments that have to be wired to something ----------------------
# Five instruments were declared but never incremented, so the panels showing
# model spend and upload volume rendered permanently empty while looking like
# a working dashboard. A counter nobody touches is worse than no panel: it
# reads as "no traffic" rather than "no instrumentation".


def _read_metrics() -> str:
    with TestClient(app) as c:
        response = c.get("/metrics/prometheus")
    assert response.status_code == 200
    return response.text


def test_a_completed_turn_counts_its_model_calls(client, monkeypatch):
    """
    The point of tracking spend is being able to see it, so the counter must
    move when a turn records usage.
    """
    from src.core.usage import UsageTracker

    before = _sample(_read_metrics(), "rag_model_calls_total")
    tracker = UsageTracker()
    for _ in range(3):
        tracker.usage.add("gpt-4o", 10, 5)
    tracker.finish()

    after = _sample(_read_metrics(), "rag_model_calls_total")
    assert after == before + 3


def test_token_counters_split_by_direction(client, monkeypatch):
    """Input and output are priced differently, so one total cannot serve both."""
    from src.api.metrics import MODEL_TOKENS
    from src.core.usage import UsageTracker

    tracker = UsageTracker()
    tracker.usage.add("gpt-4o", 200, 80)
    tracker.finish()

    body = _read_metrics()
    # Both directions must exist as distinct series, and each must have moved.
    assert _series_exists(body, "rag_model_tokens_total", {"type": "input"})
    assert _series_exists(body, "rag_model_tokens_total", {"type": "output"})
    assert _sample(body, "rag_model_tokens_total", {"type": "input"}) >= 200
    assert _sample(body, "rag_model_tokens_total", {"type": "output"}) >= 80
    assert MODEL_TOKENS is not None


def test_cumulative_cost_rises_by_the_turn_cost(client):
    """A gauge, not a counter: it holds the running total rather than a delta."""
    from src.core.usage import UsageTracker

    tracker = UsageTracker()
    tracker.usage.add("gpt-4o", 1_000_000, 0)  # prompt price is 2.50/M
    tracker.finish()

    assert _sample(_read_metrics(), "rag_model_cost_usd") >= 2.50


def test_a_turn_with_no_model_calls_records_nothing():
    """
    A turn that failed before reaching a model must not inflate the totals,
    and must not be counted as a call.
    """
    before = _sample(_read_metrics(), "rag_model_calls_total")
    from src.core.usage import UsageTracker

    UsageTracker().finish()

    assert _sample(_read_metrics(), "rag_model_calls_total") == before


def test_uploads_are_counted_by_outcome(client, monkeypatch):
    """
    Success and failure are separate series because "uploads are being
    rejected" is the signal worth alerting on, and a single total hides it.
    """
    from src.api import metrics

    before_ok = _sample(_read_metrics(), "rag_uploads_total", {"status": "success"})
    before_err = _sample(_read_metrics(), "rag_uploads_total", {"status": "error"})

    metrics.UPLOAD_COUNT.labels(status="success").inc()
    metrics.UPLOAD_COUNT.labels(status="error").inc(2)

    body = _read_metrics()
    assert _sample(body, "rag_uploads_total", {"status": "success"}) == before_ok + 1
    assert _sample(body, "rag_uploads_total", {"status": "error"}) == before_err + 2


def test_the_model_call_counter_carries_a_model_label():
    """
    The dashboard groups by model, which is the only way to tell which provider
    is spending. Declared without labels, that query is invalid and the panel
    silently returns nothing.
    """
    body = _read_metrics()
    assert "rag_model_calls_total{model=" in body


def test_no_instrument_is_declared_without_a_place_that_updates_it():
    """
    Structural guard. A declared-but-never-incremented instrument is
    indistinguishable at runtime from one reporting real zeros, so this fails
    at review time instead of on a dashboard nobody trusts.

    It searches every module, metrics.py included: an instrument updated only
    by a helper in the same file is still wired, and one updated nowhere is
    not. Only Prometheus instruments are considered, so an unrelated
    module-level constant cannot fail the build.
    """
    from pathlib import Path

    from prometheus_client import Counter, Gauge, Histogram

    from src.api import metrics as metrics_module

    instrument_types = (Counter, Gauge, Histogram)
    declared = {
        name
        for name in dir(metrics_module)
        if isinstance(getattr(metrics_module, name, None), instrument_types)
    }
    assert declared, "no instruments found; this guard is not testing anything"

    root = Path(__file__).resolve().parents[1] / "src"
    sources = "\n".join(path.read_text(encoding="utf-8") for path in root.rglob("*.py"))

    orphaned = {
        name
        for name in declared
        # Updated as `NAME.labels(...)`, `NAME.inc()` or `NAME.set(...)`.
        if not any(
            f"{name}.{call}" in sources
            for call in ("labels", "inc", "dec", "set", "observe")
        )
    }
    assert not orphaned, f"declared but never updated: {sorted(orphaned)}"


def _labels_of(line: str) -> dict:
    """
    Parse a Prometheus label set into a dict.

    The values keep the quotes the exposition format requires, so they are
    stripped here - otherwise every comparison against a bare expected value
    silently fails and a correctly wired series reads as zero.
    """
    rendered = line.split("{", 1)[1].split("}", 1)[0]
    pairs = {}
    for part in rendered.split(","):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        pairs[key.strip()] = value.strip().strip('"')
    return pairs


def _sample(body: str, name: str, labels: dict | None = None) -> float:
    """
    Sum the samples of one metric, optionally filtered by exact labels.

    A labelled series only appears in a scrape once it exists, so callers
    reading a counter that nothing has touched yet get 0.0 - which is correct,
    and indistinguishable from a series that is wired but never incremented.
    """
    total = 0.0
    for line in body.splitlines():
        if not line.startswith(name):
            continue
        if "{" in line:
            pairs = _labels_of(line)
            if labels and any(pairs.get(k) != v for k, v in labels.items()):
                continue
        elif labels:
            continue
        total += float(line.rsplit(" ", 1)[1])
    return total


def _series_exists(body: str, name: str, labels: dict) -> bool:
    """True when a scrape contains this exact label set."""
    for line in body.splitlines():
        if not line.startswith(f"{name}{{"):
            continue
        if all(_labels_of(line).get(k) == v for k, v in labels.items()):
            return True
    return False


# --- exposure ---------------------------------------------------------------
# The compose path is safe: the API port is loopback-bound and Caddy 404s this
# path. Render has no such edge, so without an in-app guard the route/latency/
# cost internals are readable by anyone who finds the hostname. Requiring a
# token moves the protection into the application, where the deployment target
# cannot route around it.
def test_prometheus_metrics_requires_the_token_when_one_is_configured(monkeypatch):
    monkeypatch.setattr(metrics_settings, "METRICS_TOKEN", "s3cret-token")
    with TestClient(app) as guarded:
        assert guarded.get("/metrics/prometheus").status_code == 401


def test_prometheus_metrics_serves_the_configured_token(monkeypatch):
    monkeypatch.setattr(metrics_settings, "METRICS_TOKEN", "s3cret-token")
    with TestClient(app) as guarded:
        response = guarded.get(
            "/metrics/prometheus", headers={"Authorization": "Bearer s3cret-token"}
        )
    assert response.status_code == 200
    assert "rag_" in response.text


def test_a_wrong_token_is_refused(monkeypatch):
    monkeypatch.setattr(metrics_settings, "METRICS_TOKEN", "s3cret-token")
    with TestClient(app) as guarded:
        assert (
            guarded.get(
                "/metrics/prometheus", headers={"Authorization": "Bearer wrong"}
            ).status_code
            == 401
        )


def test_the_token_is_not_echoed_in_the_refusal(monkeypatch):
    """A 401 body must not become an oracle for the configured token."""
    monkeypatch.setattr(metrics_settings, "METRICS_TOKEN", "s3cret-token")
    with TestClient(app) as guarded:
        response = guarded.get("/metrics/prometheus")
    assert "s3cret-token" not in response.text


def test_metrics_stay_open_when_no_token_is_configured(monkeypatch):
    """
    An unset token keeps the endpoint reachable, which is what the compose
    Prometheus scrape depends on. It is only safe there because that port is
    loopback-bound and the Caddy edge blocks the path.
    """
    monkeypatch.setattr(metrics_settings, "METRICS_TOKEN", "")
    with TestClient(app) as open_endpoint:
        assert open_endpoint.get("/metrics/prometheus").status_code == 200

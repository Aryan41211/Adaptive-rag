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

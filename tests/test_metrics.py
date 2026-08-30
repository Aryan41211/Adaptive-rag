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

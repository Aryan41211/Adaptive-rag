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


def test_rate_limit_headers_on_query(client, monkeypatch):
    """Query responses include rate limit headers."""
    from src.api import routes

    async def fake_run_query(user_id, messages):
        return "an answer", [], {"calls": 0, "total_tokens": 0, "cost_usd": 0.0}

    monkeypatch.setattr(routes, "run_query", fake_run_query)

    headers = register_and_login(client)
    response = client.post(
        "/rag/query",
        json={"query": "test", "session_id": "s1"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    assert "x-ratelimit-limit" in response.headers
    assert "x-ratelimit-remaining" in response.headers
    assert "x-ratelimit-reset" in response.headers


def test_rate_limit_headers_are_integers(client, monkeypatch):
    """Rate limit header values are valid integers."""
    from src.api import routes

    async def fake_run_query(user_id, messages):
        return "an answer", [], {"calls": 0, "total_tokens": 0, "cost_usd": 0.0}

    monkeypatch.setattr(routes, "run_query", fake_run_query)

    headers = register_and_login(client)
    response = client.post(
        "/rag/query",
        json={"query": "test", "session_id": "s2"},
        headers=headers,
    )
    assert response.status_code == 200, response.text
    limit = response.headers.get("x-ratelimit-limit")
    remaining = response.headers.get("x-ratelimit-remaining")
    reset = response.headers.get("x-ratelimit-reset")
    if limit:
        int(limit)
    if remaining:
        int(remaining)
    if reset:
        int(reset)

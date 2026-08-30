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

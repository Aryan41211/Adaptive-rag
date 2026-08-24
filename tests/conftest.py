"""
Shared test fixtures.

Environment variables are set before any application module is imported, so
settings validation sees a complete configuration. No test makes a network
call: embeddings and language models are replaced with deterministic fakes.
"""

import os

# Must precede any `src.*` import: settings are validated at import time.
os.environ["OPENAI_API_KEY"] = "sk-test-key-not-real"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-long-enough-for-validation-0123456789"
os.environ["TAVILY_API_KEY"] = ""
os.environ["MONGODB_URL"] = ""
os.environ["LOG_LEVEL"] = "WARNING"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from src.db import users  # noqa: E402
from src.main import app  # noqa: E402
from src.memory import chat_history_mongo  # noqa: E402
from src.rag import reAct_agent, vector_store  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_state():
    """Reset all process-local stores between tests."""
    users.reset_memory_store()
    chat_history_mongo.reset_memory_store()
    vector_store.reset()
    reAct_agent.reset_cache()
    yield
    users.reset_memory_store()
    chat_history_mongo.reset_memory_store()
    vector_store.reset()
    reAct_agent.reset_cache()


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """Replace OpenAI embeddings with a deterministic local fake."""
    fake = DeterministicFakeEmbedding(size=64)
    monkeypatch.setattr(vector_store, "get_embeddings", lambda: fake)
    return fake


@pytest.fixture
def client_no_raise():
    """
    A client that returns 500 responses instead of re-raising.

    Starlette's TestClient re-raises unhandled server exceptions by default,
    which hides the production behaviour of the generic exception handler.
    """
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


@pytest.fixture
def client():
    """A TestClient bound to the application."""
    with TestClient(app) as test_client:
        yield test_client


def register_and_login(client, username="alice", password="correct-horse-1"):
    """
    Register a user and return their bearer auth header.

    Args:
        client: The test client.
        username: Username to register.
        password: Password to use.

    Returns:
        A headers dict carrying the bearer token.
    """
    response = client.post(
        "/auth/register", json={"username": username, "password": password}
    )
    assert response.status_code == 201, response.text
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def auth_headers(client):
    """Bearer headers for a default registered user."""
    return register_and_login(client)

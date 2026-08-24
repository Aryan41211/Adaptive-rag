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
os.environ["QDRANT_URL"] = ""
os.environ["QDRANT_API_KEY"] = ""
os.environ["LOG_LEVEL"] = "WARNING"

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from langchain_core.embeddings import DeterministicFakeEmbedding  # noqa: E402

from src.api import ratelimit  # noqa: E402
from src.db import revoked_tokens, users  # noqa: E402
from src.main import app  # noqa: E402
from src.memory import chat_history_mongo  # noqa: E402
from src.rag import reAct_agent, vector_store  # noqa: E402
from src.rag.backends import faiss_backend, qdrant_backend  # noqa: E402

# Small enough to keep tests fast; the value only has to be consistent.
FAKE_EMBEDDING_SIZE = 64


@pytest.fixture(autouse=True)
def fake_embeddings(monkeypatch):
    """
    Replace OpenAI embeddings with a deterministic local fake.

    Each backend imports the factory by name, so both bindings are patched.
    """
    fake = DeterministicFakeEmbedding(size=FAKE_EMBEDDING_SIZE)
    for module in (faiss_backend, qdrant_backend):
        monkeypatch.setattr(module, "get_embeddings", lambda: fake)
    return fake


@pytest.fixture(autouse=True)
def _clean_state(fake_embeddings):
    """Reset all process-local stores between tests."""

    def _reset():
        users.reset_memory_store()
        chat_history_mongo.reset_memory_store()
        revoked_tokens.reset_memory_store()
        # Counters are per-process here, so without this the suite's own
        # registrations would trip the auth limit part-way through a run.
        ratelimit.reset_memory_counters()
        reAct_agent.reset_cache()
        # Drop the backend instance rather than clearing it: the next test
        # rebuilds from whatever settings it establishes.
        vector_store.set_backend(None)

    _reset()
    yield
    _reset()


@pytest.fixture
def qdrant_backend_instance():
    """A real Qdrant backend running fully in process."""
    from qdrant_client import QdrantClient

    client = QdrantClient(location=":memory:")
    backend = qdrant_backend.QdrantBackend(
        client=client, collection_name="test_documents"
    )
    vector_store.set_backend(backend)
    yield backend
    vector_store.set_backend(None)


@pytest.fixture
def faiss_backend_instance():
    """An in-memory FAISS backend."""
    backend = faiss_backend.FaissBackend()
    vector_store.set_backend(backend)
    yield backend
    vector_store.set_backend(None)


@pytest.fixture(params=["faiss", "qdrant"])
def any_backend(request):
    """
    Run a test against every backend.

    The isolation guarantees must hold identically whichever store is active.
    """
    backend = request.getfixturevalue(f"{request.param}_backend_instance")
    backend.test_name = request.param
    return backend


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

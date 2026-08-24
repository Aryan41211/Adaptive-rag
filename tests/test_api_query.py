"""
Tests for the query endpoint: validation, error translation and data scoping.
"""

import pytest

from src.core.exceptions import RetrievalError
from src.memory.chat_history_mongo import ChatHistory
from tests.conftest import register_and_login


@pytest.fixture
def stub_graph(monkeypatch):
    """Replace the graph with a recorder so no model is called."""
    import src.api.routes as routes

    calls = []

    async def fake_run_query(user_id, messages):
        calls.append({"user_id": user_id, "messages": messages})
        return "stub answer", [], {}

    monkeypatch.setattr(routes, "run_query", fake_run_query)
    return calls


def test_query_returns_an_answer(client, auth_headers, stub_graph):
    response = client.post(
        "/rag/query",
        json={"query": "what is in my document?", "session_id": "s1"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert response.json() == {
        "answer": "stub answer",
        "session_id": "s1",
        "citations": [],
        "usage": {},
    }


def test_query_is_scoped_to_the_calling_user(client, auth_headers, stub_graph):
    """The graph must receive the authenticated user, not a client-supplied id."""
    client.post(
        "/rag/query",
        json={"query": "hello", "session_id": "s1"},
        headers=auth_headers,
    )
    assert stub_graph[0]["user_id"]
    assert stub_graph[0]["user_id"] != "s1"


async def test_conversation_is_persisted(client, auth_headers, stub_graph):
    client.post(
        "/rag/query",
        json={"query": "first question", "session_id": "s1"},
        headers=auth_headers,
    )

    user_id = stub_graph[0]["user_id"]
    messages = await ChatHistory.get_session_history(user_id, "s1").get_messages()
    assert [m.content for m in messages] == ["first question", "stub answer"]


def test_history_accumulates_across_turns(client, auth_headers, stub_graph):
    for text in ("first", "second"):
        client.post(
            "/rag/query",
            json={"query": text, "session_id": "s1"},
            headers=auth_headers,
        )

    # Second call sees the first exchange plus the new question.
    assert [m.content for m in stub_graph[1]["messages"]] == [
        "first",
        "stub answer",
        "second",
    ]


def test_one_users_session_id_cannot_reach_anothers_history(client, stub_graph):
    """The IDOR that a client-supplied session_id used to allow."""
    alice = register_and_login(client, "alice", "alice-password-1")
    client.post(
        "/rag/query",
        json={"query": "alice secret", "session_id": "shared"},
        headers=alice,
    )

    bob = register_and_login(client, "bob", "bob-password-1")
    client.post(
        "/rag/query",
        json={"query": "bob question", "session_id": "shared"},
        headers=bob,
    )

    bob_sees = [m.content for m in stub_graph[1]["messages"]]
    assert bob_sees == ["bob question"]
    assert "alice secret" not in bob_sees


# --- validation ------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"query": "", "session_id": "s1"},
        {"query": "   ", "session_id": "s1"},
        {"query": "hello", "session_id": ""},
        {"query": "hello", "session_id": "has spaces"},
        {"query": "hello", "session_id": "../../etc/passwd"},
        {"query": "x" * 5000, "session_id": "s1"},
        {"session_id": "s1"},
        {"query": "hello"},
    ],
)
def test_invalid_payloads_are_rejected(client, auth_headers, stub_graph, payload):
    response = client.post("/rag/query", json=payload, headers=auth_headers)
    assert response.status_code == 422
    assert "detail" in response.json()


def test_validation_errors_name_the_offending_field(client, auth_headers):
    response = client.post(
        "/rag/query", json={"query": "", "session_id": "s1"}, headers=auth_headers
    )
    fields = [e["field"] for e in response.json()["errors"]]
    assert "query" in fields


# --- error translation -----------------------------------------------------
def test_pipeline_failure_returns_a_clean_error(client, auth_headers, monkeypatch):
    import src.api.routes as routes

    async def failing_run_query(user_id, messages):
        raise RetrievalError("Could not converge on an answer.")

    monkeypatch.setattr(routes, "run_query", failing_run_query)

    response = client.post(
        "/rag/query",
        json={"query": "hello", "session_id": "s1"},
        headers=auth_headers,
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "Could not converge on an answer."


def test_unexpected_failure_does_not_leak_internals(client_no_raise, monkeypatch):
    import src.api.routes as routes

    async def exploding_run_query(user_id, messages):
        raise RuntimeError("secret internal detail: db password is hunter2")

    monkeypatch.setattr(routes, "run_query", exploding_run_query)

    auth_headers = register_and_login(client_no_raise)
    response = client_no_raise.post(
        "/rag/query",
        json={"query": "hello", "session_id": "s1"},
        headers=auth_headers,
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Internal server error."}
    assert "hunter2" not in response.text


# --- session management ----------------------------------------------------
async def test_clear_session_removes_history(client, auth_headers, stub_graph):
    client.post(
        "/rag/query",
        json={"query": "hello", "session_id": "s1"},
        headers=auth_headers,
    )
    user_id = stub_graph[0]["user_id"]

    response = client.delete("/rag/sessions/s1", headers=auth_headers)
    assert response.status_code == 204

    messages = await ChatHistory.get_session_history(user_id, "s1").get_messages()
    assert messages == []


# --- correlation & health --------------------------------------------------
def test_request_id_is_echoed(client):
    response = client.get("/healthz", headers={"X-Request-ID": "abc123"})
    assert response.headers["X-Request-ID"] == "abc123"


def test_request_id_is_generated_when_absent(client):
    assert client.get("/healthz").headers["X-Request-ID"]


def test_health_endpoints(client):
    assert client.get("/healthz").json() == {"status": "ok"}
    assert client.get("/").json()["status"] == "running"

    ready = client.get("/readyz").json()
    assert ready["status"] == "ok"
    assert ready["persistence"] == "not-configured"
    assert ready["web_search"] == "disabled"


def test_readyz_reports_the_vector_backend(client):
    body = client.get("/readyz").json()
    assert body["vector_store"]
    assert "faiss" in body["vector_store"]


def test_readyz_is_unavailable_when_the_vector_store_is_unreachable(
    client, monkeypatch
):
    """The service cannot answer anything without its document store."""
    import src.main as main

    monkeypatch.setattr(
        main.vector_store, "health", lambda: (False, "qdrant unreachable")
    )

    response = client.get("/readyz")
    assert response.status_code == 503
    assert response.json()["status"] == "degraded"
    assert response.json()["vector_store"] == "qdrant unreachable"


@pytest.mark.parametrize("path", ["/", "/healthz", "/readyz"])
def test_health_endpoints_accept_head(client, path):
    """Many load balancers probe with HEAD; a 405 reads as unhealthy."""
    assert client.head(path).status_code == 200

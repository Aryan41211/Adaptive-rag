"""
Tests for the authentication endpoints and endpoint protection.
"""

from tests.conftest import register_and_login


def test_register_returns_token(client):
    response = client.post(
        "/auth/register", json={"username": "alice", "password": "hunter2-long"}
    )
    assert response.status_code == 201
    body = response.json()
    assert body["access_token"]
    assert body["token_type"] == "bearer"
    assert body["username"] == "alice"


def test_duplicate_registration_conflicts(client):
    payload = {"username": "alice", "password": "hunter2-long"}
    assert client.post("/auth/register", json=payload).status_code == 201
    assert client.post("/auth/register", json=payload).status_code == 409


def test_duplicate_registration_is_case_insensitive(client):
    client.post(
        "/auth/register", json={"username": "Alice", "password": "hunter2-long"}
    )
    response = client.post(
        "/auth/register", json={"username": "alice", "password": "hunter2-long"}
    )
    assert response.status_code == 409


def test_login_succeeds_with_correct_password(client):
    client.post(
        "/auth/register", json={"username": "alice", "password": "hunter2-long"}
    )
    response = client.post(
        "/auth/login", json={"username": "alice", "password": "hunter2-long"}
    )
    assert response.status_code == 200
    assert response.json()["access_token"]


def test_login_fails_with_wrong_password(client):
    client.post(
        "/auth/register", json={"username": "alice", "password": "hunter2-long"}
    )
    response = client.post(
        "/auth/login", json={"username": "alice", "password": "wrong-password"}
    )
    assert response.status_code == 401


def test_login_fails_for_unknown_user(client):
    response = client.post(
        "/auth/login", json={"username": "nobody", "password": "hunter2-long"}
    )
    assert response.status_code == 401


def test_short_password_rejected(client):
    response = client.post(
        "/auth/register", json={"username": "alice", "password": "short"}
    )
    assert response.status_code == 422


def test_invalid_username_rejected(client):
    response = client.post(
        "/auth/register", json={"username": "bad user!", "password": "hunter2-long"}
    )
    assert response.status_code == 422


# --- Endpoint protection ---------------------------------------------------
def test_query_requires_authentication(client):
    response = client.post("/rag/query", json={"query": "hello", "session_id": "s1"})
    assert response.status_code == 401


def test_upload_requires_authentication(client):
    response = client.post(
        "/rag/documents/upload",
        files={"file": ("a.txt", b"hello", "text/plain")},
        headers={"X-Description": "notes"},
    )
    assert response.status_code == 401


def test_session_delete_requires_authentication(client):
    assert client.delete("/rag/sessions/s1").status_code == 401


def test_garbage_token_rejected(client):
    response = client.post(
        "/rag/query",
        json={"query": "hello", "session_id": "s1"},
        headers={"Authorization": "Bearer not-a-real-token"},
    )
    assert response.status_code == 401


def test_valid_token_passes_authentication(client, monkeypatch):
    """A valid token must reach the handler rather than being rejected."""
    import src.api.routes as routes

    async def fake_run_query(user_id, messages):
        return "an answer", []

    monkeypatch.setattr(routes, "run_query", fake_run_query)

    headers = register_and_login(client)
    response = client.post(
        "/rag/query", json={"query": "hello", "session_id": "s1"}, headers=headers
    )
    assert response.status_code == 200

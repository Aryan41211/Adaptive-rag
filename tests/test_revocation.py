"""
Access-token revocation.

A JWT is self-contained: discarding it client-side leaves a copied token
usable until it expires. Signing out must make it unusable immediately.
"""

from datetime import datetime, timedelta, timezone

import pytest

from src.db import revoked_tokens
from tests.conftest import register_and_login


@pytest.fixture
def stub_graph(monkeypatch):
    import src.api.routes as routes

    async def fake_run_query(user_id, messages):
        return "stub answer", [], {}

    monkeypatch.setattr(routes, "run_query", fake_run_query)


# --- the store -------------------------------------------------------------
async def test_unknown_token_is_not_revoked():
    assert await revoked_tokens.is_revoked("never-seen") is False


async def test_missing_jti_is_not_revoked():
    assert await revoked_tokens.is_revoked(None) is False
    assert await revoked_tokens.is_revoked("") is False


async def test_revoked_token_is_reported():
    expiry = datetime.now(timezone.utc) + timedelta(minutes=30)
    await revoked_tokens.revoke("token-1", expiry)
    assert await revoked_tokens.is_revoked("token-1") is True


async def test_revocation_is_per_token():
    expiry = datetime.now(timezone.utc) + timedelta(minutes=30)
    await revoked_tokens.revoke("token-1", expiry)
    assert await revoked_tokens.is_revoked("token-2") is False


async def test_entry_stops_applying_once_the_token_would_have_expired():
    """The denylist only needs to outlive the token itself."""
    past = datetime.now(timezone.utc) - timedelta(minutes=1)
    await revoked_tokens.revoke("token-1", past)
    assert await revoked_tokens.is_revoked("token-1") is False


# --- the endpoint ----------------------------------------------------------
def test_logout_revokes_the_calling_token(client, stub_graph):
    headers = register_and_login(client)
    payload = {"query": "hello", "session_id": "s1"}

    assert client.post("/rag/query", json=payload, headers=headers).status_code == 200

    assert client.post("/auth/logout", headers=headers).status_code == 204

    response = client.post("/rag/query", json=payload, headers=headers)
    assert response.status_code == 401
    assert "signed out" in response.json()["detail"]


def test_logout_requires_authentication(client):
    assert client.post("/auth/logout").status_code == 401


def test_logout_does_not_affect_other_sessions(client, stub_graph):
    """Signing out one device must not sign out the others."""
    client.post(
        "/auth/register", json={"username": "alice", "password": "alice-password-1"}
    )
    first = client.post(
        "/auth/login", json={"username": "alice", "password": "alice-password-1"}
    ).json()["access_token"]
    second = client.post(
        "/auth/login", json={"username": "alice", "password": "alice-password-1"}
    ).json()["access_token"]

    assert first != second
    client.post("/auth/logout", headers={"Authorization": f"Bearer {first}"})

    payload = {"query": "hello", "session_id": "s1"}
    assert (
        client.post(
            "/rag/query", json=payload, headers={"Authorization": f"Bearer {first}"}
        ).status_code
        == 401
    )
    assert (
        client.post(
            "/rag/query", json=payload, headers={"Authorization": f"Bearer {second}"}
        ).status_code
        == 200
    )


def test_logging_in_again_after_logout_works(client, stub_graph):
    headers = register_and_login(client)
    client.post("/auth/logout", headers=headers)

    fresh = client.post(
        "/auth/login",
        json={"username": "alice", "password": "correct-horse-1"},
    )
    assert fresh.status_code == 200

    new_headers = {"Authorization": f"Bearer {fresh.json()['access_token']}"}
    assert (
        client.post(
            "/rag/query",
            json={"query": "hello", "session_id": "s1"},
            headers=new_headers,
        ).status_code
        == 200
    )


def test_revocation_check_failure_does_not_lock_users_out(
    client, monkeypatch, stub_graph
):
    """
    Failing closed here would turn a store outage into a total lockout.

    Failing open only weakens sign-out, which is the lesser harm.
    """

    class _BrokenCollection:
        async def find_one(self, *_args, **_kwargs):
            raise RuntimeError("store down")

    class _BrokenDatabase:
        def __getitem__(self, _name):
            return _BrokenCollection()

    headers = register_and_login(client)
    monkeypatch.setattr(revoked_tokens, "get_database", lambda: _BrokenDatabase())

    response = client.post(
        "/rag/query", json={"query": "hello", "session_id": "s1"}, headers=headers
    )
    assert response.status_code == 200

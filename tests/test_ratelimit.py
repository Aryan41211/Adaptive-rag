"""
Rate limiting and token revocation.

Without a limit a single account can drive unbounded model spend: every query
runs several LLM calls and every upload embeds a whole document.
"""

import pytest

from src.api import ratelimit
from src.core.config import settings
from src.core.exceptions import RateLimitExceededError
from tests.conftest import register_and_login


@pytest.fixture
def stub_graph(monkeypatch):
    """Answer queries without calling a model."""
    import src.api.routes as routes

    async def fake_run_query(user_id, messages):
        return "stub answer", []

    monkeypatch.setattr(routes, "run_query", fake_run_query)


@pytest.fixture
def tight_limits(monkeypatch):
    """Small quotas so limits are reachable in a test."""
    monkeypatch.setattr(settings, "RATE_LIMIT_QUERY_PER_MINUTE", 3)
    monkeypatch.setattr(settings, "RATE_LIMIT_UPLOAD_PER_HOUR", 2)
    monkeypatch.setattr(settings, "RATE_LIMIT_AUTH_PER_MINUTE", 3)


# --- the limiter itself ----------------------------------------------------
async def test_requests_within_quota_are_allowed():
    quota = ratelimit.Quota(limit=3, window_seconds=60, scope="test")
    for _ in range(3):
        await ratelimit.enforce(quota, "user-a")


async def test_exceeding_the_quota_raises():
    quota = ratelimit.Quota(limit=2, window_seconds=60, scope="test")
    await ratelimit.enforce(quota, "user-a")
    await ratelimit.enforce(quota, "user-a")

    with pytest.raises(RateLimitExceededError) as exc:
        await ratelimit.enforce(quota, "user-a")
    assert exc.value.status_code == 429
    assert exc.value.retry_after >= 1


async def test_quotas_are_per_caller():
    """One user exhausting their quota must not affect another."""
    quota = ratelimit.Quota(limit=1, window_seconds=60, scope="test")
    await ratelimit.enforce(quota, "user-a")

    with pytest.raises(RateLimitExceededError):
        await ratelimit.enforce(quota, "user-a")

    await ratelimit.enforce(quota, "user-b")  # unaffected


async def test_quotas_are_per_scope():
    """Uploading must not consume the query allowance."""
    queries = ratelimit.Quota(limit=1, window_seconds=60, scope="query")
    uploads = ratelimit.Quota(limit=1, window_seconds=60, scope="upload")

    await ratelimit.enforce(queries, "user-a")
    await ratelimit.enforce(uploads, "user-a")


async def test_limiting_can_be_disabled(monkeypatch):
    monkeypatch.setattr(settings, "RATE_LIMIT_ENABLED", False)
    quota = ratelimit.Quota(limit=1, window_seconds=60, scope="test")
    for _ in range(10):
        await ratelimit.enforce(quota, "user-a")


async def test_counter_outage_fails_open(monkeypatch):
    """
    A counter-store outage must degrade the protection, not the API.

    Failing closed would turn a MongoDB blip into a total outage; failing
    open only loses rate limiting while the store is down.
    """

    class _BrokenCollection:
        async def find_one_and_update(self, *_args, **_kwargs):
            raise RuntimeError("counter store down")

    class _BrokenDatabase:
        def __getitem__(self, _name):
            return _BrokenCollection()

    monkeypatch.setattr(ratelimit, "get_database", lambda: _BrokenDatabase())

    quota = ratelimit.Quota(limit=1, window_seconds=60, scope="test")
    for _ in range(5):
        await ratelimit.enforce(quota, "user-a")  # must not raise


async def test_quota_reflects_configuration_changes(monkeypatch):
    """The limit must be read per request, not frozen at import time."""
    limiter = ratelimit.query_rate_limit()
    monkeypatch.setattr(settings, "RATE_LIMIT_QUERY_PER_MINUTE", 7)
    assert limiter.quota.limit == 7
    monkeypatch.setattr(settings, "RATE_LIMIT_QUERY_PER_MINUTE", 9)
    assert limiter.quota.limit == 9


# --- enforced on the endpoints --------------------------------------------
def test_query_endpoint_enforces_its_quota(client, tight_limits, stub_graph):
    headers = register_and_login(client)
    payload = {"query": "hello", "session_id": "s1"}

    for _ in range(3):
        assert (
            client.post("/rag/query", json=payload, headers=headers).status_code == 200
        )

    response = client.post("/rag/query", json=payload, headers=headers)
    assert response.status_code == 429
    assert "Retry-After" in response.headers
    assert int(response.headers["Retry-After"]) >= 1


def test_upload_endpoint_enforces_its_quota(client, tight_limits, monkeypatch):
    import src.rag.document_upload as module

    monkeypatch.setattr(
        module, "enhance_description_with_llm", lambda text: f"about {text}"
    )
    headers = register_and_login(client)

    def _upload(index):
        return client.post(
            "/rag/documents/upload",
            files={"file": (f"n{index}.txt", b"some content", "text/plain")},
            headers={**headers, "X-Description": "notes"},
        )

    assert _upload(1).status_code == 200
    assert _upload(2).status_code == 200
    assert _upload(3).status_code == 429


def test_auth_endpoints_are_limited_by_address(client, tight_limits):
    """Slows credential guessing, which is not tied to an account."""
    codes = [
        client.post(
            "/auth/login",
            json={"username": "nobody", "password": "guess-attempt-1"},
        ).status_code
        for _ in range(5)
    ]
    assert 429 in codes
    assert codes.count(401) == 3


def test_one_users_quota_does_not_affect_another(client, tight_limits, stub_graph):
    alice = register_and_login(client, "alice", "alice-password-1")
    payload = {"query": "hello", "session_id": "s1"}

    for _ in range(3):
        client.post("/rag/query", json=payload, headers=alice)
    assert client.post("/rag/query", json=payload, headers=alice).status_code == 429

    # Registration is address-limited, so build bob's token directly.
    from src.core.security import create_access_token

    bob = {"Authorization": f"Bearer {create_access_token('bob-id', 'bob')}"}
    assert client.post("/rag/query", json=payload, headers=bob).status_code == 200


def test_rate_limit_body_explains_the_limit(client, tight_limits, stub_graph):
    headers = register_and_login(client)
    payload = {"query": "hello", "session_id": "s1"}
    for _ in range(4):
        response = client.post("/rag/query", json=payload, headers=headers)

    body = response.json()
    assert body["error"] == "RateLimitExceededError"
    assert "3 requests" in body["detail"]

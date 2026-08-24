"""
Request rate limiting.

Without this a single account can drive unbounded model spend: every query
runs several LLM calls and every upload embeds a whole document.

Counters use a fixed window. When MongoDB is configured they are shared, so
the limit applies to the deployment rather than to each worker separately -
with four workers, per-process counters would allow four times the intended
rate. Without MongoDB the fallback is per-process, which is stated in the
README rather than papered over.
"""

import time
from dataclasses import dataclass

from fastapi import Depends, Request

from src.api.deps import CurrentUser, get_current_user
from src.core.config import settings
from src.core.exceptions import RateLimitExceededError
from src.core.logger import get_logger
from src.db.mongo_client import get_database

logger = get_logger(__name__)

COLLECTION_NAME = "rate_limits"

# Fallback counters: {bucket_key: (window_start, count)}
_memory_counters: dict[str, tuple[int, int]] = {}


@dataclass(frozen=True)
class Quota:
    """A number of requests permitted within a window."""

    limit: int
    window_seconds: int
    scope: str


async def _increment(key: str, window_start: int, ttl_seconds: int) -> int:
    """
    Increment a window's counter and return its new value.

    Args:
        key: Bucket identity, unique per (scope, caller).
        window_start: Epoch second the current window began.
        ttl_seconds: How long the record should outlive the window.

    Returns:
        The number of requests seen in this window, including this one.
    """
    database = get_database()
    bucket = f"{key}:{window_start}"

    if database is None:
        seen_window, count = _memory_counters.get(bucket, (window_start, 0))
        count = count + 1 if seen_window == window_start else 1
        _memory_counters[bucket] = (window_start, count)
        _prune_memory_counters(window_start)
        return count

    try:
        document = await database[COLLECTION_NAME].find_one_and_update(
            {"_id": bucket},
            {
                "$inc": {"count": 1},
                "$setOnInsert": {"expires_at": window_start + ttl_seconds},
            },
            upsert=True,
            return_document=True,
        )
        return int(document["count"])
    except Exception as exc:  # noqa: BLE001 - never fail a request on this
        # Failing open is deliberate: a counter outage should degrade the
        # protection, not take the API down with it.
        logger.warning("Rate limit counter unavailable, allowing request: %s", exc)
        return 0


def _prune_memory_counters(window_start: int) -> None:
    """Drop counters from windows that have passed."""
    if len(_memory_counters) < 10_000:
        return
    for bucket, (seen_window, _count) in list(_memory_counters.items()):
        if seen_window < window_start:
            _memory_counters.pop(bucket, None)


async def enforce(quota: Quota, identity: str) -> None:
    """
    Consume one unit of a caller's quota.

    Args:
        quota: The limit to apply.
        identity: The caller, already namespaced by scope.

    Raises:
        RateLimitExceededError: If the caller is over the limit.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return

    now = int(time.time())
    window_start = now - (now % quota.window_seconds)
    key = f"{quota.scope}:{identity}"

    count = await _increment(key, window_start, quota.window_seconds * 2)
    if count > quota.limit:
        retry_after = window_start + quota.window_seconds - now
        logger.info(
            "Rate limit hit for scope '%s' (%d/%d)",
            quota.scope,
            count,
            quota.limit,
        )
        raise RateLimitExceededError(
            f"Rate limit exceeded: at most {quota.limit} requests per "
            f"{quota.window_seconds} seconds. Try again in "
            f"{max(retry_after, 1)}s.",
            retry_after=max(retry_after, 1),
        )


def _client_ip(request: Request) -> str:
    """
    Best-effort client address.

    X-Forwarded-For is only trusted when a proxy is expected; uvicorn must be
    run with --forwarded-allow-ips for it to be populated safely.
    """
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


class _RateLimit:
    """
    Base dependency resolving its quota per request.

    The limit is read from settings when the request arrives rather than when
    the dependency is constructed. Constructing it at import time would freeze
    the configured value into the route signature, so changing the limit would
    have no effect until the process restarted.
    """

    def __init__(self, scope: str, limit_setting: str, window_seconds: int):
        self.scope = scope
        self.limit_setting = limit_setting
        self.window_seconds = window_seconds

    @property
    def quota(self) -> Quota:
        """The quota as currently configured."""
        return Quota(
            limit=getattr(settings, self.limit_setting),
            window_seconds=self.window_seconds,
            scope=self.scope,
        )


class UserRateLimit(_RateLimit):
    """Limits an authenticated user's requests."""

    async def __call__(
        self, user: CurrentUser = Depends(get_current_user)
    ) -> CurrentUser:
        await enforce(self.quota, user.user_id)
        return user


class IpRateLimit(_RateLimit):
    """Limits unauthenticated requests by source address."""

    async def __call__(self, request: Request) -> None:
        await enforce(self.quota, _client_ip(request))


def query_rate_limit() -> UserRateLimit:
    """Per-user limit for the query endpoint."""
    return UserRateLimit("query", "RATE_LIMIT_QUERY_PER_MINUTE", 60)


def upload_rate_limit() -> UserRateLimit:
    """Per-user limit for document uploads."""
    return UserRateLimit("upload", "RATE_LIMIT_UPLOAD_PER_HOUR", 3600)


def auth_rate_limit() -> IpRateLimit:
    """Per-address limit for credential endpoints, to slow guessing."""
    return IpRateLimit("auth", "RATE_LIMIT_AUTH_PER_MINUTE", 60)


async def ensure_indexes() -> None:
    """Create the TTL index that expires spent counters."""
    database = get_database()
    if database is None:
        return
    await database[COLLECTION_NAME].create_index("expires_at", expireAfterSeconds=0)


def reset_memory_counters() -> None:
    """Clear the fallback counters. Intended for tests."""
    _memory_counters.clear()


def _quota_for(scope: str) -> Quota | None:
    """Return the configured quota for a scope. Used by tests."""
    factories = {
        "query": query_rate_limit,
        "upload": upload_rate_limit,
        "auth": auth_rate_limit,
    }
    factory = factories.get(scope)
    return factory().quota if factory else None

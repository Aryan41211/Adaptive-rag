"""
Access-token revocation.

JWTs are self-contained, so signing out cannot invalidate one on its own: a
copied token stays valid until it expires. This denylist records revoked token
identifiers until their natural expiry, at which point the entry is no longer
needed and MongoDB's TTL index removes it.

Shared through MongoDB when configured, so a sign-out on one worker is
honoured by all of them.
"""

from datetime import datetime, timezone

from src.core.logger import get_logger
from src.db.mongo_client import get_database

logger = get_logger(__name__)

COLLECTION_NAME = "revoked_tokens"

# Fallback store: {jti: expiry timestamp}
_memory_revoked: dict[str, float] = {}


async def revoke(jti: str, expires_at: datetime) -> None:
    """
    Mark a token identifier as revoked.

    Args:
        jti: The token's unique identifier claim.
        expires_at: When the token would have expired anyway.
    """
    database = get_database()
    if database is None:
        _memory_revoked[jti] = expires_at.timestamp()
        _prune_memory()
        return

    await database[COLLECTION_NAME].update_one(
        {"_id": jti},
        {"$set": {"expires_at": expires_at}},
        upsert=True,
    )


async def is_revoked(jti: str | None) -> bool:
    """
    Report whether a token has been revoked.

    Args:
        jti: The token's identifier claim, if it has one.

    Returns:
        True if the token must be rejected.
    """
    if not jti:
        return False

    database = get_database()
    if database is None:
        expiry = _memory_revoked.get(jti)
        if expiry is None:
            return False
        if expiry < datetime.now(timezone.utc).timestamp():
            _memory_revoked.pop(jti, None)
            return False
        return True

    try:
        return await database[COLLECTION_NAME].find_one({"_id": jti}) is not None
    except Exception as exc:  # noqa: BLE001
        # Fail closed would lock every user out during an outage; fail open
        # only weakens sign-out, which is the lesser harm.
        logger.warning("Revocation check unavailable, allowing token: %s", exc)
        return False


def _prune_memory() -> None:
    """Drop expired entries from the fallback store."""
    now = datetime.now(timezone.utc).timestamp()
    for jti, expiry in list(_memory_revoked.items()):
        if expiry < now:
            _memory_revoked.pop(jti, None)


async def ensure_indexes() -> None:
    """Create the TTL index that expires spent entries."""
    database = get_database()
    if database is None:
        return
    await database[COLLECTION_NAME].create_index("expires_at", expireAfterSeconds=0)


def reset_memory_store() -> None:
    """Clear the fallback store. Intended for tests."""
    _memory_revoked.clear()

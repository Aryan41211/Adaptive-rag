"""
MongoDB client management.

MongoDB is optional. When ``MONGODB_URL`` is not configured the application
falls back to in-memory storage, which keeps local development and tests
runnable without a database. The client is created lazily so that importing
this module never opens a socket.
"""

from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

_client: Optional[AsyncIOMotorClient] = None


def get_client() -> Optional[AsyncIOMotorClient]:
    """
    Return the shared Motor client, creating it on first use.

    Returns:
        The client, or None when MongoDB is not configured.
    """
    global _client
    if not settings.persistence_enabled:
        return None
    if _client is None:
        logger.info("Initialising MongoDB client")
        _client = AsyncIOMotorClient(
            settings.MONGODB_URL,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=5000,
            uuidRepresentation="standard",
        )
    return _client


def get_database() -> Optional[AsyncIOMotorDatabase]:
    """
    Return the application database handle.

    Returns:
        The database, or None when MongoDB is not configured.
    """
    client = get_client()
    return client[settings.MONGODB_DB_NAME] if client is not None else None


async def ping() -> bool:
    """
    Check connectivity to MongoDB.

    Returns:
        True if the server responded, False if unreachable or unconfigured.
    """
    client = get_client()
    if client is None:
        return False
    try:
        await client.admin.command("ping")
        return True
    except Exception as exc:  # noqa: BLE001 - surfaced as a health signal
        logger.warning("MongoDB ping failed: %s", exc)
        return False


async def ensure_indexes() -> None:
    """
    Create the indexes the application queries against.

    Without the compound chat-history index every history read is a full
    collection scan, which degrades badly as conversations accumulate.
    Index creation is idempotent.
    """
    db = get_database()
    if db is None:
        return
    try:
        await db["users"].create_index("username_lower", unique=True)
        # Matches the (user_id, session_id) filter plus the timestamp sort.
        await db["chat_history"].create_index(
            [("user_id", 1), ("session_id", 1), ("timestamp", -1)]
        )
        logger.info("MongoDB indexes ensured")
    except Exception as exc:  # noqa: BLE001 - startup must not hard-fail here
        logger.error("Could not create MongoDB indexes: %s", exc)


async def close_client() -> None:
    """Close the shared client on application shutdown."""
    global _client
    if _client is not None:
        _client.close()
        _client = None
        logger.info("MongoDB client closed")

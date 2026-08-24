"""
User storage.

Backed by MongoDB when configured, otherwise by a process-local dictionary so
that the application remains usable without a database. Both backends expose
the same async interface.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

from src.core.logger import get_logger
from src.db.mongo_client import get_database

logger = get_logger(__name__)

# Fallback store used when MongoDB is not configured. Keyed by lowercase
# username. Not durable - documented as such in the README.
_memory_users: dict[str, dict[str, Any]] = {}


def _normalise(username: str) -> str:
    """Return the canonical lookup form of a username."""
    return username.strip().lower()


async def _ensure_indexes() -> None:
    """Create the unique username index (idempotent)."""
    db = get_database()
    if db is not None:
        await db["users"].create_index("username_lower", unique=True)


async def get_user_by_username(username: str) -> dict[str, Any] | None:
    """
    Look up a user by username, case-insensitively.

    Args:
        username: The username to find.

    Returns:
        The user document, or None if no such user exists.
    """
    key = _normalise(username)
    db = get_database()
    if db is None:
        return _memory_users.get(key)
    return await db["users"].find_one({"username_lower": key})


async def create_user(username: str, password_hash: str) -> dict[str, Any]:
    """
    Persist a new user.

    Args:
        username: The requested username.
        password_hash: The bcrypt hash of the user's password.

    Returns:
        The created user document.

    Raises:
        ValueError: If the username is already taken.
    """
    key = _normalise(username)
    document = {
        "user_id": uuid.uuid4().hex,
        "username": username.strip(),
        "username_lower": key,
        "password_hash": password_hash,
        "created_at": datetime.now(timezone.utc),
    }

    db = get_database()
    if db is None:
        if key in _memory_users:
            raise ValueError("Username already registered.")
        _memory_users[key] = document
        return document

    await _ensure_indexes()
    from pymongo.errors import DuplicateKeyError

    try:
        await db["users"].insert_one(dict(document))
    except DuplicateKeyError as exc:
        raise ValueError("Username already registered.") from exc
    return document


def reset_memory_store() -> None:
    """Clear the in-memory user store. Intended for tests."""
    _memory_users.clear()


async def delete_user(user_id: str) -> bool:
    """
    Remove a user record.

    Args:
        user_id: The user to delete.

    Returns:
        True if a record was removed.
    """
    db = get_database()
    if db is None:
        for key, document in list(_memory_users.items()):
            if document["user_id"] == user_id:
                _memory_users.pop(key, None)
                return True
        return False

    result = await db["users"].delete_one({"user_id": user_id})
    return result.deleted_count > 0

"""
Tests for the MongoDB-backed storage path.

The in-memory fallback is the default in development, so without these the
production persistence configuration would be entirely untested. A mock Motor
client stands in for a real server.
"""

import pytest
from mongomock_motor import AsyncMongoMockClient

from src.db import users
from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import RegisterRequest  # noqa: F401  (schema import)


@pytest.fixture
def mongo(monkeypatch):
    """Point every storage module at a mock MongoDB database."""
    database = AsyncMongoMockClient()["adaptive_rag_test"]

    from src.db import mongo_client
    from src.memory import chat_history_mongo

    monkeypatch.setattr(mongo_client, "get_database", lambda: database)
    monkeypatch.setattr(users, "get_database", lambda: database)
    monkeypatch.setattr(chat_history_mongo, "get_database", lambda: database)
    return database


# --- users -----------------------------------------------------------------
async def test_user_is_persisted_to_mongo(mongo):
    created = await users.create_user("alice", "hashed-value")

    stored = await mongo["users"].find_one({"username_lower": "alice"})
    assert stored is not None
    assert stored["user_id"] == created["user_id"]
    assert stored["password_hash"] == "hashed-value"


async def test_lookup_is_case_insensitive_in_mongo(mongo):
    await users.create_user("Alice", "hashed-value")
    assert await users.get_user_by_username("ALICE") is not None
    assert await users.get_user_by_username("alice") is not None


async def test_duplicate_username_rejected_by_mongo(mongo):
    await users.create_user("alice", "hashed-value")
    with pytest.raises(ValueError):
        await users.create_user("alice", "other-hash")


async def test_unknown_user_returns_none_from_mongo(mongo):
    assert await users.get_user_by_username("nobody") is None


async def test_password_hash_is_stored_not_the_password(mongo):
    await users.create_user("alice", "$2b$12$fakehashvalue")
    stored = await mongo["users"].find_one({"username_lower": "alice"})
    assert "password" not in stored
    assert stored["password_hash"].startswith("$2b$")


# --- chat history ----------------------------------------------------------
async def test_messages_persist_and_read_back_in_order(mongo):
    from langchain_core.messages import AIMessage, HumanMessage

    history = ChatHistory.get_session_history("user-a", "s1")
    await history.add_message(HumanMessage(content="first"))
    await history.add_message(AIMessage(content="second"))
    await history.add_message(HumanMessage(content="third"))

    messages = await history.get_messages()
    assert [m.content for m in messages] == ["first", "second", "third"]


async def test_mongo_history_is_scoped_to_its_owner(mongo):
    """The ownership filter must be applied by the database query itself."""
    from langchain_core.messages import HumanMessage

    await ChatHistory.get_session_history("user-a", "shared").add_message(
        HumanMessage(content="alice private")
    )

    attacker = await ChatHistory.get_session_history(
        "user-b", "shared"
    ).get_messages()
    assert attacker == []


async def test_mongo_history_is_trimmed_to_the_window(mongo):
    from langchain_core.messages import HumanMessage

    from src.core.config import settings

    history = ChatHistory.get_session_history("user-a", "s1")
    total = settings.MAX_HISTORY_MESSAGES + 5
    for index in range(total):
        await history.add_message(HumanMessage(content=f"m{index}"))

    messages = await history.get_messages()
    assert len(messages) == settings.MAX_HISTORY_MESSAGES
    # The newest messages are kept, oldest first.
    assert messages[-1].content == f"m{total - 1}"


async def test_clear_deletes_only_the_owners_session(mongo):
    from langchain_core.messages import HumanMessage

    await ChatHistory.get_session_history("user-a", "s1").add_message(
        HumanMessage(content="keep")
    )
    await ChatHistory.get_session_history("user-b", "s1").add_message(
        HumanMessage(content="also keep")
    )

    await ChatHistory.get_session_history("user-a", "s1").clear()

    assert await ChatHistory.get_session_history("user-a", "s1").get_messages() == []
    remaining = await ChatHistory.get_session_history("user-b", "s1").get_messages()
    assert [m.content for m in remaining] == ["also keep"]


async def test_stored_documents_carry_owner_and_session(mongo):
    from langchain_core.messages import HumanMessage

    await ChatHistory.get_session_history("user-a", "s1").add_message(
        HumanMessage(content="hello")
    )

    stored = await mongo["chat_history"].find_one({})
    assert stored["user_id"] == "user-a"
    assert stored["session_id"] == "s1"
    assert stored["timestamp"].tzinfo is not None or stored["timestamp"] is not None


async def test_ordering_survives_identical_timestamps(mongo, monkeypatch):
    """
    Regression: a coarse system clock gives consecutive messages the same
    timestamp. Sorting on timestamp alone then keeps the oldest messages and
    returns them reversed, corrupting the conversation.
    """
    import datetime as real_datetime

    from langchain_core.messages import HumanMessage

    from src.core.config import settings
    from src.memory import chat_history_mongo

    frozen = real_datetime.datetime(2026, 1, 1, tzinfo=real_datetime.timezone.utc)

    class _FrozenDatetime(real_datetime.datetime):
        @classmethod
        def now(cls, tz=None):
            return frozen

    monkeypatch.setattr(chat_history_mongo, "datetime", _FrozenDatetime)

    history = ChatHistory.get_session_history("user-a", "s1")
    total = settings.MAX_HISTORY_MESSAGES + 5
    for index in range(total):
        await history.add_message(HumanMessage(content=f"m{index}"))

    messages = await history.get_messages()

    assert len(messages) == settings.MAX_HISTORY_MESSAGES
    # Newest kept, oldest dropped, chronological order preserved.
    assert [m.content for m in messages] == [
        f"m{i}" for i in range(total - settings.MAX_HISTORY_MESSAGES, total)
    ]


async def test_ensure_indexes_creates_the_query_indexes(mongo):
    """History reads must be index-backed, not collection scans."""
    from src.db import mongo_client

    await mongo_client.ensure_indexes()

    history_indexes = await mongo["chat_history"].index_information()
    keys = [tuple(spec["key"]) for spec in history_indexes.values()]
    assert (("user_id", 1), ("session_id", 1), ("timestamp", -1)) in keys

    user_indexes = await mongo["users"].index_information()
    assert any(
        spec["key"] == [("username_lower", 1)] and spec.get("unique")
        for spec in user_indexes.values()
    )


async def test_ensure_indexes_is_inert_without_mongo():
    from src.db import mongo_client

    await mongo_client.ensure_indexes()  # must not raise

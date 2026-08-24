"""
Chat history storage.

Every message is stored against both a ``session_id`` and the ``user_id`` that
owns it, and every read filters on both. A session identifier alone is
therefore not enough to reach another user's conversation.

MongoDB is used when configured; otherwise an in-process dictionary keeps the
application working without a database.
"""

from datetime import datetime, timezone
from typing import Any, List, Optional

from langchain_core.messages import BaseMessage, messages_from_dict

from src.core.config import settings
from src.core.logger import get_logger
from src.db.mongo_client import get_database

logger = get_logger(__name__)

COLLECTION_NAME = "chat_history"

# Fallback store: {(user_id, session_id): [message documents]}
_memory_history: dict[tuple[str, str], list[dict[str, Any]]] = {}


class ChatMessageHistory:
    """Chat history for one (user, session) pair."""

    def __init__(self, user_id: str, session_id: str):
        """
        Initialise history for a session owned by a user.

        Args:
            user_id: Owner of the session.
            session_id: Conversation identifier.
        """
        self.user_id = user_id
        self.session_id = session_id

    @property
    def _key(self) -> tuple[str, str]:
        return (self.user_id, self.session_id)

    @property
    def _filter(self) -> dict[str, str]:
        """Ownership-enforcing query filter."""
        return {"user_id": self.user_id, "session_id": self.session_id}

    async def add_message(self, message: BaseMessage) -> None:
        """
        Append a message to the conversation.

        Args:
            message: The message to store.
        """
        document = {
            **self._filter,
            "type": message.type,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
            "timestamp": datetime.now(timezone.utc),
        }

        db = get_database()
        if db is None:
            _memory_history.setdefault(self._key, []).append(document)
            return
        await db[COLLECTION_NAME].insert_one(document)

    async def get_messages(self, limit: Optional[int] = None) -> List[BaseMessage]:
        """
        Load the most recent messages for this conversation.

        Only the newest ``limit`` messages are returned. Unbounded history
        would grow the prompt without limit, raising cost on every turn and
        eventually exceeding the model's context window.

        Args:
            limit: Maximum messages to return. Defaults to
                ``MAX_HISTORY_MESSAGES``.

        Returns:
            Messages in chronological order.
        """
        window = limit or settings.MAX_HISTORY_MESSAGES

        db = get_database()
        if db is None:
            documents = _memory_history.get(self._key, [])[-window:]
        else:
            # Sorting on `timestamp` alone is not safe: the clock resolution
            # on some platforms is coarser than the gap between two messages
            # in a single turn, so consecutive writes can share a timestamp.
            # With tied keys the sort degenerates to natural order, which
            # silently keeps the OLDEST messages and returns them reversed.
            # `_id` (an ObjectId) increases monotonically per process and
            # breaks those ties deterministically.
            cursor = (
                db[COLLECTION_NAME]
                .find(self._filter)
                .sort([("timestamp", -1), ("_id", -1)])
                .limit(window)
            )
            documents = list(reversed(await cursor.to_list(length=window)))

        return messages_from_dict(
            [
                {
                    "type": document["type"],
                    "data": {
                        "content": document["content"],
                        "additional_kwargs": document.get(
                            "additional_kwargs", {}
                        ),
                    },
                }
                for document in documents
            ]
        )

    async def clear(self) -> None:
        """Delete every message in this conversation."""
        db = get_database()
        if db is None:
            _memory_history.pop(self._key, None)
            return
        await db[COLLECTION_NAME].delete_many(self._filter)


class ChatHistory:
    """Factory for owner-scoped chat histories."""

    @classmethod
    def get_session_history(
        cls, user_id: str, session_id: str
    ) -> ChatMessageHistory:
        """
        Get the history for a session belonging to a user.

        Args:
            user_id: Owner of the session.
            session_id: Conversation identifier.

        Returns:
            A :class:`ChatMessageHistory` scoped to that pair.
        """
        return ChatMessageHistory(user_id=user_id, session_id=session_id)


# Backwards-compatible alias for the previous class name.
MongoDBChatMessageHistory = ChatMessageHistory


def reset_memory_store() -> None:
    """Clear the in-memory history store. Intended for tests."""
    _memory_history.clear()

"""
Tests for chat history ownership scoping and trimming.
"""

from langchain_core.messages import AIMessage, HumanMessage

from src.core.config import settings
from src.memory.chat_history_mongo import ChatHistory


async def test_messages_round_trip():
    history = ChatHistory.get_session_history("user-a", "s1")
    await history.add_message(HumanMessage(content="hello"))
    await history.add_message(AIMessage(content="hi there"))

    messages = await history.get_messages()
    assert [m.content for m in messages] == ["hello", "hi there"]
    assert messages[0].type == "human"
    assert messages[1].type == "ai"


async def test_sessions_are_isolated():
    await ChatHistory.get_session_history("user-a", "s1").add_message(
        HumanMessage(content="in session one")
    )
    other = await ChatHistory.get_session_history("user-a", "s2").get_messages()
    assert other == []


async def test_same_session_id_is_isolated_between_users():
    """Guessing another user's session id must not expose their messages."""
    await ChatHistory.get_session_history("user-a", "shared-id").add_message(
        HumanMessage(content="alice private message")
    )

    attacker_view = await ChatHistory.get_session_history(
        "user-b", "shared-id"
    ).get_messages()

    assert attacker_view == []


async def test_history_is_trimmed_to_the_configured_window():
    """Unbounded history would grow the prompt without limit."""
    history = ChatHistory.get_session_history("user-a", "s1")
    total = settings.MAX_HISTORY_MESSAGES + 10
    for index in range(total):
        await history.add_message(HumanMessage(content=f"message {index}"))

    messages = await history.get_messages()

    assert len(messages) == settings.MAX_HISTORY_MESSAGES
    # The window keeps the most recent messages, in order.
    assert messages[-1].content == f"message {total - 1}"
    assert messages[0].content == f"message {total - settings.MAX_HISTORY_MESSAGES}"


async def test_clear_removes_only_that_conversation():
    await ChatHistory.get_session_history("user-a", "s1").add_message(
        HumanMessage(content="keep me")
    )
    await ChatHistory.get_session_history("user-a", "s2").add_message(
        HumanMessage(content="delete me")
    )

    await ChatHistory.get_session_history("user-a", "s2").clear()

    assert len(await ChatHistory.get_session_history("user-a", "s1").get_messages()) == 1
    assert await ChatHistory.get_session_history("user-a", "s2").get_messages() == []


async def test_timestamps_are_timezone_aware():
    from src.memory import chat_history_mongo

    history = ChatHistory.get_session_history("user-a", "s1")
    await history.add_message(HumanMessage(content="hello"))

    stored = chat_history_mongo._memory_history[("user-a", "s1")][0]
    assert stored["timestamp"].tzinfo is not None

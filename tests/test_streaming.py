"""
Streaming responses.

A turn runs several model calls, so the wait before the first token is
noticeable. Streaming turns that silence into visible progress.

The delivery contract matters as much as the content: a client that renders
tokens as they arrive must be told when to discard them, and a failed stream
must not enter the conversation as though it had succeeded.
"""

import json

import pytest

from src.api.routes import _sse
from src.memory.chat_history_mongo import ChatHistory
from tests.conftest import register_and_login


def _events(response) -> list[dict]:
    """Decode an SSE response body into its events."""
    events = []
    for line in response.text.splitlines():
        if line.startswith("data: "):
            events.append(json.loads(line[len("data: ") :]))
    return events


@pytest.fixture
def stub_stream(monkeypatch):
    """Replace the graph with a scripted event sequence."""
    import src.api.routes as routes

    script: list[dict] = []

    async def fake_stream_query(user_id, messages):
        for event in script:
            yield event

    monkeypatch.setattr(routes, "stream_query", fake_stream_query)
    return script


# --- wire format -----------------------------------------------------------
def test_sse_frame_format():
    frame = _sse({"type": "token", "text": "hi"})
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert json.loads(frame[6:].strip()) == {"type": "token", "text": "hi"}


def test_sse_preserves_non_ascii():
    """Escaping non-ASCII would corrupt answers in most languages."""
    frame = _sse({"type": "token", "text": "café — naïve 日本語"})
    assert "café — naïve 日本語" in frame


def test_sse_escapes_newlines_so_frames_stay_separable():
    """A literal newline in the payload would split the frame."""
    frame = _sse({"type": "token", "text": "line one\nline two"})
    assert frame.count("\n\n") == 1
    assert json.loads(frame[6:].strip())["text"] == "line one\nline two"


# --- endpoint --------------------------------------------------------------
def test_stream_returns_an_event_stream(client, auth_headers, stub_stream):
    stub_stream.extend(
        [
            {"type": "token", "text": "Hello "},
            {"type": "token", "text": "world"},
            {"type": "done", "answer": "Hello world", "citations": [], "usage": {}},
        ]
    )

    response = client.post(
        "/rag/query/stream",
        json={"query": "hi", "session_id": "s1"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    # Defeats proxy buffering, which would otherwise negate streaming.
    assert response.headers["x-accel-buffering"] == "no"

    events = _events(response)
    assert [e["type"] for e in events] == ["token", "token", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello world"


def test_stream_requires_authentication(client):
    response = client.post(
        "/rag/query/stream", json={"query": "hi", "session_id": "s1"}
    )
    assert response.status_code == 401


def test_stream_validates_its_payload(client, auth_headers, stub_stream):
    response = client.post(
        "/rag/query/stream",
        json={"query": "", "session_id": "s1"},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_stream_forwards_citations_and_usage(client, auth_headers, stub_stream):
    stub_stream.extend(
        [
            {"type": "token", "text": "answer"},
            {"type": "citations", "citations": [{"source": "a.pdf", "snippet": "x"}]},
            {"type": "usage", "usage": {"calls": 3, "total_tokens": 120}},
            {"type": "done", "answer": "answer", "citations": [], "usage": {}},
        ]
    )

    events = {
        e["type"]: e
        for e in _events(
            client.post(
                "/rag/query/stream",
                json={"query": "hi", "session_id": "s1"},
                headers=auth_headers,
            )
        )
    }

    assert events["citations"]["citations"][0]["source"] == "a.pdf"
    assert events["usage"]["usage"]["total_tokens"] == 120


async def test_completed_stream_is_persisted(client, auth_headers, stub_stream):
    stub_stream.extend(
        [
            {"type": "token", "text": "the answer"},
            {"type": "done", "answer": "the answer", "citations": [], "usage": {}},
        ]
    )

    client.post(
        "/rag/query/stream",
        json={"query": "the question", "session_id": "s1"},
        headers=auth_headers,
    )

    from src.db import users as user_store

    user_id = next(iter(user_store._memory_users.values()))["user_id"]
    messages = await ChatHistory.get_session_history(user_id, "s1").get_messages()
    assert [m.content for m in messages] == ["the question", "the answer"]


async def test_failed_stream_is_not_persisted_as_an_answer(
    client, auth_headers, stub_stream
):
    """A half-streamed failure must not enter the conversation."""
    stub_stream.extend(
        [
            {"type": "token", "text": "partial"},
            {"type": "error", "message": "the model service is unavailable"},
        ]
    )

    response = client.post(
        "/rag/query/stream",
        json={"query": "the question", "session_id": "s1"},
        headers=auth_headers,
    )
    assert response.status_code == 200
    assert _events(response)[-1]["type"] == "error"

    from src.db import users as user_store

    user_id = next(iter(user_store._memory_users.values()))["user_id"]
    messages = await ChatHistory.get_session_history(user_id, "s1").get_messages()
    # The question is recorded; no assistant turn is.
    assert [m.content for m in messages] == ["the question"]


def test_stream_counts_against_the_query_quota(client, auth_headers, stub_stream):
    """Streaming must not be a way around the rate limit."""
    from src.core.config import settings

    settings_limit = settings.RATE_LIMIT_QUERY_PER_MINUTE
    try:
        object.__setattr__(settings, "RATE_LIMIT_QUERY_PER_MINUTE", 2)
        stub_stream.append(
            {"type": "done", "answer": "a", "citations": [], "usage": {}}
        )
        payload = {"query": "hi", "session_id": "s1"}

        assert (
            client.post(
                "/rag/query/stream", json=payload, headers=auth_headers
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/rag/query/stream", json=payload, headers=auth_headers
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/rag/query/stream", json=payload, headers=auth_headers
            ).status_code
            == 429
        )
    finally:
        object.__setattr__(settings, "RATE_LIMIT_QUERY_PER_MINUTE", settings_limit)


def test_streams_are_isolated_between_users(client, stub_stream):
    alice = register_and_login(client, "alice", "alice-password-1")
    bob = register_and_login(client, "bob", "bob-password-1")

    stub_stream.append({"type": "done", "answer": "a", "citations": [], "usage": {}})

    for headers in (alice, bob):
        assert (
            client.post(
                "/rag/query/stream",
                json={"query": "hi", "session_id": "shared"},
                headers=headers,
            ).status_code
            == 200
        )


# --- generator semantics ---------------------------------------------------
async def test_stream_query_emits_tokens_then_terminal_events(monkeypatch):
    """The graph's token events become the stream's token events."""
    from src.rag import graph_builder

    class _Chunk:
        def __init__(self, content):
            self.content = content

    async def fake_events(_state, version=None, config=None):
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "generate"},
            "data": {"chunk": _Chunk("Hel")},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "generate"},
            "data": {"chunk": _Chunk("lo")},
        }
        yield {
            "event": "on_chain_end",
            "parent_ids": [],
            "data": {"output": {"messages": [], "citations": [{"source": "a.pdf"}]}},
        }

    class _Builder:
        astream_events = staticmethod(fake_events)

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    events = [e async for e in graph_builder.stream_query("user-a", [])]
    types = [e["type"] for e in events]

    assert types == ["token", "token", "citations", "usage", "done"]
    assert events[-1]["answer"] == "Hello"
    assert events[-1]["citations"] == [{"source": "a.pdf"}]


async def test_stream_query_ignores_non_answer_nodes(monkeypatch):
    """The classifier and grader also call models; their output is internal."""
    from src.rag import graph_builder

    class _Chunk:
        def __init__(self, content):
            self.content = content

    async def fake_events(_state, version=None, config=None):
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "query_analysis"},
            "data": {"chunk": _Chunk("index")},
        }
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "generate"},
            "data": {"chunk": _Chunk("real answer")},
        }
        yield {"event": "on_chain_end", "parent_ids": [], "data": {"output": {}}}

    class _Builder:
        astream_events = staticmethod(fake_events)

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    events = [e async for e in graph_builder.stream_query("user-a", [])]
    text = "".join(e["text"] for e in events if e["type"] == "token")
    assert text == "real answer"
    assert "index" not in text


async def test_regeneration_tells_the_client_to_discard(monkeypatch):
    """
    Verification can reject an answer and regenerate it.

    A client rendering tokens live has already shown the rejected text, so it
    must be told to start over rather than appending the replacement.
    """
    from src.rag import graph_builder

    class _Chunk:
        def __init__(self, content):
            self.content = content

    async def fake_events(_state, version=None, config=None):
        yield {"event": "on_chain_start", "name": "generate", "metadata": {}}
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "generate"},
            "data": {"chunk": _Chunk("wrong answer")},
        }
        yield {"event": "on_chain_start", "name": "generate", "metadata": {}}
        yield {
            "event": "on_chat_model_stream",
            "metadata": {"langgraph_node": "generate"},
            "data": {"chunk": _Chunk("right answer")},
        }
        yield {"event": "on_chain_end", "parent_ids": [], "data": {"output": {}}}

    class _Builder:
        astream_events = staticmethod(fake_events)

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    events = [e async for e in graph_builder.stream_query("user-a", [])]
    types = [e["type"] for e in events]

    assert "restart" in types
    # Only the accepted answer survives.
    assert events[-1]["answer"] == "right answer"


async def test_answers_without_a_model_call_are_still_delivered(monkeypatch):
    """The "no documents" reply is built in code and never streams."""
    from langchain_core.messages import AIMessage

    from src.rag import graph_builder

    async def fake_events(_state, version=None, config=None):
        yield {
            "event": "on_chain_end",
            "parent_ids": [],
            "data": {"output": {"messages": [AIMessage(content="Upload a file.")]}},
        }

    class _Builder:
        astream_events = staticmethod(fake_events)

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    events = [e async for e in graph_builder.stream_query("user-a", [])]
    assert events[0] == {"type": "token", "text": "Upload a file."}
    assert events[-1]["answer"] == "Upload a file."


async def test_provider_failure_becomes_an_error_event(monkeypatch):
    """Headers are already sent, so failures arrive as events, not statuses."""
    from openai import APIConnectionError

    from src.rag import graph_builder

    async def fake_events(_state, version=None, config=None):
        raise APIConnectionError(request=None)
        yield  # pragma: no cover - unreachable, makes this a generator

    class _Builder:
        astream_events = staticmethod(fake_events)

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    events = [e async for e in graph_builder.stream_query("user-a", [])]
    assert events == [
        {
            "type": "error",
            "message": "The language model service is unavailable. Please try again.",
        }
    ]


async def test_recursion_limit_becomes_an_error_event(monkeypatch):
    from langgraph.errors import GraphRecursionError

    from src.rag import graph_builder

    async def fake_events(_state, version=None, config=None):
        raise GraphRecursionError("limit")
        yield  # pragma: no cover

    class _Builder:
        astream_events = staticmethod(fake_events)

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    events = [e async for e in graph_builder.stream_query("user-a", [])]
    assert events[0]["type"] == "error"
    assert "converge" in events[0]["message"]


async def test_empty_answer_becomes_an_error_event(monkeypatch):
    from src.rag import graph_builder

    async def fake_events(_state, version=None, config=None):
        yield {"event": "on_chain_end", "parent_ids": [], "data": {"output": {}}}

    class _Builder:
        astream_events = staticmethod(fake_events)

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    events = [e async for e in graph_builder.stream_query("user-a", [])]
    assert events == [
        {"type": "error", "message": "The assistant produced an empty answer."}
    ]


# --- model configuration ---------------------------------------------------
def test_answer_model_is_configured_to_stream():
    """
    Without streaming enabled, `invoke` emits no token events and the SSE
    endpoint silently degrades to one large chunk.
    """
    from src.llms.openai import get_answer_llm

    model = get_answer_llm()
    assert model.streaming is True
    # Streamed responses report no usage unless this is set, which would make
    # the cost accounting read zero for every streamed turn.
    assert model.stream_usage is True


def test_structured_output_model_does_not_stream():
    """Streaming buys nothing for routing and adds a partial-parse risk."""
    from src.llms.openai import get_llm

    assert get_llm().streaming is False


def test_answer_nodes_use_the_streaming_model():
    """A regression here would disable streaming without failing anything."""
    import inspect

    from src.rag import graph_builder

    for node in (graph_builder.generate, graph_builder.general_llm):
        assert "get_answer_llm" in inspect.getsource(node), node.__name__


# --- integration through the real graph ------------------------------------
async def test_real_graph_streams_tokens(monkeypatch):
    """
    Exercises the actual graph and event plumbing, not a stubbed builder.

    The stubbed tests above assume the shape of LangGraph's events; this one
    proves that assumption against the compiled graph.
    """
    from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_core.runnables import RunnableLambda

    from src.models.route_identifier import RouteIdentifier
    from src.rag import graph_builder

    class _Model:
        """Routes to general knowledge, then streams a reply."""

        def __init__(self):
            self._answer = GenericFakeChatModel(
                messages=iter([AIMessage(content="the streamed reply")])
            )

        def invoke(self, value, **kwargs):
            return self._answer.invoke(value, **kwargs)

        def with_structured_output(self, _schema):
            return RunnableLambda(lambda _: RouteIdentifier(route="general"))

    model = _Model()
    monkeypatch.setattr(graph_builder, "get_llm", lambda: model)
    monkeypatch.setattr(graph_builder, "get_answer_llm", lambda: model)

    events = [
        event
        async for event in graph_builder.stream_query(
            "user-a", [HumanMessage(content="hello")]
        )
    ]
    types = [event["type"] for event in events]

    # More than one token event means the answer genuinely arrived in pieces.
    assert types.count("token") > 1, f"answer was not streamed: {types}"
    assert types[-1] == "done"
    assert events[-1]["answer"] == "the streamed reply"

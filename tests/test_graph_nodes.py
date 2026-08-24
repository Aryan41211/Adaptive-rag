"""
Tests for individual graph nodes and their degradation behaviour.

No node may raise out of the graph: an unavailable dependency must produce a
usable message, not a 500.
"""

import pytest
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from src.core.config import settings
from src.core.exceptions import RetrievalError
from src.rag import graph_builder, vector_store


def _stub_llm(monkeypatch, result):
    """Patch the module-level LLM factory with a fixed structured result."""

    class _LLM:
        def with_structured_output(self, _schema):
            return RunnableLambda(lambda _payload: result)

        def invoke(self, _payload):
            return result

    monkeypatch.setattr(graph_builder, "get_llm", lambda: _LLM())


# --- query_classifier ------------------------------------------------------
def test_classifier_never_routes_to_index_without_documents(monkeypatch):
    """The classifier cannot legitimately pick 'index' with no index."""
    from src.models.route_identifier import RouteIdentifier

    _stub_llm(monkeypatch, RouteIdentifier(route="index"))

    result = graph_builder.query_classifier(
        {"user_id": "user-a", "messages": [_msg("what is in my resume?")]}
    )
    assert result["route"] == "general"


def test_classifier_falls_back_when_web_search_unconfigured(monkeypatch):
    from src.models.route_identifier import RouteIdentifier

    _stub_llm(monkeypatch, RouteIdentifier(route="search"))
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "")

    result = graph_builder.query_classifier(
        {"user_id": "user-a", "messages": [_msg("latest news?")]}
    )
    assert result["route"] == "general"


def test_classifier_can_route_to_index_when_documents_exist(monkeypatch):
    from src.models.route_identifier import RouteIdentifier

    vector_store.add_documents(
        "user-a", [Document(page_content="my resume: python engineer")], "resume"
    )
    _stub_llm(monkeypatch, RouteIdentifier(route="index"))

    result = graph_builder.query_classifier(
        {"user_id": "user-a", "messages": [_msg("what is my job?")]}
    )
    assert result["route"] == "index"


def test_classifier_failure_degrades_to_general(monkeypatch):
    class _ExplodingLLM:
        def with_structured_output(self, _schema):
            def _raise(_payload):
                raise RuntimeError("provider down")

            return RunnableLambda(_raise)

    monkeypatch.setattr(graph_builder, "get_llm", lambda: _ExplodingLLM())

    result = graph_builder.query_classifier(
        {"user_id": "user-a", "messages": [_msg("hello")]}
    )
    assert result["route"] == "general"


def test_classifier_resets_loop_counters(monkeypatch):
    from src.models.route_identifier import RouteIdentifier

    _stub_llm(monkeypatch, RouteIdentifier(route="general"))
    result = graph_builder.query_classifier(
        {"user_id": "user-a", "messages": [_msg("hi")]}
    )
    assert result["rewrite_attempts"] == 0
    assert result["generate_attempts"] == 0


# --- retriever_node --------------------------------------------------------
def test_retriever_reports_when_nothing_is_uploaded():
    result = graph_builder.retriever_node(
        {"user_id": "user-a", "latest_query": "anything", "messages": []}
    )
    assert result["context"] == ""
    assert "upload" in result["messages"][0].content.lower()


def test_retriever_agent_failure_is_contained(monkeypatch):
    vector_store.add_documents("user-a", [Document(page_content="content")], "doc")

    class _ExplodingExecutor:
        def invoke(self, _payload):
            raise RuntimeError("agent blew up")

    monkeypatch.setattr(
        graph_builder, "get_agent_executor", lambda _user: _ExplodingExecutor()
    )

    result = graph_builder.retriever_node(
        {"user_id": "user-a", "latest_query": "q", "messages": []}
    )
    assert result["context"] == ""
    assert "try again" in result["messages"][0].content.lower()


def test_retriever_uses_tool_observations_as_context(monkeypatch):
    class _Action:
        tool = "search_uploaded_documents"
        tool_input = "q"

    class _Executor:
        def invoke(self, _payload):
            return {
                "output": "A short answer.",
                "intermediate_steps": [(_Action(), "the retrieved evidence")],
            }

    monkeypatch.setattr(graph_builder, "get_agent_executor", lambda _user: _Executor())

    result = graph_builder.retriever_node(
        {"user_id": "user-a", "latest_query": "q", "messages": []}
    )
    assert result["context"] == "the retrieved evidence"
    assert result["messages"][0].content == "A short answer."
    assert result["messages"][0].additional_kwargs["tool_calls"][0]["tool"] == (
        "search_uploaded_documents"
    )


# --- grade / generate ------------------------------------------------------
def test_empty_context_grades_as_irrelevant():
    assert graph_builder.grade({"context": "", "latest_query": "q"}) == {
        "binary_score": "no"
    }


def test_grading_failure_assumes_relevance(monkeypatch):
    """Failing closed here would spin the rewrite loop for no benefit."""

    class _ExplodingLLM:
        def with_structured_output(self, _schema):
            def _raise(_payload):
                raise RuntimeError("provider down")

            return RunnableLambda(_raise)

    monkeypatch.setattr(graph_builder, "get_llm", lambda: _ExplodingLLM())

    result = graph_builder.grade({"context": "some text", "latest_query": "q"})
    assert result["binary_score"] == "yes"


def test_generate_without_context_returns_guidance():
    result = graph_builder.generate({"context": "", "generate_attempts": 0})
    assert result["generate_attempts"] == 1
    assert "upload" in result["messages"][0].content.lower()


def test_generate_increments_its_counter(monkeypatch):
    # `generate` pipes the model directly, so the double must be a Runnable.
    def _explode(_payload):
        raise RuntimeError("down")

    monkeypatch.setattr(graph_builder, "get_llm", lambda: RunnableLambda(_explode))
    result = graph_builder.generate({"context": "raw context", "generate_attempts": 1})
    assert result["generate_attempts"] == 2
    # Falls back to the raw context rather than failing the request.
    assert result["messages"][0].content == "raw context"


# --- web_search ------------------------------------------------------------
def test_web_search_without_a_key_returns_a_clear_message(monkeypatch):
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "")
    result = graph_builder.web_search({"latest_query": "news"})
    assert result["context"] == ""
    assert "not configured" in result["messages"][0].content


def test_web_search_failure_is_contained(monkeypatch):
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "tvly-fake")

    class _ExplodingTool:
        def invoke(self, _query):
            raise RuntimeError("tavily down")

    monkeypatch.setattr(graph_builder, "_get_search_tool", lambda: _ExplodingTool())

    result = graph_builder.web_search({"latest_query": "news"})
    assert result["context"] == ""
    assert "failed" in result["messages"][0].content.lower()


def test_web_search_collects_result_contents(monkeypatch):
    monkeypatch.setattr(settings, "TAVILY_API_KEY", "tvly-fake")

    class _Tool:
        def invoke(self, _query):
            return [
                {"content": "first result"},
                {"url": "no content here"},
                {"content": "second result"},
            ]

    monkeypatch.setattr(graph_builder, "_get_search_tool", lambda: _Tool())

    result = graph_builder.web_search({"latest_query": "news"})
    assert result["context"] == "first result\n\nsecond result"


# --- run_query -------------------------------------------------------------
async def test_recursion_limit_becomes_a_clean_error(monkeypatch):
    """An unconverged graph must not surface as an unhandled 500."""
    from langgraph.errors import GraphRecursionError

    class _Builder:
        async def ainvoke(self, _state, config=None):
            raise GraphRecursionError("limit reached")

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    with pytest.raises(RetrievalError):
        await graph_builder.run_query("user-a", [_msg("hello")])


async def test_empty_answer_becomes_a_clean_error(monkeypatch):
    class _Builder:
        async def ainvoke(self, _state, config=None):
            return {"messages": []}

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    with pytest.raises(RetrievalError):
        await graph_builder.run_query("user-a", [_msg("hello")])


async def test_run_query_passes_the_user_id_into_state(monkeypatch):
    captured = {}

    class _Builder:
        async def ainvoke(self, state, config=None):
            captured.update(state)
            return {"messages": [_msg("the answer")]}

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    answer, citations = await graph_builder.run_query("user-xyz", [_msg("hello")])
    assert answer == "the answer"
    assert citations == []
    assert captured["user_id"] == "user-xyz"


# --- helpers ---------------------------------------------------------------
def _msg(content):
    from langchain_core.messages import HumanMessage

    return HumanMessage(content=content)


async def test_provider_outage_becomes_a_clean_upstream_error(monkeypatch):
    """An OpenAI outage is a 502, not an unhandled 500."""
    from openai import APIConnectionError

    class _Builder:
        async def ainvoke(self, _state, config=None):
            raise APIConnectionError(request=None)

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    with pytest.raises(RetrievalError) as exc:
        await graph_builder.run_query("user-a", [_msg("hello")])
    assert exc.value.status_code == 502


# --- citations -------------------------------------------------------------
def test_citations_carry_source_and_snippet(monkeypatch):
    """Provenance is stored at upload; this is where it becomes visible."""
    from langchain_core.documents import Document as Doc

    vector_store.add_documents(
        "user-a",
        [
            Doc(
                page_content="The mitochondria is the powerhouse of the cell.",
                metadata={"source_filename": "biology.pdf", "page": 3},
            )
        ],
        "biology notes",
    )

    citations = graph_builder._citations_for("user-a", "mitochondria")

    assert citations
    assert citations[0]["source"] == "biology.pdf"
    assert "powerhouse" in citations[0]["snippet"]
    # Page numbers are stored zero-based and surfaced one-based.
    assert citations[0]["page"] == 4


def test_citations_are_empty_without_documents():
    assert graph_builder._citations_for("user-a", "anything") == []


def test_citations_are_deduplicated_by_source_and_page():
    from langchain_core.documents import Document as Doc

    vector_store.add_documents(
        "user-a",
        [
            Doc(page_content="chunk one", metadata={"source_filename": "a.txt"}),
            Doc(page_content="chunk two", metadata={"source_filename": "a.txt"}),
        ],
        "notes",
    )

    citations = graph_builder._citations_for("user-a", "chunk")
    assert len(citations) == 1


def test_citation_failure_does_not_break_the_answer(monkeypatch):
    """Citations are best-effort; losing them must not lose the answer."""

    class _ExplodingRetriever:
        def invoke(self, _query):
            raise RuntimeError("vector store down")

    monkeypatch.setattr(
        graph_builder.vector_store, "get_retriever", lambda _u: _ExplodingRetriever()
    )
    assert graph_builder._citations_for("user-a", "anything") == []


def test_citations_never_cross_the_owner_boundary():
    from langchain_core.documents import Document as Doc

    vector_store.add_documents(
        "user-a",
        [Doc(page_content="alice secret", metadata={"source_filename": "a.txt"})],
        "a",
    )
    vector_store.add_documents(
        "user-b",
        [Doc(page_content="bob notes", metadata={"source_filename": "b.txt"})],
        "b",
    )

    citations = graph_builder._citations_for("user-b", "secret")
    assert all(c["source"] != "a.txt" for c in citations)

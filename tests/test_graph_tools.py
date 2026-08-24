"""
Tests for graph routing decisions and loop bounding.

The retrieve/rewrite cycle previously had no attempt counter, so a
persistently negative relevance grade looped until LangGraph's recursion
limit aborted the request after many paid model calls.
"""

import pytest

from src.core.config import settings
from src.tools.graph_tools import doc_tool, routing_tool, verify_answer


# --- routing ---------------------------------------------------------------
@pytest.mark.parametrize(
    "route,expected",
    [
        ("index", "retriever"),
        ("general", "general_llm"),
        ("search", "web_search"),
    ],
)
def test_routing_matches_classification(route, expected):
    assert routing_tool({"route": route}) == expected


def test_routing_falls_back_to_web_search_for_unknown_route():
    assert routing_tool({"route": None}) == "web_search"
    assert routing_tool({}) == "web_search"


# --- grading / rewrite bound ----------------------------------------------
def test_relevant_context_goes_straight_to_generate():
    assert doc_tool({"binary_score": "yes", "rewrite_attempts": 0}) == "generate"


def test_irrelevant_context_triggers_a_rewrite():
    assert doc_tool({"binary_score": "no", "rewrite_attempts": 0}) == "rewrite"


def test_rewrite_loop_is_bounded():
    """At the attempt limit the graph must stop looping and answer."""
    state = {
        "binary_score": "no",
        "rewrite_attempts": settings.MAX_REWRITE_ATTEMPTS,
    }
    assert doc_tool(state) == "generate"


def test_rewrite_loop_terminates_within_the_budget():
    """Simulate the cycle and assert it cannot run forever."""
    state = {"binary_score": "no", "rewrite_attempts": 0}
    transitions = 0
    while doc_tool(state) == "rewrite":
        state["rewrite_attempts"] += 1
        transitions += 1
        assert transitions <= settings.MAX_REWRITE_ATTEMPTS + 1, "loop is unbounded"
    assert transitions == settings.MAX_REWRITE_ATTEMPTS


# --- verification bound ----------------------------------------------------
def test_general_knowledge_answers_skip_verification():
    assert verify_answer({"route": "general"}) == "__end__"


def test_verification_skipped_without_context():
    assert verify_answer({"route": "index", "context": ""}) == "__end__"


def test_verification_budget_is_respected():
    """Once the regeneration budget is spent, the answer is returned."""
    state = {
        "route": "index",
        "context": "some retrieved context",
        "generate_attempts": settings.MAX_VERIFY_ATTEMPTS + 1,
        "messages": [],
    }
    assert verify_answer(state) == "__end__"


def test_unfaithful_answer_is_regenerated(monkeypatch):
    import src.tools.graph_tools as module
    from src.models.verification_result import VerificationResult

    monkeypatch.setattr(
        module,
        "get_llm",
        lambda: _FakeStructuredLLM(
            VerificationResult(faithful=False, explanation="not supported")
        ),
    )

    state = {
        "route": "index",
        "context": "the sky is blue",
        "latest_query": "what colour is the sky?",
        "generate_attempts": 1,
        "messages": [_FakeMessage("the sky is green")],
    }
    assert verify_answer(state) == "generate"


def test_faithful_answer_ends_the_graph(monkeypatch):
    import src.tools.graph_tools as module
    from src.models.verification_result import VerificationResult

    monkeypatch.setattr(
        module,
        "get_llm",
        lambda: _FakeStructuredLLM(
            VerificationResult(faithful=True, explanation="supported")
        ),
    )

    state = {
        "route": "index",
        "context": "the sky is blue",
        "latest_query": "what colour is the sky?",
        "generate_attempts": 1,
        "messages": [_FakeMessage("the sky is blue")],
    }
    assert verify_answer(state) == "__end__"


def test_verification_failure_does_not_break_the_request(monkeypatch):
    """A verifier outage must not fail an otherwise good answer."""
    import src.tools.graph_tools as module

    monkeypatch.setattr(module, "get_llm", lambda: _ExplodingLLM())

    state = {
        "route": "index",
        "context": "the sky is blue",
        "latest_query": "what colour is the sky?",
        "generate_attempts": 1,
        "messages": [_FakeMessage("the sky is blue")],
    }
    assert verify_answer(state) == "__end__"


# --- helpers ---------------------------------------------------------------
class _FakeMessage:
    def __init__(self, content):
        self.content = content


def _runnable(fn):
    """Wrap a callable so `prompt | fake` composes like a real chain."""
    from langchain_core.runnables import RunnableLambda

    return RunnableLambda(fn)


class _FakeStructuredLLM:
    """Stands in for a chat model whose structured output is fixed."""

    def __init__(self, result):
        self._result = result

    def with_structured_output(self, _schema):
        return _runnable(lambda _payload: self._result)


class _ExplodingLLM:
    """Stands in for a chat model whose provider is unavailable."""

    @staticmethod
    def _raise(_payload):
        raise RuntimeError("verifier unavailable")

    def with_structured_output(self, _schema):
        return _runnable(_ExplodingLLM._raise)

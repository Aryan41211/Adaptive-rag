"""
Per-user document isolation and index versioning.

Every test here runs against **both** backends via the ``any_backend``
fixture. The isolation guarantees are the security boundary between users, so
they must hold identically whether documents live in an in-process FAISS index
or in a shared Qdrant collection.

These cover the two defects that made retrieval unusable and unsafe: the
process-global index shared by every user, and the agent permanently bound to
the index that existed when the module was first imported.
"""

import pytest
from langchain_core.documents import Document

from src.rag import reAct_agent, vector_store


def _docs(*texts):
    return [Document(page_content=text) for text in texts]


def _retrieved(user_id, query):
    retriever = vector_store.get_retriever(user_id)
    return (
        [] if retriever is None else [d.page_content for d in retriever.invoke(query)]
    )


# --- empty state -----------------------------------------------------------
def test_new_user_has_no_index(any_backend):
    assert vector_store.has_documents("user-a") is False
    assert vector_store.get_retriever("user-a") is None
    assert vector_store.get_retriever_tool("user-a") is None
    assert vector_store.get_version("user-a") == 0


def test_empty_chunk_list_rejected(any_backend):
    with pytest.raises(ValueError):
        vector_store.add_documents("user-a", [], "nothing")


# --- isolation -------------------------------------------------------------
def test_documents_are_isolated_per_user(any_backend):
    """User A's upload must be invisible to user B."""
    vector_store.add_documents("user-a", _docs("alice private salary data"), "payroll")

    assert vector_store.has_documents("user-a") is True
    assert vector_store.has_documents("user-b") is False
    assert vector_store.get_retriever("user-b") is None


def test_one_users_upload_does_not_replace_anothers(any_backend):
    vector_store.add_documents("user-a", _docs("alpha content"), "a-doc")
    vector_store.add_documents("user-b", _docs("beta content"), "b-doc")

    assert _retrieved("user-a", "alpha") == ["alpha content"]
    assert _retrieved("user-b", "beta") == ["beta content"]


def test_search_never_crosses_the_owner_boundary(any_backend):
    """The filter must apply even when the other user is the better match."""
    vector_store.add_documents(
        "user-a", _docs("the treasure is buried under the oak"), "map"
    )
    vector_store.add_documents("user-b", _docs("unrelated shopping list"), "list")

    bob_sees = _retrieved("user-b", "where is the treasure buried")

    assert all("treasure" not in text for text in bob_sees)
    assert bob_sees == ["unrelated shopping list"]


def test_reset_scoped_to_one_user(any_backend):
    vector_store.add_documents("user-a", _docs("alpha"), "a")
    vector_store.add_documents("user-b", _docs("beta"), "b")

    vector_store.reset("user-a")

    assert vector_store.has_documents("user-a") is False
    assert vector_store.has_documents("user-b") is True


# --- accumulation ----------------------------------------------------------
def test_second_upload_accumulates_rather_than_replacing(any_backend):
    vector_store.add_documents("user-a", _docs("first document"), "first")
    total = vector_store.add_documents("user-a", _docs("second document"), "second")

    assert total == 2
    found = _retrieved("user-a", "document")
    assert "first document" in found
    assert "second document" in found


def test_description_combines_uploads(any_backend):
    vector_store.add_documents("user-a", _docs("one"), "a resume")
    vector_store.add_documents("user-a", _docs("two"), "a tax form")

    description = vector_store.get_description("user-a")
    assert "a resume" in description and "a tax form" in description


def test_descriptions_are_not_shared_between_users(any_backend):
    vector_store.add_documents("user-a", _docs("one"), "alice's payroll")
    vector_store.add_documents("user-b", _docs("two"), "bob's invoices")

    assert "payroll" not in vector_store.get_description("user-b")
    assert "invoices" not in vector_store.get_description("user-a")


def test_retriever_tool_mentions_the_users_own_description(any_backend):
    vector_store.add_documents("user-a", _docs("one"), "alice's resume")
    tool = vector_store.get_retriever_tool("user-a")
    assert "alice's resume" in tool.description


# --- versioning ------------------------------------------------------------
def test_version_changes_on_each_upload(any_backend):
    """The exact value is backend-specific; that it advances is not."""
    assert vector_store.get_version("user-a") == 0

    vector_store.add_documents("user-a", _docs("one"), "d1")
    first = vector_store.get_version("user-a")

    vector_store.add_documents("user-a", _docs("two"), "d2")
    second = vector_store.get_version("user-a")

    assert first > 0
    assert second > first


def test_version_is_per_user(any_backend):
    vector_store.add_documents("user-a", _docs("one"), "d1")
    assert vector_store.get_version("user-b") == 0


# --- agent cache invalidation ---------------------------------------------
def test_agent_is_none_before_any_upload(any_backend):
    assert reAct_agent.get_agent_executor("user-a") is None


def test_agent_is_built_after_upload_and_sees_the_documents(any_backend):
    vector_store.add_documents("user-a", _docs("the sky is teal"), "colours")

    executor = reAct_agent.get_agent_executor("user-a")
    assert executor is not None
    # The bound tool must query the real index, not a placeholder.
    assert "teal" in executor.tools[0].invoke("sky")


def test_agent_is_cached_between_calls_at_the_same_version(any_backend):
    vector_store.add_documents("user-a", _docs("content"), "d")
    assert reAct_agent.get_agent_executor("user-a") is (
        reAct_agent.get_agent_executor("user-a")
    )


def test_agent_is_rebuilt_after_a_new_upload(any_backend):
    """The regression that made uploaded documents unreachable."""
    vector_store.add_documents("user-a", _docs("original content"), "d1")
    first = reAct_agent.get_agent_executor("user-a")

    vector_store.add_documents("user-a", _docs("brand new content"), "d2")
    second = reAct_agent.get_agent_executor("user-a")

    assert first is not second
    assert "brand new content" in second.tools[0].invoke("brand new")


def test_agents_are_not_shared_between_users(any_backend):
    vector_store.add_documents("user-a", _docs("alpha secret"), "a")
    vector_store.add_documents("user-b", _docs("beta secret"), "b")

    a_result = reAct_agent.get_agent_executor("user-a").tools[0].invoke("secret")
    b_result = reAct_agent.get_agent_executor("user-b").tools[0].invoke("secret")

    assert "alpha secret" in a_result and "beta secret" not in a_result
    assert "beta secret" in b_result and "alpha secret" not in b_result


# --- backend selection -----------------------------------------------------
def test_backend_reports_healthy(any_backend):
    healthy, detail = vector_store.health()
    assert healthy is True
    assert detail


def test_faiss_is_the_default_backend(monkeypatch):
    from src.core.config import settings

    monkeypatch.setattr(settings, "QDRANT_URL", None)
    vector_store.set_backend(None)
    assert vector_store.get_backend().name == "faiss"


def test_qdrant_is_selected_when_configured(monkeypatch):
    from src.core.config import settings
    from src.rag.backends import qdrant_backend as module

    monkeypatch.setattr(settings, "QDRANT_URL", "http://localhost:6333")
    # Avoid opening a real connection just to check selection.
    monkeypatch.setattr(module.QdrantBackend, "__init__", lambda self, **kw: None)
    vector_store.set_backend(None)
    assert vector_store.get_backend().name == "qdrant"

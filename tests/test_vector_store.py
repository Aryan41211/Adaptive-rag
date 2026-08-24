"""
Tests for per-user document isolation and index versioning.

These cover the two defects that made retrieval unusable and unsafe:
the process-global index shared by every user, and the agent permanently
bound to the index that existed when the module was first imported.
"""

from langchain_core.documents import Document

from src.rag import reAct_agent, vector_store


def _docs(*texts):
    return [Document(page_content=text) for text in texts]


def test_new_user_has_no_index():
    assert vector_store.has_documents("user-a") is False
    assert vector_store.get_retriever("user-a") is None
    assert vector_store.get_retriever_tool("user-a") is None
    assert vector_store.get_version("user-a") == 0


def test_documents_are_isolated_per_user():
    """User A's upload must be invisible to user B."""
    vector_store.add_documents("user-a", _docs("alice private salary data"), "payroll")

    assert vector_store.has_documents("user-a") is True
    assert vector_store.has_documents("user-b") is False
    assert vector_store.get_retriever("user-b") is None


def test_one_users_upload_does_not_replace_anothers():
    vector_store.add_documents("user-a", _docs("alpha content"), "a-doc")
    vector_store.add_documents("user-b", _docs("beta content"), "b-doc")

    a_hits = vector_store.get_retriever("user-a").invoke("alpha")
    b_hits = vector_store.get_retriever("user-b").invoke("beta")

    assert [d.page_content for d in a_hits] == ["alpha content"]
    assert [d.page_content for d in b_hits] == ["beta content"]


def test_second_upload_accumulates_rather_than_replacing():
    vector_store.add_documents("user-a", _docs("first document"), "first")
    total = vector_store.add_documents("user-a", _docs("second document"), "second")

    assert total == 2
    contents = {
        d.page_content
        for d in vector_store.get_index("user-a").vectorstore.similarity_search(
            "document", k=10
        )
    }
    assert contents == {"first document", "second document"}


def test_version_increments_on_each_upload():
    vector_store.add_documents("user-a", _docs("one"), "d1")
    assert vector_store.get_version("user-a") == 1
    vector_store.add_documents("user-a", _docs("two"), "d2")
    assert vector_store.get_version("user-a") == 2


def test_description_combines_uploads():
    vector_store.add_documents("user-a", _docs("one"), "a resume")
    vector_store.add_documents("user-a", _docs("two"), "a tax form")
    description = vector_store.get_description("user-a")
    assert "a resume" in description and "a tax form" in description


def test_retriever_tool_mentions_the_users_own_description():
    vector_store.add_documents("user-a", _docs("one"), "alice's resume")
    tool = vector_store.get_retriever_tool("user-a")
    assert "alice's resume" in tool.description


def test_empty_chunk_list_rejected():
    import pytest

    with pytest.raises(ValueError):
        vector_store.add_documents("user-a", [], "nothing")


def test_reset_scoped_to_one_user():
    vector_store.add_documents("user-a", _docs("alpha"), "a")
    vector_store.add_documents("user-b", _docs("beta"), "b")

    vector_store.reset("user-a")

    assert vector_store.has_documents("user-a") is False
    assert vector_store.has_documents("user-b") is True


# --- Agent cache invalidation (the stale-retriever defect) -----------------
def test_agent_is_none_before_any_upload():
    assert reAct_agent.get_agent_executor("user-a") is None


def test_agent_is_built_after_upload_and_sees_the_documents():
    vector_store.add_documents("user-a", _docs("the sky is teal"), "colours")

    executor = reAct_agent.get_agent_executor("user-a")
    assert executor is not None

    # The bound tool must query the real index, not a placeholder.
    result = executor.tools[0].invoke("sky")
    assert "teal" in result


def test_agent_is_cached_between_calls_at_the_same_version():
    vector_store.add_documents("user-a", _docs("content"), "d")
    first = reAct_agent.get_agent_executor("user-a")
    second = reAct_agent.get_agent_executor("user-a")
    assert first is second


def test_agent_is_rebuilt_after_a_new_upload():
    """The regression that made uploaded documents unreachable."""
    vector_store.add_documents("user-a", _docs("original content"), "d1")
    first = reAct_agent.get_agent_executor("user-a")

    vector_store.add_documents("user-a", _docs("brand new content"), "d2")
    second = reAct_agent.get_agent_executor("user-a")

    assert first is not second
    assert "brand new content" in second.tools[0].invoke("brand new")


def test_agents_are_not_shared_between_users():
    vector_store.add_documents("user-a", _docs("alpha secret"), "a")
    vector_store.add_documents("user-b", _docs("beta secret"), "b")

    a_result = reAct_agent.get_agent_executor("user-a").tools[0].invoke("secret")
    b_result = reAct_agent.get_agent_executor("user-b").tools[0].invoke("secret")

    assert "alpha secret" in a_result and "beta secret" not in a_result
    assert "beta secret" in b_result and "alpha secret" not in b_result

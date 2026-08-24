"""
Document listing, deletion and account removal.

Before these existed a user could upload documents but never review or remove
them, and had no way to withdraw personal data they had uploaded.

The backend-level tests run against both vector backends, because deletion is
an ownership boundary: it must never reach another user's content.
"""

import io

import pytest
from langchain_core.documents import Document

from src.rag import reAct_agent, vector_store
from tests.conftest import register_and_login


def _docs(*texts, filename="doc.txt"):
    return [
        Document(page_content=text, metadata={"source_filename": filename})
        for text in texts
    ]


# --- listing ---------------------------------------------------------------
def test_listing_is_empty_for_a_new_user(any_backend):
    assert vector_store.list_documents("user-a") == []


def test_listing_groups_chunks_by_source_file(any_backend):
    vector_store.add_documents(
        "user-a", _docs("one", "two", filename="report.pdf"), "report"
    )
    vector_store.add_documents("user-a", _docs("three", filename="notes.txt"), "notes")

    listing = {
        entry["filename"]: entry["chunks"]
        for entry in vector_store.list_documents("user-a")
    }
    assert listing == {"report.pdf": 2, "notes.txt": 1}


def test_listing_shows_only_the_callers_documents(any_backend):
    vector_store.add_documents("user-a", _docs("alice", filename="a.txt"), "a")
    vector_store.add_documents("user-b", _docs("bob", filename="b.txt"), "b")

    names = [entry["filename"] for entry in vector_store.list_documents("user-b")]
    assert names == ["b.txt"]


# --- deletion --------------------------------------------------------------
def test_deleting_a_document_removes_its_chunks(any_backend):
    vector_store.add_documents(
        "user-a", _docs("alpha", "beta", filename="gone.txt"), "d"
    )
    vector_store.add_documents("user-a", _docs("kept", filename="stays.txt"), "d")

    removed = vector_store.delete_document("user-a", "gone.txt")

    assert removed == 2
    names = [entry["filename"] for entry in vector_store.list_documents("user-a")]
    assert names == ["stays.txt"]


def test_deleted_content_is_no_longer_retrievable(any_backend):
    vector_store.add_documents(
        "user-a", _docs("the secret code is swordfish", filename="secret.txt"), "d"
    )
    vector_store.add_documents(
        "user-a", _docs("ordinary notes", filename="ok.txt"), "d"
    )

    vector_store.delete_document("user-a", "secret.txt")

    hits = vector_store.get_retriever("user-a").invoke("secret code")
    assert all("swordfish" not in hit.page_content for hit in hits)


def test_deleting_an_unknown_document_reports_nothing_removed(any_backend):
    vector_store.add_documents("user-a", _docs("content", filename="a.txt"), "d")
    assert vector_store.delete_document("user-a", "never-uploaded.txt") == 0


def test_deletion_cannot_reach_another_users_document(any_backend):
    """Both users named their file identically; only the caller's may go."""
    vector_store.add_documents("user-a", _docs("alice data", filename="same.txt"), "a")
    vector_store.add_documents("user-b", _docs("bob data", filename="same.txt"), "b")

    vector_store.delete_document("user-a", "same.txt")

    assert vector_store.list_documents("user-a") == []
    assert vector_store.list_documents("user-b") == [
        {"filename": "same.txt", "chunks": 1}
    ]


def test_deleting_the_last_document_empties_the_index(any_backend):
    vector_store.add_documents("user-a", _docs("only one", filename="a.txt"), "d")
    vector_store.delete_document("user-a", "a.txt")

    assert vector_store.has_documents("user-a") is False
    assert vector_store.get_retriever("user-a") is None


def test_deletion_invalidates_the_cached_agent(any_backend):
    """A stale agent would keep answering from deleted content."""
    vector_store.add_documents("user-a", _docs("first", filename="a.txt"), "d")
    vector_store.add_documents("user-a", _docs("second", filename="b.txt"), "d")
    first = reAct_agent.get_agent_executor("user-a")

    vector_store.delete_document("user-a", "a.txt")
    second = reAct_agent.get_agent_executor("user-a")

    assert first is not second
    assert "first" not in second.tools[0].invoke("first")


# --- endpoints -------------------------------------------------------------
@pytest.fixture(autouse=True)
def _no_llm_description(monkeypatch):
    import src.rag.document_upload as module

    monkeypatch.setattr(
        module, "enhance_description_with_llm", lambda text: f"about {text}"
    )


def _upload(client, headers, name="notes.txt", data=b"some indexed content"):
    return client.post(
        "/rag/documents/upload",
        files={"file": (name, io.BytesIO(data), "text/plain")},
        headers={**headers, "X-Description": "notes"},
    )


def test_documents_endpoint_lists_uploads(client, auth_headers):
    _upload(client, auth_headers, "a.txt")
    _upload(client, auth_headers, "b.txt")

    body = client.get("/rag/documents", headers=auth_headers).json()

    assert body["total_chunks"] == 2
    assert {d["filename"] for d in body["documents"]} == {"a.txt", "b.txt"}


def test_documents_endpoint_requires_authentication(client):
    assert client.get("/rag/documents").status_code == 401


def test_delete_endpoint_removes_a_document(client, auth_headers):
    _upload(client, auth_headers, "a.txt")
    _upload(client, auth_headers, "b.txt")

    response = client.delete("/rag/documents/a.txt", headers=auth_headers)

    assert response.status_code == 200
    assert response.json() == {"filename": "a.txt", "chunks_deleted": 1}

    remaining = client.get("/rag/documents", headers=auth_headers).json()
    assert [d["filename"] for d in remaining["documents"]] == ["b.txt"]


def test_deleting_an_unknown_document_is_a_404(client, auth_headers):
    _upload(client, auth_headers, "a.txt")
    response = client.delete("/rag/documents/nope.txt", headers=auth_headers)
    assert response.status_code == 404


def test_delete_all_clears_the_index(client, auth_headers):
    _upload(client, auth_headers, "a.txt")
    _upload(client, auth_headers, "b.txt")

    response = client.delete("/rag/documents", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["chunks_deleted"] == 2
    assert client.get("/rag/documents", headers=auth_headers).json()["documents"] == []


def test_one_user_cannot_delete_anothers_documents(client):
    alice = register_and_login(client, "alice", "alice-password-1")
    bob = register_and_login(client, "bob", "bob-password-1")

    _upload(client, alice, "shared-name.txt")
    _upload(client, bob, "shared-name.txt")

    client.delete("/rag/documents/shared-name.txt", headers=alice)

    bob_docs = client.get("/rag/documents", headers=bob).json()
    assert [d["filename"] for d in bob_docs["documents"]] == ["shared-name.txt"]


# --- account deletion ------------------------------------------------------
async def test_account_deletion_removes_everything(client, auth_headers, monkeypatch):
    import src.api.routes as routes
    from src.memory.chat_history_mongo import ChatHistory

    async def fake_run_query(user_id, messages):
        return "stub answer", [], {}

    monkeypatch.setattr(routes, "run_query", fake_run_query)

    _upload(client, auth_headers, "private.txt")
    client.post(
        "/rag/query",
        json={"query": "something personal", "session_id": "s1"},
        headers=auth_headers,
    )

    response = client.delete("/auth/me", headers=auth_headers)
    assert response.status_code == 204

    # The token is revoked along with the account.
    assert client.get("/rag/documents", headers=auth_headers).status_code == 401

    # And the account itself is gone.
    assert (
        client.post(
            "/auth/login",
            json={"username": "alice", "password": "correct-horse-1"},
        ).status_code
        == 401
    )

    from src.db import users as user_store

    assert user_store._memory_users == {}
    assert await ChatHistory.get_session_history("any", "s1").get_messages() == []


def test_account_deletion_requires_authentication(client):
    assert client.delete("/auth/me").status_code == 401


def test_account_deletion_does_not_touch_other_accounts(client):
    alice = register_and_login(client, "alice", "alice-password-1")
    bob = register_and_login(client, "bob", "bob-password-1")

    _upload(client, bob, "bob.txt")
    client.delete("/auth/me", headers=alice)

    assert client.get("/rag/documents", headers=bob).status_code == 200
    assert (
        client.get("/rag/documents", headers=bob).json()["documents"][0]["filename"]
        == "bob.txt"
    )

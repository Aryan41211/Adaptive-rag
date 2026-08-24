"""
End-to-end tests for the document upload endpoint.
"""

import pytest

from src.core.config import settings
from src.rag import vector_store
from tests.conftest import register_and_login


@pytest.fixture(autouse=True)
def _no_llm_description(monkeypatch):
    """Skip the model call that polishes the description."""
    import src.rag.document_upload as module

    monkeypatch.setattr(
        module, "enhance_description_with_llm", lambda text: f"about {text}"
    )


def _post(client, headers, name="notes.txt", data=b"hello world", description="notes"):
    return client.post(
        "/rag/documents/upload",
        files={"file": (name, data, "text/plain")},
        headers={**headers, "X-Description": description},
    )


def test_upload_succeeds_and_reports_what_was_indexed(client, auth_headers):
    response = _post(client, auth_headers, data=b"the quick brown fox")
    assert response.status_code == 200

    body = response.json()
    assert body["filename"] == "notes.txt"
    assert body["chunks_indexed"] == 1
    assert body["total_chunks"] == 1
    assert body["description"] == "about notes"


def test_upload_requires_a_description_header(client, auth_headers):
    response = client.post(
        "/rag/documents/upload",
        files={"file": ("a.txt", b"hello", "text/plain")},
        headers=auth_headers,
    )
    assert response.status_code == 422


def test_blank_description_rejected(client, auth_headers):
    assert _post(client, auth_headers, description="").status_code == 422


def test_overlong_description_rejected(client, auth_headers):
    assert _post(client, auth_headers, description="x" * 400).status_code == 422


def test_unsupported_file_type_returns_415(client, auth_headers):
    response = _post(client, auth_headers, name="payload.exe", data=b"MZ\x90\x00")
    assert response.status_code == 415


def test_mislabelled_pdf_returns_415(client, auth_headers):
    response = _post(client, auth_headers, name="fake.pdf", data=b"not a pdf at all")
    assert response.status_code == 415


def test_empty_file_returns_422(client, auth_headers):
    assert _post(client, auth_headers, data=b"").status_code == 422


def test_oversized_file_returns_413(client, auth_headers, monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 512)
    response = _post(client, auth_headers, data=b"x" * 4096)
    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()


def test_uploads_are_isolated_between_users(client):
    """The cross-tenant leak: one user's upload must not reach another."""
    alice = register_and_login(client, "alice", "alice-password-1")
    bob = register_and_login(client, "bob", "bob-password-1")

    _post(client, alice, data=b"alice confidential salary information")

    alice_response = _post(
        client, alice, name="second.txt", data=b"alice second document"
    )
    assert alice_response.json()["total_chunks"] == 2

    # Bob's first upload starts from an empty index of his own.
    bob_response = _post(client, bob, name="bob.txt", data=b"bob own document")
    assert bob_response.json()["total_chunks"] == 1


def test_uploaded_content_is_searchable_by_its_owner_only(client):
    alice = register_and_login(client, "alice", "alice-password-1")
    bob = register_and_login(client, "bob", "bob-password-1")

    _post(client, alice, data=b"the treasure is buried under the oak tree")
    _post(client, bob, name="bob.txt", data=b"bob writes about spreadsheets")

    indexes = {
        user_id: index
        for user_id, index in _all_indexes().items()
    }
    assert len(indexes) == 2

    contents = [
        doc.page_content
        for index in indexes.values()
        for doc in index.vectorstore.similarity_search("treasure", k=5)
    ]
    # The treasure text exists exactly once, in a single user's index.
    assert sum("treasure" in text for text in contents) == 1


def _all_indexes():
    return dict(vector_store._indexes)

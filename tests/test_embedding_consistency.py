"""
Embedding-model consistency guard.

Vectors from one embedding model are meaningless to another, so a user whose
index was built with model A must get a clear error (not garbage matches)
when the service switches to model B. The check must fire before any write.
"""

import pytest
from langchain_core.documents import Document as Doc

from src.core.config import settings


def _chunks(text="the document content"):
    return [
        Doc(
            page_content=text,
            metadata={"source_filename": "doc.txt", "page": 0},
        )
    ]


def _switch_model(monkeypatch, model: str):
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_EMBEDDING_MODEL", model)


def test_uploading_with_a_different_embedding_model_is_rejected(
    any_backend, monkeypatch
):
    _switch_model(monkeypatch, "model-a")
    any_backend.add_documents("user-a", _chunks("first"), "first")

    _switch_model(monkeypatch, "model-b")
    with pytest.raises(
        ValueError, match="embedding model 'model-a'.*now uses 'model-b'"
    ):
        any_backend.add_documents("user-a", _chunks("second"), "second")


def test_the_second_user_can_use_the_new_model(any_backend, monkeypatch):
    """The guard is per user, not per store: a fresh user may use either."""
    _switch_model(monkeypatch, "model-a")
    any_backend.add_documents("user-a", _chunks(), "a")

    _switch_model(monkeypatch, "model-b")
    any_backend.add_documents("user-b", _chunks(), "b")
    assert any_backend.has_documents("user-b")


def test_same_model_uploads_keep_accumulating(any_backend, monkeypatch):
    """The guard must not reject a normal second upload."""
    _switch_model(monkeypatch, "model-a")
    any_backend.add_documents("user-a", _chunks("one"), "one")
    total = any_backend.add_documents("user-a", _chunks("two"), "two")
    assert total == 2

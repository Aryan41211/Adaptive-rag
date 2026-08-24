"""
Tests for document upload validation and indexing.
"""

import io

import pytest

from src.core.config import settings
from src.core.exceptions import (
    DocumentProcessingError,
    FileTooLargeError,
    UnsupportedFileTypeError,
)
from src.rag import vector_store
from src.rag.document_upload import process_upload


@pytest.fixture(autouse=True)
def _no_llm_description(monkeypatch):
    """Skip the model call that polishes the description."""
    import src.rag.document_upload as module

    monkeypatch.setattr(
        module, "enhance_description_with_llm", lambda text: f"about {text}"
    )


def _upload(user_id="user-a", description="notes", filename="a.txt", data=b"hello"):
    return process_upload(user_id, description, filename, io.BytesIO(data))


def test_txt_upload_is_indexed():
    result = _upload(data=b"the capital of France is Paris")

    assert result["filename"] == "a.txt"
    assert result["chunks_indexed"] == 1
    assert result["total_chunks"] == 1
    assert vector_store.has_documents("user-a")


def test_uploaded_document_is_immediately_searchable():
    """Covers the defect where uploads never reached the retriever."""
    _upload(data=b"the mitochondria is the powerhouse of the cell")

    hits = vector_store.get_retriever("user-a").invoke("mitochondria")
    assert any("powerhouse" in hit.page_content for hit in hits)


def test_source_filename_is_recorded_as_metadata():
    _upload(filename="report.txt", data=b"quarterly results")
    index = vector_store.get_index("user-a")
    docs = index.vectorstore.similarity_search("quarterly", k=1)
    assert docs[0].metadata["source_filename"] == "report.txt"


def test_unsupported_extension_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        _upload(filename="malware.exe", data=b"MZ...")


def test_missing_extension_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        _upload(filename="noextension", data=b"hello")


def test_path_traversal_filename_is_reduced_to_basename():
    result = _upload(filename="../../../../etc/passwd.txt", data=b"root:x:0:0")
    assert result["filename"] == "passwd.txt"
    assert "/" not in result["filename"] and "\\" not in result["filename"]


def test_file_renamed_to_pdf_is_rejected_by_content_check():
    """A filename suffix is client-controlled and proves nothing."""
    with pytest.raises(UnsupportedFileTypeError):
        _upload(filename="not-really.pdf", data=b"this is plain text")


def test_non_utf8_text_file_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        _upload(filename="a.txt", data=b"\xff\xfe\x00binary\x00garbage")


def test_empty_file_rejected():
    with pytest.raises(DocumentProcessingError):
        _upload(data=b"")


def test_oversized_file_rejected(monkeypatch):
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1024)
    with pytest.raises(FileTooLargeError):
        _upload(data=b"x" * 2048)


def test_oversized_upload_leaves_no_temp_file(monkeypatch, tmp_path):
    """The size guard must clean up after itself."""
    monkeypatch.setattr(settings, "MAX_UPLOAD_BYTES", 1024)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))

    with pytest.raises(FileTooLargeError):
        _upload(data=b"x" * 4096)

    assert list(tmp_path.iterdir()) == []


def test_successful_upload_leaves_no_temp_file(monkeypatch, tmp_path):
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))
    _upload(data=b"some readable content")
    assert list(tmp_path.iterdir()) == []


def test_uploads_are_private_to_the_uploader():
    _upload(user_id="user-a", data=b"alice confidential notes")
    assert vector_store.has_documents("user-b") is False


def test_second_upload_keeps_the_first_searchable():
    _upload(data=b"document about penguins")
    result = _upload(filename="b.txt", data=b"document about volcanoes")

    assert result["total_chunks"] == 2
    index = vector_store.get_index("user-a")
    found = {d.page_content for d in index.vectorstore.similarity_search("document", k=5)}
    assert any("penguins" in text for text in found)
    assert any("volcanoes" in text for text in found)


def test_upload_invalidates_the_cached_agent():
    from src.rag import reAct_agent

    _upload(data=b"first content")
    first = reAct_agent.get_agent_executor("user-a")

    _upload(filename="b.txt", data=b"second content")
    second = reAct_agent.get_agent_executor("user-a")

    assert first is not second


def test_embedding_provider_failure_is_reported_as_upstream(monkeypatch):
    """A provider outage is a 502, not a 500 and not a bad-document 422."""
    from src.core.exceptions import IndexingError
    from src.rag import vector_store as vs

    def exploding_add(**_kwargs):
        raise RuntimeError("openai unreachable")

    monkeypatch.setattr(vs, "add_documents", exploding_add)

    with pytest.raises(IndexingError) as exc:
        _upload(data=b"some content")
    assert exc.value.status_code == 502

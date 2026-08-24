"""
Qdrant-specific behaviour.

Two properties are what the migration exists for, and both are asserted here
rather than assumed:

* **Durability** - documents survive the process that indexed them.
* **Shared visibility** - a worker that did not handle an upload still sees
  the documents *and* rebuilds its cached agent.

Tests use a real Qdrant engine running in process (``:memory:``) or on disk
(``path=``), so no server is required.
"""

import warnings

import pytest
from langchain_core.documents import Document
from qdrant_client import QdrantClient

from src.rag import reAct_agent, vector_store
from src.rag.backends.qdrant_backend import USER_ID_FIELD, QdrantBackend

# Local Qdrant ignores payload indexes; filtering still applies. On a real
# server the index is what keeps filtered search off a full scan.
pytestmark = pytest.mark.filterwarnings(
    "ignore:Payload indexes have no effect:UserWarning"
)


def _docs(*texts):
    return [Document(page_content=text) for text in texts]


def _backend(client, name="test_documents"):
    return QdrantBackend(client=client, collection_name=name)


# --- collection setup ------------------------------------------------------
def test_collections_are_created_on_first_use(fake_embeddings):
    client = QdrantClient(location=":memory:")
    backend = _backend(client)

    backend.add_documents("user-a", _docs("content"), "a doc")

    names = {c.name for c in client.get_collections().collections}
    assert "test_documents" in names
    assert "test_documents__meta" in names


def test_vector_size_matches_the_embedding_model(fake_embeddings):
    """A mismatched dimension makes every upsert fail at the server."""
    from tests.conftest import FAKE_EMBEDDING_SIZE

    client = QdrantClient(location=":memory:")
    _backend(client).add_documents("user-a", _docs("content"), "a doc")

    info = client.get_collection("test_documents")
    assert info.config.params.vectors.size == FAKE_EMBEDDING_SIZE


def test_setup_is_idempotent(fake_embeddings):
    client = QdrantClient(location=":memory:")
    backend = _backend(client)

    backend.add_documents("user-a", _docs("one"), "d1")
    backend.add_documents("user-a", _docs("two"), "d2")

    assert backend.has_documents("user-a")
    assert len(client.get_collections().collections) == 2


def test_owner_is_recorded_in_the_payload(fake_embeddings):
    """The filter has nothing to match on if this tag is missing."""
    client = QdrantClient(location=":memory:")
    _backend(client).add_documents("user-a", _docs("content"), "a doc")

    points, _ = client.scroll("test_documents", limit=10, with_payload=True)
    assert points
    for point in points:
        assert point.payload["metadata"]["user_id"] == "user-a"

    assert USER_ID_FIELD == "metadata.user_id"


# --- durability ------------------------------------------------------------
def test_documents_survive_a_new_process(tmp_path, fake_embeddings):
    """
    The reason for the migration: with FAISS this data was gone on restart.

    An on-disk store is written by one client, which is then closed, and read
    back by a completely separate client.
    """
    location = str(tmp_path / "qdrant")

    writer = QdrantClient(path=location)
    _backend(writer).add_documents(
        "user-a", _docs("the mitochondria is the powerhouse"), "biology notes"
    )
    writer.close()

    reader = QdrantClient(path=location)
    backend = _backend(reader)
    try:
        assert backend.has_documents("user-a") is True
        hits = backend.get_retriever("user-a").invoke("mitochondria")
        assert any("powerhouse" in hit.page_content for hit in hits)
        # Metadata survives too, so the tool instruction is still accurate.
        assert "biology notes" in backend.get_description("user-a")
    finally:
        reader.close()


def test_isolation_survives_a_new_process(tmp_path, fake_embeddings):
    location = str(tmp_path / "qdrant")

    writer = QdrantClient(path=location)
    written = _backend(writer)
    written.add_documents("user-a", _docs("alice confidential"), "a")
    written.add_documents("user-b", _docs("bob unrelated"), "b")
    writer.close()

    reader = QdrantClient(path=location)
    backend = _backend(reader)
    try:
        hits = backend.get_retriever("user-b").invoke("confidential")
        assert all("alice" not in hit.page_content for hit in hits)
    finally:
        reader.close()


# --- shared visibility across workers --------------------------------------
def test_a_second_worker_sees_an_upload_it_did_not_handle(fake_embeddings):
    """Two backend instances on one server stand in for two workers."""
    client = QdrantClient(location=":memory:")
    worker_one = _backend(client)
    worker_two = _backend(client)

    worker_one.add_documents("user-a", _docs("uploaded by worker one"), "doc")

    assert worker_two.has_documents("user-a") is True
    hits = worker_two.get_retriever("user-a").invoke("uploaded")
    assert any("worker one" in hit.page_content for hit in hits)


def test_a_second_workers_version_advances_after_a_remote_upload(fake_embeddings):
    """
    The cache-invalidation signal must be derivable, not remembered.

    An in-process counter would leave worker two serving a stale agent that
    cannot see the new documents.
    """
    client = QdrantClient(location=":memory:")
    worker_one = _backend(client)
    worker_two = _backend(client)

    worker_one.add_documents("user-a", _docs("first"), "d1")
    before = worker_two.get_version("user-a")

    worker_one.add_documents("user-a", _docs("second"), "d2")
    after = worker_two.get_version("user-a")

    assert after > before


def test_a_second_workers_agent_rebuilds_after_a_remote_upload(fake_embeddings):
    """End to end: the stale-agent defect must not return with Qdrant."""
    client = QdrantClient(location=":memory:")
    vector_store.set_backend(_backend(client))

    vector_store.add_documents("user-a", _docs("original content"), "d1")
    first = reAct_agent.get_agent_executor("user-a")

    # A different worker indexes more documents against the same server.
    _backend(client).add_documents("user-a", _docs("added elsewhere"), "d2")

    second = reAct_agent.get_agent_executor("user-a")
    assert first is not second
    assert "added elsewhere" in second.tools[0].invoke("added elsewhere")


# --- deletion --------------------------------------------------------------
def test_reset_removes_only_the_named_user(fake_embeddings):
    client = QdrantClient(location=":memory:")
    backend = _backend(client)

    backend.add_documents("user-a", _docs("alpha"), "a")
    backend.add_documents("user-b", _docs("beta"), "b")

    backend.reset("user-a")

    assert backend.has_documents("user-a") is False
    assert backend.has_documents("user-b") is True
    # The user's metadata goes with their documents.
    assert backend.get_description("user-a") == vector_store.DEFAULT_DESCRIPTION


def test_reset_all_drops_the_collections(fake_embeddings):
    client = QdrantClient(location=":memory:")
    backend = _backend(client)
    backend.add_documents("user-a", _docs("alpha"), "a")

    backend.reset()

    assert client.get_collections().collections == []
    # The backend recreates them on next use rather than failing.
    backend.add_documents("user-a", _docs("again"), "a")
    assert backend.has_documents("user-a") is True


# --- degradation -----------------------------------------------------------
def test_health_reports_an_unreachable_server(fake_embeddings):
    class _DeadClient:
        def get_collections(self):
            raise ConnectionError("connection refused")

    healthy, detail = QdrantBackend(
        client=_DeadClient(), collection_name="x"
    ).health()

    assert healthy is False
    assert "unreachable" in detail


def test_health_reports_a_reachable_server(fake_embeddings):
    healthy, detail = _backend(QdrantClient(location=":memory:")).health()
    assert healthy is True
    assert "test_documents" in detail


def test_a_count_failure_reads_as_empty_rather_than_raising(fake_embeddings):
    """A transient outage must not turn into a 500 mid-query."""
    client = QdrantClient(location=":memory:")
    backend = _backend(client)
    backend.add_documents("user-a", _docs("content"), "d")

    def _explode(**_kwargs):
        raise ConnectionError("connection reset")

    backend._client.count = _explode

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert backend.has_documents("user-a") is False
        assert backend.get_version("user-a") == 0
        assert backend.get_retriever("user-a") is None


def test_empty_chunks_rejected(fake_embeddings):
    backend = _backend(QdrantClient(location=":memory:"))
    with pytest.raises(ValueError):
        backend.add_documents("user-a", [], "nothing")


def test_setup_failure_degrades_instead_of_raising(monkeypatch, fake_embeddings):
    """
    If the collection cannot be created the query path must still answer.

    _ensure_collections probes the embedding dimension, so a model-provider
    outage reaches this code. Raising here would turn every query into a 500
    rather than "no documents uploaded".
    """
    from src.rag.backends import qdrant_backend as module

    def _explode():
        raise RuntimeError("embedding provider unavailable")

    monkeypatch.setattr(module, "get_embeddings", _explode)

    backend = _backend(QdrantClient(location=":memory:"))

    assert backend.has_documents("user-a") is False
    assert backend.get_version("user-a") == 0
    assert backend.get_retriever("user-a") is None
    assert backend.get_description("user-a") == "documents the user has uploaded"
    backend.reset("user-a")  # must not raise

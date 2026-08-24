"""
Qdrant backend.

Documents survive restarts and are visible to every process, which is what
lifts the single-worker restriction the FAISS backend imposes.

Multi-tenancy uses the pattern Qdrant recommends: one collection partitioned
by an indexed ``user_id`` payload field, rather than a collection per user.
Every search carries a filter on that field, so a user can only ever match
their own chunks.

Two pieces of state are kept per user:

* the chunks themselves, in the main collection;
* a single metadata point in a companion collection, holding the document
  descriptions used to build the retriever tool's instruction.

The cache-invalidation ``version`` is a filtered point count rather than an
in-process counter, so a worker that did not handle an upload still observes
the change and rebuilds its cached agent.
"""

import threading
import uuid
from typing import Any

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from src.core.config import settings
from src.core.logger import get_logger
from src.llms.openai import get_embeddings
from src.rag.backends.base import DEFAULT_DESCRIPTION, VectorStoreBackend

logger = get_logger(__name__)

# LangChain nests document metadata under this payload key.
USER_ID_FIELD = "metadata.user_id"

# Stable namespace so a user's metadata point id is reproducible.
_META_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")


class QdrantBackend(VectorStoreBackend):
    """Per-user document storage in a shared Qdrant collection."""

    name = "qdrant"

    def __init__(
        self,
        client: QdrantClient | None = None,
        collection_name: str | None = None,
    ):
        """
        Args:
            client: An existing client. Built from settings when omitted.
            collection_name: Overrides the configured collection name.
        """
        self._client = client or QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
            timeout=30,
        )
        self._collection = collection_name or settings.QDRANT_COLLECTION
        self._meta_collection = f"{self._collection}__meta"
        self._lock = threading.RLock()
        self._ready = False
        self._store: QdrantVectorStore | None = None

    # --- setup ------------------------------------------------------------
    def _vector_size(self) -> int:
        """Probe the embedding dimension. Called once per process."""
        return len(get_embeddings().embed_query("dimension probe"))

    def _ensure_collections(self) -> None:
        """Create the collections and payload index. Idempotent."""
        if self._ready:
            return
        with self._lock:
            if self._ready:
                return

            existing = {c.name for c in self._client.get_collections().collections}

            if self._collection not in existing:
                size = self._vector_size()
                logger.info(
                    "Creating Qdrant collection '%s' (dim=%d)",
                    self._collection,
                    size,
                )
                self._client.create_collection(
                    collection_name=self._collection,
                    vectors_config=models.VectorParams(
                        size=size, distance=models.Distance.COSINE
                    ),
                )

            # Without this index every filtered search is a full scan.
            try:
                self._client.create_payload_index(
                    collection_name=self._collection,
                    field_name=USER_ID_FIELD,
                    field_schema=models.PayloadSchemaType.KEYWORD,
                )
            except Exception as exc:  # noqa: BLE001 - already exists
                logger.debug("Payload index already present: %s", exc)

            if self._meta_collection not in existing:
                # Metadata points are addressed by id, never searched, so the
                # smallest possible vector is enough.
                self._client.create_collection(
                    collection_name=self._meta_collection,
                    vectors_config=models.VectorParams(
                        size=1, distance=models.Distance.COSINE
                    ),
                )

            self._ready = True

    def _get_store(self) -> QdrantVectorStore:
        self._ensure_collections()
        if self._store is None:
            self._store = QdrantVectorStore(
                client=self._client,
                collection_name=self._collection,
                embedding=get_embeddings(),
            )
        return self._store

    @staticmethod
    def _user_filter(user_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key=USER_ID_FIELD,
                    match=models.MatchValue(value=user_id),
                )
            ]
        )

    @staticmethod
    def _meta_id(user_id: str) -> str:
        return str(uuid.uuid5(_META_NAMESPACE, user_id))

    # --- metadata ---------------------------------------------------------
    def _read_meta(self, user_id: str) -> dict[str, Any]:
        try:
            self._ensure_collections()
            points = self._client.retrieve(
                collection_name=self._meta_collection,
                ids=[self._meta_id(user_id)],
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001 - treat as absent
            logger.warning("Could not read document metadata: %s", exc)
            return {}
        return dict(points[0].payload) if points else {}

    def _write_meta(self, user_id: str, payload: dict[str, Any]) -> None:
        self._client.upsert(
            collection_name=self._meta_collection,
            points=[
                models.PointStruct(
                    id=self._meta_id(user_id),
                    vector=[0.0],
                    payload={**payload, "user_id": user_id},
                )
            ],
        )

    # --- interface --------------------------------------------------------
    def add_documents(
        self, user_id: str, chunks: list[Document], description: str
    ) -> int:
        if not chunks:
            raise ValueError("No content could be extracted from the document.")

        store = self._get_store()

        # The owner tag is what every later search filters on.
        for chunk in chunks:
            chunk.metadata["user_id"] = user_id
            chunk.metadata["description"] = description

        store.add_documents(chunks)

        meta = self._read_meta(user_id)
        descriptions = list(meta.get("descriptions") or [])
        if description and description not in descriptions:
            descriptions.append(description)

        total = self._count(user_id)
        self._write_meta(user_id, {"descriptions": descriptions, "chunk_count": total})

        logger.info("Indexed %d chunks in Qdrant (total=%d)", len(chunks), total)
        return total

    def get_retriever(self, user_id: str) -> VectorStoreRetriever | None:
        if not self.has_documents(user_id):
            return None
        return self._get_store().as_retriever(
            search_kwargs={
                "k": settings.RETRIEVER_TOP_K,
                "filter": self._user_filter(user_id),
            }
        )

    def get_description(self, user_id: str) -> str:
        descriptions = self._read_meta(user_id).get("descriptions") or []
        parts = [d.strip() for d in descriptions if d and d.strip()]
        return "; ".join(parts) if parts else DEFAULT_DESCRIPTION

    def get_version(self, user_id: str) -> int:
        # A filtered count is derivable by any process, unlike an in-memory
        # counter, so every worker observes an upload made by another.
        return self._count(user_id)

    def has_documents(self, user_id: str) -> bool:
        return self._count(user_id) > 0

    def reset(self, user_id: str | None = None) -> None:
        try:
            self._ensure_collections()
        except Exception as exc:  # noqa: BLE001 - nothing to reset
            logger.warning("Qdrant reset skipped, setup failed: %s", exc)
            return
        if user_id is None:
            for collection in (self._collection, self._meta_collection):
                try:
                    self._client.delete_collection(collection)
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Could not drop %s: %s", collection, exc)
            self._ready = False
            self._store = None
            return

        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(filter=self._user_filter(user_id)),
        )
        self._client.delete(
            collection_name=self._meta_collection,
            points_selector=models.PointIdsList(points=[self._meta_id(user_id)]),
        )

    def health(self) -> tuple[bool, str]:
        try:
            self._client.get_collections()
            return True, f"qdrant ({self._collection})"
        except Exception as exc:  # noqa: BLE001 - reported, not raised
            logger.warning("Qdrant health check failed: %s", exc)
            return False, f"qdrant unreachable: {type(exc).__name__}"

    # --- internals --------------------------------------------------------
    def _count(self, user_id: str) -> int:
        # Setup is attempted inside the guard: if the collection cannot be
        # created (for example the embedding probe fails because the model
        # provider is down) the query degrades to "no documents" rather than
        # raising out of every graph node.
        try:
            self._ensure_collections()
            return self._client.count(
                collection_name=self._collection,
                count_filter=self._user_filter(user_id),
                exact=True,
            ).count
        except Exception as exc:  # noqa: BLE001 - treat as empty
            logger.warning("Qdrant count unavailable: %s", exc)
            return 0

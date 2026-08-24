"""
In-process FAISS backend.

The development default: no external service, but the index lives in this
process's memory. It is lost on restart and is not visible to other workers,
which is why a deployment using this backend must run exactly one worker.
"""

import threading
from dataclasses import dataclass, field

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

from src.core.config import settings
from src.core.logger import get_logger
from src.llms.openai import get_embeddings
from src.rag.backends.base import DEFAULT_DESCRIPTION, VectorStoreBackend

logger = get_logger(__name__)


@dataclass
class UserIndex:
    """One user's index and its metadata."""

    vectorstore: FAISS
    descriptions: list[str] = field(default_factory=list)
    chunk_count: int = 0
    version: int = 1

    @property
    def description(self) -> str:
        """Combined natural-language description of the indexed documents."""
        parts = [d.strip() for d in self.descriptions if d and d.strip()]
        return "; ".join(parts) if parts else DEFAULT_DESCRIPTION


class FaissBackend(VectorStoreBackend):
    """Per-user FAISS indexes held in process memory."""

    name = "faiss"

    def __init__(self):
        # FastAPI runs sync handlers in a thread pool, so two uploads can
        # land concurrently.
        self._lock = threading.RLock()
        self._indexes: dict[str, UserIndex] = {}

    def add_documents(
        self, user_id: str, chunks: list[Document], description: str
    ) -> int:
        if not chunks:
            raise ValueError("No content could be extracted from the document.")

        embeddings = get_embeddings()

        with self._lock:
            index = self._indexes.get(user_id)
            if index is None:
                index = UserIndex(
                    vectorstore=FAISS.from_documents(
                        documents=chunks, embedding=embeddings
                    ),
                    descriptions=[description],
                    chunk_count=len(chunks),
                    version=1,
                )
                self._indexes[user_id] = index
            else:
                index.vectorstore.add_documents(chunks)
                if description and description not in index.descriptions:
                    index.descriptions.append(description)
                index.chunk_count += len(chunks)
                index.version += 1

            logger.info(
                "Indexed %d chunks (total=%d, version=%d)",
                len(chunks),
                index.chunk_count,
                index.version,
            )
            return index.chunk_count

    def get_retriever(self, user_id: str) -> VectorStoreRetriever | None:
        index = self._get_index(user_id)
        if index is None:
            return None
        return index.vectorstore.as_retriever(
            search_kwargs={"k": settings.RETRIEVER_TOP_K}
        )

    def get_description(self, user_id: str) -> str:
        index = self._get_index(user_id)
        return index.description if index else DEFAULT_DESCRIPTION

    def get_version(self, user_id: str) -> int:
        index = self._get_index(user_id)
        return index.version if index else 0

    def has_documents(self, user_id: str) -> bool:
        return self._get_index(user_id) is not None

    def list_documents(self, user_id: str) -> list[dict]:
        index = self._get_index(user_id)
        if index is None:
            return []

        counts: dict[str, int] = {}
        with self._lock:
            for document in index.vectorstore.docstore._dict.values():
                name = (document.metadata or {}).get(
                    "source_filename", "uploaded document"
                )
                counts[name] = counts.get(name, 0) + 1
        return [
            {"filename": name, "chunks": count}
            for name, count in sorted(counts.items())
        ]

    def delete_document(self, user_id: str, filename: str) -> int:
        with self._lock:
            index = self._get_index(user_id)
            if index is None:
                return 0

            store = index.vectorstore
            doomed = [
                doc_id
                for doc_id, document in store.docstore._dict.items()
                if (document.metadata or {}).get("source_filename") == filename
            ]
            if not doomed:
                return 0

            store.delete(doomed)
            index.chunk_count = max(index.chunk_count - len(doomed), 0)
            # Bump the version so the cached agent is rebuilt against the
            # reduced index rather than continuing to serve deleted content.
            index.version += 1

            if index.chunk_count == 0:
                self._indexes.pop(user_id, None)

            logger.info("Deleted %d chunks for '%s'", len(doomed), filename)
            return len(doomed)

    def reset(self, user_id: str | None = None) -> None:
        with self._lock:
            if user_id is None:
                self._indexes.clear()
            else:
                self._indexes.pop(user_id, None)

    def health(self) -> tuple[bool, str]:
        return True, "faiss (in-memory, single worker, not durable)"

    # --- internals --------------------------------------------------------
    def _get_index(self, user_id: str) -> UserIndex | None:
        with self._lock:
            index = self._indexes.get(user_id)
            return index if index and index.chunk_count > 0 else None

    def get_index(self, user_id: str) -> UserIndex | None:
        """Expose the raw index. Used by tests only."""
        return self._get_index(user_id)

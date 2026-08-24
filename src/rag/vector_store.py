"""
Per-user document storage.

Dispatches to whichever backend is configured. Callers keep the same
module-level API regardless of which one is active:

* **Qdrant** when ``QDRANT_URL`` is set - durable, shared across processes,
  so the service can run more than one worker.
* **FAISS** otherwise - in-process, lost on restart, single worker only.

Both isolate documents by owner: a user can never retrieve another user's
content, and one user's upload never displaces another's.
"""

import threading
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever

from src.core.config import settings
from src.core.logger import get_logger
from src.rag.backends.base import DEFAULT_DESCRIPTION, VectorStoreBackend

logger = get_logger(__name__)

_lock = threading.RLock()
_backend: Optional[VectorStoreBackend] = None


def get_backend() -> VectorStoreBackend:
    """
    Return the active backend, constructing it on first use.

    Returns:
        The configured :class:`VectorStoreBackend`.
    """
    global _backend
    with _lock:
        if _backend is None:
            if settings.qdrant_enabled:
                from src.rag.backends.qdrant_backend import QdrantBackend

                _backend = QdrantBackend()
                logger.info("Vector store backend: qdrant")
            else:
                from src.rag.backends.faiss_backend import FaissBackend

                _backend = FaissBackend()
                logger.info(
                    "Vector store backend: faiss (in-memory; documents are "
                    "lost on restart and the service must run one worker)"
                )
        return _backend


def set_backend(backend: Optional[VectorStoreBackend]) -> None:
    """
    Replace the active backend. Intended for tests.

    Args:
        backend: The backend to use, or None to rebuild from settings.
    """
    global _backend
    with _lock:
        _backend = backend


def add_documents(user_id: str, chunks: list[Document], description: str) -> int:
    """
    Index document chunks for one user.

    Chunks accumulate: uploading a second document keeps the first
    searchable.

    Args:
        user_id: The owning user.
        chunks: Chunked documents to index.
        description: Description of the source document.

    Returns:
        The user's new total chunk count.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    return get_backend().add_documents(user_id, chunks, description)


def get_retriever(user_id: str) -> Optional[VectorStoreRetriever]:
    """
    Return a retriever restricted to one user's documents.

    Args:
        user_id: The owning user.

    Returns:
        A retriever, or None when the user has uploaded nothing.
    """
    return get_backend().get_retriever(user_id)


def get_description(user_id: str) -> str:
    """
    Return the combined description of a user's documents.

    Args:
        user_id: The owning user.

    Returns:
        The description, or a neutral default.
    """
    return get_backend().get_description(user_id)


def get_version(user_id: str) -> int:
    """
    Return a value that changes whenever the user's documents change.

    Used to invalidate the cached ReAct agent.

    Args:
        user_id: The owning user.

    Returns:
        The current version, or 0 when nothing is indexed.
    """
    return get_backend().get_version(user_id)


def has_documents(user_id: str) -> bool:
    """
    Report whether a user has indexed any documents.

    Args:
        user_id: The owning user.

    Returns:
        True if at least one chunk is indexed.
    """
    return get_backend().has_documents(user_id)


def get_retriever_tool(user_id: str):
    """
    Build a retriever tool bound to a user's documents as they are now.

    Callers must not cache the result across an index version change.

    Args:
        user_id: The owning user.

    Returns:
        A retriever tool, or None when the user has uploaded nothing.
    """
    retriever = get_retriever(user_id)
    if retriever is None:
        return None

    return create_retriever_tool(
        retriever,
        "search_uploaded_documents",
        "Search the user's uploaded documents. Use this tool **only** to "
        f"answer questions about: {get_description(user_id)}. "
        "Do not use it for anything else.",
    )


def reset(user_id: Optional[str] = None) -> None:
    """
    Delete indexed documents.

    Args:
        user_id: Drop only this user's documents; drop all when None.
    """
    get_backend().reset(user_id)


def health() -> tuple[bool, str]:
    """
    Report backend reachability.

    Returns:
        A ``(healthy, detail)`` pair for the readiness probe.
    """
    return get_backend().health()


def get_index(user_id: str):
    """
    Return the raw FAISS index for a user. FAISS backend only; tests only.

    Args:
        user_id: The owning user.

    Returns:
        The index, or None.
    """
    backend = get_backend()
    getter = getattr(backend, "get_index", None)
    return getter(user_id) if getter else None


__all__ = [
    "DEFAULT_DESCRIPTION",
    "add_documents",
    "get_backend",
    "get_description",
    "get_index",
    "get_retriever",
    "get_retriever_tool",
    "get_version",
    "has_documents",
    "health",
    "reset",
    "set_backend",
]

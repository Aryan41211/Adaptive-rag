"""
Per-user vector store registry.

Each user owns an isolated FAISS index and an accompanying description of the
documents it holds. Previously a single process-global index and a shared
``description.txt`` on disk were used, which meant one user's upload replaced
every other user's documents and every user could retrieve every other user's
content.

Each index carries a monotonically increasing ``version``. Downstream caches
(the retriever tool and the ReAct agent) key on that version, so an upload
invalidates them and newly indexed documents become searchable immediately.
"""

import threading
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.documents import Document
from langchain_core.tools import create_retriever_tool
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_community.vectorstores import FAISS

from src.core.config import settings
from src.core.logger import get_logger
from src.llms.openai import get_embeddings

logger = get_logger(__name__)

DEFAULT_DESCRIPTION = "documents the user has uploaded"

# Guards the registry: FastAPI runs sync handlers in a thread pool, so two
# uploads can land concurrently.
_lock = threading.RLock()


@dataclass
class UserIndex:
    """One user's document index and its metadata."""

    vectorstore: FAISS
    descriptions: list[str] = field(default_factory=list)
    chunk_count: int = 0
    version: int = 1

    @property
    def description(self) -> str:
        """Combined natural-language description of the indexed documents."""
        parts = [d.strip() for d in self.descriptions if d and d.strip()]
        return "; ".join(parts) if parts else DEFAULT_DESCRIPTION


_indexes: dict[str, UserIndex] = {}


def has_documents(user_id: str) -> bool:
    """
    Report whether a user has indexed any documents.

    Args:
        user_id: The owning user.

    Returns:
        True if at least one document chunk is indexed.
    """
    with _lock:
        index = _indexes.get(user_id)
        return index is not None and index.chunk_count > 0


def get_index(user_id: str) -> Optional[UserIndex]:
    """
    Return a user's index, if one exists.

    Args:
        user_id: The owning user.

    Returns:
        The :class:`UserIndex`, or None when nothing has been uploaded.
    """
    with _lock:
        return _indexes.get(user_id)


def get_description(user_id: str) -> str:
    """
    Return the description of a user's indexed documents.

    Args:
        user_id: The owning user.

    Returns:
        The combined description, or a neutral default.
    """
    index = get_index(user_id)
    return index.description if index else DEFAULT_DESCRIPTION


def get_version(user_id: str) -> int:
    """
    Return the current index version for cache invalidation.

    Args:
        user_id: The owning user.

    Returns:
        The version number, or 0 when no index exists.
    """
    index = get_index(user_id)
    return index.version if index else 0


def add_documents(
    user_id: str, chunks: list[Document], description: str
) -> int:
    """
    Add document chunks to a user's index, creating it if necessary.

    Chunks accumulate: uploading a second document keeps the first one
    searchable.

    Args:
        user_id: The owning user.
        chunks: Chunked documents to index.
        description: Natural-language description of the source document.

    Returns:
        The new total number of indexed chunks for this user.

    Raises:
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("No content could be extracted from the document.")

    embeddings = get_embeddings()

    with _lock:
        index = _indexes.get(user_id)
        if index is None:
            index = UserIndex(
                vectorstore=FAISS.from_documents(
                    documents=chunks, embedding=embeddings
                ),
                descriptions=[description],
                chunk_count=len(chunks),
                version=1,
            )
            _indexes[user_id] = index
        else:
            index.vectorstore.add_documents(chunks)
            if description and description not in index.descriptions:
                index.descriptions.append(description)
            index.chunk_count += len(chunks)
            index.version += 1

        logger.info(
            "Indexed %d chunks for user (total=%d, version=%d)",
            len(chunks),
            index.chunk_count,
            index.version,
        )
        return index.chunk_count


def get_retriever(user_id: str) -> Optional[VectorStoreRetriever]:
    """
    Return a retriever over a user's documents.

    Args:
        user_id: The owning user.

    Returns:
        A retriever, or None when the user has uploaded nothing.
    """
    index = get_index(user_id)
    if index is None or index.chunk_count == 0:
        return None
    return index.vectorstore.as_retriever(
        search_kwargs={"k": settings.RETRIEVER_TOP_K}
    )


def get_retriever_tool(user_id: str):
    """
    Build a LangChain retriever tool bound to a user's current index.

    The tool is built against the index as it exists *now*; callers must not
    cache it across an index version change.

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
    Drop indexes.

    Args:
        user_id: Drop only this user's index; drop all when None.
    """
    with _lock:
        if user_id is None:
            _indexes.clear()
        else:
            _indexes.pop(user_id, None)

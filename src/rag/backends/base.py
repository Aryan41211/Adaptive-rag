"""
Vector store backend interface.

Both backends store documents partitioned by owner and expose the same
operations, so the rest of the application never learns which one is active.

``version`` is the cache-invalidation signal: it must change whenever a user's
documents change, and must be derivable by any process, so that a worker which
did not handle an upload still rebuilds its cached agent.
"""

from abc import ABC, abstractmethod

from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStoreRetriever

DEFAULT_DESCRIPTION = "documents the user has uploaded"


class VectorStoreBackend(ABC):
    """Storage and retrieval of per-user document chunks."""

    name: str

    @abstractmethod
    def add_documents(
        self, user_id: str, chunks: list[Document], description: str
    ) -> int:
        """
        Index chunks for a user, returning their new total chunk count.

        Chunks accumulate; a second upload does not replace the first.
        """

    @abstractmethod
    def get_retriever(self, user_id: str) -> VectorStoreRetriever | None:
        """Return a retriever restricted to one user's documents."""

    @abstractmethod
    def get_description(self, user_id: str) -> str:
        """Return the combined description of a user's documents."""

    @abstractmethod
    def get_version(self, user_id: str) -> int:
        """Return a value that changes whenever the user's documents change."""

    @abstractmethod
    def has_documents(self, user_id: str) -> bool:
        """Report whether the user has any indexed documents."""

    @abstractmethod
    def list_documents(self, user_id: str) -> list[dict]:
        """
        Summarise the documents a user has indexed.

        Returns one entry per source file, with its chunk count. Users cannot
        review or remove what they have uploaded without this.
        """

    @abstractmethod
    def delete_document(self, user_id: str, filename: str) -> int:
        """
        Remove one source document from a user's index.

        Returns the number of chunks deleted, 0 if no such document exists.
        """

    @abstractmethod
    def reset(self, user_id: str | None = None) -> None:
        """Delete one user's documents, or every user's when None."""

    def health(self) -> tuple[bool, str]:
        """
        Report backend reachability for the readiness probe.

        Returns:
            A ``(healthy, detail)`` pair.
        """
        return True, self.name

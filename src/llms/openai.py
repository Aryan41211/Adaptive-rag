"""
OpenAI model factories.

Credentials are passed explicitly from validated settings rather than relying
on ambient environment variables, so a missing key is caught at startup by
configuration validation instead of at request time by the provider.
"""

import threading
from collections import OrderedDict
from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.core.config import settings


class CachedEmbeddings:
    """
    Bounded LRU cache around ``embed_query``.

    One turn embeds the same question text several times: the classifier
    embeds it to build routing context, the ReAct retriever embeds it again,
    and citation collection embeds it a third time. Each is a blocking
    provider call, so the cache collapses a turn's repeated queries into a
    single call. Successive turns asking the same question are hits too.

    Embedding a text with the same model is deterministic, so a cached vector
    is identical to one computed again; memory stays bounded by the LRU size.
    The cached text is held in process memory for its time in the cache and
    is visible keyed by exact text, so the bound keeps that exposure small.

    ``embed_documents`` is deliberately not cached: it runs on the upload
    path, where each text is embedded exactly once.
    """

    def __init__(self, embeddings, maxsize: int = 1024):
        """
        Args:
            embeddings: An object with ``embed_query`` and ``embed_documents``.
            maxsize: Maximum number of query texts to remember.
        """
        self._embeddings = embeddings
        self._lock = threading.RLock()
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._maxsize = maxsize

    def embed_query(self, text: str) -> list[float]:
        """Return the embedding for ``text``, computed at most once."""
        with self._lock:
            cached = self._cache.get(text)
            if cached is not None:
                self._cache.move_to_end(text)
                return cached

        vector = self._embeddings.embed_query(text)

        with self._lock:
            self._cache[text] = vector
            self._cache.move_to_end(text)
            while len(self._cache) > self._maxsize:
                self._cache.popitem(last=False)
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed documents without caching."""
        return self._embeddings.embed_documents(texts)


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """
    Return the shared chat model.

    Returns:
        A configured :class:`ChatOpenAI` instance.
    """
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        timeout=60,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_answer_llm() -> ChatOpenAI:
    """
    Return the chat model used for the nodes that produce the user's answer.

    Identical to :func:`get_llm` except that it streams. ``invoke`` on a
    non-streaming model makes one blocking request and emits no token events,
    so the streaming endpoint would silently degrade to delivering the whole
    answer in a single chunk.

    Streaming is deliberately not enabled on the shared model: the classifier
    and grader use structured output, where streaming buys nothing and adds a
    partial-parse failure mode.

    ``stream_usage`` keeps token counts flowing, which the cost accounting
    depends on; without it a streamed response reports no usage.

    Returns:
        A streaming :class:`ChatOpenAI` instance.
    """
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0,
        timeout=60,
        max_retries=2,
        streaming=True,
        stream_usage=True,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> CachedEmbeddings:
    """
    Return the shared, query-caching embeddings model.

    Returns:
        A :class:`CachedEmbeddings` wrapping a configured
        :class:`OpenAIEmbeddings`; repeated queries within and across turns
        are served from the bounded cache instead of the provider.
    """
    return CachedEmbeddings(
        OpenAIEmbeddings(
            model=settings.OPENAI_EMBEDDING_MODEL,
            api_key=settings.OPENAI_API_KEY,
            timeout=60,
            max_retries=2,
        )
    )

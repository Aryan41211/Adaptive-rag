"""
Model provider dispatch.

Every other module asks for its models through this facade, which selects the
implementation behind it from the validated settings: ``LLM_PROVIDER`` and
``EMBEDDING_PROVIDER`` pick between the OpenAI-backed factories in
:mod:`src.llms.openai` and the local Ollama-backed factories in
:mod:`src.llms.ollama`.

Constructing a model never reaches the network or reads a credential file: a
:class:`langchain_ollama.ChatOllama` and a :class:`langchain_openai.ChatOpenAI`
are just configuration objects until the first call. That keeps startup fast
and offline, and means a provider outage only ever surfaces at request time,
where the graph degrades gracefully.
"""

import threading
from collections import OrderedDict
from functools import lru_cache

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
def get_llm():
    """
    Return the configured chat model for routing and grading tasks.

    Returns:
        A ``ChatOpenAI`` or ``ChatOllama`` instance depending on
        ``settings.LLM_PROVIDER``.
    """
    if settings.LLM_PROVIDER == "ollama":
        from src.llms.ollama import get_llm as _ollama_llm

        return _ollama_llm()
    from src.llms.openai import get_llm as _openai_llm

    return _openai_llm()


@lru_cache(maxsize=1)
def get_answer_llm():
    """
    Return the chat model used for the nodes that produce the user's answer.

    Same provider selection as :func:`get_llm`; the underlying factories
    differ only in streaming setup.

    Returns:
        A ``ChatOpenAI`` or ``ChatOllama`` instance depending on
        ``settings.LLM_PROVIDER``.
    """
    if settings.LLM_PROVIDER == "ollama":
        from src.llms.ollama import get_answer_llm as _ollama_llm

        return _ollama_llm()
    from src.llms.openai import get_answer_llm as _openai_llm

    return _openai_llm()


@lru_cache(maxsize=1)
def get_embeddings() -> CachedEmbeddings:
    """
    Return the configured, query-caching embeddings model.

    Returns:
        A :class:`CachedEmbeddings` wrapping the provider selected by
        ``settings.EMBEDDING_PROVIDER``.
    """
    if settings.EMBEDDING_PROVIDER == "ollama":
        from src.llms.ollama import get_embeddings as _ollama_embeddings

        return _ollama_embeddings()
    from src.llms.openai import get_embeddings as _openai_embeddings

    return _openai_embeddings()

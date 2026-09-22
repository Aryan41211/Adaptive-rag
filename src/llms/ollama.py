"""
Ollama model factories.

Chat and embeddings are served by a local Ollama server, so a deployment can
run with no API credentials and no per-token spend. Structured output is
produced through the model's tool-calling support (the default
``with_structured_output`` method); models without tool calling fall back to
the graph nodes' graceful degradation rather than failing the turn.
"""

from functools import lru_cache

from langchain_ollama import ChatOllama, OllamaEmbeddings

from src.core.config import settings
from src.llms.provider import CachedEmbeddings


@lru_cache(maxsize=1)
def get_llm() -> ChatOllama:
    """
    Return the shared Ollama chat model.

    Returns:
        A configured :class:`ChatOllama` instance.
    """
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
        timeout=60,
    )


@lru_cache(maxsize=1)
def get_answer_llm() -> ChatOllama:
    """
    Return the Ollama chat model used for the nodes that produce the answer.

    Identical to :func:`get_llm`. Note that ``invoke`` on Ollama does not
    stream token events, so the streaming endpoint delivers the answer as a
    single token chunk when this provider is active; the payload is the same,
    only the pacing differs.

    Returns:
        A configured :class:`ChatOllama` instance.
    """
    return ChatOllama(
        model=settings.OLLAMA_MODEL,
        base_url=settings.OLLAMA_BASE_URL,
        temperature=0,
        timeout=60,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> CachedEmbeddings:
    """
    Return the shared, query-caching Ollama embeddings model.

    Returns:
        A :class:`CachedEmbeddings` wrapping a configured
        :class:`OllamaEmbeddings`; repeated queries within and across turns
        are served from the bounded cache instead of the provider.
    """
    return CachedEmbeddings(
        OllamaEmbeddings(
            model=settings.OLLAMA_EMBEDDING_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
    )

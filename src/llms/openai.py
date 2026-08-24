"""
OpenAI model factories.

Credentials are passed explicitly from validated settings rather than relying
on ambient environment variables, so a missing key is caught at startup by
configuration validation instead of at request time by the provider.
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.core.config import settings


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
def get_embeddings() -> OpenAIEmbeddings:
    """
    Return the shared embeddings model.

    Returns:
        A configured :class:`OpenAIEmbeddings` instance.
    """
    return OpenAIEmbeddings(
        model=settings.OPENAI_EMBEDDING_MODEL,
        api_key=settings.OPENAI_API_KEY,
        timeout=60,
        max_retries=2,
    )

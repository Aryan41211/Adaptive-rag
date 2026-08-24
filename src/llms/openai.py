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

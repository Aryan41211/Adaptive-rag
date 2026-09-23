"""
Gemini model factories.

Credentials are passed explicitly from validated settings rather than relying
on ambient environment variables, so a missing key is caught at startup by
configuration validation instead of at request time by the provider. Like the
OpenAI factories, the key is passed on construction: without one this library
falls back to Google Application Default Credentials and raises during
construction, which would surface as a boot failure even before validation
improves the message. The providers here are only active when
``settings.LLM_PROVIDER`` / ``settings.EMBEDDING_PROVIDER`` are ``gemini``;
see :mod:`src.llms.provider`.
"""

from functools import lru_cache

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

from src.core.config import settings
from src.llms.provider import CachedEmbeddings


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """
    Return the shared chat model.

    Structured output and tool calling (used by the classifier, the grader,
    the verification step and the ReAct agent) run through Gemini's native
    function calling, which ``with_structured_output`` and ``bind_tools``
    on this model support.

    Returns:
        A configured :class:`ChatGoogleGenerativeAI` instance.
    """
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
        timeout=60,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_answer_llm() -> ChatGoogleGenerativeAI:
    """
    Return the Gemini chat model used for the nodes that produce the answer.

    Gemini has no ``streaming=True`` switch: ``invoke`` issues one blocking
    request and emits no token events, so the streaming endpoint delivers the
    answer in a single chunk when this provider is active. This is the same
    deliberate degradation as the Ollama provider's, where the payload is
    identical and only the pacing differs.

    Returns:
        A configured :class:`ChatGoogleGenerativeAI` instance.
    """
    return ChatGoogleGenerativeAI(
        model=settings.GEMINI_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0,
        timeout=60,
        max_retries=2,
    )


@lru_cache(maxsize=1)
def get_embeddings() -> CachedEmbeddings:
    """
    Return the shared, query-caching Gemini embeddings model.

    The model name must be prefixed with ``models/``: this library sends the
    field verbatim to ``BatchEmbedContents``, which rejects a bare name.

    Returns:
        A :class:`CachedEmbeddings` wrapping a configured
        :class:`GoogleGenerativeAIEmbeddings`; repeated queries within and
        across turns are served from the bounded cache instead of the provider.
    """
    return CachedEmbeddings(
        GoogleGenerativeAIEmbeddings(
            model=f"models/{settings.GEMINI_EMBEDDING_MODEL}",
            google_api_key=settings.GEMINI_API_KEY,
            request_options={"timeout": 60},
        )
    )

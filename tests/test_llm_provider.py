"""
Provider dispatch tests.

The facade must hand out the right implementation from the configured
providers, and constructing a model must never reach the network: building a
``ChatOllama`` or ``ChatOpenAI`` is config work, not an I/O call. Every
factory is lru-cached, so the tests clear the caches before each switch.
"""

import src.llms.provider as provider
from src.core.config import settings


def _clear_caches():
    provider.get_llm.cache_clear()
    provider.get_answer_llm.cache_clear()
    provider.get_embeddings.cache_clear()


def test_default_provider_constructs_openai_models():
    from langchain_openai import ChatOpenAI, OpenAIEmbeddings

    _clear_caches()
    assert isinstance(provider.get_llm(), ChatOpenAI)
    assert isinstance(provider.get_answer_llm(), ChatOpenAI)
    embeddings = provider.get_embeddings()
    assert isinstance(embeddings, provider.CachedEmbeddings)
    assert isinstance(embeddings._embeddings, OpenAIEmbeddings)


def test_ollama_provider_constructs_ollama_models(monkeypatch):
    from langchain_ollama import ChatOllama, OllamaEmbeddings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "ollama")
    _clear_caches()
    assert isinstance(provider.get_llm(), ChatOllama)
    assert isinstance(provider.get_answer_llm(), ChatOllama)
    embeddings = provider.get_embeddings()
    assert isinstance(embeddings, provider.CachedEmbeddings)
    assert isinstance(embeddings._embeddings, OllamaEmbeddings)


def test_construction_never_reaches_the_network(monkeypatch):
    """A provider outage must be a request-time problem, not a boot problem."""
    monkeypatch.setattr(settings, "LLM_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "ollama")
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://127.0.0.1:1")
    _clear_caches()
    provider.get_llm()
    provider.get_answer_llm()
    provider.get_embeddings()


def test_cached_embeddings_is_a_real_embeddings_object():
    """Vector stores only accept Embeddings instances or callables."""
    from langchain_core.embeddings import Embeddings

    from langchain_core.embeddings import DeterministicFakeEmbedding

    cached = provider.CachedEmbeddings(DeterministicFakeEmbedding(size=8))
    assert isinstance(cached, Embeddings)
    assert len(cached.embed_query("x")) == 8

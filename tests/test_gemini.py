"""
Gemini provider dispatch tests.

The facade must hand out the right implementation from the configured
providers, and constructing a model must never reach the network: building a
``ChatGoogleGenerativeAI`` or ``GoogleGenerativeAIEmbeddings`` with an explicit
key is config work, not an I/O call. Every factory is lru-cached, so the tests
clear the caches before each switch. None of these tests call the real Gemini
API; the live check lives in ``tests_live/``.
"""

import pytest

import src.llms.provider as provider
from src.core.config import settings
from src.core.exceptions import ConfigurationError


def _clear_caches():
    provider.get_llm.cache_clear()
    provider.get_answer_llm.cache_clear()
    provider.get_embeddings.cache_clear()


def _enable_gemini(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gem-test-key-not-real")


def test_gemini_provider_constructs_gemini_models(monkeypatch):
    from langchain_google_genai import (
        ChatGoogleGenerativeAI,
        GoogleGenerativeAIEmbeddings,
    )

    _enable_gemini(monkeypatch)
    _clear_caches()
    assert isinstance(provider.get_llm(), ChatGoogleGenerativeAI)
    assert isinstance(provider.get_answer_llm(), ChatGoogleGenerativeAI)
    embeddings = provider.get_embeddings()
    assert isinstance(embeddings, provider.CachedEmbeddings)
    assert isinstance(embeddings._embeddings, GoogleGenerativeAIEmbeddings)


def test_construction_never_reaches_the_network(monkeypatch):
    """A provider outage must be a request-time problem, not a boot problem."""
    _enable_gemini(monkeypatch)
    _clear_caches()
    provider.get_llm()
    provider.get_answer_llm()
    provider.get_embeddings()


def test_gemini_models_come_from_settings(monkeypatch):
    _enable_gemini(monkeypatch)
    monkeypatch.setattr(settings, "GEMINI_MODEL", "gemini-3.6-flash")
    monkeypatch.setattr(settings, "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001")
    _clear_caches()
    assert provider.get_llm().model == "models/gemini-3.6-flash"
    assert provider.get_embeddings()._embeddings.model == "models/gemini-embedding-001"


def test_mixed_providers_dispatch_independently(monkeypatch):
    """Chat and embeddings each follow their own provider setting."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_openai import OpenAIEmbeddings

    monkeypatch.setattr(settings, "LLM_PROVIDER", "gemini")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openai")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gem-test-key-not-real")
    _clear_caches()
    assert isinstance(provider.get_llm(), ChatGoogleGenerativeAI)
    assert isinstance(provider.get_embeddings()._embeddings, OpenAIEmbeddings)


def test_unknown_llm_provider_is_a_configuration_error(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "openai")
    _clear_caches()
    with pytest.raises(ConfigurationError, match="anthropic"):
        provider.get_llm()
    with pytest.raises(ConfigurationError, match="anthropic"):
        provider.get_answer_llm()


def test_unknown_embedding_provider_is_a_configuration_error(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "openai")
    monkeypatch.setattr(settings, "EMBEDDING_PROVIDER", "cohere")
    _clear_caches()
    with pytest.raises(ConfigurationError, match="cohere"):
        provider.get_embeddings()

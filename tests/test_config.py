"""
Tests for configuration validation.

A misconfigured deployment must fail loudly at startup rather than silently
defaulting a credential to an empty string and surfacing an opaque provider
error on the first user request.
"""

import pytest
from pydantic import ValidationError

from src.core.config import Settings

VALID_SECRET = "a-sufficiently-long-jwt-signing-secret-value-1234"


def _settings(**overrides):
    """Build Settings ignoring any .env file on disk."""
    base = {
        "OPENAI_API_KEY": "sk-real-looking-key",
        "JWT_SECRET_KEY": VALID_SECRET,
        "_env_file": None,
    }
    base.update(overrides)
    return Settings(**base)


def test_valid_configuration_loads():
    settings = _settings()
    assert settings.OPENAI_API_KEY == "sk-real-looking-key"
    assert settings.OPENAI_MODEL == "gpt-4o"


def test_missing_openai_key_is_rejected(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, JWT_SECRET_KEY=VALID_SECRET)


def test_empty_openai_key_is_rejected():
    with pytest.raises(ValidationError):
        _settings(OPENAI_API_KEY="")


def test_placeholder_openai_key_is_rejected():
    """A copied-but-unedited .env must not boot."""
    with pytest.raises(ValidationError):
        _settings(OPENAI_API_KEY="sk-your-openai-key-here")


def test_placeholder_jwt_secret_is_rejected():
    with pytest.raises(ValidationError):
        _settings(JWT_SECRET_KEY="change-me-generate-a-long-random-value")


def test_short_jwt_secret_is_rejected():
    with pytest.raises(ValidationError):
        _settings(JWT_SECRET_KEY="too-short")


def test_missing_jwt_secret_is_rejected(monkeypatch):
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, OPENAI_API_KEY="sk-real-looking-key")


# --- optional integrations -------------------------------------------------
def test_web_search_disabled_without_a_key():
    assert _settings(TAVILY_API_KEY="").web_search_enabled is False
    assert _settings(TAVILY_API_KEY="   ").web_search_enabled is False


def test_web_search_enabled_with_a_key():
    assert _settings(TAVILY_API_KEY="tvly-abc").web_search_enabled is True


def test_blank_mongo_url_means_not_configured():
    """An empty MONGODB_URL must not be treated as a connection string."""
    settings = _settings(MONGODB_URL="")
    assert settings.MONGODB_URL is None
    assert settings.persistence_enabled is False


def test_mongo_url_enables_persistence():
    settings = _settings(MONGODB_URL="mongodb://localhost:27017")
    assert settings.persistence_enabled is True


# --- bounds ----------------------------------------------------------------
@pytest.mark.parametrize(
    "field,value",
    [
        ("MAX_HISTORY_MESSAGES", 0),
        ("MAX_HISTORY_MESSAGES", 10_000),
        ("MAX_UPLOAD_BYTES", 0),
        ("MAX_REWRITE_ATTEMPTS", -1),
        ("MAX_REWRITE_ATTEMPTS", 99),
        ("AGENT_MAX_ITERATIONS", 0),
        ("RETRIEVER_TOP_K", 0),
        ("ACCESS_TOKEN_EXPIRE_MINUTES", 0),
    ],
)
def test_out_of_range_values_rejected(field, value):
    with pytest.raises(ValidationError):
        _settings(**{field: value})


def test_missing_prompt_key_names_the_available_prompts():
    from src.config.settings import Config

    with pytest.raises(KeyError) as exc:
        Config().prompt("no_such_prompt")
    assert "no_such_prompt" in str(exc.value)
    assert "classify_prompt" in str(exc.value)


def test_all_prompts_used_by_the_code_exist():
    """Guards against a prompt being renamed in YAML but not in code."""
    from src.config.settings import Config

    config = Config()
    for key in (
        "system_prompt",
        "classify_prompt",
        "grading_prompt",
        "rewrite_prompt",
        "generate_prompt",
        "verify_prompt",
    ):
        assert config.prompt(key).strip()


# --- vector store selection ------------------------------------------------
def test_faiss_is_the_default_vector_backend():
    settings = _settings()
    assert settings.qdrant_enabled is False
    assert settings.vector_backend == "faiss"


def test_blank_qdrant_url_means_not_configured():
    """An empty QDRANT_URL must not be treated as an endpoint."""
    settings = _settings(QDRANT_URL="", QDRANT_API_KEY="")
    assert settings.QDRANT_URL is None
    assert settings.QDRANT_API_KEY is None
    assert settings.vector_backend == "faiss"


def test_qdrant_url_selects_the_persistent_backend():
    settings = _settings(QDRANT_URL="http://localhost:6333")
    assert settings.qdrant_enabled is True
    assert settings.vector_backend == "qdrant"


def test_qdrant_collection_has_a_default():
    assert _settings().QDRANT_COLLECTION == "adaptive_rag_documents"

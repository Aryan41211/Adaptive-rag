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
    """A copied-but-unedited .env must not boot when OpenAI is selected."""
    with pytest.raises(ValidationError):
        _settings(OPENAI_API_KEY="sk-your-openai-key-here")


# --- Gemini provider ---------------------------------------------------------
def test_gemini_provider_without_a_key_is_rejected():
    with pytest.raises(ValidationError):
        _settings(LLM_PROVIDER="gemini")
    with pytest.raises(ValidationError):
        _settings(EMBEDDING_PROVIDER="gemini")


def test_placeholder_gemini_key_is_rejected():
    """A copied-but-unedited .env must not boot when Gemini is selected."""
    with pytest.raises(ValidationError):
        _settings(LLM_PROVIDER="gemini", GEMINI_API_KEY="your-gemini-api-key-here")


def test_gemini_provider_with_a_key_is_valid():
    settings = _settings(
        LLM_PROVIDER="gemini",
        EMBEDDING_PROVIDER="gemini",
        OPENAI_API_KEY="",
        GEMINI_API_KEY="gem-real-looking-key",
    )
    assert settings.GEMINI_MODEL == "gemini-3.6-flash"
    assert settings.GEMINI_EMBEDDING_MODEL == "gemini-embedding-001"


def test_ollama_and_openai_do_not_require_a_gemini_key():
    """No Gemini key may be demanded when Gemini is not selected."""
    _settings(LLM_PROVIDER="ollama", EMBEDDING_PROVIDER="ollama", OPENAI_API_KEY="")
    _settings(OPENAI_API_KEY="sk-real-looking-key")


def test_gemini_model_names_follow_the_provider():
    gemini = _settings(
        LLM_PROVIDER="gemini",
        EMBEDDING_PROVIDER="gemini",
        OPENAI_API_KEY="",
        GEMINI_API_KEY="gem-real-looking-key",
    )
    assert gemini.chat_model_name == "gemini-3.6-flash"
    assert gemini.embedding_model_name == "gemini-embedding-001"


# --- provider selection -----------------------------------------------------
def test_providers_default_to_openai():
    settings = _settings()
    assert settings.LLM_PROVIDER == "openai"
    assert settings.EMBEDDING_PROVIDER == "openai"
    assert settings.free_local_mode is False


def test_ollama_providers_allow_an_empty_api_key():
    """Free local mode must not demand an OpenAI key."""
    settings = _settings(
        OPENAI_API_KEY="", LLM_PROVIDER="ollama", EMBEDDING_PROVIDER="ollama"
    )
    assert settings.free_local_mode is True
    assert settings.OPENAI_API_KEY == ""


def test_ollama_settings_have_defaults():
    settings = _settings()
    assert settings.OLLAMA_BASE_URL == "http://localhost:11434"
    assert settings.OLLAMA_MODEL == "qwen2.5:7b"
    assert settings.OLLAMA_EMBEDDING_MODEL == "nomic-embed-text"


@pytest.mark.parametrize("field", ["LLM_PROVIDER", "EMBEDDING_PROVIDER"])
def test_unknown_provider_is_rejected(field):
    with pytest.raises(ValidationError):
        _settings(**{field: "anthropic"})


def test_active_model_names_follow_the_provider():
    openai = _settings()
    assert openai.chat_model_name == "gpt-4o"
    assert openai.embedding_model_name == "text-embedding-3-small"

    ollama = _settings(LLM_PROVIDER="ollama", EMBEDDING_PROVIDER="ollama")
    assert ollama.chat_model_name == "qwen2.5:7b"
    assert ollama.embedding_model_name == "nomic-embed-text"


def test_placeholder_jwt_secret_is_rejected():
    for placeholder in (
        "change-me-generate-a-long-random-value",
        "replace-with-a-long-random-secret",
    ):
        with pytest.raises(ValidationError):
            _settings(JWT_SECRET_KEY=placeholder)


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


# --- API schema exposure ----------------------------------------------------
def test_api_docs_are_served_by_default():
    """They are how the API is explored during development."""
    assert _settings().ENABLE_API_DOCS is True


@pytest.mark.parametrize("value", ["false", "0", "no", False])
def test_api_docs_can_be_disabled(value):
    assert _settings(ENABLE_API_DOCS=value).ENABLE_API_DOCS is False


@pytest.mark.parametrize("route", ["/docs", "/redoc", "/openapi.json"])
def test_disabling_docs_removes_the_route_entirely(route, monkeypatch):
    """
    Serving an empty page would still confirm the endpoint exists and leave
    the schema reachable by another name; the route must be absent.
    """
    import importlib

    import src.main
    from src.core.config import settings as live_settings

    monkeypatch.setattr(live_settings, "ENABLE_API_DOCS", False)
    try:
        rebuilt = importlib.reload(src.main)
        assert route not in {r.path for r in rebuilt.app.routes}
    finally:
        # Restore the module other tests hold a reference into.
        monkeypatch.undo()
        importlib.reload(src.main)


def test_docs_routes_are_present_when_enabled():
    from src.main import app

    paths = {r.path for r in app.routes}
    assert {"/docs", "/redoc", "/openapi.json"} <= paths


# --- secrets from files -----------------------------------------------------
def test_a_setting_can_be_supplied_as_a_file(tmp_path, monkeypatch):
    """
    The convention used by Docker secrets and Kubernetes secret volumes: a
    file named after the setting, containing only its value.
    """
    # The suite exports these for every other test; a file cannot be observed
    # while the environment still wins.
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    (tmp_path / "OPENAI_API_KEY").write_text("sk-from-a-secret-file", encoding="utf-8")
    (tmp_path / "JWT_SECRET_KEY").write_text(VALID_SECRET, encoding="utf-8")

    settings = Settings(_env_file=None, _secrets_dir=str(tmp_path))
    assert settings.OPENAI_API_KEY == "sk-from-a-secret-file"
    assert settings.JWT_SECRET_KEY == VALID_SECRET


def test_the_environment_wins_over_a_secret_file(tmp_path, monkeypatch):
    """An explicit override must not be silently ignored in favour of a file."""
    (tmp_path / "OPENAI_API_KEY").write_text("sk-from-the-file", encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-from-the-environment")

    settings = Settings(
        _env_file=None, _secrets_dir=str(tmp_path), JWT_SECRET_KEY=VALID_SECRET
    )
    assert settings.OPENAI_API_KEY == "sk-from-the-environment"


def test_a_secret_file_still_faces_validation(tmp_path, monkeypatch):
    """A weak secret in a file is no safer than one in the environment."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("JWT_SECRET_KEY", raising=False)

    (tmp_path / "OPENAI_API_KEY").write_text("sk-real-looking-key", encoding="utf-8")
    (tmp_path / "JWT_SECRET_KEY").write_text("too-short", encoding="utf-8")

    with pytest.raises(ValidationError):
        Settings(_env_file=None, _secrets_dir=str(tmp_path))


def test_a_missing_secrets_directory_is_not_an_error():
    """The default path does not exist off-container; startup must not care."""
    settings = Settings(
        _env_file=None,
        _secrets_dir=None,
        OPENAI_API_KEY="sk-real-looking-key",
        JWT_SECRET_KEY=VALID_SECRET,
    )
    assert settings.OPENAI_API_KEY == "sk-real-looking-key"

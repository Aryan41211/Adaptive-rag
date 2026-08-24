"""
Core application configuration.

Settings are loaded from the environment (and a local `.env` file) and
validated at import time so that a misconfigured deployment fails fast and
loudly, instead of surfacing as an opaque 401 from a downstream provider on
the first user request.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Placeholder values shipped in .env.example. Treated as "not configured" so a
# copied-but-unedited .env fails validation rather than booting insecurely.
_PLACEHOLDER_SECRETS = {
    "",
    "change-me-generate-a-long-random-value",
    "sk-your-openai-key-here",
}


class Settings(BaseSettings):
    """Validated application settings."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Required ---------------------------------------------------------
    OPENAI_API_KEY: str = Field(
        ...,
        description="OpenAI API key used for chat completions and embeddings.",
    )
    JWT_SECRET_KEY: str = Field(
        ...,
        description="Secret used to sign JWT access tokens.",
    )

    # --- Optional integrations -------------------------------------------
    TAVILY_API_KEY: str = ""
    MONGODB_URL: str | None = None
    MONGODB_DB_NAME: str = "adaptive_rag"

    # --- Vector store -----------------------------------------------------
    # When QDRANT_URL is set the vector store is persistent and shared across
    # processes. Otherwise an in-process FAISS index is used, which is lost on
    # restart and confines the service to a single worker.
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_COLLECTION: str = "adaptive_rag_documents"

    # --- Models -----------------------------------------------------------
    OPENAI_MODEL: str = "gpt-4o"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # --- Behaviour limits -------------------------------------------------
    MAX_HISTORY_MESSAGES: int = Field(default=20, ge=2, le=200)
    MAX_UPLOAD_BYTES: int = Field(default=10 * 1024 * 1024, ge=1024)
    MAX_QUERY_LENGTH: int = Field(default=4000, ge=1)
    MAX_REWRITE_ATTEMPTS: int = Field(default=2, ge=0, le=5)
    MAX_VERIFY_ATTEMPTS: int = Field(default=1, ge=0, le=5)
    AGENT_MAX_ITERATIONS: int = Field(default=5, ge=1, le=15)
    RETRIEVER_TOP_K: int = Field(default=4, ge=1, le=25)

    # --- Auth -------------------------------------------------------------
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=60, ge=1)
    JWT_ALGORITHM: str = "HS256"

    # --- Rate limiting ----------------------------------------------------
    # Bounds model spend per user. Counters are shared through MongoDB when
    # configured, so the limit is a whole-deployment limit rather than a
    # per-worker one.
    RATE_LIMIT_ENABLED: bool = True
    RATE_LIMIT_QUERY_PER_MINUTE: int = Field(default=20, ge=1)
    RATE_LIMIT_UPLOAD_PER_HOUR: int = Field(default=20, ge=1)
    RATE_LIMIT_AUTH_PER_MINUTE: int = Field(default=10, ge=1)

    # --- HTTP -------------------------------------------------------------
    # Comma-separated browser origins allowed to call the API. Empty means no
    # cross-origin access, which is correct for the server-rendered Streamlit
    # UI and wrong for a browser SPA.
    CORS_ALLOW_ORIGINS: str = ""
    # Comma-separated hostnames the API will answer to. "*" disables the check.
    ALLOWED_HOSTS: str = "*"

    # --- Tracing ----------------------------------------------------------
    # Tracing is off unless an endpoint is set and the OpenTelemetry packages
    # from requirements-tracing.txt are installed.
    OTEL_EXPORTER_OTLP_ENDPOINT: str | None = None
    OTEL_SERVICE_NAME: str = "adaptive-rag"
    # Keeps a collector outage from stalling shutdown or a request.
    OTEL_EXPORT_TIMEOUT_SECONDS: int = Field(default=5, ge=1, le=60)

    # --- Ops --------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    APP_VERSION: str = "1.0.0"

    @field_validator("OPENAI_API_KEY")
    @classmethod
    def _require_openai_key(cls, value: str) -> str:
        if value.strip() in _PLACEHOLDER_SECRETS:
            raise ValueError(
                "OPENAI_API_KEY is not set. Copy .env.example to .env and "
                "provide a real key."
            )
        return value.strip()

    @field_validator("JWT_SECRET_KEY")
    @classmethod
    def _require_strong_jwt_secret(cls, value: str) -> str:
        candidate = value.strip()
        if candidate in _PLACEHOLDER_SECRETS:
            raise ValueError(
                "JWT_SECRET_KEY is not set. Generate one with: "
                'python -c "import secrets; print(secrets.token_urlsafe(48))"'
            )
        if len(candidate) < 32:
            raise ValueError(
                "JWT_SECRET_KEY must be at least 32 characters to resist "
                "brute-force token forgery."
            )
        return candidate

    @field_validator(
        "MONGODB_URL",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "OTEL_EXPORTER_OTLP_ENDPOINT",
    )
    @classmethod
    def _blank_means_unset(cls, value: str | None) -> str | None:
        # An empty string in .env means "not configured", not "connect to ''".
        return value.strip() or None if value else None

    @property
    def web_search_enabled(self) -> bool:
        """True when a Tavily key is configured."""
        return bool(self.TAVILY_API_KEY.strip())

    @property
    def persistence_enabled(self) -> bool:
        """True when MongoDB is configured for durable storage."""
        return self.MONGODB_URL is not None

    @property
    def tracing_configured(self) -> bool:
        """True when a trace collector endpoint is configured."""
        return self.OTEL_EXPORTER_OTLP_ENDPOINT is not None

    @property
    def qdrant_enabled(self) -> bool:
        """True when a persistent, shared vector store is configured."""
        return self.QDRANT_URL is not None

    @property
    def cors_origins(self) -> list[str]:
        """Browser origins permitted to call the API."""
        return [
            origin.strip()
            for origin in self.CORS_ALLOW_ORIGINS.split(",")
            if origin.strip()
        ]

    @property
    def allowed_hosts(self) -> list[str]:
        """Hostnames the API will answer to."""
        hosts = [h.strip() for h in self.ALLOWED_HOSTS.split(",") if h.strip()]
        return hosts or ["*"]

    @property
    def vector_backend(self) -> str:
        """Name of the active vector store backend."""
        return "qdrant" if self.qdrant_enabled else "faiss"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached, validated settings instance."""
    return Settings()


settings = get_settings()

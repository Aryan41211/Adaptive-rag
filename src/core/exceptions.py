"""
Domain-level exceptions.

These carry an HTTP status so the API layer can translate them into clean
responses without leaking internal details or stack traces to the client.
"""


class AdaptiveRagError(Exception):
    """Base class for all application errors."""

    status_code: int = 500
    default_message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None):
        self.message = message or self.default_message
        super().__init__(self.message)


class ConfigurationError(AdaptiveRagError):
    """Raised when a required integration is not configured."""

    status_code = 503
    default_message = "The requested capability is not configured."


class DocumentProcessingError(AdaptiveRagError):
    """Raised when an uploaded document cannot be parsed or indexed."""

    status_code = 422
    default_message = "The uploaded document could not be processed."


class UnsupportedFileTypeError(DocumentProcessingError):
    """Raised when an uploaded file is not a supported type."""

    status_code = 415
    default_message = "Only PDF and TXT files are supported."


class FileTooLargeError(DocumentProcessingError):
    """Raised when an uploaded file exceeds the configured size limit."""

    status_code = 413
    default_message = "The uploaded file is too large."


class AuthenticationError(AdaptiveRagError):
    """Raised when credentials are missing or invalid."""

    status_code = 401
    default_message = "Invalid or missing credentials."


class UserAlreadyExistsError(AdaptiveRagError):
    """Raised when registering a username that is already taken."""

    status_code = 409
    default_message = "That username is already registered."


class RetrievalError(AdaptiveRagError):
    """Raised when the retrieval pipeline fails."""

    status_code = 502
    default_message = "The retrieval pipeline failed to produce an answer."


class IndexingError(AdaptiveRagError):
    """Raised when the embedding provider fails while indexing a document."""

    status_code = 502
    default_message = (
        "The document could not be indexed because the embedding service is "
        "unavailable. Please try again."
    )


class RateLimitExceededError(AdaptiveRagError):
    """Raised when a caller exceeds their request quota."""

    status_code = 429
    default_message = "Too many requests. Please slow down."

    def __init__(self, message: str | None = None, retry_after: int = 60):
        super().__init__(message)
        self.retry_after = retry_after


class TokenRevokedError(AuthenticationError):
    """Raised when a token has been explicitly signed out."""

    default_message = "This session has been signed out."


class DocumentNotFoundError(AdaptiveRagError):
    """Raised when a requested document is not in the caller's index."""

    status_code = 404
    default_message = "No such document."

"""
Application logging.

Provides a single ``configure_logging()`` entry point (called once at
application startup) and a ``get_logger()`` helper for modules.

Log records carry the request id when one is available, so a single user
request can be traced across the API, graph and retrieval layers.
"""

import logging
import sys
from contextvars import ContextVar

# Correlation id for the in-flight request, set by the API middleware.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")

_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | req=%(request_id)s | %(message)s"
)

_configured = False


class _RequestIdFilter(logging.Filter):
    """Inject the current request id into every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True


def configure_logging(level: str | None = None) -> None:
    """
    Configure root logging. Safe to call more than once.

    Args:
        level: Log level name. Defaults to the configured ``LOG_LEVEL``.
    """
    global _configured
    if _configured:
        return

    from src.core.config import settings

    resolved = (level or settings.LOG_LEVEL).upper()

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    handler.addFilter(_RequestIdFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, resolved, logging.INFO))

    # These libraries are extremely chatty at INFO and leak request payloads.
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-scoped logger.

    Args:
        name: Usually ``__name__``.

    Returns:
        A configured :class:`logging.Logger`.
    """
    return logging.getLogger(name)


# Backwards-compatible module-level logger.
logger = get_logger(__name__)

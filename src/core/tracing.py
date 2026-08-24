"""
Optional distributed tracing.

Structured logs say what happened; a trace says where the time went. A RAG
turn fans out into several model calls, a vector search and two database
round-trips, and without a trace an "it was slow" report has no answer.

Tracing is off unless ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set *and* the
OpenTelemetry packages are installed (``requirements-tracing.txt``). Neither
being present is a normal, supported configuration, so every import is guarded
and every failure degrades to running untraced rather than refusing to start.
"""

from contextlib import contextmanager
from typing import Any

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger(__name__)

_tracer: Any = None
_provider: Any = None
_configured = False


def tracing_available() -> bool:
    """
    Report whether the OpenTelemetry packages are importable.

    Returns:
        True if tracing could be enabled.
    """
    try:
        import opentelemetry.sdk.trace  # noqa: F401
    except ImportError:
        return False
    return True


def tracing_enabled() -> bool:
    """
    Report whether tracing is both configured and available.

    Returns:
        True if spans are being recorded.
    """
    return _tracer is not None


def configure_tracing(app: Any = None) -> bool:
    """
    Set up tracing, if it is configured and available.

    Safe to call more than once.

    Args:
        app: The FastAPI application to instrument, if any.

    Returns:
        True if tracing was enabled.
    """
    global _tracer, _provider, _configured

    if _configured:
        return tracing_enabled()
    _configured = True

    if not settings.OTEL_EXPORTER_OTLP_ENDPOINT:
        logger.debug("Tracing not configured; OTEL_EXPORTER_OTLP_ENDPOINT unset")
        return False

    if not tracing_available():
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but the OpenTelemetry packages "
            "are not installed. Install requirements-tracing.txt to enable "
            "tracing; continuing without it."
        )
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": settings.OTEL_SERVICE_NAME,
                    "service.version": settings.APP_VERSION,
                }
            )
        )
        # Bounded timeouts throughout: an unreachable collector must not be
        # able to stall a request or hold up shutdown. The SDK defaults retry
        # for up to 30 seconds, which turns a telemetry outage into what looks
        # like an application hang.
        endpoint = settings.OTEL_EXPORTER_OTLP_ENDPOINT.rstrip("/")
        exporter = OTLPSpanExporter(
            endpoint=endpoint + "/v1/traces",
            timeout=settings.OTEL_EXPORT_TIMEOUT_SECONDS,
        )
        provider.add_span_processor(
            BatchSpanProcessor(
                exporter,
                export_timeout_millis=settings.OTEL_EXPORT_TIMEOUT_SECONDS * 1000,
            )
        )
        # set_tracer_provider is a one-shot global: if anything else already
        # set one, this call is ignored with a warning. Take the tracer from
        # the provider we built rather than from the global, so our spans
        # always reach our exporter even when the global belongs to someone
        # else.
        trace.set_tracer_provider(provider)
        _provider = provider
        _tracer = provider.get_tracer("adaptive-rag")

        _instrument(app)

        logger.info(
            "Tracing enabled, exporting to %s",
            settings.OTEL_EXPORTER_OTLP_ENDPOINT,
        )
        return True
    except Exception as exc:  # noqa: BLE001 - never block startup on telemetry
        logger.error("Could not enable tracing, continuing without it: %s", exc)
        _tracer = None
        _provider = None
        return False


def _instrument(app: Any) -> None:
    """Attach the library instrumentations that are installed."""
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            # Health probes are polled constantly and would swamp the traces.
            FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz,readyz")
        except Exception as exc:  # noqa: BLE001
            logger.warning("FastAPI instrumentation unavailable: %s", exc)

    for name, module_path, attribute in (
        ("httpx", "opentelemetry.instrumentation.httpx", "HTTPXClientInstrumentor"),
        ("pymongo", "opentelemetry.instrumentation.pymongo", "PymongoInstrumentor"),
    ):
        try:
            module = __import__(module_path, fromlist=[attribute])
            getattr(module, attribute)().instrument()
        except Exception as exc:  # noqa: BLE001 - optional
            logger.debug("%s instrumentation unavailable: %s", name, exc)


@contextmanager
def span(name: str, **attributes: Any):
    """
    Record a span, or do nothing when tracing is disabled.

    Args:
        name: Span name.
        **attributes: Attributes to attach.

    Yields:
        The span, or None when tracing is disabled.
    """
    if _tracer is None:
        yield None
        return

    with _tracer.start_as_current_span(name) as current:
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
        yield current


def set_attributes(target: Any, **attributes: Any) -> None:
    """
    Attach attributes to a span, tolerating a disabled tracer.

    Args:
        target: The span returned by :func:`span`, which may be None.
        **attributes: Attributes to attach.
    """
    if target is None:
        return
    for key, value in attributes.items():
        if value is not None:
            target.set_attribute(key, value)


def annotate_current_span(**attributes: Any) -> None:
    """
    Attach attributes to whichever span is currently active.

    The FastAPI instrumentation already opens a span per request, so the
    pipeline enriches that rather than nesting another one covering almost
    exactly the same period. No-ops when tracing is disabled.

    Args:
        **attributes: Attributes to attach; None values are skipped.
    """
    if _tracer is None:
        return
    try:
        from opentelemetry import trace

        current = trace.get_current_span()
        if current is None or not current.is_recording():
            return
        for key, value in attributes.items():
            if value is not None:
                current.set_attribute(key, value)
    except Exception as exc:  # noqa: BLE001 - telemetry is never fatal
        logger.debug("Could not annotate span: %s", exc)


def shutdown() -> None:
    """Flush pending spans on application shutdown."""
    if _tracer is None:
        return
    try:
        # Flush the provider we created, not the global one, which may belong
        # to another library.
        provider = _provider
        # force_flush takes an explicit bound; shutdown() alone falls back to
        # the SDK default and can block for tens of seconds when the collector
        # is down.
        if hasattr(provider, "force_flush"):
            provider.force_flush(
                timeout_millis=settings.OTEL_EXPORT_TIMEOUT_SECONDS * 1000
            )
        if hasattr(provider, "shutdown"):
            provider.shutdown()
    except Exception as exc:  # noqa: BLE001
        logger.debug("Tracer shutdown failed: %s", exc)


def reset() -> None:
    """Forget the configured state. Intended for tests."""
    global _tracer, _provider, _configured
    _tracer = None
    _provider = None
    _configured = False

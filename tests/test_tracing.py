"""
Optional distributed tracing.

Tracing is an operational convenience, never a dependency. The behaviour that
matters most is what happens when it is unconfigured, uninstalled or broken:
in every case the application must run untraced rather than fail.
"""

import pytest

from src.core import tracing
from src.core.config import settings


@pytest.fixture(autouse=True)
def _reset_tracing():
    tracing.reset()
    yield
    tracing.reset()


# --- disabled by default ---------------------------------------------------
def test_tracing_is_off_without_an_endpoint():
    assert settings.OTEL_EXPORTER_OTLP_ENDPOINT is None
    assert tracing.configure_tracing() is False
    assert tracing.tracing_enabled() is False


def test_blank_endpoint_is_treated_as_unset():
    """An empty value in .env must not look like a configured collector."""
    from src.core.config import Settings

    configured = Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-real-looking-key",
        JWT_SECRET_KEY="a-sufficiently-long-jwt-signing-secret-value-1234",
        OTEL_EXPORTER_OTLP_ENDPOINT="",
    )
    assert configured.OTEL_EXPORTER_OTLP_ENDPOINT is None
    assert configured.tracing_configured is False


def test_endpoint_marks_tracing_configured():
    from src.core.config import Settings

    configured = Settings(
        _env_file=None,
        OPENAI_API_KEY="sk-real-looking-key",
        JWT_SECRET_KEY="a-sufficiently-long-jwt-signing-secret-value-1234",
        OTEL_EXPORTER_OTLP_ENDPOINT="http://collector:4318",
    )
    assert configured.tracing_configured is True


# --- the no-op path --------------------------------------------------------
def test_span_is_a_no_op_when_disabled():
    with tracing.span("anything", key="value") as current:
        assert current is None


def test_annotating_when_disabled_does_nothing():
    tracing.annotate_current_span(**{"rag.route": "index"})  # must not raise


def test_set_attributes_tolerates_a_missing_span():
    tracing.set_attributes(None, key="value")  # must not raise


def test_shutdown_when_disabled_does_nothing():
    tracing.shutdown()  # must not raise


def test_configure_is_idempotent():
    assert tracing.configure_tracing() is False
    assert tracing.configure_tracing() is False


# --- degradation -----------------------------------------------------------
def test_missing_packages_disable_tracing_rather_than_failing(monkeypatch):
    """A configured endpoint without the packages installed must not crash."""
    monkeypatch.setattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://c:4318")
    monkeypatch.setattr(tracing, "tracing_available", lambda: False)

    assert tracing.configure_tracing() is False
    assert tracing.tracing_enabled() is False


def test_a_broken_exporter_does_not_block_startup(monkeypatch):
    monkeypatch.setattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://c:4318")
    monkeypatch.setattr(tracing, "tracing_available", lambda: True)

    def explode(*_args, **_kwargs):
        raise RuntimeError("collector unreachable")

    monkeypatch.setattr(tracing, "_instrument", explode)

    assert tracing.configure_tracing() is False
    assert tracing.tracing_enabled() is False


def test_the_app_starts_with_a_configured_but_unreachable_collector(monkeypatch):
    """
    Startup must not depend on a telemetry backend being up.

    The exporter is swapped for an in-memory one so the test does not spend
    the export timeout retrying against a port with nothing behind it; what
    is under test is that startup proceeds, not the exporter's networking.
    """
    pytest.importorskip("opentelemetry.sdk.trace")

    import opentelemetry.exporter.otlp.proto.http.trace_exporter as otlp
    from fastapi.testclient import TestClient
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    monkeypatch.setattr(
        otlp, "OTLPSpanExporter", lambda **kwargs: InMemorySpanExporter()
    )
    monkeypatch.setattr(
        settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:59999"
    )
    tracing.reset()

    from src.main import app

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200

    tracing.shutdown()


# --- enabled path ----------------------------------------------------------
def test_spans_record_when_tracing_is_enabled(monkeypatch):
    """Exercises the real SDK with an in-memory exporter, not a mock."""
    pytest.importorskip("opentelemetry.sdk.trace")

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))

    with tracing.span("rag.turn", **{"rag.route": "index"}) as current:
        assert current is not None
        tracing.set_attributes(current, **{"rag.total_tokens": 120})

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["rag.turn"]
    assert spans[0].attributes["rag.route"] == "index"
    assert spans[0].attributes["rag.total_tokens"] == 120


def test_none_attributes_are_skipped(monkeypatch):
    """A None attribute is rejected by the SDK and would log a warning."""
    pytest.importorskip("opentelemetry.sdk.trace")

    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_tracer", provider.get_tracer("test"))

    with tracing.span("rag.turn", present="yes", absent=None):
        pass

    attributes = exporter.get_finished_spans()[0].attributes
    assert "present" in attributes
    assert "absent" not in attributes


# --- readiness reporting ---------------------------------------------------
def test_readyz_reports_tracing_state(client):
    assert client.get("/readyz").json()["tracing"] == "disabled"


# --- the enabled path ------------------------------------------------------
@pytest.fixture
def offline_tracing(monkeypatch):
    """
    Run the real configure path with an in-memory exporter.

    Constructing a genuine OTLP exporter here would make the suite depend on a
    collector being up: the batch processor retries on flush, so shutdown
    blocks for as long as the export timeout allows.
    """
    pytest.importorskip("opentelemetry.sdk.trace")

    import opentelemetry.exporter.otlp.proto.http.trace_exporter as otlp
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    monkeypatch.setattr(otlp, "OTLPSpanExporter", lambda **kwargs: exporter)
    monkeypatch.setattr(settings, "OTEL_EXPORTER_OTLP_ENDPOINT", "http://c:4318")
    tracing.reset()
    return exporter


def test_configure_enables_tracing_with_an_endpoint(offline_tracing):
    """The production wiring: real provider, real batch processor."""
    assert tracing.configure_tracing() is True
    assert tracing.tracing_enabled() is True

    with tracing.span("probe", **{"rag.route": "index"}) as current:
        assert current is not None

    tracing.shutdown()
    assert [s.name for s in offline_tracing.get_finished_spans()] == ["probe"]


def test_configure_instruments_the_app(offline_tracing):
    pytest.importorskip("opentelemetry.instrumentation.fastapi")
    from fastapi import FastAPI

    app = FastAPI()
    assert tracing.configure_tracing(app) is True
    # FastAPIInstrumentor marks the app once it has wrapped it.
    assert getattr(app, "_is_instrumented_by_opentelemetry", False) is True
    tracing.shutdown()


def test_instrumentation_failure_does_not_disable_tracing(offline_tracing):
    """A broken optional instrumentation must not take tracing down with it."""

    class _Unusable:
        def __getattr__(self, _name):
            raise RuntimeError("instrumentation broken")

    assert tracing.configure_tracing(_Unusable()) is True
    assert tracing.tracing_enabled() is True
    tracing.shutdown()


def test_export_timeout_is_bounded():
    """An unreachable collector must not stall shutdown for tens of seconds."""
    assert 1 <= settings.OTEL_EXPORT_TIMEOUT_SECONDS <= 60


def test_annotate_reaches_the_active_span(monkeypatch):
    pytest.importorskip("opentelemetry.sdk.trace")
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(tracing, "_tracer", tracer)

    with tracer.start_as_current_span("rag.turn"):
        tracing.annotate_current_span(
            **{"rag.route": "index", "rag.total_tokens": 512, "rag.absent": None}
        )

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes["rag.route"] == "index"
    assert attributes["rag.total_tokens"] == 512
    assert "rag.absent" not in attributes


async def test_a_traced_turn_records_its_cost(monkeypatch):
    """The attributes that make a slow or expensive request diagnosable."""
    pytest.importorskip("opentelemetry.sdk.trace")
    from langchain_core.messages import AIMessage
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from src.rag import graph_builder

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("test")
    monkeypatch.setattr(tracing, "_tracer", tracer)

    class _Builder:
        async def ainvoke(self, _state, config=None):
            return {
                "messages": [AIMessage(content="the answer")],
                "route": "index",
                "citations": [{"source": "a.pdf"}],
            }

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    with tracer.start_as_current_span("request"):
        await graph_builder.run_query("user-a", [])

    attributes = exporter.get_finished_spans()[0].attributes
    assert attributes["rag.route"] == "index"
    assert attributes["rag.citations"] == 1
    assert attributes["rag.streamed"] is False

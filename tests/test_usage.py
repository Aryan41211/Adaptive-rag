"""
Token and cost accounting.

A RAG turn makes several model calls, so per-request spend is not visible from
the outside. Accounting must be accurate enough to be useful and must never be
able to fail a request.
"""

from types import SimpleNamespace

import pytest

from src.core.usage import MODEL_PRICES, TOTALS, Usage, UsageTracker, estimate_cost


@pytest.fixture(autouse=True)
def _reset_totals():
    TOTALS.reset()
    yield
    TOTALS.reset()


def _response(model="gpt-4o", prompt=100, completion=50):
    return SimpleNamespace(
        llm_output={
            "model_name": model,
            "token_usage": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
            },
        },
        generations=[],
    )


# --- pricing ---------------------------------------------------------------
def test_known_model_is_priced():
    # gpt-4o: $2.50 per 1M input, $10.00 per 1M output.
    assert estimate_cost("gpt-4o", 1_000_000, 0) == pytest.approx(2.50)
    assert estimate_cost("gpt-4o", 0, 1_000_000) == pytest.approx(10.00)


def test_versioned_model_names_match_their_family():
    """Providers append dates; the price table should still apply."""
    assert estimate_cost("gpt-4o-2024-11-20", 1_000_000, 0) == pytest.approx(2.50)


def test_unknown_model_reports_zero_rather_than_guessing():
    assert estimate_cost("some-future-model", 1_000_000, 1_000_000) == 0.0


def test_every_price_entry_is_well_formed():
    for name, prices in MODEL_PRICES.items():
        assert len(prices) == 2, name
        assert all(price >= 0 for price in prices), name


# --- accumulation ----------------------------------------------------------
def test_usage_accumulates_across_calls():
    usage = Usage()
    usage.add("gpt-4o", 100, 50)
    usage.add("gpt-4o", 200, 100)

    assert usage.calls == 2
    assert usage.input_tokens == 300
    assert usage.output_tokens == 150
    assert usage.total_tokens == 450
    assert usage.cost_usd > 0


def test_usage_merges():
    first, second = Usage(), Usage()
    first.add("gpt-4o", 100, 50)
    second.add("gpt-4o", 100, 50)
    first.merge(second)

    assert first.calls == 2
    assert first.total_tokens == 300


def test_usage_serialises():
    usage = Usage()
    usage.add("gpt-4o", 1000, 500)
    body = usage.as_dict()

    assert set(body) == {
        "calls",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
    }
    assert body["total_tokens"] == 1500


# --- tracker ---------------------------------------------------------------
def test_tracker_records_a_completed_call():
    tracker = UsageTracker()
    tracker.on_llm_end(_response())

    assert tracker.usage.calls == 1
    assert tracker.usage.input_tokens == 100
    assert tracker.usage.output_tokens == 50


def test_tracker_covers_every_call_in_a_turn():
    """A turn makes several model calls; one call is not the turn's cost."""
    tracker = UsageTracker()
    for _ in range(4):
        tracker.on_llm_end(_response())

    assert tracker.usage.calls == 4
    assert tracker.usage.total_tokens == 600


def test_tracker_falls_back_to_generation_metadata():
    """Some providers report usage on the generation, not llm_output."""
    message = SimpleNamespace(usage_metadata={"input_tokens": 70, "output_tokens": 30})
    response = SimpleNamespace(
        llm_output={"model_name": "gpt-4o"},
        generations=[[SimpleNamespace(message=message)]],
    )

    tracker = UsageTracker()
    tracker.on_llm_end(response)

    assert tracker.usage.input_tokens == 70
    assert tracker.usage.output_tokens == 30


def test_malformed_response_does_not_raise():
    """Accounting must never be able to fail the request it is measuring."""
    tracker = UsageTracker()
    tracker.on_llm_end(SimpleNamespace(llm_output=None, generations=[]))
    tracker.on_llm_end(object())

    assert tracker.usage.calls >= 0  # no exception is the assertion


def test_finish_folds_into_the_process_totals():
    tracker = UsageTracker()
    tracker.on_llm_end(_response())
    usage = tracker.finish()

    snapshot = TOTALS.snapshot()
    assert usage.calls == 1
    assert snapshot["requests"] == 1
    assert snapshot["total_tokens"] == 150


def test_totals_accumulate_across_requests():
    for _ in range(3):
        tracker = UsageTracker()
        tracker.on_llm_end(_response())
        tracker.finish()

    snapshot = TOTALS.snapshot()
    assert snapshot["requests"] == 3
    assert snapshot["calls"] == 3
    assert snapshot["total_tokens"] == 450


# --- endpoint --------------------------------------------------------------
def test_metrics_endpoint_reports_totals(client, auth_headers):
    tracker = UsageTracker()
    tracker.on_llm_end(_response())
    tracker.finish()

    body = client.get("/metrics", headers=auth_headers).json()

    assert body["vector_backend"] == "faiss"
    assert body["usage"]["requests"] == 1
    assert body["usage"]["total_tokens"] == 150


def test_metrics_endpoint_requires_authentication(client):
    """Token counts and spend are operational detail, not public."""
    assert client.get("/metrics").status_code == 401


def test_query_response_includes_usage(client, auth_headers, monkeypatch):
    import src.api.routes as routes

    async def fake_run_query(user_id, messages):
        return "answer", [], {"calls": 3, "total_tokens": 450, "cost_usd": 0.001}

    monkeypatch.setattr(routes, "run_query", fake_run_query)

    body = client.post(
        "/rag/query",
        json={"query": "hello", "session_id": "s1"},
        headers=auth_headers,
    ).json()

    assert body["usage"]["calls"] == 3
    assert body["usage"]["total_tokens"] == 450

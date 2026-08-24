"""
The evaluation harness.

The harness itself is tested here; running it against a real provider is a
deliberate, paid action and is not part of this suite. What is verified is
that the dataset is valid, the scoring is correct, and the runner wires the
pipeline up properly — so that when a real run does happen, its numbers mean
something.
"""

import json

import pytest

from evals import dataset as dataset_module
from evals import report, runner
from evals.dataset import Case, Dataset, Document
from evals.metrics import CaseResult, score, summarise


def _case(**overrides) -> Case:
    defaults = {
        "id": "c1",
        "question": "q?",
        "expected_route": "index",
        "expected_source": "doc.txt",
        "must_include": [],
        "must_not_include": [],
    }
    defaults.update(overrides)
    return Case(**defaults)


def _result(**overrides) -> CaseResult:
    defaults = {
        "case_id": "c1",
        "question": "q?",
        "answer": "an answer",
        "route": "index",
        "citations": [{"source": "doc.txt"}],
    }
    defaults.update(overrides)
    return CaseResult(**defaults)


# --- the shipped dataset ---------------------------------------------------
def test_shipped_dataset_loads():
    data = dataset_module.load()
    assert len(data) > 0
    assert data.documents


def test_shipped_dataset_covers_every_route():
    routes = {case.expected_route for case in dataset_module.load().cases}
    assert "index" in routes
    assert "general" in routes


def test_shipped_dataset_includes_unanswerable_cases():
    """Fabrication is the failure mode that matters most; it must be measured."""
    cases = dataset_module.load().cases
    assert any(case.must_not_include for case in cases)


def test_shipped_dataset_facts_appear_in_their_documents():
    """
    A required fact absent from the source would be unpassable.

    This catches a dataset edited out of step with its documents.
    """
    data = dataset_module.load()
    contents = {doc.filename: doc.content.lower() for doc in data.documents}

    for case in data.cases:
        if not case.expected_source or not case.must_include:
            continue
        source = contents[case.expected_source]
        for fact in case.must_include:
            assert fact.lower() in source, f"{case.id}: '{fact}' not in its source"


# --- dataset validation ----------------------------------------------------
def test_dataset_rejects_an_unknown_route(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "cases:\n  - id: a\n    question: q\n    expected_route: sideways\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown route"):
        dataset_module.load(path)


def test_dataset_rejects_duplicate_ids(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text(
        "cases:\n"
        "  - id: a\n    question: q\n    expected_route: general\n"
        "  - id: a\n    question: r\n    expected_route: general\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Duplicate case id"):
        dataset_module.load(path)


def test_dataset_rejects_a_source_that_is_not_indexed(tmp_path):
    """Otherwise the case could never pass and the score would be misleading."""
    path = tmp_path / "bad.yaml"
    path.write_text(
        "cases:\n"
        "  - id: a\n    question: q\n    expected_route: index\n"
        "    expected_source: missing.txt\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="not one of the dataset's documents"):
        dataset_module.load(path)


def test_dataset_rejects_an_empty_file(tmp_path):
    path = tmp_path / "empty.yaml"
    path.write_text("documents: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no cases"):
        dataset_module.load(path)


# --- scoring ---------------------------------------------------------------
def test_a_fully_correct_case_passes():
    result = score(
        _case(must_include=["812"]),
        _result(answer="It orbits at 812 kilometres."),
    )
    assert result.passed
    assert result.routed_correctly
    assert result.retrieved_correctly


def test_wrong_route_fails():
    result = score(_case(), _result(route="general"))
    assert not result.routed_correctly
    assert not result.passed


def test_wrong_document_fails_retrieval():
    result = score(_case(), _result(citations=[{"source": "other.txt"}]))
    assert result.retrieved_correctly is False
    assert not result.passed


def test_missing_fact_fails():
    result = score(
        _case(must_include=["812"]), _result(answer="It orbits rather high.")
    )
    assert result.facts_missing == ["812"]
    assert not result.passed


def test_fact_matching_is_case_insensitive():
    result = score(
        _case(must_include=["Petrosyan"]), _result(answer="dr alina PETROSYAN")
    )
    assert result.facts_missing == []


def test_a_fabricated_fact_fails():
    """The failure mode for unanswerable questions is confident invention."""
    result = score(
        _case(must_not_include=["$"]),
        _result(answer="The launch cost about $50 million."),
    )
    assert result.forbidden_present == ["$"]
    assert not result.passed


def test_declining_to_answer_an_unanswerable_question_passes():
    result = score(
        _case(must_not_include=["$"]),
        _result(answer="The documents do not state the launch cost."),
    )
    assert result.passed


def test_an_errored_case_does_not_pass():
    result = score(_case(), _result(answer="", error="APIConnectionError"))
    assert not result.answered
    assert not result.passed


def test_retrieval_is_unscored_when_nothing_was_cited():
    """A routing failure must not also be counted as a retrieval failure."""
    result = score(_case(), _result(route="general", citations=[]))
    assert result.retrieved_correctly is None


# --- summarising -----------------------------------------------------------
def test_summary_counts_and_rates():
    results = [
        score(_case(id="a", must_include=["x"]), _result(case_id="a", answer="x")),
        score(_case(id="b", must_include=["y"]), _result(case_id="b", answer="no")),
    ]
    summary = summarise(results)

    assert summary.total == 2
    assert summary.passed == 1
    assert summary.pass_rate == 0.5
    assert summary.routing_accuracy == 1.0


def test_summary_counts_fabrications_separately():
    results = [
        score(
            _case(must_not_include=["$"]),
            _result(answer="about $50 million"),
        )
    ]
    assert summarise(results).hallucinated == 1


def test_summary_totals_usage():
    results = [
        _result(usage={"total_tokens": 100, "cost_usd": 0.01}),
        _result(usage={"total_tokens": 250, "cost_usd": 0.02}),
    ]
    summary = summarise(results)
    assert summary.total_tokens == 350
    assert summary.cost_usd == pytest.approx(0.03)


def test_rates_are_zero_rather_than_dividing_by_zero():
    summary = summarise([])
    assert summary.pass_rate == 0.0
    assert summary.retrieval_accuracy == 0.0
    assert summary.fact_accuracy == 0.0


# --- reporting -------------------------------------------------------------
def test_text_report_explains_failures():
    results = [
        score(
            _case(id="a", must_include=["812"]),
            _result(case_id="a", answer="unknown", route="general"),
        )
    ]
    text = report.render_text(results, summarise(results))

    assert "FAIL" in text
    assert "routed to 'general'" in text
    assert "'812'" in text


def test_text_report_shows_the_headline_metrics():
    results = [score(_case(), _result())]
    text = report.render_text(results, summarise(results))
    for label in ("Routing accuracy", "Retrieval accuracy", "Fact accuracy", "Tokens"):
        assert label in text


def test_json_report_round_trips(tmp_path):
    results = [score(_case(must_include=["812"]), _result(answer="812 km"))]
    path = report.write_json(results, summarise(results), tmp_path / "out.json")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["summary"]["total"] == 1
    assert payload["cases"][0]["status"] == "PASS"
    assert payload["cases"][0]["facts_found"] == ["812"]


# --- runner ----------------------------------------------------------------
async def test_runner_scores_a_full_dataset(monkeypatch, fake_embeddings):
    """
    Drives the runner end to end with the graph stubbed.

    Real retrieval quality needs a real model; what is proved here is that the
    runner indexes, routes each case through the pipeline, and scores what
    comes back.
    """
    data = Dataset(
        documents=[
            Document(
                filename="doc.txt",
                description="a test document",
                content="The altitude is 812 kilometres.",
            )
        ],
        cases=[
            _case(id="hit", must_include=["812"]),
            _case(id="miss", must_include=["999"]),
        ],
    )

    class _Builder:
        async def ainvoke(self, state, config=None):
            from langchain_core.messages import AIMessage

            return {
                "messages": [AIMessage(content="The altitude is 812 kilometres.")],
                "route": "index",
                "citations": [{"source": "doc.txt"}],
            }

    import src.rag.graph_builder as graph_builder

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    results, summary = await runner.run(data)

    assert [r.case_id for r in results] == ["hit", "miss"]
    assert summary.total == 2
    assert summary.passed == 1
    assert summary.routing_accuracy == 1.0


async def test_runner_records_a_failure_rather_than_stopping(
    monkeypatch, fake_embeddings
):
    """One broken case must not abandon the rest of the run."""
    data = Dataset(
        documents=[Document(filename="doc.txt", description="d", content="text")],
        cases=[_case(id="a"), _case(id="b")],
    )

    class _Builder:
        def __init__(self):
            self.calls = 0

        async def ainvoke(self, state, config=None):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("provider exploded")
            from langchain_core.messages import AIMessage

            return {
                "messages": [AIMessage(content="fine")],
                "route": "index",
                "citations": [{"source": "doc.txt"}],
            }

    import src.rag.graph_builder as graph_builder

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    results, summary = await runner.run(data)

    assert len(results) == 2
    assert "provider exploded" in results[0].error
    assert summary.errors == 1
    assert results[1].passed


async def test_runner_cleans_up_its_scratch_user(monkeypatch, fake_embeddings):
    """A run must not leave indexed documents behind."""
    from src.rag import vector_store

    data = Dataset(
        documents=[Document(filename="doc.txt", description="d", content="text")],
        cases=[_case(id="a")],
    )

    seen_users = []

    class _Builder:
        async def ainvoke(self, state, config=None):
            from langchain_core.messages import AIMessage

            seen_users.append(state["user_id"])
            return {"messages": [AIMessage(content="x")], "route": "index"}

    import src.rag.graph_builder as graph_builder

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    await runner.run(data)

    assert seen_users, "the runner never invoked the graph"
    assert vector_store.has_documents(seen_users[0]) is False


async def test_runner_reports_progress(monkeypatch, fake_embeddings):
    data = Dataset(
        documents=[Document(filename="doc.txt", description="d", content="text")],
        cases=[_case(id="a"), _case(id="b")],
    )

    class _Builder:
        async def ainvoke(self, state, config=None):
            from langchain_core.messages import AIMessage

            return {"messages": [AIMessage(content="x")], "route": "index"}

    import src.rag.graph_builder as graph_builder

    monkeypatch.setattr(graph_builder, "builder", _Builder())

    seen = []
    await runner.run(data, on_result=seen.append)
    assert [r.case_id for r in seen] == ["a", "b"]


# --- CLI -------------------------------------------------------------------
def test_cli_parses_its_arguments():
    from evals.__main__ import _parse_args

    args = _parse_args(["--fail-under", "0.8", "--quiet"])
    assert args.fail_under == 0.8
    assert args.quiet is True


def test_cli_reports_a_failure_cleanly(capsys):
    """A CLI should not traceback at the user."""
    from evals.__main__ import main

    assert main(["--dataset", "does-not-exist.yaml"]) == 2
    assert "Evaluation failed" in capsys.readouterr().err

"""
Evaluation reporting.

A run's value is in comparing it to the previous one, so the report is written
both as readable text and as JSON that can be diffed or tracked over time.
"""

import json
from pathlib import Path

from evals.metrics import CaseResult, Summary


def _status(result: CaseResult) -> str:
    if result.error:
        return "ERROR"
    return "PASS" if result.passed else "FAIL"


def _reasons(result: CaseResult) -> str:
    """Explain a failure in terms of what was expected."""
    if result.error:
        return result.error
    reasons = []
    if not result.answered:
        reasons.append("no answer")
    if not result.routed_correctly:
        reasons.append(f"routed to '{result.route}'")
    if result.retrieved_correctly is False:
        cited = (
            ", ".join(sorted({c.get("source", "?") for c in result.citations}))
            or "nothing"
        )
        reasons.append(f"cited {cited}")
    if result.facts_missing:
        reasons.append("missing " + ", ".join(repr(f) for f in result.facts_missing))
    if result.forbidden_present:
        reasons.append(
            "fabricated " + ", ".join(repr(f) for f in result.forbidden_present)
        )
    return "; ".join(reasons)


def render_text(results: list[CaseResult], summary: Summary) -> str:
    """
    Render a human-readable report.

    Args:
        results: Scored case results.
        summary: Their aggregate.

    Returns:
        The report text.
    """
    lines = ["", "=" * 72, "RAG EVALUATION", "=" * 72, ""]

    width = max((len(r.case_id) for r in results), default=8)
    for result in results:
        status = _status(result)
        marker = {"PASS": "PASS ", "FAIL": "FAIL ", "ERROR": "ERROR"}[status]
        lines.append(f"  {marker}  {result.case_id.ljust(width)}")
        if status != "PASS":
            lines.append(f"         {_reasons(result)}")

    metrics = summary.as_dict()
    lines += [
        "",
        "-" * 72,
        f"  Cases              {metrics['passed']}/{metrics['total']} passed "
        f"({metrics['pass_rate']:.0%})",
        f"  Routing accuracy   {metrics['routing_accuracy']:.0%}",
        f"  Retrieval accuracy {metrics['retrieval_accuracy']:.0%}"
        f"  (over {summary.retrieval_attempted} retrieval cases)",
        f"  Fact accuracy      {metrics['fact_accuracy']:.0%}"
        f"  (over {metrics['answered']} answered cases)",
        f"  Fabrications       {metrics['hallucinated']}",
        f"  Errors             {metrics['errors']}",
        f"  Tokens             {metrics['total_tokens']:,}"
        f"  (~${metrics['cost_usd']:.4f})",
        "-" * 72,
        "",
    ]
    return "\n".join(lines)


def write_json(results: list[CaseResult], summary: Summary, path: Path | str) -> Path:
    """
    Write the full run as JSON.

    Args:
        results: Scored case results.
        summary: Their aggregate.
        path: Destination file.

    Returns:
        The path written.
    """
    payload = {
        "summary": summary.as_dict(),
        "cases": [
            {
                "id": result.case_id,
                "question": result.question,
                "status": _status(result),
                "route": result.route,
                "answer": result.answer,
                "citations": result.citations,
                "facts_found": result.facts_found,
                "facts_missing": result.facts_missing,
                "fabricated": result.forbidden_present,
                "error": result.error,
            }
            for result in results
        ],
    }

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return destination

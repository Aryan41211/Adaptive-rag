"""
Scoring for evaluation cases.

Every metric here is deterministic and computed from the pipeline's own
output. There is deliberately no LLM judge: a judge would make the score
depend on a second model's mood, and would cost money on every run. The
trade-off is that these metrics measure whether the right facts are present,
not whether the prose is good.
"""

from dataclasses import dataclass, field

from evals.dataset import Case


@dataclass
class CaseResult:
    """How one case scored."""

    case_id: str
    question: str
    answer: str
    route: str
    citations: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    error: str = ""

    # Scores, filled in by `score`.
    routed_correctly: bool = False
    retrieved_correctly: bool | None = None
    facts_found: list[str] = field(default_factory=list)
    facts_missing: list[str] = field(default_factory=list)
    forbidden_present: list[str] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        """True when the turn produced an answer at all."""
        return bool(self.answer) and not self.error

    @property
    def facts_correct(self) -> bool:
        """True when every required fact appeared and no forbidden one did."""
        return not self.facts_missing and not self.forbidden_present

    @property
    def passed(self) -> bool:
        """True when routing, retrieval and facts are all correct."""
        return (
            self.answered
            and self.routed_correctly
            and self.retrieved_correctly is not False
            and self.facts_correct
        )


def score(case: Case, result: CaseResult) -> CaseResult:
    """
    Score one result against its case.

    Args:
        case: The expectation.
        result: The observed output, mutated in place with its scores.

    Returns:
        The same result, scored.
    """
    result.routed_correctly = result.route == case.expected_route

    if case.expected_source:
        sources = {citation.get("source") for citation in result.citations}
        # None rather than False when nothing was cited: on a general-knowledge
        # route there is nothing to retrieve, and scoring it as a retrieval
        # failure would misattribute a routing error.
        result.retrieved_correctly = (
            case.expected_source in sources if sources else None
        )

    haystack = result.answer.lower()
    result.facts_found = [
        fact for fact in case.must_include if fact.lower() in haystack
    ]
    result.facts_missing = [
        fact for fact in case.must_include if fact.lower() not in haystack
    ]
    result.forbidden_present = [
        fact for fact in case.must_not_include if fact.lower() in haystack
    ]
    return result


@dataclass
class Summary:
    """Aggregate scores across a run."""

    total: int = 0
    answered: int = 0
    passed: int = 0
    routed_correctly: int = 0
    retrieval_attempted: int = 0
    retrieved_correctly: int = 0
    facts_correct: int = 0
    hallucinated: int = 0
    errors: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    @property
    def pass_rate(self) -> float:
        """Fraction of cases that were fully correct."""
        return self._rate(self.passed, self.total)

    @property
    def routing_accuracy(self) -> float:
        """Fraction of cases sent down the right branch."""
        return self._rate(self.routed_correctly, self.total)

    @property
    def retrieval_accuracy(self) -> float:
        """Fraction of retrieval cases that cited the right document."""
        return self._rate(self.retrieved_correctly, self.retrieval_attempted)

    @property
    def fact_accuracy(self) -> float:
        """Fraction of answered cases containing the expected facts."""
        return self._rate(self.facts_correct, self.answered)

    def as_dict(self) -> dict:
        """Return a serialisable summary."""
        return {
            "total": self.total,
            "answered": self.answered,
            "passed": self.passed,
            "errors": self.errors,
            "pass_rate": self.pass_rate,
            "routing_accuracy": self.routing_accuracy,
            "retrieval_accuracy": self.retrieval_accuracy,
            "fact_accuracy": self.fact_accuracy,
            "hallucinated": self.hallucinated,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 4),
        }


def summarise(results: list[CaseResult]) -> Summary:
    """
    Aggregate scored results.

    Args:
        results: Scored case results.

    Returns:
        The aggregate summary.
    """
    summary = Summary(total=len(results))

    for result in results:
        if result.error:
            summary.errors += 1
        if result.answered:
            summary.answered += 1
        if result.routed_correctly:
            summary.routed_correctly += 1
        if result.retrieved_correctly is not None:
            summary.retrieval_attempted += 1
            if result.retrieved_correctly:
                summary.retrieved_correctly += 1
        if result.answered and result.facts_correct:
            summary.facts_correct += 1
        if result.forbidden_present:
            summary.hallucinated += 1
        if result.passed:
            summary.passed += 1

        summary.total_tokens += int(result.usage.get("total_tokens", 0))
        summary.cost_usd += float(result.usage.get("cost_usd", 0.0))

    return summary

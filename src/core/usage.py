"""
Token and cost accounting.

A RAG turn makes several model calls, so the cost of a request is not obvious
from the outside. This records tokens and estimated spend per request and in
running totals, which is what makes an unexpected bill diagnosable rather than
merely surprising.

Prices are per million tokens and are a configuration detail, not a fact: they
change, and they differ per model. ``MODEL_PRICES`` is a best-effort table and
an unknown model simply reports zero cost rather than guessing.
"""

import threading
from dataclasses import dataclass, field

from langchain_core.callbacks import BaseCallbackHandler

from src.core.logger import get_logger, request_id_var

logger = get_logger(__name__)

# USD per million tokens, (input, output). Update alongside provider pricing.
MODEL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """
    Estimate the cost of a call in USD.

    Args:
        model: The model name reported by the provider.
        input_tokens: Prompt tokens consumed.
        output_tokens: Completion tokens produced.

    Returns:
        The estimated cost, or 0.0 when the model has no known price.
    """
    for name, (prompt_price, completion_price) in MODEL_PRICES.items():
        if model.startswith(name):
            return (
                input_tokens * prompt_price + output_tokens * completion_price
            ) / 1_000_000
    return 0.0


@dataclass
class Usage:
    """Accumulated token counts and cost."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.input_tokens + self.output_tokens

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        """Record one model call."""
        self.calls += 1
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens
        self.cost_usd += estimate_cost(model, input_tokens, output_tokens)

    def merge(self, other: "Usage") -> None:
        """Fold another tally into this one."""
        self.calls += other.calls
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd

    def as_dict(self) -> dict:
        """Return a serialisable summary."""
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": round(self.cost_usd, 6),
        }


@dataclass
class _Totals:
    """Process-wide running totals."""

    lock: threading.Lock = field(default_factory=threading.Lock)
    usage: Usage = field(default_factory=Usage)
    requests: int = 0

    def record(self, usage: Usage) -> None:
        with self.lock:
            self.usage.merge(usage)
            self.requests += 1

    def snapshot(self) -> dict:
        with self.lock:
            return {"requests": self.requests, **self.usage.as_dict()}

    def reset(self) -> None:
        with self.lock:
            self.usage = Usage()
            self.requests = 0


TOTALS = _Totals()


class UsageTracker(BaseCallbackHandler):
    """
    LangChain callback accumulating token usage for one request.

    Attach it to a graph invocation; every model call underneath reports
    through it, so the tally covers the whole turn rather than a single call.
    """

    def __init__(self):
        self.usage = Usage()

    def on_llm_end(self, response, **kwargs) -> None:
        """Record the tokens reported by a completed model call."""
        try:
            output = response.llm_output or {}
            token_usage = output.get("token_usage") or {}
            model = output.get("model_name") or "unknown"

            input_tokens = int(token_usage.get("prompt_tokens", 0))
            output_tokens = int(token_usage.get("completion_tokens", 0))

            if not (input_tokens or output_tokens):
                # Some providers report usage on the generation instead.
                for generations in response.generations:
                    for generation in generations:
                        metadata = getattr(generation, "message", None)
                        usage = getattr(metadata, "usage_metadata", None) or {}
                        input_tokens += int(usage.get("input_tokens", 0))
                        output_tokens += int(usage.get("output_tokens", 0))

            self.usage.add(model, input_tokens, output_tokens)
        except Exception as exc:  # noqa: BLE001 - accounting is never fatal
            logger.debug("Could not record token usage: %s", exc)

    def finish(self) -> Usage:
        """
        Fold this request's usage into the process totals and log it.

        Returns:
            The usage for this request.
        """
        TOTALS.record(self.usage)
        if self.usage.calls:
            logger.info(
                "Request usage: %d model calls, %d tokens, ~$%.5f (req=%s)",
                self.usage.calls,
                self.usage.total_tokens,
                self.usage.cost_usd,
                request_id_var.get(),
            )
        return self.usage

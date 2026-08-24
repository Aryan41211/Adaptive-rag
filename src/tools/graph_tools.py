"""
Conditional-edge functions for the adaptive RAG graph.

These decide which node runs next. Every loop-forming branch is bounded by an
attempt counter carried in the graph state: without one, a persistently
negative relevance grade drives an unbounded retrieve/rewrite cycle that only
terminates at LangGraph's recursion limit, after many paid model calls.
"""

from typing import Literal

from langchain_core.prompts import PromptTemplate

from src.config.settings import Config
from src.core.config import settings
from src.core.logger import get_logger
from src.llms.openai import get_llm
from src.models.state import State
from src.models.verification_result import VerificationResult

logger = get_logger(__name__)

config = Config()


def routing_tool(state: State) -> Literal["retriever", "general_llm", "web_search"]:
    """
    Route to the node matching the classifier's decision.

    Args:
        state: The current graph state.

    Returns:
        The name of the next node.
    """
    route = state.get("route")
    if route == "index":
        return "retriever"
    if route == "general":
        return "general_llm"
    return "web_search"


def doc_tool(state: State) -> Literal["rewrite", "generate"]:
    """
    Decide whether to rewrite the query or generate the answer.

    Args:
        state: The current graph state.

    Returns:
        ``"generate"`` when the retrieved context was graded relevant or the
        rewrite budget is exhausted, otherwise ``"rewrite"``.
    """
    score = state.get("binary_score")
    attempts = state.get("rewrite_attempts", 0)

    if score == "yes":
        return "generate"

    if attempts >= settings.MAX_REWRITE_ATTEMPTS:
        logger.info(
            "Rewrite budget exhausted after %d attempts; generating from the "
            "best available context",
            attempts,
        )
        return "generate"

    return "rewrite"


def verify_answer(state: State) -> Literal["__end__", "generate"]:
    """
    Check the generated answer against the retrieved context.

    Compares the *answer* with the *context it was drawn from*. Bounded by
    ``MAX_VERIFY_ATTEMPTS`` so an unverifiable answer cannot loop forever.

    Args:
        state: The current graph state.

    Returns:
        ``"__end__"`` to finish, or ``"generate"`` to regenerate the answer.
    """
    # General-knowledge answers have no retrieved context to verify against.
    if state.get("route") == "general":
        return "__end__"

    context = (state.get("context") or "").strip()
    if not context:
        return "__end__"

    # generate_attempts counts answer generations; the first one is not a
    # retry, so retries performed so far is one fewer.
    retries_done = max(state.get("generate_attempts", 1) - 1, 0)
    if retries_done >= settings.MAX_VERIFY_ATTEMPTS:
        logger.info("Verification budget exhausted; returning current answer")
        return "__end__"

    messages = state.get("messages") or []
    if not messages:
        return "__end__"
    final_answer = str(messages[-1].content)

    verify_prompt = PromptTemplate(
        template=config.prompt("verify_prompt"),
        input_variables=["question", "context", "final_answer"],
    )
    chain = verify_prompt | get_llm().with_structured_output(VerificationResult)

    try:
        result = chain.invoke(
            {
                "question": state.get("latest_query", ""),
                "context": context,
                "final_answer": final_answer,
            }
        )
    except Exception as exc:  # noqa: BLE001 - verification is best-effort
        logger.warning("Answer verification failed, accepting answer: %s", exc)
        return "__end__"

    if result.faithful:
        return "__end__"

    logger.info("Answer judged unfaithful, regenerating: %s", result.explanation)
    return "generate"

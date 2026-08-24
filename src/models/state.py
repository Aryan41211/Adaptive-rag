"""
Graph state schema for the adaptive RAG system.
"""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages


class State(TypedDict, total=False):
    """
    State carried through the RAG graph.

    Attributes:
        messages: Conversation messages; new messages are appended.
        user_id: Owner of the request. Scopes retrieval to that user's
            private document index.
        latest_query: The (possibly rewritten) query driving retrieval.
        route: Classifier decision for the current turn.
        binary_score: Relevance grade of the retrieved context.
        context: The retrieved text the answer must be grounded in.
        citations: Provenance of the retrieved chunks, surfaced to the
            caller so an answer can be checked against its sources.
        rewrite_attempts: Number of query rewrites performed this turn.
        generate_attempts: Number of answer generations performed this turn.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    latest_query: str | None
    route: Literal["index", "general", "search"] | None
    binary_score: Literal["yes", "no"] | None
    context: str | None
    citations: list[dict]
    rewrite_attempts: int
    generate_attempts: int

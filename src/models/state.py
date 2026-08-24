"""
Graph state schema for the adaptive RAG system.
"""

from typing import Annotated, Literal, Optional, TypedDict

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
        rewrite_attempts: Number of query rewrites performed this turn.
        generate_attempts: Number of answer generations performed this turn.
    """

    messages: Annotated[list[AnyMessage], add_messages]
    user_id: str
    latest_query: Optional[str]
    route: Optional[Literal["index", "general", "search"]]
    binary_score: Optional[Literal["yes", "no"]]
    context: Optional[str]
    rewrite_attempts: int
    generate_attempts: int

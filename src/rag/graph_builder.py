"""
Adaptive RAG graph.

    START
      -> query_analysis            classify the turn
         -> retriever              answer from the user's own documents
            -> grade -> rewrite    bounded relevance-driven retry
                     -> generate
         -> web_search -> generate
         -> general_llm -> END
      generate -> verify -> END    bounded faithfulness check

Every node is scoped to ``state["user_id"]`` so retrieval only ever touches
that user's private index.
"""

import functools
import os
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langgraph.constants import END, START
from langgraph.graph.state import StateGraph

from src.config.settings import Config
from src.core.config import settings
from src.core.logger import get_logger
from src.llms.openai import get_answer_llm, get_llm
from src.models.grade import Grade
from src.models.route_identifier import RouteIdentifier
from src.models.state import State
from src.rag import vector_store
from src.rag.reAct_agent import get_agent_executor
from src.tools.graph_tools import doc_tool, routing_tool, verify_answer

logger = get_logger(__name__)

config = Config()

NO_DOCUMENTS_MESSAGE = (
    "I don't have any uploaded documents to search yet. "
    "Upload a PDF or TXT file and ask again."
)
WEB_SEARCH_UNAVAILABLE_MESSAGE = (
    "I can't look that up right now: web search is not configured on this deployment."
)


# Enough to identify the passage without returning the whole chunk.
CITATION_SNIPPET_CHARS = 240


def _citations_for(user_id: str, query: str) -> list[dict]:
    """
    Collect the provenance of the chunks a query matches.

    The agent's tool observations are plain strings, so source metadata is
    gathered with one direct retrieval. That is a vector search with no model
    call, which is negligible next to the several LLM calls in a turn.

    Args:
        user_id: The owning user.
        query: The query to attribute.

    Returns:
        Citation dictionaries, deduplicated by source and page.
    """
    retriever = vector_store.get_retriever(user_id)
    if retriever is None:
        return []

    try:
        documents = retriever.invoke(query)
    except Exception as exc:  # noqa: BLE001 - citations are best-effort
        logger.warning("Could not collect citations: %s", exc)
        return []

    citations: list[dict] = []
    seen: set[tuple] = set()
    for document in documents:
        metadata = document.metadata or {}
        page = metadata.get("page")
        key = (metadata.get("source_filename"), page)
        if key in seen:
            continue
        seen.add(key)
        citations.append(
            {
                "source": metadata.get("source_filename") or "uploaded document",
                "snippet": document.page_content[:CITATION_SNIPPET_CHARS].strip(),
                "page": (page + 1) if isinstance(page, int) else None,
            }
        )
    return citations


def _latest_question(state: State) -> str:
    """Return the text of the most recent user message."""
    messages = state.get("messages") or []
    return str(messages[-1].content) if messages else ""


@functools.lru_cache(maxsize=1)
def _get_search_tool():
    """Build the Tavily search tool once, if a key is configured."""
    from langchain_community.tools import TavilySearchResults

    # TavilySearchResults reads its credential from the environment.
    os.environ["TAVILY_API_KEY"] = settings.TAVILY_API_KEY
    return TavilySearchResults(max_results=5)


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------
def query_classifier(state: State) -> dict:
    """
    Decide how this turn should be answered.

    Retrieval runs only when the user actually has an index, which avoids
    building a throwaway index (and paying for its embeddings) on every query
    from a user who has uploaded nothing.

    Args:
        state: The current graph state.

    Returns:
        State update with ``route``, ``latest_query`` and reset counters.
    """
    user_id = state["user_id"]
    question = _latest_question(state)

    retriever = vector_store.get_retriever(user_id)
    if retriever is None:
        context = "(the user has not uploaded any documents)"
        has_documents = False
    else:
        documents = retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in documents)
        has_documents = True
        logger.debug("Retrieved %d candidate chunks for routing", len(documents))

    classify_prompt = PromptTemplate(
        template=config.prompt("classify_prompt"),
        input_variables=["question", "context"],
    )
    chain = classify_prompt | get_llm().with_structured_output(RouteIdentifier)

    try:
        route = chain.invoke({"question": question, "context": context}).route
    except Exception as exc:  # noqa: BLE001 - degrade to a safe default
        logger.warning("Query classification failed, defaulting to general: %s", exc)
        route = "general"

    # The classifier cannot legitimately choose "index" with no index.
    if route == "index" and not has_documents:
        route = "general"

    if route == "search" and not settings.web_search_enabled:
        logger.info("Web search unavailable; falling back to general knowledge")
        route = "general"

    logger.info("Routed query to '%s'", route)
    return {
        "route": route,
        "latest_query": question,
        "rewrite_attempts": 0,
        "generate_attempts": 0,
        "context": None,
        "binary_score": None,
        "citations": [],
    }


def general_llm(state: State) -> dict:
    """
    Answer from the model's general knowledge.

    Args:
        state: The current graph state.

    Returns:
        State update containing the answer message.
    """
    result = get_answer_llm().invoke(state["messages"])
    return {"messages": [result]}


def retriever_node(state: State) -> dict:
    """
    Answer from the user's own documents via the ReAct agent.

    Args:
        state: The current graph state.

    Returns:
        State update with the agent's answer and the context it used.
    """
    user_id = state["user_id"]
    query = state.get("latest_query") or _latest_question(state)

    executor = get_agent_executor(user_id)
    if executor is None:
        logger.info("Retriever requested but user has no indexed documents")
        return {
            "messages": [AIMessage(content=NO_DOCUMENTS_MESSAGE)],
            "context": "",
        }

    try:
        result = executor.invoke({"input": query})
    except Exception as exc:  # noqa: BLE001 - surfaced as a graded answer
        logger.exception("Retrieval agent failed: %s", exc)
        return {
            "messages": [
                AIMessage(
                    content="I couldn't search the documents just now. "
                    "Please try again."
                )
            ],
            "context": "",
        }

    output = str(result.get("output", "")).strip()

    # Observations from the tool calls are the evidence the answer must be
    # grounded in; keep them for grading and verification.
    observations = [
        str(observation)
        for _action, observation in result.get("intermediate_steps", [])
    ]
    context = "\n\n".join(observations) if observations else output

    tool_calls = [
        {"tool": action.tool, "input": action.tool_input}
        for action, _observation in result.get("intermediate_steps", [])
    ]

    return {
        "messages": [
            AIMessage(
                content=output or NO_DOCUMENTS_MESSAGE,
                additional_kwargs={"tool_calls": tool_calls},
            )
        ],
        "context": context,
        "citations": _citations_for(user_id, query),
    }


def grade(state: State) -> dict:
    """
    Grade whether the retrieved context answers the question.

    Args:
        state: The current graph state.

    Returns:
        State update with ``binary_score``.
    """
    context = state.get("context") or ""
    if not context.strip():
        return {"binary_score": "no"}

    grading_prompt = PromptTemplate(
        template=config.prompt("grading_prompt"),
        input_variables=["question", "context"],
    )
    chain = grading_prompt | get_llm().with_structured_output(Grade)

    try:
        score = chain.invoke(
            {"question": state.get("latest_query", ""), "context": context}
        ).binary_score
    except Exception as exc:  # noqa: BLE001 - assume relevant, don't loop
        logger.warning("Grading failed, treating context as relevant: %s", exc)
        score = "yes"

    logger.info("Context graded relevant=%s", score)
    return {"binary_score": score}


def rewrite_query(state: State) -> dict:
    """
    Rewrite the query to improve retrieval, counting the attempt.

    Args:
        state: The current graph state.

    Returns:
        State update with the rewritten query and incremented counter.
    """
    attempts = state.get("rewrite_attempts", 0) + 1
    rewrite_prompt = PromptTemplate(
        template=config.prompt("rewrite_prompt"),
        input_variables=["query"],
    )
    chain = rewrite_prompt | get_llm()

    try:
        rewritten = str(chain.invoke({"query": state["latest_query"]}).content)
    except Exception as exc:  # noqa: BLE001 - keep the original query
        logger.warning("Query rewrite failed, reusing original query: %s", exc)
        rewritten = state.get("latest_query", "")

    logger.info("Rewrote query (attempt %d)", attempts)
    return {"latest_query": rewritten, "rewrite_attempts": attempts}


def generate(state: State) -> dict:
    """
    Produce the user-facing answer from the gathered context.

    Args:
        state: The current graph state.

    Returns:
        State update with the answer and incremented generation counter.
    """
    attempts = state.get("generate_attempts", 0) + 1
    context = state.get("context") or ""

    if not context.strip():
        return {
            "messages": [AIMessage(content=NO_DOCUMENTS_MESSAGE)],
            "generate_attempts": attempts,
        }

    generate_prompt = PromptTemplate(
        template=config.prompt("generate_prompt"),
        input_variables=["context"],
    )
    chain = generate_prompt | get_answer_llm()

    try:
        answer = str(chain.invoke({"context": context}).content)
    except Exception as exc:  # noqa: BLE001 - fall back to raw context
        logger.exception("Answer generation failed: %s", exc)
        answer = context

    return {
        "messages": [AIMessage(content=answer)],
        "generate_attempts": attempts,
    }


def web_search(state: State) -> dict:
    """
    Search the web for the current query.

    Args:
        state: The current graph state.

    Returns:
        State update with the search results as context.
    """
    if not settings.web_search_enabled:
        return {
            "messages": [AIMessage(content=WEB_SEARCH_UNAVAILABLE_MESSAGE)],
            "context": "",
        }

    try:
        results = _get_search_tool().invoke(state.get("latest_query", ""))
    except Exception as exc:  # noqa: BLE001 - degrade, do not fail the request
        logger.exception("Web search failed: %s", exc)
        return {
            "messages": [AIMessage(content="Web search failed. Please try again.")],
            "context": "",
        }

    contents = [
        item["content"]
        for item in (results or [])
        if isinstance(item, dict) and item.get("content")
    ]
    logger.info("Web search returned %d results", len(contents))
    return {"context": "\n\n".join(contents)}


# --------------------------------------------------------------------------
# Graph wiring
# --------------------------------------------------------------------------
def build_graph():
    """
    Build and compile the adaptive RAG graph.

    Returns:
        The compiled LangGraph application.
    """
    graph = StateGraph(State)

    graph.add_node("query_analysis", query_classifier)
    graph.add_node("retriever", retriever_node)
    graph.add_node("grade", grade)
    graph.add_node("generate", generate)
    graph.add_node("rewrite", rewrite_query)
    graph.add_node("web_search", web_search)
    graph.add_node("general_llm", general_llm)

    graph.add_edge(START, "query_analysis")
    graph.add_conditional_edges(
        "query_analysis",
        routing_tool,
        {
            "retriever": "retriever",
            "general_llm": "general_llm",
            "web_search": "web_search",
        },
    )
    graph.add_edge("retriever", "grade")
    graph.add_conditional_edges(
        "grade", doc_tool, {"rewrite": "rewrite", "generate": "generate"}
    )
    graph.add_edge("rewrite", "retriever")
    graph.add_edge("web_search", "generate")
    # Verification can send the answer back for one bounded regeneration.
    graph.add_conditional_edges(
        "generate", verify_answer, {"__end__": END, "generate": "generate"}
    )
    graph.add_edge("general_llm", END)

    return graph.compile()


builder = build_graph()


async def run_query(user_id: str, messages: list) -> tuple[str, list[dict], dict]:
    """
    Run one turn of the graph and return the answer text.

    Args:
        user_id: The requesting user; scopes retrieval to their documents.
        messages: Conversation history, oldest first.

    Returns:
        The assistant's answer, the sources it was grounded in, and the token
        usage for the whole turn. The source list is empty for
        general-knowledge and web-search answers.

    Raises:
        RetrievalError: If the graph fails or produces no answer.
    """
    from langgraph.errors import GraphRecursionError
    from openai import OpenAIError

    from src.core.exceptions import RetrievalError
    from src.core.usage import UsageTracker

    # One tracker per turn: a turn makes several model calls and the cost of
    # any single one says little about the cost of the request.
    tracker = UsageTracker()

    try:
        result = await builder.ainvoke(
            {"messages": messages, "user_id": user_id},
            config={"recursion_limit": 25, "callbacks": [tracker]},
        )
    except GraphRecursionError as exc:
        tracker.finish()
        logger.error("Graph hit its recursion limit: %s", exc)
        raise RetrievalError(
            "The assistant could not converge on an answer. Please rephrase "
            "your question."
        ) from exc
    except OpenAIError as exc:
        tracker.finish()
        # An upstream provider failure is a 502, not an internal 500.
        logger.error("Model provider call failed: %s", exc)
        raise RetrievalError(
            "The language model service is unavailable. Please try again."
        ) from exc

    output: str | None = None
    if result.get("messages"):
        output = str(result["messages"][-1].content)

    usage = tracker.finish()

    if not output:
        raise RetrievalError("The assistant produced an empty answer.")
    return output, list(result.get("citations") or []), usage.as_dict()


# Nodes that produce the user-facing answer. Only their tokens are streamed;
# the classifier and grader also call models, but their output is structured
# routing data the user must never see.
ANSWER_NODES = {"generate", "general_llm"}


async def stream_query(user_id: str, messages: list) -> AsyncIterator[dict]:
    """
    Run one turn, yielding the answer as it is produced.

    Events are dictionaries with a ``type``:

    * ``token``    - a fragment of the answer
    * ``restart``  - discard the answer so far and start again, emitted when
                     verification rejects an answer and it is regenerated
    * ``citations``- the sources the answer was grounded in
    * ``usage``    - token counts and estimated cost for the turn
    * ``error``    - the turn failed; a message is included
    * ``done``     - the turn is complete, with the full answer text

    Args:
        user_id: The requesting user; scopes retrieval to their documents.
        messages: Conversation history, oldest first.

    Yields:
        Event dictionaries in the order above.
    """
    from langgraph.errors import GraphRecursionError
    from openai import OpenAIError

    from src.core.usage import UsageTracker

    tracker = UsageTracker()
    parts: list[str] = []
    final_state: dict = {}
    generate_starts = 0

    try:
        async for event in builder.astream_events(
            {"messages": messages, "user_id": user_id},
            version="v2",
            config={"recursion_limit": 25, "callbacks": [tracker]},
        ):
            kind = event.get("event")
            node = (event.get("metadata") or {}).get("langgraph_node")

            if kind == "on_chain_start" and event.get("name") == "generate":
                generate_starts += 1
                # A second entry into generate means verification rejected the
                # first answer. Anything already sent is now wrong.
                if generate_starts > 1 and parts:
                    parts.clear()
                    yield {"type": "restart"}

            elif kind == "on_chat_model_stream" and node in ANSWER_NODES:
                chunk = event.get("data", {}).get("chunk")
                text = str(getattr(chunk, "content", "") or "")
                if text:
                    parts.append(text)
                    yield {"type": "token", "text": text}

            elif kind == "on_chain_end" and not event.get("parent_ids"):
                output = event.get("data", {}).get("output")
                if isinstance(output, dict):
                    final_state = output

    except GraphRecursionError as exc:
        tracker.finish()
        logger.error("Graph hit its recursion limit: %s", exc)
        yield {
            "type": "error",
            "message": "The assistant could not converge on an answer. "
            "Please rephrase your question.",
        }
        return
    except OpenAIError as exc:
        tracker.finish()
        logger.error("Model provider call failed: %s", exc)
        yield {
            "type": "error",
            "message": "The language model service is unavailable. Please try again.",
        }
        return
    except Exception as exc:  # noqa: BLE001 - the client is mid-stream
        tracker.finish()
        logger.exception("Streaming turn failed: %s", exc)
        yield {"type": "error", "message": "The assistant failed to answer."}
        return

    answer = "".join(parts).strip()

    # Some paths produce an answer without a model call - the "no documents"
    # message, for instance - so nothing streamed. Send it in one piece.
    if not answer and final_state.get("messages"):
        answer = str(final_state["messages"][-1].content)
        if answer:
            yield {"type": "token", "text": answer}

    if not answer:
        yield {"type": "error", "message": "The assistant produced an empty answer."}
        return

    citations = list(final_state.get("citations") or [])
    if citations:
        yield {"type": "citations", "citations": citations}

    usage = tracker.finish().as_dict()
    yield {"type": "usage", "usage": usage}
    yield {"type": "done", "answer": answer, "citations": citations, "usage": usage}

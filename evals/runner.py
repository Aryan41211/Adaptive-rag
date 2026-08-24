"""
Evaluation runner.

Indexes the dataset's documents into a scratch user, runs every case through
the real pipeline, and scores the results. The scratch user is removed
afterwards, so a run never touches real user data.

Running this costs money: every case makes several model calls. It is not part
of the test suite for that reason, and is invoked deliberately.
"""

import uuid
from collections.abc import Callable

from langchain_core.messages import HumanMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter

from evals.dataset import Case, Dataset
from evals.metrics import CaseResult, Summary, score, summarise
from src.core.logger import get_logger
from src.rag import reAct_agent, vector_store

logger = get_logger(__name__)


def index_documents(dataset: Dataset, user_id: str) -> int:
    """
    Index the dataset's documents for a user.

    Args:
        dataset: The dataset whose documents to index.
        user_id: The scratch user to index them under.

    Returns:
        The total number of chunks indexed.
    """
    from langchain_core.documents import Document as LangChainDocument

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    total = 0

    for document in dataset.documents:
        chunks = splitter.split_documents(
            [
                LangChainDocument(
                    page_content=document.content,
                    metadata={"source_filename": document.filename},
                )
            ]
        )
        total = vector_store.add_documents(user_id, chunks, document.description)

    reAct_agent.reset_cache(user_id)
    logger.info("Indexed %d chunks for evaluation", total)
    return total


async def run_case(case: Case, user_id: str) -> CaseResult:
    """
    Run one case through the pipeline.

    Args:
        case: The case to run.
        user_id: The scratch user holding the indexed documents.

    Returns:
        The scored result.
    """
    from src.rag.graph_builder import builder

    result = CaseResult(case_id=case.id, question=case.question, answer="", route="")

    try:
        state = await builder.ainvoke(
            {
                "messages": [HumanMessage(content=case.question)],
                "user_id": user_id,
            },
            config={"recursion_limit": 25},
        )
    except Exception as exc:  # noqa: BLE001 - a failure is a result, not a stop
        result.error = f"{type(exc).__name__}: {exc}"
        return score(case, result)

    messages = state.get("messages") or []
    result.answer = str(messages[-1].content) if messages else ""
    result.route = str(state.get("route") or "")
    result.citations = list(state.get("citations") or [])
    return score(case, result)


async def run(
    dataset: Dataset,
    on_result: Callable[[CaseResult], None] | None = None,
) -> tuple[list[CaseResult], Summary]:
    """
    Run every case in a dataset and summarise the outcome.

    Args:
        dataset: The dataset to run.
        on_result: Optional callback invoked as each case completes, so a
            long run can report progress instead of appearing to hang.

    Returns:
        The individual results and their summary.
    """
    # A random scratch user keeps the run isolated from real data and from any
    # concurrent run.
    user_id = f"eval-{uuid.uuid4().hex[:12]}"
    results: list[CaseResult] = []

    try:
        index_documents(dataset, user_id)

        for case in dataset.cases:
            result = await run_case(case, user_id)
            results.append(result)
            if on_result:
                on_result(result)
    finally:
        vector_store.reset(user_id)
        reAct_agent.reset_cache(user_id)

    return results, summarise(results)

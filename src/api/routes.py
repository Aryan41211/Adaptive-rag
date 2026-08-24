"""
RAG API routes.

Both endpoints require authentication and operate strictly on the calling
user's own data: their conversation history and their private document index.
"""

from fastapi import APIRouter, Depends, File, Header, UploadFile
from fastapi.concurrency import run_in_threadpool
from langchain_core.messages import AIMessage, HumanMessage

from src.api.deps import CurrentUser, get_current_user
from src.api.ratelimit import query_rate_limit, upload_rate_limit
from src.core.config import settings
from src.core.logger import get_logger
from src.memory.chat_history_mongo import ChatHistory
from src.models.query_request import (
    Citation,
    DeletionResponse,
    DocumentListResponse,
    DocumentSummary,
    QueryRequest,
    QueryResponse,
    UploadResponse,
)
from src.rag import vector_store
from src.rag.document_upload import process_upload
from src.rag.graph_builder import run_query

logger = get_logger(__name__)

router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/query", response_model=QueryResponse)
async def rag_query(
    req: QueryRequest,
    user: CurrentUser = Depends(query_rate_limit()),
) -> QueryResponse:
    """
    Answer a question using the adaptive RAG pipeline.

    Args:
        req: The query and the conversation it belongs to.
        user: The authenticated caller.

    Returns:
        The assistant's answer.
    """
    history = ChatHistory.get_session_history(user.user_id, req.session_id)
    await history.add_message(HumanMessage(content=req.query))

    messages = await history.get_messages()

    # run_query awaits the graph, so long model calls never block the loop.
    answer, citations, usage = await run_query(user_id=user.user_id, messages=messages)

    await history.add_message(AIMessage(content=answer))

    return QueryResponse(
        answer=answer,
        session_id=req.session_id,
        citations=[Citation(**citation) for citation in citations],
        usage=usage,
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def clear_session(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
) -> None:
    """
    Delete the caller's conversation history for a session.

    Args:
        session_id: The conversation to clear.
        user: The authenticated caller.
    """
    await ChatHistory.get_session_history(user.user_id, session_id).clear()


@router.post("/documents/upload", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    description: str = Header(
        ...,
        alias="X-Description",
        min_length=1,
        max_length=300,
        description="Short description of the document being uploaded.",
    ),
    user: CurrentUser = Depends(upload_rate_limit()),
) -> UploadResponse:
    """
    Index a PDF or TXT document into the caller's private knowledge base.

    Args:
        file: The uploaded file.
        description: A short description of its contents.
        user: The authenticated caller.

    Returns:
        A summary of what was indexed.
    """
    # Reject oversized uploads before reading a byte, when the client
    # declared a length.
    if file.size is not None and file.size > settings.MAX_UPLOAD_BYTES:
        from src.core.exceptions import FileTooLargeError

        raise FileTooLargeError(
            f"File exceeds the "
            f"{settings.MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit."
        )

    # Parsing and embedding are blocking; run them off the event loop.
    result = await run_in_threadpool(
        process_upload,
        user_id=user.user_id,
        description=description,
        filename=file.filename,
        stream=file.file,
    )
    return UploadResponse(**result)


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents(
    user: CurrentUser = Depends(get_current_user),
) -> DocumentListResponse:
    """
    List the documents the caller has indexed.

    Args:
        user: The authenticated caller.

    Returns:
        One entry per source document, with its chunk count.
    """
    documents = await run_in_threadpool(vector_store.list_documents, user.user_id)
    return DocumentListResponse(
        documents=[DocumentSummary(**document) for document in documents],
        total_chunks=sum(document["chunks"] for document in documents),
    )


@router.delete("/documents/{filename:path}", response_model=DeletionResponse)
async def delete_document(
    filename: str,
    user: CurrentUser = Depends(get_current_user),
) -> DeletionResponse:
    """
    Remove one of the caller's documents from the index.

    Args:
        filename: The source filename to remove.
        user: The authenticated caller.

    Returns:
        How many chunks were removed.

    Raises:
        DocumentNotFoundError: If the caller has no such document.
    """
    removed = await run_in_threadpool(
        vector_store.delete_document, user.user_id, filename
    )
    if removed == 0:
        from src.core.exceptions import DocumentNotFoundError

        raise DocumentNotFoundError(f"No indexed document named '{filename}'.")

    # The cached agent is keyed on index version, which deletion advances.
    from src.rag.reAct_agent import reset_cache

    reset_cache(user.user_id)

    logger.info("Deleted document '%s' (%d chunks)", filename, removed)
    return DeletionResponse(filename=filename, chunks_deleted=removed)


@router.delete("/documents", response_model=DeletionResponse)
async def delete_all_documents(
    user: CurrentUser = Depends(get_current_user),
) -> DeletionResponse:
    """
    Remove every document the caller has indexed.

    Args:
        user: The authenticated caller.

    Returns:
        How many chunks were removed.
    """
    documents = await run_in_threadpool(vector_store.list_documents, user.user_id)
    total = sum(document["chunks"] for document in documents)

    await run_in_threadpool(vector_store.reset, user.user_id)

    from src.rag.reAct_agent import reset_cache

    reset_cache(user.user_id)

    logger.info("Cleared all documents (%d chunks)", total)
    return DeletionResponse(chunks_deleted=total)

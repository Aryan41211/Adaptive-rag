"""
Document ingestion.

Uploaded files are validated (extension, declared size, and actual magic
bytes), streamed to a temporary file under a size cap, parsed, chunked and
indexed into the *uploading user's* private vector store.

This module is synchronous and CPU/IO bound; the API layer runs it in a worker
thread so it never blocks the event loop.
"""

import os
import tempfile
from pathlib import Path
from typing import BinaryIO

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.core.config import settings
from src.core.exceptions import (
    DocumentProcessingError,
    FileTooLargeError,
    IndexingError,
    UnsupportedFileTypeError,
)
from src.core.logger import get_logger
from src.rag import vector_store
from src.tools.common_tools import enhance_description_with_llm

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
PDF_MAGIC = b"%PDF-"
_READ_CHUNK = 1024 * 1024  # 1 MiB


def _safe_extension(filename: str | None) -> str:
    """
    Extract a validated lowercase extension from an untrusted filename.

    Args:
        filename: The client-supplied filename.

    Returns:
        The extension, including the leading dot.

    Raises:
        UnsupportedFileTypeError: If the extension is missing or unsupported.
    """
    # Take the basename only: the client controls this string and it may
    # contain path separators.
    base = os.path.basename((filename or "").replace("\\", "/")).strip()
    extension = Path(base).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            "Only PDF and TXT files are supported "
            f"(received: {base or 'unnamed file'})."
        )
    return extension


def _stream_to_tempfile(source: BinaryIO, extension: str) -> str:
    """
    Copy an upload to a temporary file, enforcing the size limit.

    Args:
        source: The uploaded file's binary stream.
        extension: Validated file extension for the temp file suffix.

    Returns:
        Path to the temporary file.

    Raises:
        FileTooLargeError: If the stream exceeds ``MAX_UPLOAD_BYTES``.
    """
    limit = settings.MAX_UPLOAD_BYTES
    written = 0

    handle = tempfile.NamedTemporaryFile(delete=False, suffix=extension)
    try:
        while True:
            chunk = source.read(_READ_CHUNK)
            if not chunk:
                break
            written += len(chunk)
            if written > limit:
                raise FileTooLargeError(
                    f"File exceeds the {limit // (1024 * 1024)} MB upload limit."
                )
            handle.write(chunk)
    except BaseException:
        handle.close()
        os.unlink(handle.name)
        raise
    handle.close()

    if written == 0:
        os.unlink(handle.name)
        raise DocumentProcessingError("The uploaded file is empty.")

    return handle.name


def _verify_content(path: str, extension: str) -> None:
    """
    Confirm the file's actual bytes match its claimed type.

    A filename suffix is client-controlled and proves nothing.

    Args:
        path: Path to the temporary file.
        extension: The claimed extension.

    Raises:
        UnsupportedFileTypeError: If the content does not match the extension.
    """
    with open(path, "rb") as handle:
        header = handle.read(len(PDF_MAGIC))

    if extension == ".pdf":
        if header != PDF_MAGIC:
            raise UnsupportedFileTypeError(
                "File is named .pdf but its contents are not a valid PDF."
            )
        return

    # .txt: must decode as UTF-8 to be usable downstream.
    try:
        with open(path, "r", encoding="utf-8") as handle:
            handle.read(_READ_CHUNK)
    except UnicodeDecodeError as exc:
        raise UnsupportedFileTypeError(
            "Text files must be UTF-8 encoded."
        ) from exc


def process_upload(
    user_id: str,
    description: str,
    filename: str | None,
    stream: BinaryIO,
) -> dict:
    """
    Validate, parse and index an uploaded document for one user.

    Args:
        user_id: The uploading user; the document is private to them.
        description: User-supplied description of the document.
        filename: Client-supplied filename (used only for type detection).
        stream: The uploaded file's binary stream.

    Returns:
        A summary containing ``filename``, ``chunks_indexed``,
        ``total_chunks`` and the stored ``description``.

    Raises:
        UnsupportedFileTypeError: Unsupported or mismatched file type.
        FileTooLargeError: File exceeds the configured size limit.
        DocumentProcessingError: File could not be parsed or produced no text.
    """
    extension = _safe_extension(filename)
    safe_name = os.path.basename((filename or "document").replace("\\", "/"))

    temp_path = _stream_to_tempfile(stream, extension)
    try:
        _verify_content(temp_path, extension)

        loader = (
            PyPDFLoader(temp_path)
            if extension == ".pdf"
            else TextLoader(temp_path, encoding="utf-8")
        )
        try:
            docs = loader.load()
        except Exception as exc:  # noqa: BLE001 - normalised below
            logger.warning("Failed to parse upload %s: %s", safe_name, exc)
            raise DocumentProcessingError(
                "The document could not be parsed. It may be corrupt or "
                "password protected."
            ) from exc
    finally:
        os.unlink(temp_path)

    # Keep the original filename as provenance for future citation support.
    for doc in docs:
        doc.metadata["source_filename"] = safe_name

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )
    chunks = splitter.split_documents(docs)
    if not chunks:
        raise DocumentProcessingError(
            "No readable text was found in the document. Scanned PDFs "
            "without a text layer are not supported."
        )

    enhanced_description = enhance_description_with_llm(description)

    try:
        total = vector_store.add_documents(
            user_id=user_id, chunks=chunks, description=enhanced_description
        )
    except ValueError as exc:
        raise DocumentProcessingError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - upstream embedding failure
        # Distinguish "their service is down" (502) from "your file is bad"
        # (422) and from "we have a bug" (500).
        logger.exception("Embedding provider failed while indexing: %s", exc)
        raise IndexingError() from exc

    # Invalidate the cached agent so the new chunks are immediately searchable.
    from src.rag.reAct_agent import reset_cache

    reset_cache(user_id)

    logger.info(
        "Indexed upload '%s': %d chunks (user total %d)",
        safe_name,
        len(chunks),
        total,
    )
    return {
        "filename": safe_name,
        "chunks_indexed": len(chunks),
        "total_chunks": total,
        "description": enhanced_description,
    }

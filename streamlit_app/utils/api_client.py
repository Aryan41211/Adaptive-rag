"""
HTTP client for the Adaptive RAG API.

Talks only to the FastAPI backend. The previous version called an external
Rust auth service on port 8080 that does not exist in this repository, which
made login impossible.

Every request carries an explicit timeout: without one a stalled backend
leaves the UI hanging forever.
"""

import logging
import os
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")

# Auth is quick; a RAG turn runs several model calls; indexing embeds a whole
# document.
AUTH_TIMEOUT = 15
QUERY_TIMEOUT = 180
UPLOAD_TIMEOUT = 300


class ApiError(Exception):
    """Raised when the backend returns an error the user should see."""


def _auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def _detail(response: requests.Response) -> str:
    """Extract a human-readable message from an error response."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"

    # Field-level validation errors are more actionable than the generic
    # summary that accompanies them.
    if body.get("errors"):
        first = body["errors"][0]
        return f"{first.get('field', 'input')}: {first.get('message', 'invalid')}"

    detail = body.get("detail")
    if isinstance(detail, str):
        return detail
    return f"HTTP {response.status_code}"


def _post(
    path: str,
    timeout: int,
    *,
    json: Optional[dict] = None,
    files: Optional[dict] = None,
    headers: Optional[dict] = None,
) -> dict[str, Any]:
    """
    POST to the API and return the decoded body.

    Args:
        path: API path, e.g. ``/auth/login``.
        timeout: Request timeout in seconds.
        json: Optional JSON body.
        files: Optional multipart payload.
        headers: Optional extra headers.

    Returns:
        The decoded JSON response.

    Raises:
        ApiError: On connection failure, timeout or an error status.
    """
    url = f"{API_BASE_URL}{path}"
    try:
        response = requests.post(
            url, json=json, files=files, headers=headers, timeout=timeout
        )
    except requests.Timeout as exc:
        logger.warning("Timeout calling %s", url)
        raise ApiError("The server took too long to respond.") from exc
    except requests.RequestException as exc:
        logger.warning("Failed calling %s: %s", url, exc)
        raise ApiError(
            f"Could not reach the API at {API_BASE_URL}. Is the backend running?"
        ) from exc

    if not response.ok:
        raise ApiError(_detail(response))

    try:
        return response.json()
    except ValueError as exc:
        raise ApiError("The server returned an unreadable response.") from exc


def register(username: str, password: str) -> dict[str, Any]:
    """
    Create an account.

    Args:
        username: Desired username.
        password: Desired password.

    Returns:
        The token payload.

    Raises:
        ApiError: If registration is refused.
    """
    return _post(
        "/auth/register",
        AUTH_TIMEOUT,
        json={"username": username, "password": password},
    )


def login(username: str, password: str) -> dict[str, Any]:
    """
    Authenticate and obtain an access token.

    Args:
        username: The username.
        password: The password.

    Returns:
        The token payload.

    Raises:
        ApiError: If the credentials are rejected.
    """
    return _post(
        "/auth/login",
        AUTH_TIMEOUT,
        json={"username": username, "password": password},
    )


def query_backend(query: str, session_id: str, token: str) -> str:
    """
    Ask the RAG pipeline a question.

    Args:
        query: The user's question.
        session_id: The conversation identifier.
        token: The caller's access token.

    Returns:
        The assistant's answer.

    Raises:
        ApiError: If the request fails.
    """
    body = _post(
        "/rag/query",
        QUERY_TIMEOUT,
        json={"query": query, "session_id": session_id},
        headers=_auth_header(token),
    )
    return body["answer"]


def upload_document(file, description: str, token: str) -> dict[str, Any]:
    """
    Upload and index a document.

    Args:
        file: A Streamlit ``UploadedFile``.
        description: Short description of the document.
        token: The caller's access token.

    Returns:
        The upload summary.

    Raises:
        ApiError: If the upload is rejected.
    """
    return _post(
        "/rag/documents/upload",
        UPLOAD_TIMEOUT,
        files={"file": (file.name, file.getvalue(), file.type)},
        headers={**_auth_header(token), "X-Description": description},
    )


def api_available() -> bool:
    """
    Check whether the backend is reachable.

    Returns:
        True if the health endpoint responds.
    """
    try:
        return requests.get(f"{API_BASE_URL}/healthz", timeout=5).ok
    except requests.RequestException:
        return False

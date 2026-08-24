"""
Request and response models for the API.
"""

import re

from pydantic import BaseModel, Field, field_validator

# Session identifiers appear in database queries and logs; restrict them to a
# conservative character set.
_SESSION_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class QueryRequest(BaseModel):
    """A single RAG question."""

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's question.",
    )
    session_id: str = Field(
        ...,
        min_length=1,
        max_length=128,
        description="Conversation identifier, scoped to the calling user.",
    )

    @field_validator("query")
    @classmethod
    def _reject_blank_query(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Query must not be blank.")
        return stripped

    @field_validator("session_id")
    @classmethod
    def _validate_session_id(cls, value: str) -> str:
        candidate = value.strip()
        if not _SESSION_ID_PATTERN.fullmatch(candidate):
            raise ValueError(
                "session_id may only contain letters, digits, '.', '_', ':' and '-'."
            )
        return candidate


class Citation(BaseModel):
    """A document chunk an answer was drawn from."""

    source: str = Field(description="Filename the chunk came from.")
    snippet: str = Field(description="Excerpt of the chunk text.")
    page: int | None = Field(default=None, description="Page number, for PDF sources.")


class QueryResponse(BaseModel):
    """The assistant's answer to a query."""

    answer: str
    session_id: str
    citations: list[Citation] = Field(
        default_factory=list,
        description=(
            "Sources the answer was grounded in. Empty for answers from "
            "general knowledge or web search."
        ),
    )
    usage: dict = Field(
        default_factory=dict,
        description=(
            "Token counts and estimated cost for this turn, across every "
            "model call it made."
        ),
    )


class UploadResponse(BaseModel):
    """Result of a document upload."""

    filename: str
    chunks_indexed: int
    total_chunks: int
    description: str


class DocumentSummary(BaseModel):
    """One indexed source document."""

    filename: str
    chunks: int


class DocumentListResponse(BaseModel):
    """The documents a user has indexed."""

    documents: list[DocumentSummary] = Field(default_factory=list)
    total_chunks: int = 0


class DeletionResponse(BaseModel):
    """Result of removing indexed content."""

    filename: str | None = None
    chunks_deleted: int


class RegisterRequest(BaseModel):
    """New account details."""

    username: str = Field(..., min_length=3, max_length=64)
    password: str = Field(..., min_length=8, max_length=72)

    @field_validator("username")
    @classmethod
    def _validate_username(cls, value: str) -> str:
        candidate = value.strip()
        if not re.fullmatch(r"[A-Za-z0-9._-]{3,64}", candidate):
            raise ValueError(
                "Username may only contain letters, digits, '.', '_' and '-'."
            )
        return candidate


class LoginRequest(BaseModel):
    """Login credentials."""

    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=72)


class TokenResponse(BaseModel):
    """An issued access token."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int
    username: str

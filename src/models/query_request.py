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
                "session_id may only contain letters, digits, '.', '_', "
                "':' and '-'."
            )
        return candidate


class QueryResponse(BaseModel):
    """The assistant's answer to a query."""

    answer: str
    session_id: str


class UploadResponse(BaseModel):
    """Result of a document upload."""

    filename: str
    chunks_indexed: int
    total_chunks: int
    description: str


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
